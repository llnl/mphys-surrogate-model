"""
Model factory for composable encoder/decoder/dynamics architecture.
Allows mixing and matching different encoder types, decoder types, and dynamics models.
"""

import math

import torch
from torch import nn

from src import models, nwi


class ComposedModel(nn.Module):
    """
    Generic composed model with encoder, decoder, and optional dynamics.
    Provides consistent interface regardless of component choices.
    """

    def __init__(
        self,
        encoder,
        decoder,
        dynamics=None,
        dynamics_type="none",
        number_aware=False,
        rbin_median=None,
        water_density=1000.0,
        encoder_type=None,
        M_scale=1.0,
        N_scale=1.0,
        ratio_tolerance=1.0e-12,
    ):
        super(ComposedModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dzdt = dynamics  # Can be None for pure autoencoders
        self.dynamics_type = dynamics_type
        self._number_aware = number_aware
        self.encoder_type = encoder_type
        self.M_scale = M_scale
        self.N_scale = N_scale
        self.ratio_tolerance = ratio_tolerance
        if number_aware:
            if rbin_median is None:
                raise ValueError("number-aware models require rbin_median")
            radii = torch.as_tensor(rbin_median, dtype=torch.get_default_dtype())
            if radii.ndim != 1 or radii.numel() == 0 or torch.any(radii <= 0):
                raise ValueError("rbin_median must be a nonempty 1D array of positive radii")
            if radii.numel() != getattr(decoder, "n_bins", radii.numel()):
                raise ValueError("rbin_median length must match the decoder bin count")
            droplet_mass = (4.0 / 3.0) * torch.pi * water_density * radii**3
        else:
            droplet_mass = torch.empty(0, dtype=torch.get_default_dtype())
        self.register_buffer("droplet_mass", droplet_mass)

    @property
    def number_aware(self):
        return self._number_aware

    def _shape_coordinates(self, latent_mass):
        """Extract dimensionless shared spectral coordinates from ``[h, M]``."""
        if self.encoder_type == "nwi":
            mass = latent_mass[..., -1:].clamp_min(1.0e-12)
            return latent_mass[..., :-1] / mass
        return latent_mass[..., :-1]

    def _decoder_state(self, latent_with_number, total_scale):
        """Build ``[Z, total, log(M_physical/N_physical)]`` for a decoder."""
        latent_mass = latent_with_number[..., :-1]
        M = latent_mass[..., -1]
        N = latent_with_number[..., -1]
        ratio = torch.log(
            (M * self.M_scale).clamp_min(self.ratio_tolerance)
            / (N * self.N_scale).clamp_min(self.ratio_tolerance)
        ).unsqueeze(-1)
        shape = self._shape_coordinates(latent_mass)
        if self.encoder_type == "nwi":
            base = torch.cat(
                [shape * total_scale.unsqueeze(-1), total_scale.unsqueeze(-1)], dim=-1
            )
        else:
            base = torch.cat([shape, total_scale.unsqueeze(-1)], dim=-1)
        return torch.cat([base, ratio], dim=-1)

    def decode_mass(self, latent_with_number):
        """Decode mass from state ``[Z, M, N]``, conditioned on log(M/N)."""
        M = latent_with_number[..., -2]
        return self.decoder(self._decoder_state(latent_with_number, M))

    def diagnose_number_dsd(self, reconstructed_mass, tol=None):
        """Infer normalized number DSD from reconstructed mass-bin amounts.

        Each mass bin is divided by the mass of one water droplet at that bin's
        geometric-mean radius, then the resulting number weights are normalized.
        """
        if not self.number_aware:
            raise RuntimeError("This model was not created with number_aware=True")
        tol = self.ratio_tolerance if tol is None else tol
        mass = torch.clamp(reconstructed_mass, min=0.0)
        number_weights = mass / self.droplet_mass.to(dtype=mass.dtype)
        return number_weights / number_weights.sum(dim=-1, keepdim=True).clamp_min(tol)

    def reconstruct_number(self, latent_with_number):
        """Decode mass from ``[Z, M, N]`` and diagnose normalized number DSD."""
        if latent_with_number.shape[-1] < 3:
            raise ValueError("number-aware latent state must contain M and N")
        return self.diagnose_number_dsd(self.decode_mass(latent_with_number))

    def encode_with_number(self, mass_state, total_number):
        """Encode mass and append N, producing ``[coordinates, M, N]``."""
        latent = self.encoder(mass_state)
        if total_number is None:
            raise ValueError("total_number is required for number-aware models")
        return torch.cat([latent, total_number.unsqueeze(-1)], dim=-1)

    def forward(self, x, *args):
        """
        Forward pass depends on dynamics type.
        Encoder outputs Z = [z, M], decoder takes Z and outputs dimensioned DSD.

        - autoregressive: forward(x, M) -> (reconstructed x at t+1, predicted M)
        - sindy/nn_dzdt: forward(x, M) -> dZ/dt where Z = [z, M]
        - none: forward(x, N) -> reconstructed x for number-aware models
        """
        if self.dynamics_type == "autoregressive":
            # x is lagged dimensioned DSDs, args[0] is M (redundant but kept for compatibility)
            n_lag = getattr(self.dzdt, "n_lag", 1)
            # Encode each lag step
            Z_lags = []
            for t in range(n_lag):
                Z_lags.append(self.encoder(x[:, t, :]).unsqueeze(1))
            Z0 = torch.cat(Z_lags, dim=2)  # Concatenate lags: [batch, 1, n_lag*(n_latent+1)]

            # Predict next latent state
            Z1 = self.dzdt(Z0)

            # Decode to get next DSD
            x1 = self.decoder(Z1)
            M1 = Z1[:, :, -1]  # Extract predicted mass
            return x1, M1

        elif self.dynamics_type in ["sindy", "nn_dzdt"]:
            # x is current dimensioned DSD
            Z = self.encoder(x)
            if self.number_aware:
                if not args:
                    raise ValueError("number-aware dynamics require total_number")
                Z = torch.cat([Z, args[0].unsqueeze(-1)], dim=-1)
            dZdt = self.dzdt(Z)
            return dZdt

        elif self.dynamics_type == "none":
            # Pure autoencoder
            Z = self.encoder(x)
            if self.number_aware:
                if not args:
                    raise ValueError("number-aware autoencoders require total_number")
                Z = torch.cat([Z, args[0].unsqueeze(-1)], dim=-1)
                x_recon = self.decode_mass(Z)
            else:
                x_recon = self.decoder(Z)
            return x_recon

        else:
            raise ValueError(f"Unknown dynamics_type: {self.dynamics_type}")


# Registry of available components
ENCODER_REGISTRY = {
    "ffnn": models.FFNNEncoder,
    "nwi": nwi.LinearEncoder,
    # "cnn": None,  # Placeholder for future CNN encoder
}

DECODER_REGISTRY = {
    "ffnn": models.FFNNDecoder,
    "nwi_simple": nwi.SimpleDecoder,
    "nwi_deep": nwi.DeepDecoder,
}

DYNAMICS_REGISTRY = {
    "sindy": models.SINDyDeriv,
    "nn_dzdt": models.NNDerivatives,
    "autoregressive": models.Autoregressive,
    "none": None,
}


def create_encoder(encoder_type, n_bins, n_latent, params):
    """
    Create encoder based on type.

    Args:
        encoder_type: "ffnn", "nwi", or "cnn"
        n_bins: Number of distribution bins
        n_latent: Latent dimension
        params: Additional parameters (unused for most encoders)

    Returns:
        Encoder module
    """
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Available: {list(ENCODER_REGISTRY.keys())}")

    encoder_cls = ENCODER_REGISTRY[encoder_type]

    if encoder_type == "ffnn":
        return encoder_cls(n_bins=n_bins, n_latent=n_latent)
    elif encoder_type == "nwi":
        return encoder_cls(n_bins=n_bins, n_latent=n_latent)
    else:
        raise NotImplementedError(f"Encoder type {encoder_type} not yet implemented")


def create_decoder(decoder_type, n_bins, n_latent, params):
    """
    Create decoder based on type.

    Args:
        decoder_type: "ffnn", "nwi_simple", or "nwi_deep"
        n_bins: Number of distribution bins
        n_latent: Latent dimension
        params: Additional parameters (hidden_size, num_blocks for NWI)

    Returns:
        Decoder module
    """
    if decoder_type not in DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder_type: {decoder_type}. Available: {list(DECODER_REGISTRY.keys())}")

    decoder_cls = DECODER_REGISTRY[decoder_type]

    if decoder_type == "ffnn":
        return decoder_cls(
            n_bins=n_bins,
            n_latent=n_latent,
            distribution=True,
            condition_on_number=params.get("number_aware", False),
        )
    elif decoder_type in ["nwi_simple", "nwi_deep"]:
        hidden_size = params.get("hidden_size", 128)
        num_blocks = params.get("num_blocks", 3)
        return decoder_cls(
            n_bins=n_bins,
            n_latent=n_latent,
            hidden_features=hidden_size,
            num_blocks=num_blocks,
            condition_on_number=params.get("number_aware", False),
        )
    else:
        raise NotImplementedError(f"Decoder type {decoder_type} not yet implemented")


def create_dynamics(dynamics_type, n_latent, params):
    """
    Create dynamics model based on type.

    Args:
        dynamics_type: "sindy", "nn_dzdt", "autoregressive", or "none"
        n_latent: Latent dimension
        params: Additional parameters (poly_order for SINDy, layer_size for NN, n_lag for AR)

    Returns:
        Dynamics module or None
    """
    if dynamics_type not in DYNAMICS_REGISTRY:
        raise ValueError(f"Unknown dynamics_type: {dynamics_type}. Available: {list(DYNAMICS_REGISTRY.keys())}")

    if dynamics_type == "none":
        return None

    dynamics_cls = DYNAMICS_REGISTRY[dynamics_type]

    state_size = n_latent + 1 + int(params.get("number_aware", False))
    if dynamics_type == "sindy":
        poly_order = params.get("poly_order", 2)
        return dynamics_cls(
            n_latent=state_size,
            poly_order=poly_order,
            use_thresholds=False,
            nonneg=params['process'] == 'sedimentation'
        )
    elif dynamics_type == "nn_dzdt":
        layer_size = params.get("layer_size", (100, 100, 100))
        return dynamics_cls(
            n_latent=state_size,
            layer_size=layer_size,
            nonneg=params['process'] == 'sedimentation'
        )
    elif dynamics_type == "autoregressive":
        n_lag = params.get("n_lag", 1)
        layer_size = params.get("layer_size", (100, 100, 100))
        return dynamics_cls(
            n_bins=state_size,
            n_bins_in=state_size * n_lag,
            layer_size=layer_size,
        )
    else:
        raise NotImplementedError(f"Dynamics type {dynamics_type} not yet implemented")


def create_model(encoder_type, decoder_type, dynamics_type, params, n_bins):
    """
    Factory function to create a composed model from components.

    Args:
        encoder_type: "ffnn", "nwi", or "cnn"
        decoder_type: "ffnn", "nwi_simple", or "nwi_deep"
        dynamics_type: "sindy", "nn_dzdt", "autoregressive", or "none"
        params: Dict with model parameters (latent_dim, poly_order, layer_size, etc.)
        n_bins: Number of distribution bins

    Returns:
        ComposedModel with encoder, decoder, and dynamics components

    Notes:
        - For dynamics_type in ["sindy", "nn_dzdt"], forward() returns (dzdt, dMdt)
          where dMdt should be ~0 for mass-conserving processes like coalescence
        - For dynamics_type "autoregressive", forward() returns (bin1, dM)
        - For dynamics_type "none", forward() returns x_recon

    Example:
        # AE-SINDy with FFNN encoder
        model = create_model("ffnn", "ffnn", "sindy", params, 64)

        # AE-SINDy with NWI encoder
        model = create_model("nwi", "nwi_simple", "sindy", params, 64)

        # Pure NNWI autoencoder
        model = create_model("nwi", "nwi_simple", "none", params, 64)

        # AE-AR with FFNN
        model = create_model("ffnn", "ffnn", "autoregressive", params, 64)
    """
    n_latent = params["latent_dim"]
    number_aware = params.get("number_aware", False)
    if number_aware:
        missing_scales = [name for name in ("M_scale", "N_scale") if name not in params]
        if missing_scales:
            raise ValueError(
                "number-aware models require dataset normalization scale(s): "
                + ", ".join(missing_scales)
            )
        invalid_scales = [
            name
            for name in ("M_scale", "N_scale")
            if not math.isfinite(params[name]) or params[name] <= 0
        ]
        if invalid_scales:
            raise ValueError(
                "number-aware model scales must be positive and finite: "
                + ", ".join(invalid_scales)
            )

    # Create components
    encoder = create_encoder(encoder_type, n_bins, n_latent, params)
    decoder = create_decoder(decoder_type, n_bins, n_latent, params)
    dynamics = create_dynamics(dynamics_type, n_latent, params)

    # Compose into single model
    model = ComposedModel(
        encoder,
        decoder,
        dynamics,
        dynamics_type,
        number_aware=number_aware,
        rbin_median=params.get("rbin_median"),
        water_density=params.get("water_density", 1000.0),
        encoder_type=encoder_type,
        M_scale=params["M_scale"] if number_aware else params.get("M_scale", 1.0),
        N_scale=params["N_scale"] if number_aware else params.get("N_scale", 1.0),
        ratio_tolerance=params.get("tol", 1.0e-12),
    )

    return model
