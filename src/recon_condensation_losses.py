"""
Loss functions for condensation surrogate models.
Provides loss computation for different dynamics types and condensation targets (ERF vs theory).
"""

import torch

from src.condensation_utils import (
    compute_dhdt,
    compute_theoretical_condensation,
    eulerian_from_lagrangian,
)
from src.reconstruction_losses import (
    _dlnr_tensor,
    _reconstruction_loss_dict,
    _reconstruction_losses,
    _weighted_reconstruction_loss,
)


def _cond_target_form(params):
    """'eulerian' (default): compare the projected latent tendency against the
    Eulerian bin tendency, which is what compute_dhdt represents.
    'lagrangian': previous behaviour, compares against ERF's per-bin mass gain."""
    form = params.get("cond_target", "eulerian")
    if form not in ("eulerian", "lagrangian"):
        raise ValueError(f"cond_target must be 'eulerian' or 'lagrangian', got {form}")
    return form

mse = torch.nn.MSELoss()


def _project_condensation_tendency(model, latent, batch, params, device):
    """Project the theoretical latent condensation tendency into mass-bin space."""
    batch_temp = batch["temp"].reshape(-1, 1, 1)
    batch_sat = batch["rh"].reshape(-1, 1, 1)
    rbin_median = torch.as_tensor(
        params["rbin_median"], dtype=latent.dtype, device=device
    ).reshape(1, 1, -1)
    decoder = model.decode_mass if model.number_aware else model.decoder
    dhdt = compute_dhdt(
        latent,
        batch_temp,
        batch_sat,
        rbin_median,
        model.encoder,
        decoder,
        m_scale=params["M_scale"],
        use_kohler=False,
    )
    latent_for_jvp = latent.detach().requires_grad_(True)
    _, projected_tendency = torch.func.jvp(
        decoder, (latent_for_jvp,), (dhdt,)
    )
    return projected_tendency, rbin_median


def compute_condensation_erf_term(model, batch, latent, params, device):
    """Return only the unweighted projected-vs-ERF condensation tendency MSE.

    This term does not encode, decode a reconstruction for KL, apply a loss
    weight, or combine itself with reconstruction losses.

    Args:
        model: Composed model with encoder and mass decoder.
        batch: Dictionary batch from ``AllProcessesDataset``.
        latent: Previously computed state ``[Z, M]`` or ``[Z, M, N]``.
        params: Parameters containing ``rbin_median``, ``dlnr``, and ``M_scale``.
        device: Compute device.

    Returns:
        Scalar, unweighted condensation tendency MSE.
    """
    if "dgdt_cond" not in batch:
        raise ValueError("dgdt_cond is required for the ERF condensation term")
    dgdt_proj, rbin_median = _project_condensation_tendency(
        model, latent, batch, params, device
    )

    # The file stores dgdt per ln(r). Convert it to the decoder representation:
    # physical bin-mass tendency divided by M_scale.
    dlnr = _dlnr_tensor(params, dgdt_proj)
    batch_dgdt = batch["dgdt_cond"].unsqueeze(1) * dlnr / params["M_scale"]

    # ERF's dgdt_cond is Lagrangian (mass gain of droplets in each bin); the projected
    # tendency is Eulerian, so convert the target unless the old behaviour is requested.
    if _cond_target_form(params) == "eulerian":
        uniform_dlnr = float(
            torch.log(rbin_median[..., 1] / rbin_median[..., 0]).reshape(-1)[0]
        )
        batch_dgdt = eulerian_from_lagrangian(batch_dgdt, uniform_dlnr)
    return mse(dgdt_proj, batch_dgdt)


def compute_condensation_loss_erf(model, batch, params, device):
    """Combine reconstruction losses with the weighted ERF condensation term."""
    latent, _, recon_losses = _reconstruction_losses(model, batch, params)
    loss_recon = _weighted_reconstruction_loss(recon_losses, params)
    erf_weight = params.get("loss_weight_cond_dx", 0.0)

    # Allow reconstruction-only training without loading the ERF tendency.
    if erf_weight == 0.0:
        loss_cond_dx = loss_recon.new_zeros(())
    else:
        if "dgdt_cond" not in batch:
            raise ValueError(
                "dgdt_cond is required when loss_weight_cond_dx is nonzero"
            )
        loss_cond_dx = compute_condensation_erf_term(
            model, batch, latent, params, device
        )

    loss = loss_recon + erf_weight * loss_cond_dx
    loss_dict = {
        "total": loss,
        "cond_dx": erf_weight * loss_cond_dx,
    }
    loss_dict.update(_reconstruction_loss_dict(recon_losses, params))

    return loss, loss_dict


def compute_condensation_theory_term(
    model, batch, latent, reconstructed_mass, params, device
):
    """Return only the unweighted projected-vs-theory condensation MSE.

    The theoretical tendency is evaluated in physical units and converted back
    to the model's scaled bin-mass representation before comparison. This term
    does not compute or combine reconstruction losses and applies no weight.

    Args:
        model: Composed model with encoder and mass decoder.
        batch: Dictionary batch from ``AllProcessesDataset``.
        latent: Previously computed state ``[Z, M]`` or ``[Z, M, N]``.
        reconstructed_mass: Previously decoded scaled mass-bin amounts.
        params: Parameters containing ``rbin_median``, ``dlnr``, and ``M_scale``.
        device: Compute device.

    Returns:
        Scalar, unweighted condensation tendency MSE.
    """
    batch_temp = batch["temp"].reshape(-1, 1, 1)
    batch_sat = batch["rh"].reshape(-1, 1, 1)
    dgdt_proj, rbin_median = _project_condensation_tendency(
        model, latent, batch, params, device
    )

    # Evaluate theory in physical units. The model reconstructs scaled bin
    # mass, x = dmdlnr * dlnr / M_scale. Condensation theory requires
    # physical dmdlnr [kg m-3 per ln(r)], so convert before evaluating it and
    # convert the result back to the decoder/JVP representation afterwards.
    dlnr = _dlnr_tensor(params, reconstructed_mass)
    dmdlnr_physical = reconstructed_mass * params["M_scale"] / dlnr
    dgdt_theory_physical = compute_theoretical_condensation(
        dmdlnr_physical,
        batch_temp,
        batch_sat,
        rbin_median,
        use_kohler=False,
        form=_cond_target_form(params),
        scheme=params.get("cond_theory_scheme", "central"),
    )
    dgdt_theory = dgdt_theory_physical * dlnr / params["M_scale"]
    return mse(dgdt_proj, dgdt_theory)


def compute_condensation_loss_theory(model, batch, params, device):
    """Combine reconstruction losses with the weighted theory condensation term."""
    latent, reconstructed_mass, recon_losses = _reconstruction_losses(
        model, batch, params
    )
    loss_recon = _weighted_reconstruction_loss(recon_losses, params)
    theory_weight = params.get("loss_weight_cond_dx", 0.0)

    if theory_weight == 0.0:
        loss_cond_dx = loss_recon.new_zeros(())
    else:
        loss_cond_dx = compute_condensation_theory_term(
            model, batch, latent, reconstructed_mass, params, device
        )

    loss = loss_recon + theory_weight * loss_cond_dx
    loss_dict = {
        "total": loss,
        "cond_dx": theory_weight * loss_cond_dx,
    }
    loss_dict.update(_reconstruction_loss_dict(recon_losses, params))

    return loss, loss_dict


def compute_autoencoder_loss(model, batch, params, device):
    """
    Loss for pure autoencoder (no dynamics).

    Args:
        model: ComposedModel with encoder and decoder only
        batch: Dictionary batch from ``AllProcessesDataset``.
        params: Parameters dict with loss weights
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    _, _, recon_losses = _reconstruction_losses(model, batch, params)
    loss = _weighted_reconstruction_loss(recon_losses, params)
    loss_dict = {"total": loss, **_reconstruction_loss_dict(recon_losses, params)}

    return loss, loss_dict


def get_loss_function(cond_loss_type="erf"):
    """
    Get the appropriate loss function for condensation models.

    Note: Condensation models train only the autoencoder (encoder + decoder).
    There is no separate dynamics model. The condensation tendency is either
    from ERF data or computed theoretically, then encoded and projected through
    the decoder to ensure the autoencoder preserves condensation information.

    Args:
        cond_loss_type: "erf" (train against ERF data) or "theory" (train against theoretical condensation)
                       or "none" (pure autoencoder, no condensation penalty)

    Returns:
        Loss function callable
    """
    if cond_loss_type == "erf":
        return compute_condensation_loss_erf
    elif cond_loss_type == "theory":
        return compute_condensation_loss_theory
    elif cond_loss_type == "none":
        return compute_autoencoder_loss
    else:
        raise ValueError(
            f"Invalid cond_loss_type: {cond_loss_type}. Must be 'erf', 'theory', or 'none'"
        )
