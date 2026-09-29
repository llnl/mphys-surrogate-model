"""
Loss functions for condensation surrogate models.
Provides loss computation for different dynamics types and condensation targets (ERF vs theory).
"""

import torch

from src.condensation_utils import compute_theoretical_condensation, compute_dhdt

# Global loss functions
mse = torch.nn.MSELoss()
divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)


def compute_condensation_loss_erf(model, batch, params, device):
    """
    Loss for condensation-aware autoencoder trained against ERF ground truth.

    No dynamics model is trained. Instead, we compute dhdt from thermodynamics
    using the dhdt function (condensation velocity + NWI prefactor), project it
    through the decoder, and compare to ERF dgdt_cond.

    Args:
        model: ComposedModel with encoder and decoder (no dynamics)
        batch: Tuple of (batch_x, batch_dgdt, batch_temp, batch_sat, batch_m)
               - batch_x: dimensioned DSD [batch, 1, n_bins]
               - batch_dgdt: target condensation tendency from ERF [batch, 1, n_bins]
               - batch_temp: temperature [batch, 1, 1] in Kelvin
               - batch_sat: saturation ratio [batch, 1, 1]
               - batch_m: total mass [batch, 1, 1]
        params: Parameters dict with loss weights and 'rbin_median'
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_x, batch_dgdt, batch_temp, batch_sat, batch_m = batch

    # 1. Encode DSD to latent space
    Z = model.encoder(batch_x)

    # 2. Reconstruction loss (in normalized DSD space)
    pred_x_recon = model.decoder(Z)
    loss_recon = divergence(
        torch.log(pred_x_recon / (pred_x_recon.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
        torch.log(batch_x / (batch_x.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
    )

    # 3. Compute dhdt from thermodynamics using the dhdt function
    # This uses: condensation velocity, reconstructed DSD (de-normalized), and NWI prefactor
    rbin_median = torch.from_numpy(params["rbin_median"]).to(device).float().reshape(1, 1, -1)
    m_scale = params["m_scale"]
    dhdt_computed = compute_dhdt(
        Z, batch_temp, batch_sat, rbin_median,
        model.encoder, model.decoder, m_scale=m_scale, use_kohler=False
    )

    # 4. Project dhdt to physical space using JVP
    Z_detached = Z.clone().detach().requires_grad_(True)
    _, dgdt_proj = torch.func.jvp(model.decoder, (Z_detached,), (dhdt_computed,))

    # 5. Scale dgdt_proj to match batch_dgdt scaling
    # dgdt_proj is scaled by m_scale (same as decoder output g)
    # batch_dgdt is scaled by cond_scale
    # Convert: (physical / m_scale) * (m_scale / cond_scale) = (physical / cond_scale)
    m_scale = params["m_scale"]
    cond_scale = params["cond_scale"]
    dgdt_proj_scaled = dgdt_proj * m_scale / cond_scale

    # 6. Condensation tendency loss (projected vs ERF ground truth, both scaled by cond_scale)
    loss_cond_dx = mse(dgdt_proj_scaled, batch_dgdt)

    # Weighted total loss
    loss = (
        params["loss_weight_recon"] * loss_recon
        + params["loss_weight_cond_dx"] * loss_cond_dx
    )

    loss_dict = {
        "total": loss,
        "recon": params["loss_weight_recon"] * loss_recon,
        "cond_dx": params["loss_weight_cond_dx"] * loss_cond_dx,
    }

    return loss, loss_dict


def compute_condensation_loss_theory(model, batch, params, device):
    """
    Loss for condensation-aware autoencoder trained against theoretical condensation.

    No dynamics model is trained. Instead, we compute dhdt from thermodynamics
    using the dhdt function, project it through the decoder, and compare to
    theoretical dgdt computed from the reconstructed DSD.

    IMPORTANT: Theoretical condensation must be computed in PHYSICAL (de-normalized) space,
    not normalized space, because the thermodynamic equations require actual kg/m³ units.

    Args:
        model: ComposedModel with encoder and decoder (no dynamics)
        batch: Tuple of (batch_x, batch_dgdt, batch_temp, batch_sat, batch_m)
               - batch_x: normalized DSD [batch, 1, n_bins] (divided by m_scale)
               - batch_dgdt: target condensation tendency (not used in theory variant)
               - batch_temp: temperature [batch, 1, 1] in Kelvin
               - batch_sat: saturation ratio [batch, 1, 1] (dimensionless)
               - batch_m: normalized total mass [batch, 1, 1] (divided by m_scale)
        params: Parameters dict with loss weights, 'rbin_median', 'm_scale', 'cond_scale'
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_x, batch_dgdt, batch_temp, batch_sat, batch_m = batch

    # 1. Encode DSD to latent space
    Z = model.encoder(batch_x)

    # 2. Reconstruction loss (in normalized DSD space)
    pred_x_recon = model.decoder(Z)
    loss_recon = divergence(
        torch.log(pred_x_recon / (pred_x_recon.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
        torch.log(batch_x / (batch_x.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
    )

    # 3. Compute dhdt from thermodynamics using the dhdt function, in scaled mass units
    # This uses: condensation velocity, reconstructed DSD (de-normalized), and NWI prefactor
    rbin_median = torch.from_numpy(params["rbin_median"]).to(device).float().reshape(1, 1, -1)
    m_scale = params["m_scale"]
    dhdt_computed = compute_dhdt(
        Z, batch_temp, batch_sat, rbin_median,
        model.encoder, model.decoder, m_scale=m_scale, use_kohler=False
    )

    # 4. Project dhdt to scaled physical space using JVP
    Z_detached = Z.clone().detach().requires_grad_(True)
    _, dgdt_proj = torch.func.jvp(model.decoder, (Z_detached,), (dhdt_computed,))

    # 5. Compute the theoretical rate of condensation in scaled mass units
    dgdt_theory = compute_theoretical_condensation(
        pred_x_recon, batch_temp, batch_sat, rbin_median, use_kohler=False
    )

    # 6. Condensation tendency loss: projected vs theoretical with scaled mass units 
    # Scale by cond_scale to be consistent with ERF loss term
    loss_cond_dx = mse(dgdt_proj, dgdt_theory) / params["cond_scale"]**2 * params["m_scale"]**2

    # Weighted total loss
    loss = (
        params["loss_weight_recon"] * loss_recon
        + params["loss_weight_cond_dx"] * loss_cond_dx
    )

    loss_dict = {
        "total": loss,
        "recon": params["loss_weight_recon"] * loss_recon,
        "cond_dx": params["loss_weight_cond_dx"] * loss_cond_dx,
    }

    return loss, loss_dict


def compute_autoencoder_loss(model, batch, params, device):
    """
    Loss for pure autoencoder (no dynamics).

    Args:
        model: ComposedModel with encoder and decoder only
        batch: Tuple of (batch_x, batch_dgdt, batch_temp, batch_sat, batch_m)
               Only batch_x is used
        params: Parameters dict with loss weights
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_x, batch_dgdt, batch_temp, batch_sat, batch_m = batch

    # Forward pass for reconstruction
    pred_x_recon = model.decoder(model.encoder(batch_x))

    # Reconstruction loss in normalized space for KL divergence
    loss_recon = divergence(
        torch.log(pred_x_recon / (pred_x_recon.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
        torch.log(batch_x / (batch_x.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
    )

    loss = loss_recon

    loss_dict = {
        "total": loss,
        "recon": loss_recon,
    }

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
