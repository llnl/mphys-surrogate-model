"""Reusable reconstruction losses for mass- and number-aware spectra."""

import torch


kl_divergence = torch.nn.KLDivLoss(reduction="batchmean")


def normalized_spectrum(x, tol=1.0e-8):
    """Return a nonnegative discrete spectrum normalized over its last axis."""
    x = torch.clamp(x, min=0.0)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(tol)


def spectrum_kl(prediction, target, tol=1.0e-8):
    """PyTorch ``KL(target || prediction)`` for discrete binned spectra.

    ``KLDivLoss`` expects log probabilities as its input and probabilities as
    its target. Only the prediction needs a numerical floor; zero-probability
    target bins are handled by PyTorch and contribute zero to the divergence.
    """
    pred = normalized_spectrum(prediction, tol).clamp_min(tol)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    truth = normalized_spectrum(target, tol)
    return kl_divergence(torch.log(pred), truth)


def number_reconstruction_losses(model, reconstructed_mass, number_dsd, tol=1.0e-8):
    """Diagnose number shape from reconstructed mass and compute its KL loss.

    Args:
        model: Number-aware ``ComposedModel``.
        reconstructed_mass: Reconstructed mass-bin amounts.
        number_dsd: Target normalized number DSD summing to one over bins.
        tol: Numerical floor used by the KL divergence.

    Returns:
        ``(predicted_number_dsd, number_KL)``.
    """
    predicted = model.diagnose_number_dsd(reconstructed_mass, tol)
    loss_kl = spectrum_kl(predicted, number_dsd, tol)
    return predicted, loss_kl


def _dlnr_tensor(params, reference):
    """Return log-radius bin widths broadcastable to a spectral tensor."""
    return torch.as_tensor(
        params["dlnr"], dtype=reference.dtype, device=reference.device
    ).reshape((1,) * (reference.ndim - 1) + (-1,))


def _reconstruction_losses(model, batch, params):
    """Encode a mass DSD and compute mass and optional number-shape KL losses."""
    mass_dsd = batch["mass_dsd"].unsqueeze(1)
    mass_state = mass_dsd * batch["M"].reshape(-1, 1, 1)
    if model.number_aware:
        latent = model.encode_with_number(mass_state, batch["N"].unsqueeze(1))
        prediction = model.decode_mass(latent)
    else:
        latent = model.encoder(mass_state)
        prediction = model.decoder(latent)

    losses = {"mass_recon": spectrum_kl(prediction, mass_dsd, params["tol"])}
    if model.number_aware:
        _, loss_number = number_reconstruction_losses(
            model,
            prediction,
            batch["number_dsd"].unsqueeze(1),
            params["tol"],
        )
        losses["number_recon"] = loss_number
    return latent, prediction, losses


def _weighted_reconstruction_loss(losses, params):
    """Combine unweighted reconstruction terms using configured loss weights."""
    mass_weight = params.get(
        "loss_weight_mass_recon", params.get("loss_weight_recon", 1.0)
    )
    total = mass_weight * losses["mass_recon"]
    if "number_recon" in losses:
        total = total + params.get("loss_weight_number_recon", 1.0) * losses[
            "number_recon"
        ]
    return total


def _reconstruction_loss_dict(losses, params):
    """Return reconstruction components after applying configured weights."""
    mass_weight = params.get(
        "loss_weight_mass_recon", params.get("loss_weight_recon", 1.0)
    )
    result = {"mass_recon": mass_weight * losses["mass_recon"]}
    if "number_recon" in losses:
        result["number_recon"] = params.get(
            "loss_weight_number_recon", 1.0
        ) * losses["number_recon"]
    return result
