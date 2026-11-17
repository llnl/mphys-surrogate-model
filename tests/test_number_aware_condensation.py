"""Number-aware reconstruction tests for condensation and parcel diagnostics."""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from online_tests.parcel_condensation import (
    BinGrid,
    LatentCondensation,
    ParcelForcing,
    ParcelState0,
    ROMParcel,
)
from src import model_factory
from src.condensation_utils import compute_theoretical_condensation
from src.recon_condensation_losses import (
    compute_autoencoder_loss,
    compute_condensation_loss_erf,
    compute_condensation_loss_theory,
)
from src.reconstruction_losses import spectrum_kl


def make_model():
    edges = np.geomspace(1.0e-6, 5.0e-3, 65)
    params = {
        "latent_dim": 3,
        "process": "condensation",
        "hidden_size": 32,
        "num_blocks": 1,
        "number_aware": True,
        "rbin_median": np.sqrt(edges[:-1] * edges[1:]),
        "M_scale": 1.0e-3,
        "N_scale": 1.0e6,
    }
    return model_factory.create_model("nwi", "nwi_simple", "none", params, 64)


def make_batch(batch_size=8):
    mass = torch.softmax(torch.randn(batch_size, 64), dim=-1)
    number = torch.softmax(torch.randn(batch_size, 64), dim=-1)
    return {
        "mass_dsd": mass,
        "number_dsd": number,
        "M": torch.linspace(0.1, 1.0, batch_size),
        "N": torch.linspace(0.2, 2.0, batch_size),
        "temp": torch.full((batch_size,), 285.0),
        "rh": torch.full((batch_size,), 1.001),
        "dgdt_cond": torch.zeros(batch_size, 64),
        "dgdt_coal": torch.zeros(batch_size, 64),
        "vt_mass_flux": torch.zeros(batch_size, 64),
        "sample_index": torch.arange(batch_size),
    }


def loss_params():
    return {
        "tol": 1.0e-8,
        "number_aware": True,
        "loss_weight_mass_recon": 1.0,
        "loss_weight_number_recon": 2.0,
        "M_scale": 1.0e-3,
        "dlnr": np.full(64, 1.0),
    }


def test_spectrum_kl_is_zero_for_identical_bin_fractions():
    p = torch.softmax(torch.randn(5, 64), dim=-1)
    torch.testing.assert_close(spectrum_kl(p, p), torch.tensor(0.0), atol=1e-7, rtol=0)


def test_number_aware_condensation_loss_has_expected_components_and_gradients():
    model = make_model()
    loss, components = compute_autoencoder_loss(model, make_batch(), loss_params(), "cpu")
    assert set(components) == {"total", "mass_recon", "number_recon"}
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.decoder.parameters())
    assert not hasattr(model, "number_decoder")


def test_number_dsd_is_diagnosed_from_mass_and_geometric_mean_radii():
    model = make_model()
    reconstructed_mass = torch.ones(2, 1, 64)
    number_dsd = model.diagnose_number_dsd(reconstructed_mass)
    expected = model.droplet_mass.reciprocal()
    expected = expected / expected.sum()
    torch.testing.assert_close(number_dsd, expected.expand_as(number_dsd))
    torch.testing.assert_close(
        number_dsd.sum(dim=-1), torch.ones(2, 1), atol=1e-6, rtol=0
    )


def test_number_aware_model_requires_total_number():
    model = make_model()
    batch = make_batch()
    latent = model.encoder(batch["mass_dsd"] * batch["M"].unsqueeze(-1))
    try:
        model.encode_with_number(batch["mass_dsd"] * batch["M"].unsqueeze(-1), None)
    except ValueError as error:
        assert "total_number is required" in str(error)
    else:
        raise AssertionError("number-aware reconstruction accepted missing N")


def test_number_aware_model_requires_explicit_dataset_scales():
    edges = np.geomspace(1.0e-6, 5.0e-3, 65)
    params = {
        "latent_dim": 3,
        "process": "condensation",
        "number_aware": True,
        "rbin_median": np.sqrt(edges[:-1] * edges[1:]),
    }
    try:
        model_factory.create_model("nwi", "nwi_simple", "none", params, 64)
    except ValueError as error:
        assert "M_scale" in str(error) and "N_scale" in str(error)
    else:
        raise AssertionError("number-aware model accepted missing dataset scales")


def test_condensation_theory_uses_physical_dmdlnr_then_returns_scaled_bin_tendency():
    model = make_model()
    batch = make_batch(batch_size=4)
    edges = np.geomspace(1.0e-6, 5.0e-3, 65)
    dlnr = np.diff(np.log(edges))
    params = {
        **loss_params(),
        "rbin_median": np.sqrt(edges[:-1] * edges[1:]),
        "dlnr": dlnr,
        "loss_weight_cond_dx": 1.0,
        "cond_target": "eulerian",
        "cond_theory_scheme": "central",
    }
    loss, components = compute_condensation_loss_theory(model, batch, params, "cpu")
    assert torch.isfinite(loss)
    assert torch.isfinite(components["cond_dx"])
    loss.backward()

    # Direct unit conversion check independent of the randomly initialized model:
    # scaled bin mass -> physical dmdlnr -> physical theory -> scaled bin tendency.
    x_scaled = (batch["mass_dsd"] * batch["M"].unsqueeze(-1)).unsqueeze(1)
    dlnr_t = torch.tensor(dlnr, dtype=x_scaled.dtype).reshape(1, 1, -1)
    g_physical = x_scaled * params["M_scale"] / dlnr_t
    T = batch["temp"].reshape(-1, 1, 1)
    S = batch["rh"].reshape(-1, 1, 1)
    radius = torch.tensor(params["rbin_median"], dtype=x_scaled.dtype)
    tendency_physical = compute_theoretical_condensation(
        g_physical, T, S, radius, form="eulerian"
    )
    tendency_scaled = tendency_physical * dlnr_t / params["M_scale"]
    torch.testing.assert_close(
        tendency_scaled * params["M_scale"] / dlnr_t,
        tendency_physical,
    )


def test_erf_tendency_is_optional_when_its_weight_is_zero():
    model = make_model()
    batch = make_batch(batch_size=4)
    del batch["dgdt_cond"]
    params = {**loss_params(), "loss_weight_cond_dx": 0.0}
    loss, components = compute_condensation_loss_erf(model, batch, params, "cpu")
    assert torch.isfinite(loss)
    assert components["cond_dx"].item() == 0.0


def test_erf_tendency_is_required_when_its_weight_is_nonzero():
    model = make_model()
    batch = make_batch(batch_size=4)
    del batch["dgdt_cond"]
    params = {**loss_params(), "loss_weight_cond_dx": 1.0}
    try:
        compute_condensation_loss_erf(model, batch, params, "cpu")
    except ValueError as error:
        assert "dgdt_cond is required" in str(error)
    else:
        raise AssertionError("ERF loss accepted a missing nonzero-weight target")


def test_parcel_number_diagnosis_preserves_carried_total_number():
    model = make_model()
    grid = BinGrid.log_uniform()
    op = LatentCondensation(
        model.encoder,
        model.decoder,
        grid,
        model=model,
        m_scale=1.0e-3,
        n_scale=1.0e6,
    )
    g = np.exp(-0.5 * ((np.log(grid.r_center) - np.log(10e-6)) / 0.5) ** 2)
    g = g / g.sum() * 1.0e-3
    Z = op.encode(g, 8.0e7)
    np.testing.assert_allclose(Z[-2], (g * grid.dlnr).sum(), rtol=1e-6)
    np.testing.assert_allclose(op.liquid_mass(Z), (g * grid.dlnr).sum(), rtol=1e-6)
    fractions, N = op.decode_number(Z)
    np.testing.assert_allclose(fractions.sum(axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(N, 8.0e7, rtol=1e-6)


def test_number_aware_parcel_box_smoke_run():
    """A short fixed-S integration preserves finite mass and diagnostic N."""
    model = make_model()
    grid = BinGrid.log_uniform()
    op = LatentCondensation(
        model.encoder,
        model.decoder,
        grid,
        model=model,
        m_scale=1.0e-3,
        n_scale=1.0e6,
    )
    g0 = np.exp(-0.5 * ((np.log(grid.r_center) - np.log(10e-6)) / 0.5) ** 2)
    g0 = g0 / (g0 * grid.dlnr).sum() * 1.0e-3
    forcing = ParcelForcing(S_of_t=lambda _: 1.001)
    result = ROMParcel(op, forcing).run(
        ParcelState0(), g0, t_end=2.0, dt_out=1.0,
        N0_per_kg=8.0e7, fixed_dt=0.1
    )
    assert np.all(np.isfinite(result["g"]))
    assert np.all(np.isfinite(result["ql"]))
    assert np.all(result["ql"] > 0.0)

    fractions, N = op.decode_number(result["Z"])
    np.testing.assert_allclose(fractions.sum(axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(N, 8.0e7, rtol=1e-6)


if __name__ == "__main__":
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for name, test in tests:
        test()
        print(f"PASS: {name}")
    print(f"\n{len(tests)} tests passed")
