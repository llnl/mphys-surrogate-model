"""Consistency tests for online_tests/parcel_condensation.py.

Uses a randomly initialised NWI autoencoder, so these check plumbing and
conservation, not model skill. Run with: pytest online_tests -q
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from online_tests import parcel_condensation as pc  # noqa: E402
from src import model_factory  # noqa: E402
from src.condensation_utils import G_func, compute_dhdt  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    params = {"latent_dim": 3, "process": "condensation", "hidden_size": 64, "num_blocks": 2}
    return model_factory.create_model("nwi", "nwi_simple", "none", params, 64)


@pytest.fixture(scope="module")
def grid():
    return pc.BinGrid.log_uniform()


@pytest.fixture(scope="module")
def op(model, grid):
    return pc.LatentCondensation(model.encoder, model.decoder, grid, m_scale=1.0,
                                 prefactor="autograd")


def _lognormal_g(grid):
    r, xi = pc.superdroplets_from_lognormal(grid, pc.LognormalIC(9e7, 6e-6, 1.3), 8)
    return pc.rebin(grid, r, xi)[0], r, xi


def test_fd_prefactor_matches_autograd(model, grid, op):
    op_fd = pc.LatentCondensation(model.encoder, model.decoder, grid, prefactor="fd",
                                  fd_grid="encoder")
    interior = slice(2, -2)  # one-sided differences at the ends
    np.testing.assert_allclose(op_fd.P[:, interior], op.P[:, interior], rtol=2e-2, atol=1e-6)


def test_tendency_matches_repo_compute_dhdt(model, grid, op):
    g, *_ = _lognormal_g(grid)
    Z = op.encode(g)[None, :]
    T, S = 285.0, 1.004
    ours = op.tendency(Z, T, S, rho=1.0)
    repo = compute_dhdt(
        torch.tensor(Z, dtype=torch.float32)[:, None, :],
        torch.tensor([[[T]]], dtype=torch.float32),
        torch.tensor([[[S]]], dtype=torch.float32),
        torch.tensor(grid.r_eval, dtype=torch.float32).reshape(1, 1, -1),
        model.encoder, model.decoder,
    ).detach().numpy()[:, 0, :]
    np.testing.assert_allclose(ours, repo, rtol=2e-3)


def test_tendency_degree_one_homogeneous(grid, op):
    g, *_ = _lognormal_g(grid)
    Z = op.encode(g)[None, :]
    d1 = op.tendency(Z, 285.0, 1.003)
    d10 = op.tendency(10 * Z, 285.0, 1.003)
    np.testing.assert_allclose(d10, 10 * d1, rtol=1e-4)


def test_mass_channel_is_total_mass(grid, op):
    g, r, xi = _lognormal_g(grid)
    ql = (xi * pc.droplet_mass(r)).sum()
    assert np.isclose(grid.dlnr0 * op.encode(g)[-1], ql, rtol=1e-10)


def test_reference_growth_law(grid):
    lp = pc.LagrangianParcel(grid, pc.ParcelForcing())
    r = np.array([2e-6, 10e-6, 50e-6])
    dx = lp._dx_dt(r**2, np.asarray(285.0), np.asarray(1.002))
    np.testing.assert_allclose(dx, 2 * G_func(np.asarray(285.0), r) * 0.002, rtol=1e-12)


@pytest.mark.parametrize("which", ["rom", "ref"])
def test_coupled_parcel_conserves_water_and_lse(grid, op, which):
    forcing = pc.ParcelForcing(w=lambda t: 1.0)
    s0 = pc.ParcelState0(S=1.0)
    g, r, xi = _lognormal_g(grid)
    if which == "rom":
        out = pc.ROMParcel(op, forcing).run(s0, g, 120.0, 10.0)
    else:
        out = pc.LagrangianParcel(grid, forcing).run(s0, r, xi, 120.0, 10.0)
    qt = out["qv"] + out["ql"]
    lse = pc.Cp_d * out["T"] + pc.GRAV * out["z"] - pc.L_v * out["ql"]
    np.testing.assert_allclose(qt, qt[0], rtol=1e-6)
    np.testing.assert_allclose(lse, lse[0], rtol=1e-8)
    assert out["ql"][-1] > out["ql"][0]


def test_box_mode_has_no_ascent(grid, op):
    forcing = pc.ParcelForcing(S_of_t=lambda t: 1.002)
    g, *_ = _lognormal_g(grid)
    out = pc.ROMParcel(op, forcing).run(pc.ParcelState0(), g, 60.0, 10.0)
    for k in ("p", "T", "qv", "z"):
        assert np.ptp(out[k]) == 0.0
    np.testing.assert_allclose(out["S"], 1.002)
