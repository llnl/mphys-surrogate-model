"""
Tests for src/condensation_utils.py:

1. compute_theoretical_condensation solves the full growth equation
   dg/dt = -d(v g)/d(ln r) + 3 v g (not just the 3 v g source term).
2. Representative radius is the geometric bin center.
3. G_func / drdt are pure torch for tensor inputs: differentiable in T and
   safe to torch.jit.trace.

Run with: pytest tests/test_condensation_utils.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import model_factory  # noqa: E402
from src.condensation_utils import (  # noqa: E402
    G_func,
    bin_faces_from_centers,
    compute_dhdt,
    compute_theoretical_condensation,
    condensation_tendency_fv,
    drdt,
)

N_BINS = 64
EDGES = np.geomspace(1e-6, 5e-3, N_BINS + 1)
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])
DLNR = np.log(EDGES[1] / EDGES[0])


def gaussian_in_lnr(mu=np.log(10e-6), s=0.5):
    x = np.log(CENTERS)
    g = np.exp(-0.5 * ((x - mu) / s) ** 2)
    dg_dlnr = -(x - mu) / s**2 * g
    return g, dg_dlnr


# --------------------------------------------------------------------------- #
# 1. Full growth equation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme,tol", [("central", 2e-2), ("upwind", 2.5e-1)])
def test_fv_matches_analytic_for_constant_v(scheme, tol):
    """With v = d(ln r)/dt constant, g(ln r, t) = exp(3 v t) g0(ln r - v t),
    so dg/dt = 3 v g - v dg/d(ln r) exactly."""
    v = 1e-3
    g, dg_dlnr = gaussian_in_lnr()
    exact = 3 * v * g - v * dg_dlnr
    fv = condensation_tendency_fv(g, np.full(N_BINS, v), np.full(N_BINS + 1, v), DLNR, scheme)
    assert np.abs(fv - exact).max() / np.abs(exact).max() < tol


def test_redistribution_term_is_present():
    """The old source-only form (3 v g) has no mass flux between bins."""
    v = 1e-3
    g, dg_dlnr = gaussian_in_lnr()
    fv = condensation_tendency_fv(g, np.full(N_BINS, v), np.full(N_BINS + 1, v), DLNR)
    source_only = 3 * v * g
    # for a Gaussian in ln r, max|v dg/dlnr| / max|3 v g| = 1 / (3 s sqrt(e)) ~ 0.4 at s = 0.5
    assert np.abs(fv - source_only).max() > 0.3 * np.abs(source_only).max()


def test_total_mass_tendency_is_source_term_only():
    """Fluxes are conservative with zero boundary flux: sum(dg/dt) = sum(3 v g)."""
    g, _ = gaussian_in_lnr()
    T = torch.tensor([[[285.0]]], dtype=torch.float64)
    S = torch.tensor([[[1.004]]], dtype=torch.float64)
    gt = torch.tensor(g)[None, None, :]
    rc = torch.tensor(CENTERS)
    dgdt = compute_theoretical_condensation(gt, T, S, rc)
    v_c = drdt(rc.reshape(1, 1, -1), T, S) / rc.reshape(1, 1, -1)
    torch.testing.assert_close(dgdt.sum(), (3 * v_c * gt).sum(), rtol=1e-12, atol=0)


def test_torch_and_numpy_paths_agree():
    g, _ = gaussian_in_lnr()
    T, S = np.array([[[285.0]]]), np.array([[[1.004]]])
    a = compute_theoretical_condensation(g[None, None, :], T, S, CENTERS)
    b = compute_theoretical_condensation(
        torch.tensor(g)[None, None, :], torch.tensor(T), torch.tensor(S), torch.tensor(CENTERS)
    ).numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_theory_target_consistent_with_latent_projection():
    """Encoding the (fixed) theoretical tendency should agree with compute_dhdt,
    which is the same equation integrated by parts against the weight functions.
    This is what the 'theory' loss implicitly assumes."""
    torch.manual_seed(0)
    model = model_factory.create_model(
        "nwi", "nwi_simple", "none",
        {"latent_dim": 3, "process": "condensation", "hidden_size": 64, "num_blocks": 2}, N_BINS,
    )
    g, _ = gaussian_in_lnr(s=0.6)
    x = torch.tensor(g, dtype=torch.float32)[None, None, :]
    T = torch.tensor([[[285.0]]])
    S = torch.tensor([[[1.004]]])
    rc = torch.tensor(CENTERS, dtype=torch.float32).reshape(1, 1, -1)
    with torch.no_grad():
        Z = model.encoder(x)
        g_dec = model.decoder(Z)
    dhdt = compute_dhdt(Z, T, S, rc, model.encoder, model.decoder).detach()
    enc_theory = model.encoder(compute_theoretical_condensation(g_dec, T, S, rc)).detach()
    enc_old = model.encoder(3 * g_dec * drdt(rc, T, S) / rc).detach()  # previous formula
    rel = lambda a, b: (a - b).abs().max() / b.abs().max()  # noqa: E731
    assert rel(enc_theory, dhdt) < 3e-2
    assert rel(enc_old, dhdt) > 3 * rel(enc_theory, dhdt)


# --------------------------------------------------------------------------- #
# 2. Bin geometry
# --------------------------------------------------------------------------- #
def test_faces_from_centers_roundtrip():
    faces, dlnr = bin_faces_from_centers(CENTERS)
    np.testing.assert_allclose(faces, EDGES, rtol=1e-12)
    assert np.isclose(dlnr, DLNR)
    faces_t, _ = bin_faces_from_centers(torch.tensor(CENTERS))
    np.testing.assert_allclose(faces_t.numpy(), EDGES, rtol=1e-12)


# --------------------------------------------------------------------------- #
# 3. Pure-torch G_func
# --------------------------------------------------------------------------- #
def test_G_func_numpy_and_torch_agree():
    T = np.array([[[265.0]], [[290.0]]])
    R = CENTERS.reshape(1, 1, -1)
    np.testing.assert_allclose(G_func(T, R), G_func(torch.tensor(T), torch.tensor(R)).numpy(),
                               rtol=1e-12)


def test_G_func_differentiable_in_T():
    T = torch.tensor([[[285.0]]], dtype=torch.float64, requires_grad=True)
    R = torch.tensor(CENTERS, dtype=torch.float64).reshape(1, 1, -1)
    assert torch.autograd.gradcheck(lambda t: G_func(t, R), (T,))


def test_drdt_traces_with_live_temperature():
    R = torch.tensor([[[5e-6, 2e-5]]])
    S = torch.tensor([[[1.003]]])

    class M(torch.nn.Module):
        def forward(self, T):
            return drdt(R, T, S)

    traced = torch.jit.trace(M(), torch.tensor([[[285.0]]]))
    T_new = torch.tensor([[[265.0]]])
    torch.testing.assert_close(traced(T_new), M()(T_new))
