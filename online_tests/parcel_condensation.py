"""
Adiabatic parcel test for the latent-space (NWI) condensation dynamic.

Two microphysics representations are run under *identical* parcel thermodynamics,
so that every difference between them is attributable to the ROM:

    ROMParcel        latent state Z = [h_1..h_L, M] (per kg dry air), advanced with
                     dZ_j/dt = sum_i P_ji g_i(Z) v_i(T, S),
                     P = df/dlnr + 3f (NWI weight functions), v = (dr/dt)/r,
                     g = decoder(Z). This is the same tendency as
                     src.condensation_utils.compute_dhdt, with the prefactor
                     precomputed once instead of on every call.

    LagrangianParcel reference: K "superdroplets" per bin with fixed multiplicity,
                     each growing by d(r^2)/dt = 2 G(T, r) (S - 1 [+ Kohler]),
                     using the *same* G_func / esat as the ROM. Rebinned onto the
                     ROM's 64 bins for comparison. With no coalescence this is
                     the exact solution of the condensation problem up to ODE
                     tolerance (no numerical diffusion in radius space).

Two configurations:

    parcel  (default) adiabatic parcel lifted at w(t); S diagnosed from
            (p, T, qv), with latent heating and vapor depletion feeding back.
            This is the actual parcel test.
    box     NOT a parcel: no ascent (w is ignored) and T, p fixed, with S(t)
            imposed as if by an infinite vapor reservoir. A 0D growth test that
            isolates the latent tendency from the thermodynamic feedback.

Thermodynamics are deliberately simple (constant L_v and cp of dry air,
hydrostatic pressure) and shared by both parcels; liquid static energy
Cp_d*T + g*z - L_v*q_l and total water q_v + q_l are conserved exactly.

Units: g [kg / kg dry air / ln r], q in kg/kg dry air. The latent tendency is
homogeneous of degree 1 in Z (the decoder is mass-equivariant and the tendency
is linear in g), so the per-kg <-> per-m^3 conversion and m_scale only matter
for float32 precision and the LOG_TOLERANCE epsilon inside the decoder.

Usage
-----
    # from a training run directory containing params.json + model .pth
    python online_tests/parcel_condensation.py --model-dir trained_models/condensation/<run>

    # box (fixed-S growth) test instead of the parcel
    python online_tests/parcel_condensation.py --model-dir <run> --mode box --S-box 1.005

    # or from TorchScript encoder/decoder (prefactor via finite differences)
    python online_tests/parcel_condensation.py \
        --encoder-ts data/ftorch_weights/nwi_encoder.pt \
        --decoder-ts data/ftorch_weights/nwi_decoder.pt --m-scale 1e-2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import model_factory  # noqa: E402
from src.condensation_utils import (  # noqa: E402
    Cp_d,
    L_v,
    Rd,
    Rv,
    a as KOHLER_A,
    b as KOHLER_B,
    compute_nwi_prefactor,
    drdt as repo_drdt,
    get_esat,
    rho_l,
)

GRAV = 9.81
EPS = Rd / Rv
LNR_MIN, LNR_MAX = np.log(1e-6), np.log(5e-3)  # hardcoded in compute_nwi_prefactor


# --------------------------------------------------------------------------- #
# Bin grid
# --------------------------------------------------------------------------- #
@dataclass
class BinGrid:
    """Log-uniform radius bins. ``r_eval`` is where v_cond is evaluated."""

    edges: np.ndarray  # [N+1], m
    r_eval_choice: str = "center"  # match the training convention (params["rbin_eval"])

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=float)
        self.lnr_edges = np.log(self.edges)
        self.dlnr = np.diff(self.lnr_edges)
        if not np.allclose(self.dlnr, self.dlnr[0], rtol=1e-6):
            raise ValueError("ROM assumes uniform ln r spacing (M = sum g_i).")
        self.dlnr0 = float(self.dlnr[0])
        self.r_left, self.r_right = self.edges[:-1], self.edges[1:]
        self.r_center = np.sqrt(self.r_left * self.r_right)
        if self.r_eval_choice == "right":
            self.r_eval = self.r_right
        elif self.r_eval_choice == "center":
            self.r_eval = self.r_center
        else:
            raise ValueError("r_eval_choice must be 'right' or 'center'")

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    @classmethod
    def log_uniform(cls, n_bins=64, r_min=1e-6, r_max=5e-3, r_eval_choice="center"):
        return cls(np.geomspace(r_min, r_max, n_bins + 1), r_eval_choice)

    @classmethod
    def from_centers(cls, r_center, r_eval_choice="center"):
        """Rebuild edges from geometric centers (rbin_eval == "center")."""
        r_center = np.asarray(r_center, dtype=float)
        dl = np.log(r_center[1] / r_center[0])
        edges = np.concatenate([[r_center[0] * np.exp(-dl / 2)], r_center * np.exp(dl / 2)])
        return cls(edges, r_eval_choice)

    @classmethod
    def from_right_edges(cls, r_right, r_eval_choice="right"):
        """Rebuild edges from rbin_median (= rbin_r) saved in params.json."""
        r_right = np.asarray(r_right, dtype=float)
        dl = np.log(r_right[1] / r_right[0])
        edges = np.concatenate([[r_right[0] * np.exp(-dl)], r_right])
        return cls(edges, r_eval_choice)

    @classmethod
    def from_dataset(cls, path, r_eval_choice="center"):
        import xarray as xr

        ds = xr.open_dataset(path)
        edges = np.concatenate([ds["rbin_l"].values, ds["rbin_r"].values[-1:]])
        return cls(edges, r_eval_choice)


def droplet_mass(r):
    return 4.0 / 3.0 * np.pi * rho_l * r**3


# --------------------------------------------------------------------------- #
# Latent condensation operator
# --------------------------------------------------------------------------- #
class LatentCondensation:
    """Wraps an NWI encoder/decoder and evaluates dZ/dt for condensation."""

    def __init__(self, encoder, decoder, grid: BinGrid, m_scale: float = 1.0,
                 prefactor: str = "autograd", fd_grid: str = "encoder",
                 use_kohler: bool = False):
        self.encoder, self.decoder = encoder, decoder
        self.grid, self.m_scale, self.use_kohler = grid, float(m_scale), use_kohler
        for m in (encoder, decoder):
            m.eval()
        with torch.no_grad():
            self.wf = self.encoder(torch.eye(grid.n_bins)).double().numpy()  # [N, L+1]
        self.n_latent = self.wf.shape[1]
        if prefactor == "autograd":
            if not hasattr(encoder, "wfs"):
                raise ValueError("autograd prefactor needs a Python NWI encoder; use 'fd'.")
            P = compute_nwi_prefactor(encoder).double().numpy()
        elif prefactor == "fd":
            P = self.fd_prefactor(fd_grid)
        else:
            raise ValueError("prefactor must be 'autograd' or 'fd'")
        self.P = P  # [L+1, N]

    def fd_prefactor(self, fd_grid="encoder"):
        """P = d f/d ln r + 3 f by finite differences on ln f.

        fd_grid="encoder": the coordinate compute_nwi_prefactor assumes
            (64 points spanning [ln 1um, ln 5mm] inclusive, step span/63).
        fd_grid="bins":    the actual bin spacing (step span/64 for edges
            1um..5mm). The two differ by 63/64 in the df/dlnr term.
        """
        n = self.grid.n_bins
        if fd_grid == "encoder":
            lnr = np.linspace(LNR_MIN, LNR_MAX, n)
        elif fd_grid == "bins":
            lnr = np.log(self.grid.r_center)
        else:
            raise ValueError("fd_grid must be 'encoder' or 'bins'")
        f = self.wf
        dlnf = np.gradient(np.log(f), lnr, axis=0)
        dfdlnr = f * dlnf
        dfdlnr[:, -1] = 0.0  # mass weight function is identically 1
        return (dfdlnr + 3.0 * f).T

    # --- encode / decode in physical per-kg units ------------------------- #
    def encode(self, g_kg: np.ndarray) -> np.ndarray:
        """[..., N] per-kg mass density -> [..., L+1] per-kg latent (linear)."""
        return np.asarray(g_kg) @ self.wf

    def decode(self, Z_kg: np.ndarray, rho: float | np.ndarray = 1.0) -> np.ndarray:
        Z_kg = np.atleast_2d(Z_kg)
        rho = np.broadcast_to(np.asarray(rho, dtype=float), (Z_kg.shape[0],))
        Zm = torch.as_tensor(Z_kg * rho[:, None] / self.m_scale, dtype=torch.float32)
        with torch.no_grad():
            g = self.decoder(Zm).reshape(Z_kg.shape[0], -1).double().numpy()
        return g * self.m_scale / rho[:, None]

    def v_cond(self, T, S):
        """dlnr/dt [B, N] at r_eval, using the repo's drdt (same G_func, esat)."""
        R = self.grid.r_eval[None, :]
        T = np.asarray(T, dtype=float).reshape(-1, 1)
        S = np.asarray(S, dtype=float).reshape(-1, 1)
        return repo_drdt(R, T, S, use_kohler=self.use_kohler) / R

    def tendency(self, Z_kg, T, S, rho=1.0):
        """dZ_kg/dt [B, L+1]. Homogeneous of degree 1 in Z_kg."""
        g = self.decode(Z_kg, rho)
        return (g * self.v_cond(T, S)) @ self.P.T


# --------------------------------------------------------------------------- #
# Shared parcel thermodynamics
# --------------------------------------------------------------------------- #
def esat_pa(T):
    return get_esat(np.asarray(T, dtype=float) - 273.16) * 100.0


def vapor_pressure(p, qv):
    return p * qv / (EPS + qv)


def saturation_ratio(p, T, qv):
    return vapor_pressure(p, qv) / esat_pa(T)


def rho_dry(p, T, qv):
    return (p - vapor_pressure(p, qv)) / (Rd * T)


def rho_moist(p, T, qv, ql):
    return rho_dry(p, T, qv) * (1.0 + qv + ql)


def qv_from_S(p, T, S):
    e = S * esat_pa(T)
    return EPS * e / (p - e)


@dataclass
class ParcelForcing:
    """Parcel if S_of_t is None; otherwise a fixed-T/p box with imposed S(t),
    in which w is ignored (nothing is lifted)."""

    w: Callable[[float], float] = lambda t: 1.0  # m/s, parcel only
    S_of_t: Optional[Callable[[float], float]] = None  # box mode if set

    @property
    def coupled(self):
        return self.S_of_t is None


def thermo_rhs(t, z, p, T, qv, ql, dql_dt, forcing: ParcelForcing):
    """Returns dz, dp, dT, dqv. All zero in box mode (no ascent, fixed T/p)."""
    if not forcing.coupled:
        return 0.0, 0.0, 0.0, 0.0
    w = forcing.w(t)
    dz = w
    dp = -rho_moist(p, T, qv, ql) * GRAV * w
    dT = -GRAV * w / Cp_d + L_v / Cp_d * dql_dt
    dqv = -dql_dt
    return dz, dp, dT, dqv


def current_S(t, p, T, qv, forcing: ParcelForcing):
    return forcing.S_of_t(t) if not forcing.coupled else saturation_ratio(p, T, qv)


# --------------------------------------------------------------------------- #
# Initial spectra
# --------------------------------------------------------------------------- #
@dataclass
class LognormalIC:
    N_per_kg: float = 100e6 / 1.1  # ~100 cm^-3 at rho ~ 1.1 kg/m^3
    r_g: float = 5e-6
    sigma_g: float = 1.3

    def number_density_lnr(self, r):
        s = np.log(self.sigma_g)
        return self.N_per_kg / (np.sqrt(2 * np.pi) * s) * np.exp(-np.log(r / self.r_g) ** 2 / (2 * s * s))


def superdroplets_from_lognormal(grid: BinGrid, ic: LognormalIC, k_per_bin=16, min_frac=1e-10):
    """K log-uniform superdroplets per bin; returns (r [m], xi [#/kg])."""
    r, xi = [], []
    for l, rr in zip(grid.r_left, grid.r_right):
        sub_edges = np.geomspace(l, rr, k_per_bin + 1)
        rc = np.sqrt(sub_edges[:-1] * sub_edges[1:])
        r.append(rc)
        xi.append(ic.number_density_lnr(rc) * np.diff(np.log(sub_edges)))
    r, xi = np.concatenate(r), np.concatenate(xi)
    mass = xi * droplet_mass(r)
    keep = mass > min_frac * mass.max()
    return r[keep], xi[keep]


def superdroplets_from_binned(grid: BinGrid, g_kg, k_per_bin=16):
    """Split binned mass density equally (in mass) across K sub-bins."""
    r, xi = [], []
    for i, (l, rr) in enumerate(zip(grid.r_left, grid.r_right)):
        if g_kg[i] <= 0:
            continue
        sub_edges = np.geomspace(l, rr, k_per_bin + 1)
        rc = np.sqrt(sub_edges[:-1] * sub_edges[1:])
        r.append(rc)
        xi.append(g_kg[i] * grid.dlnr[i] / k_per_bin / droplet_mass(rc))
    return np.concatenate(r), np.concatenate(xi)


def rebin(grid: BinGrid, r, xi):
    """Superdroplets -> per-kg mass density per ln r on the grid.
    Returns (g [N], ql_below_range, ql_above_range)."""
    m = xi * droplet_mass(r)
    idx = np.searchsorted(grid.edges, r, side="right") - 1
    inside = (idx >= 0) & (idx < grid.n_bins)
    g = np.bincount(idx[inside], weights=m[inside], minlength=grid.n_bins) / grid.dlnr
    return g, m[idx < 0].sum(), m[idx >= grid.n_bins].sum()


# --------------------------------------------------------------------------- #
# Parcels
# --------------------------------------------------------------------------- #
@dataclass
class ParcelState0:
    z: float = 0.0
    p: float = 90000.0
    T: float = 285.0
    S: float = 1.0  # initial saturation ratio (sets qv)

    def qv(self):
        return qv_from_S(self.p, self.T, self.S)


def _integrate(rhs, y0, t_end, dt_out, method, rtol, atol, fixed_dt):
    t_eval = np.arange(0.0, t_end + 0.5 * dt_out, dt_out)
    if fixed_dt is None:
        sol = solve_ivp(rhs, (0.0, t_end), y0, method=method, t_eval=t_eval,
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.t, sol.y.T, sol.nfev
    # classical RK4 with a host-like fixed step (to probe stiffness/stability)
    n_sub = int(round(dt_out / fixed_dt))
    if not np.isclose(n_sub * fixed_dt, dt_out):
        raise ValueError("dt_out must be a multiple of fixed_dt")
    ys, y, t, nfev = [y0.copy()], y0.copy(), 0.0, 0
    for _ in range(len(t_eval) - 1):
        for _ in range(n_sub):
            k1 = rhs(t, y)
            k2 = rhs(t + fixed_dt / 2, y + fixed_dt / 2 * k1)
            k3 = rhs(t + fixed_dt / 2, y + fixed_dt / 2 * k2)
            k4 = rhs(t + fixed_dt, y + fixed_dt * k3)
            y = y + fixed_dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += fixed_dt
            nfev += 4
            if not np.all(np.isfinite(y)):
                raise FloatingPointError(f"RK4 blew up at t={t:.1f}s with dt={fixed_dt}s")
        ys.append(y.copy())
    return t_eval, np.array(ys), nfev


class ROMParcel:
    """State y = [z, p, T, qv, Z_kg (L+1)]."""

    def __init__(self, op: LatentCondensation, forcing: ParcelForcing):
        self.op, self.forcing = op, forcing

    def ql(self, Z_kg):
        return self.op.grid.dlnr0 * Z_kg[..., -1]

    def rhs(self, t, y):
        z, p, T, qv, Z = y[0], y[1], y[2], y[3], y[4:]
        ql = self.ql(Z)
        S = current_S(t, p, T, qv, self.forcing)
        rho = rho_dry(p, T, qv)
        dZ = self.op.tendency(Z[None, :], T, S, rho)[0]
        dql = self.op.grid.dlnr0 * dZ[-1]
        return np.concatenate([thermo_rhs(t, z, p, T, qv, ql, dql, self.forcing), dZ])

    def run(self, state0: ParcelState0, g0_kg, t_end, dt_out=1.0, method="LSODA",
            rtol=1e-6, fixed_dt=None):
        Z0 = self.op.encode(g0_kg)
        y0 = np.concatenate([[state0.z, state0.p, state0.T, state0.qv()], Z0])
        atol = np.maximum(1e-8 * np.abs(y0), 1e-14)
        t, Y, nfev = _integrate(self.rhs, y0, t_end, dt_out, method, rtol, atol, fixed_dt)
        out = _unpack_thermo(t, Y, self.forcing)
        out["Z"] = Y[:, 4:]
        out["ql"] = self.ql(out["Z"])
        out["g"] = self.op.decode(out["Z"], rho_dry(out["p"], out["T"], out["qv"]))
        out["nfev"] = nfev
        return out


class LagrangianParcel:
    """Reference. State y = [z, p, T, qv, x_k = r_k^2]."""

    def __init__(self, grid: BinGrid, forcing: ParcelForcing, use_kohler=False,
                 r_floor=0.1e-6):
        self.grid, self.forcing, self.use_kohler = grid, forcing, use_kohler
        self.x_floor = r_floor**2

    def _dx_dt(self, x, T, S):
        r = np.sqrt(x)
        # repo drdt = G (S - 1 [- a/(rT) + b/r^3]) / r  ->  d(r^2)/dt = 2 r drdt
        dx = 2.0 * r * repo_drdt(r, np.asarray(T, dtype=float), np.asarray(S, dtype=float),
                                 use_kohler=self.use_kohler)
        # evaporated droplets park at r_floor (counted as below-range liquid)
        return np.where((x <= self.x_floor) & (dx < 0), 0.0, dx)

    def rhs(self, t, y):
        z, p, T, qv, x = y[0], y[1], y[2], y[3], np.maximum(y[4:], self.x_floor)
        ql = np.sum(self.xi * droplet_mass(np.sqrt(x)))
        S = current_S(t, p, T, qv, self.forcing)
        dx = self._dx_dt(x, T, S)
        dql = np.sum(self.xi * 2.0 * np.pi * rho_l * np.sqrt(x) * dx)  # dm/dt = 2 pi rho r d(r^2)/dt
        return np.concatenate([thermo_rhs(t, z, p, T, qv, ql, dql, self.forcing), dx])

    def run(self, state0: ParcelState0, r0, xi, t_end, dt_out=1.0, method="LSODA",
            rtol=1e-8, fixed_dt=None):
        self.xi = np.asarray(xi, dtype=float)
        y0 = np.concatenate([[state0.z, state0.p, state0.T, state0.qv()], np.asarray(r0) ** 2])
        atol = np.concatenate([[1e-6, 1e-4, 1e-8, 1e-12], np.full(len(r0), 1e-18)])
        t, Y, nfev = _integrate(self.rhs, y0, t_end, dt_out, method, rtol, atol, fixed_dt)
        out = _unpack_thermo(t, Y, self.forcing)
        r = np.sqrt(np.maximum(Y[:, 4:], self.x_floor))
        out["r"] = r
        out["ql"] = (self.xi * droplet_mass(r)).sum(axis=1)
        binned = [rebin(self.grid, ri, self.xi) for ri in r]
        out["g"] = np.array([b[0] for b in binned])
        out["ql_below"] = np.array([b[1] for b in binned])
        out["ql_above"] = np.array([b[2] for b in binned])
        out["N"] = self.xi.sum()
        out["nfev"] = nfev
        return out


def _unpack_thermo(t, Y, forcing):
    out = {"t": t, "z": Y[:, 0], "p": Y[:, 1], "T": Y[:, 2], "qv": Y[:, 3]}
    if forcing.coupled:
        out["S"] = saturation_ratio(out["p"], out["T"], out["qv"])
    else:
        out["S"] = np.array([forcing.S_of_t(ti) for ti in t])
    return out


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def spectral_moments(grid: BinGrid, g):
    """Number [#/kg] and effective radius [m] diagnosed from binned g (bin centers)."""
    g = np.atleast_2d(g)
    rc = grid.r_center
    N = (g * grid.dlnr / droplet_mass(rc)).sum(axis=-1)
    r_eff = g.sum(axis=-1) / np.maximum((g / rc).sum(axis=-1), 1e-30)
    return N, r_eff


def compare(op: LatentCondensation, rom: dict, ref: dict) -> dict:
    """Split ROM error into (i) autoencoder projection error and (ii) latent
    trajectory error, by encoding the reference spectra."""
    rho = rho_dry(ref["p"], ref["T"], ref["qv"])
    Z_ref = op.encode(ref["g"])
    g_ae = op.decode(Z_ref, rho)  # best the AE can do on the true spectrum

    def l1(a, b):
        return np.abs(a - b).sum(axis=-1) / np.maximum(np.abs(b).sum(axis=-1), 1e-30)

    ql_ref_in_range = op.grid.dlnr0 * ref["g"].sum(axis=1)
    N_ref, reff_ref = spectral_moments(op.grid, ref["g"])
    N_rom, reff_rom = spectral_moments(op.grid, rom["g"])
    return {
        "N_ref": N_ref, "N_rom": N_rom, "reff_ref": reff_ref, "reff_rom": reff_rom,
        "Z_ref": Z_ref,
        "g_ae": g_ae,
        "err_ae_l1": l1(g_ae, ref["g"]),            # projection error only
        "err_rom_l1": l1(rom["g"], ref["g"]),       # total ROM error
        "err_traj_Z": np.abs(rom["Z"] - Z_ref) / np.maximum(np.abs(Z_ref), 1e-30),
        "dS": rom["S"] - ref["S"],
        "dql_rel": (rom["ql"] - ql_ref_in_range) / np.maximum(ql_ref_in_range, 1e-30),
        "ql_ref_in_range": ql_ref_in_range,
    }


def plot_comparison(op, rom, ref, cmp, path=None, spectra_times=None):
    t = ref["t"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()
    ax[0].plot(t, 100 * (ref["S"] - 1), "k", label="reference")
    ax[0].plot(t, 100 * (rom["S"] - 1), "C0--", label="ROM")
    ax[0].set(xlabel="t [s]", ylabel="S - 1 [%]", title="Supersaturation")
    ax[0].legend()
    ax[1].plot(t, 1e3 * cmp["ql_ref_in_range"], "k", label="reference (1um-5mm)")
    ax[1].plot(t, 1e3 * rom["ql"], "C0--", label="ROM")
    ax[1].set(xlabel="t [s]", ylabel="q_l [g/kg]", title="Liquid water")
    ax[1].legend()
    for j in range(op.n_latent):
        ax[2].semilogy(t, np.abs(cmp["Z_ref"][:, j]), color=f"C{j}", label=f"Enc(ref) Z{j}")
        ax[2].semilogy(t, np.abs(rom["Z"][:, j]), "--", color=f"C{j}")
    ax[2].set(xlabel="t [s]", title="Latent trajectories (solid ref, dashed ROM)")
    ax[2].legend(fontsize=7)
    ax[3].plot(t, cmp["err_ae_l1"], "C1", label="AE projection: |Dec(Enc(ref)) - ref|")
    ax[3].plot(t, cmp["err_rom_l1"], "C0", label="ROM total: |Dec(Z_rom) - ref|")
    ax[3].set(xlabel="t [s]", ylabel="relative L1", title="Spectral error decomposition")
    ax[3].legend(fontsize=8)
    if spectra_times is None:
        spectra_times = np.linspace(t[0], t[-1], 4)
    r = op.grid.r_center * 1e6
    for k, ts in enumerate(spectra_times):
        i = np.argmin(np.abs(t - ts))
        ax[4].semilogx(r, 1e3 * ref["g"][i], "k", alpha=0.4 + 0.6 * k / len(spectra_times))
        ax[4].semilogx(r, 1e3 * cmp["g_ae"][i], ":", color=f"C{k}")
        ax[4].semilogx(r, 1e3 * rom["g"][i], "--", color=f"C{k}", label=f"t={t[i]:.0f}s")
    ax[4].set(xlabel="r [um]", ylabel="g [g/kg/ln r]", xlim=(1, 100),
              title="Spectra (black ref, dotted AE, dashed ROM)")
    ax[4].legend(fontsize=8)
    ax[5].plot(t, 1e6 * cmp["reff_ref"], "k", label="r_eff ref")
    ax[5].plot(t, 1e6 * cmp["reff_rom"], "C0--", label="r_eff ROM (decoded)")
    ax[5].set(xlabel="t [s]", ylabel="r_eff [um]", title="Effective radius / number")
    a2 = ax[5].twinx()
    a2.plot(t, 1e-6 * cmp["N_ref"], "k:", label="N ref")
    a2.plot(t, 1e-6 * cmp["N_rom"], "C1:", label="N ROM (decoded)")
    a2.set_ylabel("N [1/mg]")
    ax[5].legend(loc="upper left", fontsize=8)
    a2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_from_checkpoint(model_dir, r_eval_choice=None, prefactor="autograd", **kw):
    """Training-run directory with params.json and a single .pth.

    r_eval_choice=None uses the radius convention the model was trained with:
    params["rbin_eval"] ("center"), or "right" for checkpoints that predate it.
    """
    model_dir = Path(model_dir)
    params = json.loads((model_dir / "params.json").read_text())
    pths = sorted(model_dir.glob("*.pth"))
    if len(pths) != 1:
        raise FileNotFoundError(f"expected one .pth in {model_dir}, found {pths}")
    n_bins = len(params["rbin_median"]) if "rbin_median" in params else 64
    model = model_factory.create_model(params["encoder_type"], params["decoder_type"],
                                       params.get("dynamics_type", "none"), params, n_bins)
    model.load_state_dict(torch.load(pths[0], map_location="cpu"))
    trained_eval = params.get("rbin_eval", "right")
    r_eval_choice = r_eval_choice or trained_eval
    if "rbin_median" in params:
        build = BinGrid.from_centers if trained_eval == "center" else BinGrid.from_right_edges
        grid = build(params["rbin_median"], r_eval_choice)
    else:
        grid = BinGrid.log_uniform(n_bins, r_eval_choice=r_eval_choice)
    return LatentCondensation(model.encoder, model.decoder, grid,
                              m_scale=params.get("m_scale", 1.0), prefactor=prefactor, **kw)


def load_from_torchscript(enc_path, dec_path, grid: BinGrid, m_scale=1.0, **kw):
    enc, dec = torch.jit.load(enc_path), torch.jit.load(dec_path)
    return LatentCondensation(enc, dec, grid, m_scale=m_scale, prefactor="fd", **kw)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir")
    src.add_argument("--encoder-ts")
    ap.add_argument("--decoder-ts")
    ap.add_argument("--m-scale", type=float, default=1.0)
    ap.add_argument("--bins-file", help="netCDF with rbin_l/rbin_r (else params.json or 1um-5mm)")
    ap.add_argument("--r-eval", choices=["right", "center"], default=None,
                    help="radius for v_cond (default: training convention; 'center' for TorchScript)")
    ap.add_argument("--prefactor", choices=["autograd", "fd"], default=None)
    ap.add_argument("--fd-grid", choices=["encoder", "bins"], default="encoder")
    ap.add_argument("--mode", choices=["parcel", "box"], default="parcel",
                    help="parcel: lifted at --w; box: no ascent, fixed T/p, S = --S-box")
    ap.add_argument("--w", type=float, default=1.0, help="updraft [m/s]")
    ap.add_argument("--descend-after", type=float, default=None,
                    help="reverse w after this many seconds (evaporation leg)")
    ap.add_argument("--S-box", type=float, default=1.005, help="saturation ratio (box mode)")
    ap.add_argument("--t-end", type=float, default=600.0)
    ap.add_argument("--dt-out", type=float, default=2.0)
    ap.add_argument("--fixed-dt", type=float, default=None, help="RK4 step for the ROM (host-like)")
    ap.add_argument("--N-cm3", type=float, default=100.0)
    ap.add_argument("--r-g-um", type=float, default=5.0)
    ap.add_argument("--sigma-g", type=float, default=1.3)
    ap.add_argument("--p0", type=float, default=90000.0)
    ap.add_argument("--T0", type=float, default=285.0)
    ap.add_argument("--S0", type=float, default=1.0)
    ap.add_argument("--k-per-bin", type=int, default=16)
    ap.add_argument("--out", default="parcel_out")
    args = ap.parse_args(argv)

    if args.model_dir:
        op = load_from_checkpoint(args.model_dir, args.r_eval, args.prefactor or "autograd",
                                  fd_grid=args.fd_grid)
        if args.bins_file:
            op.grid = BinGrid.from_dataset(args.bins_file, args.r_eval or op.grid.r_eval_choice)
    else:
        r_eval = args.r_eval or "center"
        grid = (BinGrid.from_dataset(args.bins_file, r_eval) if args.bins_file
                else BinGrid.log_uniform(r_eval_choice=r_eval))
        op = load_from_torchscript(args.encoder_ts, args.decoder_ts, grid, args.m_scale,
                                   fd_grid=args.fd_grid)

    if args.mode == "box":
        forcing = ParcelForcing(S_of_t=lambda t: args.S_box)
    elif args.descend_after is not None:
        forcing = ParcelForcing(w=lambda t: args.w if t < args.descend_after else -args.w)
    else:
        forcing = ParcelForcing(w=lambda t: args.w)

    s0 = ParcelState0(p=args.p0, T=args.T0, S=args.S0)
    rho0 = rho_dry(s0.p, s0.T, s0.qv())
    ic = LognormalIC(args.N_cm3 * 1e6 / rho0, args.r_g_um * 1e-6, args.sigma_g)
    r0, xi = superdroplets_from_lognormal(op.grid, ic, args.k_per_bin)
    g0, *_ = rebin(op.grid, r0, xi)

    ref = LagrangianParcel(op.grid, forcing, use_kohler=op.use_kohler).run(
        s0, r0, xi, args.t_end, args.dt_out)
    rom = ROMParcel(op, forcing).run(s0, g0, args.t_end, args.dt_out, fixed_dt=args.fixed_dt)
    cmp = compare(op, rom, ref)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "parcel_results.npz",
             **{f"ref_{k}": v for k, v in ref.items() if isinstance(v, np.ndarray)},
             **{f"rom_{k}": v for k, v in rom.items() if isinstance(v, np.ndarray)},
             **{f"cmp_{k}": v for k, v in cmp.items()},
             r_edges=op.grid.edges)
    plot_comparison(op, rom, ref, cmp, out / "parcel_comparison.png")

    i = np.argmax(ref["S"])
    print(f"ref: S_max-1 = {100*(ref['S'][i]-1):.3f}% at t={ref['t'][i]:.0f}s, "
          f"ql_end = {1e3*ref['ql'][-1]:.3f} g/kg, nfev={ref['nfev']}")
    j = np.argmax(rom["S"])
    print(f"ROM: S_max-1 = {100*(rom['S'][j]-1):.3f}% at t={rom['t'][j]:.0f}s, "
          f"ql_end = {1e3*rom['ql'][-1]:.3f} g/kg, nfev={rom['nfev']}")
    print(f"spectral L1 at end: AE projection {cmp['err_ae_l1'][-1]:.3f}, "
          f"ROM total {cmp['err_rom_l1'][-1]:.3f}")
    print(f"wrote {out/'parcel_results.npz'} and {out/'parcel_comparison.png'}")


if __name__ == "__main__":
    main()
