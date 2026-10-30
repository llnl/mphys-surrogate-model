"""
Condensation physics utilities for computing theoretical condensation rates.

Implements thermodynamic equations for droplet growth/evaporation following
Köhler theory with Knudsen corrections.
"""

import torch
import numpy as np

# Thermodynamic constants
K = 0.024  # W / m / K, thermal conductivity for air
Rd = 287.0  # J/kg-K, dry air gas constant
Rv = 461.505  # J/kg-K, water vapor gas constant
Cp_d = 1004.5  # J/kg-K, specific heat of dry air
Cp_v = 1859.0  # J/kg-K, specific heat of water vapor
Cp_l = 4200.0  # J/kg-K, specific heat of liquid water
D_v = 2.21e-5  # m²/s, vapor diffusivity
rho_l = 1000.0  # kg/m³, liquid water density
L_v = 2.5e6  # J/kg, latent heat of vaporization

# Köhler theory parameters
a = 3.3e-7  # m·K, curvature effect parameter
b = 2.7e-23  # m³, solute effect parameter


def get_esat(dtt):
    """
    Compute saturation vapor pressure using polynomial approximation.

    Args:
        dtt: Temperature in Celsius (can be numpy array or torch tensor)

    Returns:
        Saturation vapor pressure in hPa
    """
    # Polynomial coefficients
    coeffs = [
        6.11239921,
        0.443987641,
        0.142986287e-1,
        0.264847430e-3,
        0.302950461e-5,
        0.206739458e-7,
        0.640689451e-10,
        -0.952447341e-13,
        -0.976195544e-15
    ]

    # Horner's method for polynomial evaluation
    result = coeffs[-1]
    for coeff in reversed(coeffs[:-1]):
        result = result * dtt + coeff

    return result


def G_func(T, R):
    """
    Compute growth coefficient G with Knudsen corrections.

    Pure torch when either input is a tensor (traceable/exportable and
    differentiable w.r.t. T); pure numpy otherwise.

    Args:
        T: Temperature [batch, 1, 1] in Kelvin (torch tensor or numpy array)
        R: Droplet radius [1, 1, n_bins] in meters (torch tensor or numpy array)

    Returns:
        G: Growth coefficient [batch, 1, n_bins] in m²/s
    """
    use_torch = torch.is_tensor(T) or torch.is_tensor(R)
    sqrt = torch.sqrt if use_torch else np.sqrt
    if use_torch and not torch.is_tensor(T):
        T = torch.as_tensor(T, dtype=R.dtype, device=R.device)

    # Mean free path of vapor molecules
    lambda_v = 2 * D_v / sqrt(8 * T * Rv / np.pi)

    # Knudsen number
    Kn = lambda_v / R

    # Diffusion correction factor
    dcf = (1 + Kn) / (1 + 2 * Kn * (1 + Kn))

    # Thermal factor F_k
    F_k = (L_v / Rv / T - 1) * L_v * rho_l / K / T

    # Saturation vapor pressure (polynomial in °C; works for tensors and arrays).
    # Previously this round-tripped through numpy for tensors, which detached T
    # from autograd and made torch.jit.trace bake in the example T as a constant.
    es_T = get_esat(T - 273.16) * 100  # hPa -> Pa

    # Diffusion factor F_d
    F_d = rho_l * Rv * T / (dcf * D_v * es_T)

    # Growth coefficient
    G = 1 / (F_k + F_d)

    return G


def drdt(R, T, S, use_kohler=False):
    """
    Compute droplet radius growth rate dr/dt.

    Args:
        R: Droplet radius [1, 1, n_bins] in meters
        T: Temperature [batch, 1, 1] in Kelvin
        S: Saturation ratio [batch, 1, 1] (dimensionless, e.g., 1.01 for 1% supersaturation)
        use_kohler: If True, include Köhler theory corrections for curvature and solute effects

    Returns:
        dr/dt: Radius growth rate [batch, 1, n_bins] in m/s
    """
    G = G_func(T, R)

    # Start with supersaturation term
    drdt = S - 1

    # Add Köhler corrections if requested
    if use_kohler:
        drdt = drdt - a / (R * T) + b / (R ** 3)

    # Final growth rate
    drdt = G * drdt / R

    return drdt


def bin_faces_from_centers(rbin_median):
    """
    Bin edges (faces) for log-uniform bins, reconstructed from geometric centers.

    Args:
        rbin_median: Geometric bin centers [n_bins] (or any shape ending in n_bins)

    Returns:
        (r_faces [n_bins + 1], dlnr) in the same array type as the input
    """
    rc = rbin_median.reshape(-1)
    if torch.is_tensor(rc):
        dlnr = torch.log(rc[1] / rc[0])
        faces = torch.cat([rc[:1] * torch.exp(-dlnr / 2), rc * torch.exp(dlnr / 2)])
    else:
        dlnr = np.log(rc[1] / rc[0])
        faces = np.concatenate([rc[:1] * np.exp(-dlnr / 2), rc * np.exp(dlnr / 2)])
    return faces, dlnr


def condensation_tendency_fv(g, v_center, v_face, dlnr, scheme="central"):
    """
    Finite-volume condensation tendency of a mass density g(ln r):

        dg/dt = -d(v g)/d(ln r) + 3 v g,     v = (dr/dt) / r = d(ln r)/dt

    The first term moves mass between bins (shape change); the second is the
    mass gain from growth (m ∝ r^3) and is the only term that changes total mass.
    Boundary fluxes at the smallest/largest faces are zero, consistent with the
    integration by parts behind compute_nwi_prefactor.

    Args:
        g: [..., n_bins] mass density per ln r
        v_center: [..., n_bins] d(ln r)/dt at bin centers (for the 3 v g term)
        v_face: [..., n_bins + 1] d(ln r)/dt at bin faces (for the flux term)
        dlnr: bin width in ln r (uniform)
        scheme: "central" (default; second order for smooth g, and the form whose
                encoding matches the latent projection P (v g)) or "upwind" (first
                order, monotone). Use "upwind" for *time integration* of g: central
                produces negative tendencies in empty bins next to spikes and blows
                up under strong evaporation.

    Returns:
        dgdt: [..., n_bins]
    """
    is_t = torch.is_tensor(g)
    vf = v_face[..., 1:-1]
    if scheme == "central":
        g_face = 0.5 * (g[..., :-1] + g[..., 1:])
    elif scheme == "upwind":
        g_face = torch.where(vf > 0, g[..., :-1], g[..., 1:]) if is_t else np.where(vf > 0, g[..., :-1], g[..., 1:])
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    F_int = vf * g_face
    if is_t:
        zero = torch.zeros_like(F_int[..., :1])
        F = torch.cat([zero, F_int, zero], dim=-1)
    else:
        zero = np.zeros_like(F_int[..., :1])
        F = np.concatenate([zero, F_int, zero], axis=-1)
    return -(F[..., 1:] - F[..., :-1]) / dlnr + 3 * v_center * g


def eulerian_from_lagrangian(dgdt_lag, dlnr, scheme="central"):
    """
    Convert a per-bin *Lagrangian* condensation tendency into the *Eulerian* bin
    tendency dg/dt.

    ERF's dgdt_cond is the mass-gain rate of the droplets that are currently in
    each bin, sum_{p in bin i} dm_p/dt / dlnr = 3 (v g)_i. It does not include
    droplets crossing bin edges. The Eulerian tendency (what the latent state
    h = Enc(g) actually feels) adds that flux:

        dg/dt = -d(v g)/d(ln r) + 3 v g,   with (v g)_i = dgdt_lag_i / 3

    No representative radius is needed: (v g)_i comes straight from the data.

    Args:
        dgdt_lag: [..., n_bins] Lagrangian per-bin tendency (e.g. ERF dgdt_cond)
        dlnr: bin width in ln r
        scheme: face value of (v g): "central" (default; Enc of the result equals
            P (dgdt_lag / 3) to second order, so it is the consistent training target)
            or "upwind" (donor cell; monotone, use for time integration)

    Returns:
        dgdt_euler: [..., n_bins]
    """
    vg = dgdt_lag / 3
    is_t = torch.is_tensor(vg)
    if scheme == "central":
        F_int = 0.5 * (vg[..., :-1] + vg[..., 1:])
    elif scheme == "upwind":
        # direction from the sign of v, which is uniform within a cell (sign of S - 1
        # without Kohler); vg alone is 0 in empty bins, so use the pair sum
        up = (vg[..., :-1] + vg[..., 1:]) > 0
        F_int = torch.where(up, vg[..., :-1], vg[..., 1:]) if is_t else np.where(up, vg[..., :-1], vg[..., 1:])
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    if is_t:
        zero = torch.zeros_like(F_int[..., :1])
        F = torch.cat([zero, F_int, zero], dim=-1)
    else:
        zero = np.zeros_like(F_int[..., :1])
        F = np.concatenate([zero, F_int, zero], axis=-1)
    return -(F[..., 1:] - F[..., :-1]) / dlnr + dgdt_lag


def compute_theoretical_condensation(g, T, S, rbin_median, use_kohler=False, r_faces=None,
                                     form="lagrangian", scheme="central"):
    """
    Theoretical condensation tendency from a binned DSD.

    Two different quantities, both called "dg/dt" in practice:

    form="lagrangian" (default)
        3 v g: mass-gain rate of the droplets currently in each bin. This is how
        ERF's dgdt_cond is defined, so use it when comparing against that data.
        It has the right total mass tendency but no transfer between bins, so
        integrating it in time grows each bin in place.

    form="eulerian"
        -d(v g)/d(ln r) + 3 v g: the rate of change of the binned DSD itself.
        This is what h = Enc(g) evolves with (compute_dhdt is its projection),
        and what a time integration of g needs.

    Here v = (dr/dt) / r = d(ln r)/dt.

    Args:
        g: DSD [batch, 1, n_bins] in kg/m³/ln(R)
        T: Temperature [batch, 1, 1] in Kelvin
        S: Saturation ratio [batch, 1, 1] (dimensionless)
        rbin_median: Geometric bin centers [n_bins] or [1, 1, n_bins] in meters
        use_kohler: If True, include Köhler theory corrections
        r_faces: Optional bin edges [n_bins + 1] (eulerian form); reconstructed
            from rbin_median (assumed log-uniform centers) if not given
        form: "lagrangian" or "eulerian"
        scheme: face interpolation for the eulerian form, "central" (default) or
            "upwind" (use for time integration; see condensation_tendency_fv)

    Returns:
        dgdt: Condensation tendency [batch, 1, n_bins] in kg/m³/s/ln(R)
    """
    lead = (1,) * (g.ndim - 1)
    rc = rbin_median.reshape(lead + (-1,))
    v_center = drdt(rc, T, S, use_kohler=use_kohler) / rc
    if form == "lagrangian":
        return 3 * v_center * g
    if form != "eulerian":
        raise ValueError(f"Unknown form: {form}")

    faces, dlnr = bin_faces_from_centers(rbin_median)
    if r_faces is not None:
        faces = r_faces.reshape(-1)
    rf = faces.reshape(lead + (-1,))
    v_face = drdt(rf, T, S, use_kohler=use_kohler) / rf
    return condensation_tendency_fv(g, v_center, v_face, dlnr, scheme=scheme)


def compute_projected_condensation(dhdt, Z, decoder):
    """
    Project latent condensation tendency to physical space using Jacobian-vector product.

    Computes: dg/dt = ∂g/∂h * dh/dt

    Args:
        dhdt: Predicted condensation tendency in latent space [batch, 1, n_latent+1]
        Z: Latent state [batch, 1, n_latent+1] (must require grad)
        decoder: NWI decoder model

    Returns:
        dgdt: Projected condensation tendency [batch, 1, n_bins] in physical space
    """
    # Use Jacobian-vector product to compute dgdt = ∂g/∂h * dhdt
    _, dgdt = torch.func.jvp(decoder, (Z,), (dhdt,))

    return dgdt


def compute_nwi_prefactor(encoder):
    """
    Compute the prefactor for NWI-based dhdt calculation.

    The prefactor is: df/d(lnr) + 3*f
    where f are the NWI weight functions.

    This is used to compute dh/dt from condensation velocities:
    dh_j/dt = sum_i [prefactor_j(r_i) * g(r_i) * v_cond(r_i)]

    Note: Only computes for the dimensionless latent variables (not mass).
    The last dimension (mass) has prefactor = 3 (from d(sum)/d(lnr) + 3*1).

    Args:
        encoder: NWI encoder with weight functions

    Returns:
        prefactor: [n_latent+1, n_bins] array
    """

    # Get weight function matrix
    f = encoder.wf_mat().detach()  # [n_bins, n_latent+1]
    n_bins = f.shape[0]
    n_latent_total = f.shape[1]
    n_latent = n_latent_total - 1  # Exclude mass dimension

    # Compute derivatives of weight functions w.r.t. log(r)
    # The encoder uses a transformed coordinate: x = 2*(ln(r) - ln_min)/(ln_max - ln_min) - 1
    # So d/d(lnr) = dx/d(lnr) * d/dx
    lnr_min = np.log(1e-6)
    lnr_max = np.log(5e-3)
    dx_dlnr = 2.0 / (lnr_max - lnr_min)  # Coordinate transformation factor

    # For each weight function, compute numerical derivative
    df_dlnr = torch.zeros_like(f)

    # Only loop over the actual weight functions (not including mass)
    for j in range(n_latent):
        # Get the log-space bin coordinates for this weight function
        wf = encoder.wfs[j]
        log_bin_mass = wf.log_bin_mass.detach()  # [n_bins, 1]

        # Compute Jacobian of this weight function w.r.t. its input
        # This gives us df_j/dx for each bin
        jacobian = torch.autograd.functional.jacobian(
            lambda x: wf.stack(x),
            log_bin_mass
        )

        # Extract diagonal (derivative at each bin)
        for i in range(n_bins):
            df_dlnr[i, j] = jacobian[i, 0, i, 0] * dx_dlnr * f[i, j]

    # For the mass dimension (last column), the weight function is constant 1
    # So df/d(lnr) = 0, and prefactor = 0 + 3*1 = 3
    df_dlnr[:, -1] = 0.0

    # Compute prefactor: df/d(lnr) + 3*f
    prefactor = df_dlnr + 3.0 * f

    return prefactor.T  # Return as [n_latent+1, n_bins] for easier indexing


def compute_dhdt(Z, T, S, rbin_median, encoder, decoder, m_scale=1.0, use_kohler=False):
    """
    Compute dh/dt (condensation tendency in latent space) from thermodynamics.

    This is the key function from condensation_test.ipynb that computes the
    latent-space condensation tendency using:
    - Condensation velocity v_cond from thermodynamics
    - Reconstructed DSD g from decoder (in NORMALIZED units, same as training)
    - NWI weight function derivatives (prefactor)

    dh_j/dt = sum_i [prefactor_j(r_i) * g(r_i) * v_cond(r_i)]

    IMPORTANT: The encoder was trained on normalized DSDs (divided by m_scale).
    To be consistent, we compute dhdt using normalized g, so the result is in
    the same "normalized latent space" units as encoder(dgdt_normalized).

    Args:
        Z: Latent state [batch, 1, n_latent+1]
        T: Temperature [batch, 1, 1] in Kelvin
        S: Saturation ratio [batch, 1, 1]
        rbin_median: Median bin radii [1, 1, n_bins] in meters
        encoder: NWI encoder (to get prefactor from weight functions)
        decoder: NWI decoder (to reconstruct DSD)
        m_scale: Mass scale (kept for API compatibility but not used)
        use_kohler: Whether to include Köhler corrections

    Returns:
        dhdt: Condensation tendency in latent space [batch, 1, n_latent+1]
              In "normalized latent space" units (consistent with encoder training)
    """

    # Get reconstructed DSD (normalized - same as what encoder was trained on)
    g = decoder(Z)  # [batch, 1, n_bins] - normalized by m_scale

    # Compute condensation velocity for each bin
    # v_cond = dr/dt / r (units: 1/s)
    drdt_val = drdt(rbin_median, T, S, use_kohler=use_kohler)  # [batch, 1, n_bins]
    v_cond = drdt_val / rbin_median  # [batch, 1, n_bins]

    # Get prefactor from NWI weight functions (dimensionless)
    prefactor = compute_nwi_prefactor(encoder)  # [n_latent+1, n_bins]
    prefactor = prefactor.to(Z.device).float()

    # Compute dh/dt for each latent dimension
    # dh_j/dt = sum_i [prefactor_j(r_i) * g(r_i) * v_cond(r_i)]
    # Note: g is normalized, so dhdt is in "normalized latent space" units
    n_latent = Z.shape[-1]

    dhdt = torch.zeros_like(Z)  # [batch, 1, n_latent+1]

    for j in range(n_latent):
        # prefactor[j, :] is [n_bins] (dimensionless)
        # g is [batch, 1, n_bins] (normalized, dimensionless)
        # v_cond is [batch, 1, n_bins] (1/s)
        # Result: sum over bins, units: 1/s (in normalized space)
        dhdt[:, :, j] = torch.sum(
            prefactor[j:j+1, :] * g * v_cond,
            dim=-1,
            keepdim=True
        ).squeeze(-1)

    return dhdt
