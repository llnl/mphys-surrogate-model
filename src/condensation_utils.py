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

    Args:
        T: Temperature [batch, 1, 1] in Kelvin (torch tensor or numpy array)
        R: Droplet radius [1, 1, n_bins] in meters (torch tensor or numpy array)

    Returns:
        G: Growth coefficient [batch, 1, n_bins] in m²/s
    """
    is_numpy = isinstance(T, np.ndarray)

    if is_numpy:
        sqrt = np.sqrt
        pi = np.pi
    else:
        sqrt = torch.sqrt
        pi = np.pi

    # Mean free path of vapor molecules
    lambda_v = 2 * D_v / sqrt(8 * T * Rv / pi)

    # Knudsen number
    Kn = lambda_v / R

    # Diffusion correction factor
    dcf = (1 + Kn) / (1 + 2 * Kn * (1 + Kn))

    # Thermal factor F_k
    F_k = (L_v / Rv / T - 1) * L_v * rho_l / K / T

    # Saturation vapor pressure (convert T from K to °C)
    if is_numpy:
        es_T = get_esat(T - 273.16)
    else:
        # For torch tensors, convert to numpy for esat computation
        es_T_np = get_esat((T - 273.16).cpu().numpy())
        es_T = torch.from_numpy(es_T_np).to(T.device)

    es_T = es_T * 100  # Convert from hPa to Pa

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


def compute_theoretical_condensation(g, T, S, rbin_median, use_kohler=False):
    """
    Compute theoretical condensation tendency dg/dt from reconstructed DSD.

    The condensation tendency is computed as:
        dg/dt = 3 * g * (dr/dt) / r

    where dr/dt is the droplet radius growth rate from Köhler theory.

    Args:
        g: Reconstructed DSD [batch, 1, n_bins] in kg/m³/ln(R)
        T: Temperature [batch, 1, 1] in Kelvin
        S: Saturation ratio [batch, 1, 1] (dimensionless)
        rbin_median: Median bin radii [1, 1, n_bins] in meters (torch tensor or numpy array)
        use_kohler: If True, include Köhler theory corrections

    Returns:
        dgdt: Condensation tendency [batch, 1, n_bins] in kg/m³/s/ln(R)
    """
    # Ensure rbin_median has the right shape
    if len(rbin_median.shape) == 1:
        # Reshape from [n_bins] to [1, 1, n_bins]
        if isinstance(rbin_median, np.ndarray):
            rbin_median = rbin_median.reshape(1, 1, -1)
        else:
            rbin_median = rbin_median.reshape(1, 1, -1)

    # Compute dr/dt
    drdt_val = drdt(rbin_median, T, S, use_kohler=use_kohler)

    # Compute dg/dt = 3 * g * (dr/dt) / r
    dgdt = 3 * g * drdt_val / rbin_median

    return dgdt


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
    import torch

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
    import torch

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
