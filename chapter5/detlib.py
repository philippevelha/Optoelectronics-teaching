"""
detlib.py -- small, transparent toolbox for the Chapter 5 companion
(Photodetectors and image sensors).

Units: SI throughout (m, m^-1, s, W, A, V, F), EXCEPT photon and band energies, which are in eV.
The slides quote absorption coefficients in m^-1 (as here) but ionisation coefficients and fields in
cm^-1 and V cm^-1: the APD functions say which unit they expect.

Contents
--------
- plotting style            : use_style()
- photons and bands         : photon_energy_eV(), wavelength_from_eV(), cutoff_wavelength(), varshni()
- absorption                : MATERIALS, alpha(), penetration_depth(), alpha_direct_model()
- efficiency, responsivity  : responsivity(), quantum_efficiency(), max_qe_point(), fresnel_T(),
                              photocurrent_structure(), pin_photocurrent()
- carrier transport         : drift_velocity(), diffusion_time(), transit_time()
- Shockley-Ramo             : ramo_single(), pin_impulse_response(), transit_response(), pin_bandwidth()
- avalanche photodiodes     : chynoweth(), IONIZATION, apd_gain_uniform(), apd_gain_profile(), miller_gain(),
                              excess_noise_mcintyre(), excess_noise_power()
- photoconductors           : photoconductive_gain(), photoconductivity()
- circuits                  : diode_current(), operating_point(), junction_capacitance(), rc_cutoff(), rise_time()
- noise                     : shot_noise(), thermal_noise(), snr(), nep(), detectivity(), p_min_ideal(), apd_snr(),
                              apd_optimum_gain()
- receivers                 : ber_from_q(), q_from_ber(), sensitivity_pin(), sensitivity_apd(), quantum_limit()
- solar cells               : solar_iv(), solar_operating_point(), solar_mpp(), shockley_queisser()
- image sensors             : cte_remaining(), sensor_snr(), simulate_exposure()
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants and units
# ---------------------------------------------------------------------------
c0 = 299_792_458.0
h = 6.62607015e-34
qe = 1.602176634e-19
kB = 1.380649e-23
eps0 = 8.8541878128e-12
sigma_SB = 5.670374419e-8
HC_EVUM = h * c0 / qe * 1e6      # 1.23984 eV µm

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def use_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "figure.figsize": (7.5, 4.2), "figure.dpi": 110,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "lines.linewidth": 2, "xtick.color": INK2, "ytick.color": INK2,
        "legend.frameon": False, "font.size": 10.5,
        "axes.titlesize": 11.5, "axes.titleweight": "bold",
    })


def kT(T=300.0):
    """Thermal energy k_B T in eV (0.02585 eV at 300 K)."""
    return kB * T / qe


def photon_energy_eV(wl):
    """hν in eV for a free-space wavelength wl (m): 1.2398/λ[µm]."""
    return h * c0 / np.asarray(wl, dtype=float) / qe


def wavelength_from_eV(E):
    return h * c0 / (np.asarray(E, dtype=float) * qe)


def cutoff_wavelength(Eg):
    """Absorption cut-off λ_g (m) for a band gap Eg (eV): λ_g[µm] ≈ 1.24/Eg[eV] (slide 18)."""
    return wavelength_from_eV(Eg)


def varshni(T, Eg0, a, b):
    """Band gap versus temperature, Eg(T) = Eg0 − aT²/(T + b) (eV)."""
    T = np.asarray(T, dtype=float)
    return Eg0 - a * T**2 / (T + b)


def dB(x):
    return 10 * np.log10(x)


# ---------------------------------------------------------------------------
# Absorption (slides 16-20)
# ---------------------------------------------------------------------------
# Absorption coefficients at 300 K, tables of (λ in µm, log10 α in m^-1).
# Si: M. A. Green, Sol. Energy Mater. Sol. Cells 92, 1305 (2008), which is also the curve of slide 16.
# Ge, GaAs, InP, In0.7Ga0.3As0.64P0.36: digitised from the figure of slide 16 (Kasap, Fig. 5.5).
# In0.53Ga0.47As: literature values (≈ 7×10^5 m^-1 at 1.55 µm); the slide-16 curve is lower near 1.55 µm.
# Varshni parameters (Eg0, a, b) from Vurgaftman et al. / Sze; the alloys use typical values.
MATERIALS = {
    "Si": dict(Eg=1.12, kind="indirect", varshni=(1.170, 4.73e-4, 636), n=3.6, table=[
        (0.30, 8.26), (0.35, 8.02), (0.40, 6.98), (0.45, 6.41), (0.50, 6.05), (0.55, 5.81), (0.60, 5.62),
        (0.65, 5.45), (0.70, 5.28), (0.75, 5.11), (0.80, 4.93), (0.85, 4.73), (0.90, 4.49), (0.95, 4.20),
        (1.00, 3.81), (1.05, 3.21), (1.10, 2.54), (1.15, 1.43), (1.20, 0.34)]),
    "Ge": dict(Eg=0.66, kind="indirect", varshni=(0.7437, 4.774e-4, 235), n=4.0, table=[
        (0.35, 8.00), (0.40, 7.73), (0.50, 7.60), (0.55, 7.44), (0.60, 7.19), (0.70, 6.73), (0.80, 6.48),
        (0.90, 6.29), (1.00, 6.15), (1.20, 6.05), (1.30, 5.98), (1.40, 5.88), (1.50, 5.70), (1.52, 5.45),
        (1.54, 4.70), (1.56, 4.30), (1.60, 4.05), (1.70, 3.62), (1.80, 3.20), (1.87, 2.0)]),
    "GaAs": dict(Eg=1.42, kind="direct", varshni=(1.519, 5.405e-4, 204), n=3.6, table=[
        (0.40, 7.10), (0.50, 6.90), (0.57, 6.73), (0.60, 6.57), (0.70, 6.07), (0.80, 5.78), (0.85, 5.57),
        (0.87, 5.28), (0.875, 4.70), (0.88, 3.79), (0.89, 3.12), (0.90, 2.0)]),
    "InP": dict(Eg=1.35, kind="direct", varshni=(1.421, 4.9e-4, 327), n=3.2, table=[
        (0.40, 7.0), (0.50, 6.70), (0.60, 6.45), (0.70, 6.30), (0.80, 6.15), (0.90, 6.00), (0.91, 5.53),
        (0.92, 5.11), (0.94, 4.29), (0.96, 3.04), (0.97, 2.0)]),
    "In0.7Ga0.3As0.64P0.36": dict(Eg=0.89, kind="direct", varshni=(0.96, 4.3e-4, 250), n=3.4, table=[
        (0.80, 6.80), (1.00, 6.60), (1.145, 6.44), (1.25, 6.03), (1.30, 5.94), (1.35, 5.61), (1.38, 5.30),
        (1.39, 4.50), (1.40, 3.80), (1.41, 3.25), (1.42, 2.0)]),
    "In0.53Ga0.47As": dict(Eg=0.75, kind="direct", varshni=(0.81, 4.9e-4, 301), n=3.6, table=[
        (0.80, 6.50), (1.00, 6.30), (1.20, 6.15), (1.30, 6.08), (1.40, 6.00), (1.50, 5.90), (1.55, 5.85),
        (1.60, 5.72), (1.63, 5.50), (1.65, 5.10), (1.67, 4.30), (1.69, 3.20), (1.72, 2.0)]),
}


def alpha(material, wl):
    """Absorption coefficient α (m^-1) of a material of MATERIALS at the wavelength wl (m), 300 K.
    Interpolated in log10 α; zero beyond the last tabulated point (no band-to-band absorption)."""
    tab = np.array(MATERIALS[material]["table"])
    x = np.asarray(wl, dtype=float) * 1e6
    la = np.interp(x, tab[:, 0], tab[:, 1], left=np.nan, right=-np.inf)
    la = np.where(np.isnan(la), tab[0, 1], la)
    return 10.0 ** la


def penetration_depth(material, wl):
    """δ = 1/α (m), the depth where the intensity falls to 1/e (slide 16)."""
    a = alpha(material, wl)
    with np.errstate(divide="ignore"):
        return 1.0 / a


def alpha_direct_model(E, Eg, A=3e6, EU=0.007):
    """Generic direct-gap edge: α = A√(E − Eg) (m^-1, A in m^-1 eV^-1/2) above the gap, joined to an
    exponential Urbach tail α ∝ exp((E − Eg)/E_U) below it."""
    E = np.asarray(E, dtype=float)
    E1 = Eg + EU / 2                       # where the slopes of √ and exp match
    a1 = A * np.sqrt(EU / 2)
    return np.where(E > E1, A * np.sqrt(np.clip(E - Eg, 0, None)), a1 * np.exp((E - E1) / EU))


# ---------------------------------------------------------------------------
# Quantum efficiency and responsivity (slides 21-28, 41-43, 139-145, 150)
# ---------------------------------------------------------------------------
def responsivity(eta, wl):
    """R = η e λ/(hc) (A/W) (slide 22)."""
    return eta * qe * np.asarray(wl, dtype=float) / (h * c0)


def quantum_efficiency(R, wl):
    """External QE η_e = R hc/(eλ) (slide 21)."""
    return np.asarray(R, dtype=float) * h * c0 / (qe * np.asarray(wl, dtype=float))


def max_qe_point(wl, R):
    """Point where η = Rhc/eλ is maximum, i.e. where the tangent to R(λ) passes through the origin
    (dR/dλ = R/λ, slide 26). Returns (λ, R, η)."""
    eta = quantum_efficiency(R, wl)
    i = int(np.nanargmax(eta))
    return wl[i], R[i], eta[i]


def fresnel_T(n, n0=1.0):
    """Normal-incidence power transmission into a medium of index n: 1 − ((n − n0)/(n + n0))²."""
    return 1 - ((n - n0) / (n + n0)) ** 2


def photocurrent_structure(Po, wl, a, lp, W, Le=0.0, Lh=0.0, T=1.0, eta_i=1.0):
    """Photocurrent of a p⁺n (or pin) photodiode lit from the p side (slide 144):
    I_ph = (eη_iTP_o/hν){exp[−α(ℓ_p − L_e)] − exp[−α(ℓ_p + W + L_h)]}.
    Carriers generated within a diffusion length of the depletion region are collected; a = α (m^-1)."""
    hv = h * c0 / np.asarray(wl, dtype=float)
    start = np.maximum(lp - Le, 0.0)
    return qe * eta_i * T * Po / hv * (np.exp(-a * start) - np.exp(-a * (lp + W + Lh)))


def pin_photocurrent(Po, wl, a, W, T=1.0, eta_i=1.0):
    """Steady-state photocurrent of a pin with a very thin p⁺ layer (slide 43): (eη_iTP_o/hν)[1 − exp(−αW)]."""
    return photocurrent_structure(Po, wl, a, 0.0, W, T=T, eta_i=eta_i)


# ---------------------------------------------------------------------------
# Carrier transport (slides 31-32, 37-40)
# ---------------------------------------------------------------------------
# Caughey–Thomas model v = µE/[1 + (µE/v_sat)^β]^(1/β) (Jacoboni et al. 1977, Si at 300 K).
DRIFT = {
    ("Si", "e"): dict(mu=0.135, vsat=1.07e5, beta=1.11),
    ("Si", "h"): dict(mu=0.048, vsat=8.37e4, beta=1.21),
    ("InGaAs", "e"): dict(mu=1.0, vsat=7e4, beta=1.0),     # no overshoot modelled; v_sat typical
    ("InGaAs", "h"): dict(mu=0.03, vsat=5e4, beta=1.0),
}


def drift_velocity(E, carrier="e", mat="Si"):
    """Drift velocity (m/s) at the field E (V/m), Caughey–Thomas model (slide 31)."""
    p = DRIFT[(mat, carrier)]
    E = np.abs(np.asarray(E, dtype=float))
    v0 = p["mu"] * E
    return v0 / (1 + (v0 / p["vsat"]) ** p["beta"]) ** (1 / p["beta"])


def diffusion_time(ell, D):
    """Mean time to diffuse a distance ℓ, t = ℓ²/(2D) (slide 32)."""
    return ell**2 / (2 * D)


def transit_time(W, v):
    return W / v


# ---------------------------------------------------------------------------
# Shockley–Ramo theorem and transit-time response (slides 11-15)
# ---------------------------------------------------------------------------
def ramo_single(t, l, L, ve, vh):
    """External currents (A) of one EHP created at x = l between electrodes at 0 and L (slides 12-15).
    The electron drifts to x = L, the hole to x = 0. Returns (i_e, i_h)."""
    t = np.asarray(t, dtype=float)
    ie = np.where((t >= 0) & (t < (L - l) / ve), qe * ve / L, 0.0)
    ih = np.where((t >= 0) & (t < l / vh), qe * vh / L, 0.0)
    return ie, ih


def _generation_profile(W, a, nx):
    x = (np.arange(nx) + 0.5) * W / nx
    if a <= 0:
        g = np.ones(nx)
    else:
        g = np.exp(-a * x)
    return x, g / g.sum()


def pin_impulse_response(t, W, a, ve, vh, nx=400):
    """Current per photogenerated EHP (A, i.e. charge e in total) after a δ-pulse absorbed with the
    profile exp(−αx) inside an i-layer of width W (light enters at x = 0, electrons drift to x = W).
    Returns (i_e, i_h); a = 0 gives uniform generation."""
    t = np.asarray(t, dtype=float)[:, None]
    x, g = _generation_profile(W, a, nx)
    ie = (g * ((t >= 0) & (t < (W - x) / ve))).sum(axis=1) * qe * ve / W
    ih = (g * ((t >= 0) & (t < x / vh))).sum(axis=1) * qe * vh / W
    return ie, ih


def transit_response(f, W, a, ve, vh, nx=400):
    """Normalised transit-time frequency response H(f) of a pin (complex), Fourier transform of
    pin_impulse_response: each carrier contributes (ev/W)(1 − e^{−jωT})/(jω)."""
    f = np.asarray(f, dtype=float)[:, None]
    w = 2 * np.pi * np.where(f == 0, 1e-30, f)
    x, g = _generation_profile(W, a, nx)
    Te, Th = (W - x) / ve, x / vh
    H = (g * (ve / W * (1 - np.exp(-1j * w * Te)) + vh / W * (1 - np.exp(-1j * w * Th))) / (1j * w)).sum(axis=1)
    return H / (g * (ve * Te + vh * Th) / W).sum()


def f3dB(f, H):
    """First frequency where |H| falls below 1/√2."""
    mag = np.abs(H) / np.abs(H[0])
    i = np.argmax(mag < 1 / np.sqrt(2))
    return f[i] if mag[i] < 1 / np.sqrt(2) else np.nan


def junction_capacitance(A, W, eps_r=11.9):
    """Depletion (parallel-plate) capacitance C = ε₀ε_rA/W (slide 30)."""
    return eps0 * eps_r * A / W


def pin_bandwidth(W, A, RL, a=0.0, ve=1e5, vh=7e4, eps_r=11.9, C_par=0.0):
    """−3 dB bandwidth of a pin limited by both transit time and RC: |H_tr(f)|·|1/(1 + j2πfRC)|.
    Returns (f_total, f_transit, f_RC)."""
    C = junction_capacitance(A, W, eps_r) + C_par
    fRC = 1 / (2 * np.pi * RL * C)
    f = np.linspace(0, 2.0 * max(ve, vh) / W, 1500)          # transit response vanishes beyond ~v/W
    Htr = transit_response(f, W, a, ve, vh, nx=150)
    return f3dB(f, Htr / (1 + 1j * f / fRC)), f3dB(f, Htr), fRC


# ---------------------------------------------------------------------------
# Avalanche photodiodes (slides 44-60, 99-100)
# ---------------------------------------------------------------------------
# Chynoweth's law α = A exp(−B/E), A in cm^-1 and B in V/cm.
# Si: values of the slide-58 example. InP: Cook, Bedair et al. (1982), widely used in APD design.
IONIZATION = {
    "Si": {"e": (0.740e6, 1.16e6), "h": (0.725e6, 2.20e6)},
    "InP": {"e": (1.12e7, 3.11e6), "h": (4.79e6, 2.55e6)},
}


def chynoweth(E_Vcm, mat="Si", carrier="e"):
    """Impact-ionisation coefficient (cm^-1) at the field E (V/cm) (slide 50)."""
    A, B = IONIZATION[mat][carrier]
    return A * np.exp(-B / np.asarray(E_Vcm, dtype=float))


def apd_gain_uniform(alpha_e, k, w):
    """Multiplication for electron injection into a uniform avalanche region of width w (slide 51):
    M = (1 − k)/(exp[−(1 − k)α_e w] − k). α_e and w in consistent units (e.g. cm^-1 and cm).
    Returns inf at and beyond breakdown."""
    alpha_e, k = np.asarray(alpha_e, dtype=float), np.asarray(k, dtype=float)
    one = np.isclose(k, 1.0)
    kk = np.where(one, 0.5, k)                                  # placeholder, k = 1 handled below
    M = (1 - kk) / (np.exp(-(1 - kk) * alpha_e * w) - kk)
    M = np.where(one, 1 / (1 - alpha_e * w), M)
    return np.where(M >= 1, M, np.inf)                           # M < 1 or M < 0 means beyond breakdown


def apd_gain_profile(x, ae, ah):
    """Multiplication for electrons injected at x[0] and drifting to x[-1] through a field profile
    with local coefficients ae(x), ah(x) (McIntyre): M = 1/(1 − ∫ α_e exp[−∫(α_e − α_h)dx'] dx)."""
    x, ae, ah = map(np.asarray, (x, ae, ah))
    d = np.concatenate(([0.0], np.cumsum(0.5 * (ae[1:] - ah[1:] + ae[:-1] - ah[:-1]) * np.diff(x))))
    integrand = ae * np.exp(-d)
    I = np.trapezoid(integrand, x) if hasattr(np, "trapezoid") else np.trapz(integrand, x)
    return 1 / (1 - I) if I < 1 else np.inf


def miller_gain(V, Vbr, m=3.0):
    """Empirical multiplication M = 1/[1 − (V_r/V_br)^m] (slide 46)."""
    V = np.asarray(V, dtype=float)
    return 1 / (1 - (V / Vbr) ** m)


def excess_noise_mcintyre(M, k):
    """McIntyre excess noise factor for pure electron injection, F = kM + (1 − k)(2 − 1/M)."""
    M = np.asarray(M, dtype=float)
    return k * M + (1 - k) * (2 - 1 / M)


def excess_noise_power(M, x):
    """Empirical F ≈ M^x (slide 100)."""
    return np.asarray(M, dtype=float) ** x


# ---------------------------------------------------------------------------
# Photoconductors (slides 69-74)
# ---------------------------------------------------------------------------
def photoconductive_gain(tau, L, V, mu_e, mu_h=0.0):
    """G = τ/t_e + τ/t_h with t = L²/(µV) (slide 74)."""
    return tau * V * (mu_e + mu_h) / L**2


def photoconductivity(I_opt, wl, d, tau, mu_e, mu_h, eta_i=1.0):
    """Δσ = eη_iIλτ(µ_e + µ_h)/(hcd) for an intensity I (W/m²) absorbed in a thickness d (slide 72)."""
    return qe * eta_i * I_opt * wl * tau * (mu_e + mu_h) / (h * c0 * d)


# ---------------------------------------------------------------------------
# Circuits (slides 75-85, 122-129)
# ---------------------------------------------------------------------------
def diode_current(V, Io, Iph=0.0, n=1.0, T=300.0):
    """Illuminated diode, I = Io[exp(eV/nkT) − 1] − I_ph (positive I and V as for forward bias)."""
    return Io * np.expm1(np.asarray(V, dtype=float) / (n * kT(T))) - Iph


def operating_point(Iph, Io, Vr, RL, n=1.0, T=300.0):
    """Photodiode reverse biased by V_r (> 0) through R_L (slide 76): solves
    Io[exp(V/nkT) − 1] − I_ph = −(V + V_r)/R_L for V (diode voltage, < 0 in photodetection). Returns (V, I)."""
    from scipy.optimize import brentq
    f = lambda V: diode_current(V, Io, Iph, n, T) + (V + Vr) / RL
    V = brentq(f, -Vr - 1e-9, 2.0)
    return V, diode_current(V, Io, Iph, n, T)


def rc_cutoff(R, C):
    """f_c = 1/(2πRC) (slide 80)."""
    return 1 / (2 * np.pi * R * C)


def rise_time(R, C):
    """10–90 % rise (or fall) time of an RC circuit, 2.2RC, and f_c = 0.35/τ_r (slide 84)."""
    return 2.2 * R * C


# ---------------------------------------------------------------------------
# Noise (slides 86-107)
# ---------------------------------------------------------------------------
def shot_noise(I, B, M=1.0, F=1.0):
    """RMS shot-noise current [2eI M²F B]^1/2, I the unmultiplied current (slides 89, 99)."""
    return np.sqrt(2 * qe * np.asarray(I, dtype=float) * M**2 * F * B)


def thermal_noise(R, B, T=300.0):
    """RMS Johnson noise current of a resistor, (4kTB/R)^1/2 (slide 92)."""
    return np.sqrt(4 * kB * T * B / R)


def snr(Iph, Id, B, RL=np.inf, T=300.0, M=1.0, F=1.0, i_amp=0.0):
    """Power SNR = (M I_ph)²/[2e(I_d + I_ph)M²FB + 4kTB/R_L + i_amp²] (slides 94, 107).
    I_ph and I_d are the primary (unmultiplied) currents; i_amp is an RMS amplifier noise current."""
    th = 0.0 if np.isinf(RL) else 4 * kB * T * B / RL
    return (M * Iph) ** 2 / (2 * qe * (Id + Iph) * M**2 * F * B + th + i_amp**2)


def nep(R, Id, Iph=0.0):
    """Shot-noise NEP = [2e(I_d + I_ph)]^1/2/R (W Hz^-1/2) (slide 95)."""
    return np.sqrt(2 * qe * (Id + Iph)) / R


def detectivity(A, NEP):
    """Specific detectivity D* = A^1/2/NEP; with A in cm² it is in cm Hz^1/2 W^-1 (Jones) (slide 96)."""
    return np.sqrt(A) / NEP


def p_min_ideal(wl, B, eta=1.0):
    """Optical power for SNR = 1 of an ideal, noiseless-amplifier detector with no dark current,
    P₁ = 2hcB/(ηλ) (slide 101)."""
    return 2 * h * c0 * B / (eta * wl)


def apd_snr(Po, R1, M, Id0, B, x=None, k=None, RL=np.inf, T=300.0, i_amp=0.0):
    """SNR of an APD of unmultiplied responsivity R1 (A/W), with F = M^x or McIntyre(k)."""
    F = excess_noise_power(M, x) if x is not None else excess_noise_mcintyre(M, k)
    return snr(R1 * Po, Id0, B, RL, T, M, F, i_amp)


def apd_optimum_gain(Po, R1, Id0, B, RL, T=300.0, x=None, k=None, Mmax=1000):
    """Gain that maximises the APD SNR when thermal noise is present. Returns (M_opt, SNR_opt)."""
    M = np.logspace(0, np.log10(Mmax), 4000)
    s = apd_snr(Po, R1, M, Id0, B, x, k, RL, T)
    i = int(np.argmax(s))
    return M[i], s[i]


# ---------------------------------------------------------------------------
# Digital receivers (Going further, links to Ch. 2 and Ch. 4)
# ---------------------------------------------------------------------------
def ber_from_q(Q):
    """BER = ½ erfc(Q/√2) for on–off keying with Gaussian noise."""
    from scipy.special import erfc
    return 0.5 * erfc(np.asarray(Q, dtype=float) / np.sqrt(2))


def q_from_ber(ber):
    from scipy.special import erfcinv
    return np.sqrt(2) * erfcinv(2 * np.asarray(ber, dtype=float))


def sensitivity_pin(Q, R, B, RL=None, T=300.0, Id=0.0, i_amp=None):
    """Average received power (W) for a target Q with a pin receiver, OOK with zero-power '0':
    Q = I₁/(σ₁ + σ₀), σ₀² = σ_T² + 2eI_dB, σ₁² = σ₀² + 2eI₁B, I₁ = 2RP̄.
    σ_T from R_L (4kTB/R_L) or directly i_amp (A rms)."""
    sT2 = (i_amp**2 if i_amp is not None else 4 * kB * T * B / RL)
    s02 = sT2 + 2 * qe * Id * B
    # Solve I1 = Q(σ1 + σ0)  ->  I1 = Q(√(σ0² + 2eBI1) + σ0): quadratic in I1
    s0 = np.sqrt(s02)
    I1 = 2 * qe * B * Q**2 + 2 * Q * s0
    return I1 / (2 * R)


def sensitivity_apd(Q, R1, B, M, F, RL=None, T=300.0, i_amp=None):
    """Average received power for a target Q with an APD (dark current neglected):
    I₁ = Q(σ₁ + σ_T), σ₁² = σ_T² + 2eM²F R1P₁B, I₁ = MR1P₁, P̄ = P₁/2."""
    sT = np.sqrt(i_amp**2 if i_amp is not None else 4 * kB * T * B / RL)
    # M R1 P1 = Q(√(σT² + 2e M² F R1 P1 B) + σT)  -> quadratic in y = M R1 P1
    y = Q**2 * 2 * qe * F * M * B + 2 * Q * sT
    return y / (M * R1) / 2


def quantum_limit(ber, wl, B):
    """Ideal photon-counting receiver: BER = ½exp(−N₁) with N₁ photons in a '1' bit;
    returns (average photons per bit N̄ = N₁/2, average power N̄hνB)."""
    N1 = -np.log(2 * np.asarray(ber, dtype=float))
    Nbar = N1 / 2
    return Nbar, Nbar * h * c0 / wl * B


# ---------------------------------------------------------------------------
# Solar cells (slides 121-137)
# ---------------------------------------------------------------------------
def _lambertw_exp(logx):
    """W(exp(logx)) without overflow (asymptotic series for large arguments)."""
    from scipy.special import lambertw
    logx = np.asarray(logx, dtype=float)
    small = logx < 500
    w = np.empty_like(logx)
    w[small] = np.real(lambertw(np.exp(logx[small])))
    L = logx[~small]
    w[~small] = L - np.log(L) + np.log(L) / L
    return w


def solar_iv(V, Iph, Io, n=1.0, Rs=0.0, Rp=np.inf, T=300.0):
    """Current I(V) (A) of the one-diode solar-cell model (slide 128), sign convention of slide 123
    (I < 0 when the cell delivers power): I = Io[exp((V − IR_s)/nkT) − 1] + (V − IR_s)/R_p − I_ph.
    Solved exactly with the Lambert W function."""
    V = np.asarray(V, dtype=float)
    Vt = n * kT(T)
    if Rs == 0:
        return Io * np.expm1(V / Vt) + (0 if np.isinf(Rp) else V / Rp) - Iph
    G = 0.0 if np.isinf(Rp) else 1 / Rp
    # (V − IRs) = Vd ; I = Io(e^{Vd/Vt} − 1) + G Vd − Iph ; V = Vd + Rs I
    # => V + Rs(Iph + Io) = Vd(1 + Rs G) + Rs Io e^{Vd/Vt}
    a = 1 + Rs * G
    rhs = V + Rs * (Iph + Io)
    Vd = rhs / a - Vt * _lambertw_exp(np.log(Rs * Io / (a * Vt)) + rhs / (a * Vt))
    return (V - Vd) / Rs


def solar_operating_point(R, Iph, Io, n=1.0, Rs=0.0, Rp=np.inf, T=300.0):
    """Intersection of the cell I–V with the load line I = −V/R (slide 125). Returns (V, I)."""
    from scipy.optimize import brentq
    Voc = brentq(lambda V: solar_iv(V, Iph, Io, n, Rs, Rp, T), 0, 5 * n * kT(T) * np.log(Iph / Io + 1) + 1e-3)
    V = brentq(lambda V: solar_iv(V, Iph, Io, n, Rs, Rp, T) + V / R, 0, Voc)
    return V, -V / R


def solar_mpp(Iph, Io, n=1.0, Rs=0.0, Rp=np.inf, T=300.0, npts=4000):
    """Maximum power point, V_oc, I_sc and fill factor (slide 126). Returns a dict."""
    from scipy.optimize import brentq
    Voc = brentq(lambda V: solar_iv(V, Iph, Io, n, Rs, Rp, T), 0, 5 * n * kT(T) * np.log(Iph / Io + 1) + 1e-3)
    Isc = -float(solar_iv(0.0, Iph, Io, n, Rs, Rp, T))
    V = np.linspace(0, Voc, npts)
    P = -V * solar_iv(V, Iph, Io, n, Rs, Rp, T)
    i = int(np.argmax(P))
    return dict(Voc=Voc, Isc=Isc, Vm=V[i], Im=P[i] / V[i], Pm=P[i], FF=P[i] / (Voc * Isc), R_opt=V[i] ** 2 / P[i])


def _photon_flux_bb(E, T):
    """Spectral photon flux of a black body into a hemisphere, per eV (m^-2 s^-1 eV^-1)."""
    E = np.asarray(E, dtype=float)
    return 2 * np.pi / (h**3 * c0**2) * (E * qe) ** 2 / np.expm1(E / kT(T)) * qe


def shockley_queisser(Eg, T_sun=5778.0, T_cell=300.0, concentration=1.0):
    """Detailed-balance efficiency limit of a single-junction cell under black-body sunlight.
    The Sun subtends a solid angle that dilutes its radiance by f = (R_sun/AU)² ≈ 2.16×10⁻⁵, times the
    concentration (max 1/f). Returns a dict with Jsc, J0, Voc, efficiency."""
    f = (6.957e8 / 1.496e11) ** 2 * concentration
    E = np.linspace(Eg, Eg + 40 * kT(T_sun), 4000)
    Jsc = qe * f * np.trapezoid(_photon_flux_bb(E, T_sun), E)
    Ec = np.linspace(Eg, Eg + 40 * kT(T_cell), 4000)
    J0 = qe * np.trapezoid(_photon_flux_bb(Ec, T_cell), Ec)
    Pin = f * sigma_SB * T_sun**4
    Vt = kT(T_cell)
    Voc = Vt * np.log(Jsc / J0 + 1)
    V = np.linspace(0, Voc, 4000)
    P = V * (Jsc - J0 * np.expm1(V / Vt))
    i = int(np.argmax(P))
    return dict(Jsc=Jsc, J0=J0, Voc=Voc, Vm=V[i], Pin=Pin, eff=P[i] / Pin, FF=P[i] / (Voc * Jsc))


# ---------------------------------------------------------------------------
# Image sensors (slides 108-120)
# ---------------------------------------------------------------------------
def cte_remaining(cte, n_transfers):
    """Fraction of a charge packet left after n transfers with charge-transfer efficiency cte."""
    return cte ** np.asarray(n_transfers, dtype=float)


def sensor_snr(photons, qe_pix=0.6, read_noise=3.0, dark=0.0, full_well=20000):
    """Pixel SNR with photon shot noise, dark-current shot noise and read noise (electrons rms):
    SNR = S/√(S + D + σ_r²), S = QE × photons (clipped at the full well)."""
    S = np.minimum(qe_pix * np.asarray(photons, dtype=float), full_well)
    return S / np.sqrt(S + dark + read_noise**2)


def simulate_exposure(scene, photons_max, qe_pix=0.6, read_noise=3.0, full_well=20000, bits=12, rng=None):
    """Simulate a raw image: scene (0..1) → Poisson photo-electrons → Gaussian read noise →
    full-well clipping → ADC. Returns digital numbers (0 .. 2^bits − 1)."""
    rng = np.random.default_rng(rng)
    e = rng.poisson(qe_pix * photons_max * np.clip(scene, 0, None)).astype(float)
    e = np.minimum(e + rng.normal(0, read_noise, e.shape), full_well)
    return np.clip(np.round(e / full_well * (2**bits - 1)), 0, 2**bits - 1)


# ---------------------------------------------------------------------------
# Schottky, MSM and phototransistors (slides 61-68)
# ---------------------------------------------------------------------------
A_STAR = {"Si": 1.12e6, "GaAs": 8.16e4, "GaP": 1.0e6, "InGaAs": 5.0e4}    # effective Richardson constants, A m^-2 K^-2


def schottky_dark(phiB, T=300.0, A_star=8.16e4):
    """Thermionic-emission saturation current density J = A*T² exp(−Φ_B/kT) (A/m²) (slide 63)."""
    return A_star * T**2 * np.exp(-phiB / kT(T))


def fowler_qe(E, phiB, C1=0.2):
    """Internal photoemission yield over a Schottky barrier, η = C₁(hν − Φ_B)²/hν (C₁ in eV^-1)."""
    E = np.asarray(E, dtype=float)
    return np.where(E > phiB, C1 * (E - phiB) ** 2 / E, 0.0)


def msm_capacitance(area, finger, gap, eps_r=12.9):
    """Capacitance (F) of an interdigitated MSM detector covering `area` (m²), conformal mapping
    (Lim & Moore): C = ε₀(1 + ε_r)K(k)/K(k')·area/(finger + gap), k = tan²(π finger/(4(finger + gap)))."""
    from scipy.special import ellipk
    k = np.tan(np.pi * finger / (4 * (finger + gap))) ** 2
    return eps0 * (1 + eps_r) * ellipk(k**2) / ellipk(1 - k**2) * area / (finger + gap)


def phototransistor_bandwidth(Ipho, beta, C, T=300.0):
    """−3 dB bandwidth of a phototransistor limited by the base charging, f ≈ 1/(2π β r_e C),
    r_e = kT/(eI_C) with I_C = βI_pho: f = I_pho/(2π kT/e C). It grows with the light level."""
    return Ipho / (2 * np.pi * kT(T) * C)


def sensitivity_preamp(Q, R, Be, Bo, G, NF_dB, wl, sigma_T=0.0):
    """Average input power (W) for a target Q with an optical preamplifier (gain G, noise figure NF) before a
    pin (responsivity R), OOK with zero-power '0' (Ch. 4 EDFA noise). Per polarisation ASE density
    S = n_sp(G − 1)hν with n_sp = NF·G/(2(G − 1)); noise variances (Agrawal):
    σ²_s-sp = 4R²G P₁ S B_e, σ²_sp-sp = 4R²S²B_oB_e, plus signal shot noise and σ_T²."""
    from scipy.optimize import brentq
    hv = h * c0 / wl
    nsp = 10 ** (NF_dB / 10) * G / (2 * (G - 1))
    S = nsp * (G - 1) * hv
    s0 = np.sqrt(4 * R**2 * S**2 * Bo * Be + sigma_T**2 + 2 * qe * R * S * Bo * Be)
    def q_of(P1):
        I1 = R * G * P1
        s1 = np.sqrt(s0**2 + 4 * R**2 * G * P1 * S * Be + 2 * qe * I1 * Be)
        return I1 / (s1 + s0) - Q
    return brentq(q_of, 1e-15, 1e-2) / 2
