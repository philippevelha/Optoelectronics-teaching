"""
laserlib.py -- small, transparent toolbox for the Chapter 4 companion
(Stimulated emission devices: optical amplifiers and lasers).

Units: SI throughout (m, m^-3, m^2, s, W, A), EXCEPT photon and band energies, which are in eV.
The slides quote concentrations in cm^-3 and cross sections in cm^2: multiply by CM3 (= 1e6)
and CM2 (= 1e-4) to get SI values.

Contents
--------
- plotting style            : use_style()
- black-body radiation      : mode_density(), planck_rho(), rayleigh_jeans_rho(), planck_emittance()
- Einstein coefficients     : A_over_B(), stim_over_spont()
- level systems             : three_level(), four_level(), pump_power_3level()
- line shapes               : lorentzian(), gaussian(), voigt(), doppler_fwhm()
- erbium fiber amplifier    : er_cross_sections(), edfa_gain_uniform(), edfa_propagate(), edfa_nf()
- Fabry-Perot cavity        : mode_number(), mode_spacing(), threshold_gain(), photon_lifetime(),
                              threshold_inversion(), fp_transmission(), finesse()
- pulsed lasers             : qswitch_energy_fraction(), qswitch_pulse(), mode_locked_field()
- semiconductor gain        : fermi_half(), quasi_fermi(), gain_spectrum_bulk(), peak_gain_slide()
- laser diode               : ld_threshold(), ld_power(), ld_efficiencies(), ld_rate_equations(),
                              ld_step_response(), ld_small_signal(), turn_on_delay()
- quantum wells             : qw_infinite(), qw_finite()
- gratings & mirrors        : bragg_wavelength(), dfb_modes(), grating_rt(), dbr_reflectance()
- amplifiers                : fp_amplifier_gain(), saturated_gain()
- beams & resonators        : hermite_gauss(), resonator_g(), transverse_mode_freqs()
- linewidth                 : schawlow_townes()
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants and units
# ---------------------------------------------------------------------------
c0 = 299_792_458.0
h = 6.62607015e-34
hbar = h / (2 * np.pi)
qe = 1.602176634e-19
kB = 1.380649e-23
me = 9.1093837015e-31
NA = 6.02214076e23
CM3 = 1e6          # 1 cm^-3 = 1e6 m^-3
CM2 = 1e-4         # 1 cm^2  = 1e-4 m^2

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
    return h * c0 / np.asarray(wl) / qe


def wavelength_from_eV(E):
    return h * c0 / (np.asarray(E) * qe)


def dB(x):
    return 10 * np.log10(x)


# ---------------------------------------------------------------------------
# Radiation in a box (slides 4-7, 16)
# ---------------------------------------------------------------------------
def mode_density(nu, n=1.0):
    """Number of EM modes per unit volume per unit frequency, both polarisations: 8π n³ ν²/c³."""
    return 8 * np.pi * n**3 * np.asarray(nu, dtype=float) ** 2 / c0**3


def count_modes_box(L, nu_max):
    """Brute-force count of standing-wave modes (kx, ky, kz = π n_i/L, n_i ≥ 1) of a cubic box with
    frequency below nu_max, one polarisation. Compare with (4π/3)(2Lν/c)³/8 = (π/6)(2Lν/c)³."""
    R = 2 * L * nu_max / c0
    n = np.arange(1, int(R) + 1)
    nx, ny, nz = np.meshgrid(n, n, n, indexing="ij", sparse=True)
    return int(np.count_nonzero(nx**2 + ny**2 + nz**2 <= R**2))


def planck_rho(nu, T):
    """Planck spectral energy density ρ(ν) = 8πhν³/c³ / (exp(hν/kT) − 1), in J m^-3 Hz^-1."""
    nu = np.asarray(nu, dtype=float)
    return 8 * np.pi * h * nu**3 / c0**3 / np.expm1(h * nu / (kB * T))


def rayleigh_jeans_rho(nu, T):
    """Classical limit: each mode carries kT, ρ(ν) = 8π kT ν²/c³ (diverges at high ν)."""
    return mode_density(nu) * kB * T


def planck_emittance(wl, T):
    """Spectral emittance of a black body, M_λ = 2πhc²/λ⁵ / (exp(hc/λkT) − 1), in W m^-2 m^-1."""
    wl = np.asarray(wl, dtype=float)
    return 2 * np.pi * h * c0**2 / wl**5 / np.expm1(h * c0 / (wl * kB * T))


# ---------------------------------------------------------------------------
# Einstein coefficients (slides 15-19)
# ---------------------------------------------------------------------------
def A_over_B(nu, n=1.0):
    """A21/B21 = 8π h ν³ n³/c³ (B defined with the energy density ρ(ν))."""
    return 8 * np.pi * h * np.asarray(nu) ** 3 * n**3 / c0**3


def stim_over_spont(nu, T):
    """Ratio of stimulated to spontaneous emission rates in thermal equilibrium: 1/(exp(hν/kT) − 1)."""
    return 1 / np.expm1(h * np.asarray(nu) / (kB * T))


# ---------------------------------------------------------------------------
# Three- and four-level systems (slides 11-14, 20-21)
# ---------------------------------------------------------------------------
def three_level(Wp, tau21):
    """Steady state of an ideal 3-level system (fast E3→E2 decay) with pump rate Wp (s^-1 per ion in E1)
    and upper-level lifetime tau21. Returns (N1, N2) as fractions of N0.
    dN2/dt = Wp N1 − N2/τ = 0  →  N2/N0 = Wpτ/(1 + Wpτ); inversion needs Wpτ > 1."""
    x = np.asarray(Wp) * tau21
    return 1 / (1 + x), x / (1 + x)


def four_level(Wp, tau32, tau10):
    """Steady state of an ideal 4-level system: pump from E0 (rate Wp), upper laser level E2 with
    lifetime tau32 (to E1), lower laser level E1 emptied with time tau10. Returns (N0, N1, N2)/Ntot."""
    Wp = np.asarray(Wp, dtype=float)
    # N2 = Wp N0 τ32 ; N1 = N2 τ10/τ32 = Wp N0 τ10
    N0 = 1 / (1 + Wp * tau32 + Wp * tau10)
    return N0, Wp * tau10 * N0, Wp * tau32 * N0


def pump_power_3level(N0, volume, E_pump_eV, tau_sp):
    """Minimum absorbed pump power for inversion in a 3-level laser (slide 20, eq. 4.2.12):
    P = V (N0/2) hν13 / τ_sp."""
    return volume * N0 / 2 * E_pump_eV * qe / tau_sp


# ---------------------------------------------------------------------------
# Line shapes (slides 54-60, 76-83)
# ---------------------------------------------------------------------------
def lorentzian(nu, nu0, fwhm):
    """Normalised Lorentzian, ∫g dν = 1. Slide 78 writes γ = fwhm/2."""
    g = fwhm / 2
    return (g / np.pi) / ((np.asarray(nu) - nu0) ** 2 + g**2)


def gaussian(nu, nu0, fwhm):
    """Normalised Gaussian, ∫g dν = 1. Slide 80 writes Γ = fwhm/(2√ln2)."""
    G = fwhm / (2 * np.sqrt(np.log(2)))
    return np.exp(-((np.asarray(nu) - nu0) / G) ** 2) / (np.sqrt(np.pi) * G)


def voigt(nu, nu0, fwhm_G, fwhm_L):
    """Convolution of a Gaussian and a Lorentzian (both normalised)."""
    from scipy.special import voigt_profile
    return voigt_profile(np.asarray(nu) - nu0, fwhm_G / (2 * np.sqrt(2 * np.log(2))), fwhm_L / 2)


def doppler_fwhm(nu0, T, M_gmol):
    """Doppler FWHM Δν = 2ν0 √(2kT ln2/(Mc²)) of a gas line (slide 49), M in g/mol."""
    M = M_gmol * 1e-3 / NA
    return 2 * nu0 * np.sqrt(2 * kB * T * np.log(2) / (M * c0**2))


def lifetime_fwhm(tau):
    """Natural (lifetime) FWHM of a transition whose upper level decays with time constant tau:
    the field decays as exp(−t/2τ), so Δν = 1/(2πτ)."""
    return 1 / (2 * np.pi * tau)


# ---------------------------------------------------------------------------
# Erbium-doped fiber amplifier (slides 22-44)
# ---------------------------------------------------------------------------
# Model emission cross section of Er3+ in alumino-silicate glass, as a sum of Gaussians in λ (nm),
# shaped after Pedersen et al. (JLT 9, 1105, 1991) and scaled so that σem(1550) = 3.2e-21 cm²
# (slide 39). Absorption follows from McCumber, σab = σem exp[(hν − ε)/kT], with ε chosen so that
# σab(1550) = 2.4e-21 cm² (slide 39).
_ER_EM = [(1530.5, 4.2, 3.30), (1535.5, 7.0, 1.20), (1548.0, 11.0, 1.70),
          (1560.0, 14.0, 1.30), (1585.0, 22.0, 0.45), (1508.0, 12.0, 0.45),
          (1485.0, 16.0, 0.30)]
ER_EPS = None                                  # McCumber energy (eV), set below


def _er_em_raw(wl_nm):
    wl_nm = np.asarray(wl_nm, dtype=float)
    return sum(a * np.exp(-0.5 * ((wl_nm - c) / w) ** 2) for c, w, a in _ER_EM)


_ER_SCALE = 3.2 / _er_em_raw(1550.0)
ER_EPS = photon_energy_eV(1550e-9) - kT(295) * np.log(2.4 / 3.2)


def er_cross_sections(wl):
    """(σab, σem) of Er3+ in m² at free-space wavelength(s) wl (m), C/L band model (1450-1650 nm),
    plus the 980 nm pump band (absorption only, σab(980) = 2.2e-25 m²)."""
    wl_nm = np.asarray(wl, dtype=float) * 1e9
    em = _ER_SCALE * _er_em_raw(wl_nm) * 1e-21 * CM2
    ab = em * np.exp((photon_energy_eV(wl_nm * 1e-9) - ER_EPS) / kT(295))
    pump = 2.2e-25 * np.exp(-0.5 * ((wl_nm - 978.0) / 7.0) ** 2)
    band = (wl_nm > 1400) & (wl_nm < 1700)
    return np.where(band, ab, pump), np.where(band, em, 0.0)


def edfa_gain_uniform(x, sig_em, sig_ab, N0, L):
    """Small-signal gain of a uniformly inverted fiber (slides 39-40): g = [σem x − σab(1 − x)] N0,
    G = exp(gL). x = N2/N0 is the fractional inversion. Returns (g in m^-1, G linear)."""
    g = (sig_em * x - sig_ab * (1 - x)) * N0
    return g, np.exp(g * L)


def edfa_propagate(L, Pp0, Ps0, wl_s=1550e-9, wl_p=980e-9, N0=1e25, tau=10e-3, core_d=5e-6,
                   Gp=0.7, Gs=0.7, loss_dB_km=0.0, nz=600, backward_pump=False):
    """Pump and signal powers along a two-level EDFA (steady state, no ASE).

    Local inversion: N2/N0 = Σ σab_k Φ_k / (Σ (σab_k + σem_k) Φ_k + 1/τ), with photon flux densities
    Φ_k = Γ_k P_k /(hν_k A). Powers: dP_k/dz = ±Γ_k [σem_k N2 − σab_k N1] P_k − α P_k.
    Returns dict with z, Pp, Ps, x (N2/N0). Backward pumping is solved by shooting on Pp(0)."""
    A = np.pi * core_d**2 / 4
    ab_s, em_s = er_cross_sections(wl_s)
    ab_p, em_p = er_cross_sections(wl_p)
    hv_s, hv_p = h * c0 / wl_s, h * c0 / wl_p
    alpha = loss_dB_km / 4.343 / 1e3
    z = np.linspace(0, L, nz + 1)
    dz = z[1] - z[0]

    def inversion(Pp, Ps):
        fp, fs = Gp * Pp / (hv_p * A), Gs * Ps / (hv_s * A)
        return (ab_p * fp + ab_s * fs) / ((ab_p + em_p) * fp + (ab_s + em_s) * fs + 1 / tau)

    def deriv(Pp, Ps, sgn):
        x = inversion(Pp, Ps)
        dPp = sgn * (Gp * (em_p * x - ab_p * (1 - x)) * N0 - alpha) * Pp
        dPs = (Gs * (em_s * x - ab_s * (1 - x)) * N0 - alpha) * Ps
        return dPp, dPs, x

    def run(Pp_start):
        Pp, Ps = np.empty_like(z), np.empty_like(z)
        Pp[0], Ps[0] = Pp_start, Ps0
        sgn = -1.0 if backward_pump else 1.0
        for i in range(nz):                     # RK4 in z
            k1 = deriv(Pp[i], Ps[i], sgn)
            k2 = deriv(Pp[i] + dz / 2 * k1[0], Ps[i] + dz / 2 * k1[1], sgn)
            k3 = deriv(Pp[i] + dz / 2 * k2[0], Ps[i] + dz / 2 * k2[1], sgn)
            k4 = deriv(Pp[i] + dz * k3[0], Ps[i] + dz * k3[1], sgn)
            Pp[i + 1] = max(Pp[i] + dz / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 0.0)
            Ps[i + 1] = max(Ps[i] + dz / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 0.0)
        return Pp, Ps

    if not backward_pump:
        Pp, Ps = run(Pp0)
    else:                                       # find Pp(0) such that Pp(L) = Pp0 (pump enters at z = L)
        lo, hi = 0.0, Pp0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if run(mid)[0][-1] < Pp0: lo = mid
            else: hi = mid
        Pp, Ps = run(0.5 * (lo + hi))
    return dict(z=z, Pp=Pp, Ps=Ps, x=inversion(Pp, Ps))


def edfa_nsp(x, sig_em, sig_ab):
    """Spontaneous-emission (population-inversion) factor n_sp = σem N2 / (σem N2 − σab N1)."""
    return sig_em * x / (sig_em * x - sig_ab * (1 - x))


def edfa_nf(G, nsp):
    """Signal–spontaneous-beat-limited noise figure of an optical amplifier,
    NF = 2 n_sp (G − 1)/G + 1/G  →  2 n_sp ≥ 2 (3 dB) at high gain."""
    return 2 * nsp * (G - 1) / G + 1 / G


# ---------------------------------------------------------------------------
# Fabry-Perot laser cavity (slides 48-50, 59, 63-74, 101)
# ---------------------------------------------------------------------------
def mode_number(L, wl, n=1.0):
    """Longitudinal mode index m = 2nL/λ (not rounded)."""
    return 2 * n * L / wl


def mode_spacing(L, wl=None, n=1.0):
    """Mode spacing: Δν = c/2nL (Hz); if wl is given, also Δλ = λ²/2nL (m). Returns (Δν, Δλ)."""
    dnu = c0 / (2 * n * L)
    return dnu, (None if wl is None else wl**2 / (2 * n * L))


def threshold_gain(alpha_s, L, R1, R2=None):
    """g_th = α_t = α_s + (1/2L) ln(1/R1R2) (slide 65), m^-1."""
    R2 = R1 if R2 is None else R2
    return alpha_s + np.log(1 / (R1 * R2)) / (2 * L)


def photon_lifetime(alpha_t, L, n=1.0, exact=True):
    """Photon cavity lifetime. exact: τ = 2nL/[c(1 − exp(−2α_tL))] (eq. 4.6.12); else n/(cα_t)."""
    if exact:
        return 2 * n * L / (c0 * (1 - np.exp(-2 * alpha_t * L)))
    return n / (c0 * alpha_t)


def threshold_inversion(g_th, nu0, tau_sp, dnu, n=1.0):
    """(N2 − N1)_th = g_th 8π n² ν0² τ_sp Δν / c² (slide 66), m^-3."""
    return g_th * 8 * np.pi * n**2 * nu0**2 * tau_sp * dnu / c0**2


def gas_laser_output(Nph, area, R1, nu0, n=1.0):
    """Output power through mirror R1: P_o = ½ A (1 − R1) hν0 N_ph c/n (slide 71)."""
    return 0.5 * area * (1 - R1) * h * nu0 * Nph * c0 / n


def finesse(R1, R2=None):
    """Coefficient finesse F = π (R1R2)^{1/4} / (1 − √(R1R2))."""
    R2 = R1 if R2 is None else R2
    r = np.sqrt(R1 * R2)
    return np.pi * np.sqrt(r) / (1 - r)


def fp_transmission(nu, L, R1, R2=None, n=1.0, g=0.0, alpha=0.0):
    """Power transmission of a Fabry–Perot etalon with single-pass net amplitude gain exp[(g − α)L/2]
    (g, α are power coefficients). With g = 0 this is a passive resonator; near threshold it peaks
    sharply at the cavity modes."""
    R2 = R1 if R2 is None else R2
    Gs = np.exp((g - alpha) * L)
    phi = 2 * np.pi * np.asarray(nu) * n * L / c0
    rr = np.sqrt(R1 * R2) * Gs
    return (1 - R1) * (1 - R2) * Gs / ((1 - rr) ** 2 + 4 * rr * np.sin(phi) ** 2)


# ---------------------------------------------------------------------------
# Q-switching and mode-locking (slides 84-88)
# ---------------------------------------------------------------------------
def qswitch_energy_fraction(r):
    """Fraction η of the initial inversion extracted by a Q-switched pulse, r = N_i/N_th:
    solves r η = −ln(1 − η)  (Wagner & Lengyel / Siegman)."""
    from scipy.optimize import brentq
    r = float(r)
    if r <= 1: return 0.0
    return brentq(lambda e: r * e + np.log(1 - e), 1e-12, 1 - 1e-15)


def qswitch_pulse(r, tau_c, t_end=None, n_t=4000, phi0=1e-8):
    """Normalised Q-switch rate equations of a four-level medium after the switch (pumping and
    spontaneous decay neglected): dn/dt = −n φ/τ_c, dφ/dt = (n − 1) φ/τ_c, with n = ΔN/ΔN_th, φ the
    photon density in units of ΔN_th and τ_c the photon lifetime of the high-Q cavity.
    Returns t, n(t), φ(t)."""
    from scipy.integrate import solve_ivp
    t_end = t_end or 60 * tau_c * max(1.0, 3 / max(r - 1, 0.05))

    def f(t, y):
        n, p = y
        return [-n * p / tau_c, (n - 1) * p / tau_c]
    t = np.linspace(0, t_end, n_t)
    s = solve_ivp(f, (0, t_end), [r, phi0], t_eval=t, rtol=1e-9, atol=1e-14, method="LSODA")
    return s.t, s.y[0], s.y[1]


def mode_locked_field(t, N, dnu, phases=None, nu_c=0.0):
    """Total field of N equal-amplitude longitudinal modes spaced by dnu, with given phases
    (default 0 = locked). Returns the intensity |E|² normalised to N² (peak of a locked pulse = 1)."""
    k = np.arange(N) - (N - 1) / 2
    ph = np.zeros(N) if phases is None else np.asarray(phases)
    E = np.exp(1j * (2 * np.pi * np.outer(np.asarray(t), (nu_c + k * dnu)) + ph[None, :])).sum(axis=1)
    return np.abs(E) ** 2 / N**2


# ---------------------------------------------------------------------------
# Semiconductor optical gain (slides 90-91, 128-130)
# ---------------------------------------------------------------------------
SEMI = {   # Eg (eV, 300 K), me*, mh* (heavy hole), n, B (m^3/s)
    "GaAs":         dict(Eg=1.424, me=0.067, mh=0.50, n=3.6, B=2e-16),
    "InGaAsP-1300": dict(Eg=0.954, me=0.058, mh=0.46, n=3.5, B=1e-16),
    "InGaAsP-1550": dict(Eg=0.800, me=0.045, mh=0.45, n=3.6, B=1e-16),
}


def fermi_half(eta):
    """Normalised Fermi–Dirac integral F_{1/2}(η) = (2/√π) ∫ √x/(1+e^{x−η}) dx."""
    x = np.linspace(0, 60 + max(0.0, float(np.max(eta))), 6000)
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    f = np.sqrt(x)[None, :] * 0.5 * (1 - np.tanh((x[None, :] - eta[:, None]) / 2))
    out = 2 / np.sqrt(np.pi) * np.trapezoid(f, x, axis=1)
    return out if out.size > 1 else out[0]


def eff_dos(m_eff, T=300.0):
    return 2 * (2 * np.pi * m_eff * me * kB * T / h**2) ** 1.5


def quasi_fermi(n, m_eff, T=300.0):
    """Quasi-Fermi level measured from the band edge into the band (eV), for carrier density n (m^-3):
    solves n = N_c F_{1/2}(η). Positive values mean degenerate (inside the band)."""
    from scipy.optimize import brentq
    Nc = eff_dos(m_eff, T)
    eta = brentq(lambda e: Nc * fermi_half(e) - n, -30, 200)
    return eta * kT(T)


def gain_spectrum_bulk(E, n, mat="GaAs", T=300.0, p=None, alpha0=1.2e6, broadening=0.0):
    """Free-carrier (parabolic band, k-conserving) gain of a bulk active layer, m^-1, vs photon
    energy E (eV):  g(hν) = α0 √((hν − Eg)/0.1 eV) [f_c(E2) − f_v(E1)],
    with E2 − Ec = (hν − Eg) m_r/m_e, Ev − E1 = (hν − Eg) m_r/m_h. α0 sets the absorption 0.1 eV above
    the gap of the unpumped crystal (1.2e4 cm^-1 for GaAs). p defaults to n (undoped active layer).
    Optional intraband broadening: Gaussian of FWHM `broadening` (eV); a Lorentzian would add a
    spurious absorption tail below the gap. Gain > 0 only for Eg < hν < ΔF = E_Fn − E_Fp."""
    P = SEMI[mat]
    p = n if p is None else p
    Fn = quasi_fermi(n, P["me"], T)                 # above Ec
    Fp = quasi_fermi(p, P["mh"], T)                 # below Ev
    E = np.asarray(E, dtype=float)

    def g0(Ex):
        x = np.clip(Ex - P["Eg"], 0, None)
        mr = P["me"] * P["mh"] / (P["me"] + P["mh"])
        e2 = x * mr / P["me"]                       # electron kinetic energy
        e1 = x * mr / P["mh"]                       # hole kinetic energy
        fc = 0.5 * (1 - np.tanh((e2 - Fn) / (2 * kT(T))))
        fv_empty = 0.5 * (1 - np.tanh((e1 - Fp) / (2 * kT(T))))   # probability that the VB state is empty
        # f_c − f_v = f_c + (1 − f_v) − 1, with 1 − f_v the hole occupation
        return alpha0 * np.sqrt(x / 0.1) * (fc + fv_empty - 1)
    if broadening <= 0:
        return g0(E)
    Eg_ = np.linspace(P["Eg"] - 0.1, P["Eg"] + 0.6, 3000)
    G0 = g0(Eg_)
    s = broadening / (2 * np.sqrt(2 * np.log(2)))
    K = np.exp(-0.5 * ((E[:, None] - Eg_[None, :]) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
    return np.trapezoid(K * G0[None, :], Eg_, axis=1)


def delta_F(n, mat="GaAs", T=300.0, p=None):
    """Quasi-Fermi level separation E_Fn − E_Fp (eV) for injected density n (= p by default)."""
    P = SEMI[mat]
    p = n if p is None else p
    return P["Eg"] + quasi_fermi(n, P["me"], T) + quasi_fermi(p, P["mh"], T)


# Peak gain versus carrier density digitised (approximately) from slide 130, n in 1e18 cm^-3, g in cm^-1
PEAK_GAIN_SLIDE = {
    "GaAs (860 nm)":         ([1.65, 1.80, 2.00, 2.20, 2.50], [20, 33, 55, 78, 140]),
    "InGaAsP (1300 nm)":     ([1.18, 1.30, 1.40, 1.55, 1.80], [33, 70, 105, 160, 280]),
    "InGaAsP (1500 nm)":     ([0.50, 1.00, 1.50, 2.00, 2.50, 3.00], [68, 160, 270, 400, 560, 730]),
}


def peak_gain_slide(n_cm3, curve="GaAs (860 nm)"):
    """Peak gain (cm^-1) at n (cm^-3) by log-linear interpolation of the slide-130 curves."""
    x, y = PEAK_GAIN_SLIDE[curve]
    return np.exp(np.interp(np.asarray(n_cm3) / 1e18, x, np.log(y), left=np.nan, right=np.nan))


def n_for_gain_slide(g_cm, curve="GaAs (860 nm)"):
    """Inverse of peak_gain_slide: carrier density (cm^-3) that gives peak gain g (cm^-1)."""
    x, y = PEAK_GAIN_SLIDE[curve]
    return np.interp(np.log(g_cm), np.log(y), x, left=np.nan, right=np.nan) * 1e18


# ---------------------------------------------------------------------------
# Laser diode: threshold, power, efficiencies (slides 108-134)
# ---------------------------------------------------------------------------
def facet_R(n, n0=1.0):
    return ((n - n0) / (n + n0)) ** 2


def ld_threshold(n_th, d, L, W, B=None, tau_r=None):
    """Threshold current density and current from I_th/(eLWd) = n_th/τ_r (slide 125).
    τ_r = 1/(B n_th) if B is given (strong injection, slide 131). Returns dict(J_th, I_th, tau_r)."""
    tau_r = 1 / (B * n_th) if tau_r is None else tau_r
    J = n_th * qe * d / tau_r
    return dict(J_th=J, I_th=J * L * W, tau_r=tau_r)


def ld_power(I, I_th, tau_ph, R, n, wl, L):
    """Laser diode equation P_o = [hc²τ_ph(1 − R)/(2enλL)] (I − I_th) (slide 126); 0 below threshold."""
    k = h * c0**2 * tau_ph * (1 - R) / (2 * qe * n * wl * L)
    return k * np.clip(np.asarray(I, dtype=float) - I_th, 0, None)


def ld_efficiencies(Po, I, V, I_th, wl):
    """Slope, PCE, EQE and EDQE of a laser diode at one operating point (slides 108-115, 118)."""
    hv = h * c0 / wl
    return dict(slope=Po / (I - I_th), PCE=Po / (I * V),
                EQE=(Po / hv) / (I / qe), EDQE=(Po / hv) / ((I - I_th) / qe))


def extraction_efficiency(alpha_s, L, R1, R2=None):
    """η_EE = (1/2L) ln(1/R1) / α_t (slide 113): fraction of the losses that leaves through facet 1."""
    R2 = R1 if R2 is None else R2
    return np.log(1 / R1) / (2 * L) / threshold_gain(alpha_s, L, R1, R2)


def ld_rate_equations(t, y, I_of_t, V, tau, tau_ph, n_th, Gamma=1.0, beta=1e-4, tau_r=None):
    """Rate equations of slide 124 (with spontaneous-emission coupling β for start-up):
    dn/dt = I/(eV) − n/τ − C n N_ph,  dN_ph/dt = C n N_ph − N_ph/τ_ph + β n/τ_r, C = 1/(n_th τ_ph)."""
    n, N = y
    C = 1 / (n_th * tau_ph)
    tau_r = tau if tau_r is None else tau_r
    return [I_of_t(t) / (qe * V) - n / tau - C * n * N,
            C * n * N - N / tau_ph + beta * n / tau_r]


def ld_step_response(I1, I2, I_th, tau, tau_ph, V, t_end=5e-9, beta=1e-4, n_t=5000):
    """Carrier density and photon density after a current step I1 → I2 at t = 0 (slide 164).
    Starts from the steady state at I1. Returns t, n, N_ph (SI)."""
    from scipy.integrate import solve_ivp
    n_th = I_th * tau / (qe * V)
    C = 1 / (n_th * tau_ph)
    # steady state at I1 (solve the quadratic including spontaneous coupling)
    n1 = min(I1 * tau / (qe * V), n_th * 0.999999)
    N1 = beta * n1 / tau / max(1 / tau_ph - C * n1, 1e-30)
    if I1 > I_th:
        n1, N1 = n_th, tau_ph * (I1 - I_th) / (qe * V)
    f = lambda t, y: ld_rate_equations(t, y, lambda _t: I2, V, tau, tau_ph, n_th, beta=beta)
    t = np.linspace(0, t_end, n_t)
    s = solve_ivp(f, (0, t_end), [n1, N1], t_eval=t, method="LSODA", rtol=1e-8, atol=[1e10, 1e6],
                  max_step=tau_ph * 5)
    return s.t, s.y[0], s.y[1]


def ld_small_signal(f, I1, I_th, tau, tau_ph):
    """Small-signal modulation response of the rate equations (no gain compression):
    H(f) = ω_r²/(ω_r² − ω² + jωγ), ω_r² = (I1/I_th − 1)/(τ τ_ph) (slide 163), γ = (I1/I_th)/τ."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    wr2 = (I1 / I_th - 1) / (tau * tau_ph)
    gam = (I1 / I_th) / tau
    return wr2 / (wr2 - w**2 + 1j * w * gam)


def relaxation_frequency(I1, I_th, tau, tau_ph):
    """f_r ≈ [(I1/I_th − 1)/(τ τ_ph)]^{1/2} / 2π (slide 163)."""
    return np.sqrt(np.clip(I1 / I_th - 1, 0, None) / (tau * tau_ph)) / (2 * np.pi)


def turn_on_delay(I1, I2, I_th, tau):
    """Turn-on delay after a step I1 → I2 (I1 < I_th < I2): t_d = τ ln[(I2 − I1)/(I2 − I_th)]."""
    return tau * np.log((I2 - I1) / (I2 - I_th))


# ---------------------------------------------------------------------------
# Quantum wells (slides 135-139)
# ---------------------------------------------------------------------------
def qw_infinite(n, d, m_eff):
    """Levels of an infinite well, ΔE_n = h² n²/(8 m* d²), in eV."""
    return h**2 * np.asarray(n) ** 2 / (8 * m_eff * me * d**2) / qe


def qw_finite(d, V0, m_w, m_b=None):
    """Bound levels (eV above the well bottom) of a finite square well (BenDaniel–Duke conditions)."""
    from scipy.optimize import brentq
    m_b = m_w if m_b is None else m_b

    def k_(E): return np.sqrt(2 * m_w * me * E * qe) / hbar
    def K_(E): return np.sqrt(2 * m_b * me * (V0 - E) * qe) / hbar

    def even(E): k = k_(E); return (k / m_w) * np.sin(k * d / 2) - (K_(E) / m_b) * np.cos(k * d / 2)
    def odd(E):  k = k_(E); return (k / m_w) * np.cos(k * d / 2) + (K_(E) / m_b) * np.sin(k * d / 2)

    E = np.linspace(1e-9, V0 * (1 - 1e-9), 20000)
    levels = []
    for f in (even, odd):
        y = f(E)
        for i in np.nonzero(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]:
            levels.append(brentq(f, E[i], E[i + 1]))
    return np.sort(levels)


# ---------------------------------------------------------------------------
# Gratings, DFB, DBR mirrors (slides 140-151)
# ---------------------------------------------------------------------------
def bragg_wavelength(period, n_eff, order=1):
    """λ_B = 2 n Λ / q."""
    return 2 * n_eff * period / order


def dfb_modes(period, n_eff, L, m=(0, 1, 2), order=1):
    """Kasap's DFB mode wavelengths λ_m = λ_B ± λ_B²(m + 1)/(2nL) (slide 142)."""
    lB = bragg_wavelength(period, n_eff, order)
    d = lB**2 / (2 * n_eff * L) * (np.asarray(m) + 1)
    return lB, lB - d, lB + d


def grating_rt(wl, period, n_eff, dn, L, g=0.0, alpha=0.0, shift=0.0):
    """Amplitude reflection and transmission of a uniform index grating (coupled-mode theory, transfer
    matrix). dn: index modulation amplitude (κ = π dn/λ), g/α: power gain/loss coefficients,
    shift: phase shift (rad) inserted in the middle (π/2 for a quarter-wave-shifted DFB, i.e. a λ/4 step).
    Returns (r, t) complex arrays."""
    wl = np.atleast_1d(np.asarray(wl, dtype=float))
    beta = 2 * np.pi * n_eff / wl
    delta = beta - np.pi / period + 1j * (g - alpha) / 2           # gain: complex detuning (e^{−jβz} convention)
    kappa = np.pi * dn / wl

    def section(Ls):
        s = np.sqrt(kappa**2 - delta**2 + 0j)
        ch, sh = np.cosh(s * Ls), np.sinh(s * Ls)
        M = np.empty((wl.size, 2, 2), dtype=complex)
        M[:, 0, 0] = ch - 1j * delta / s * sh
        M[:, 0, 1] = -1j * kappa / s * sh
        M[:, 1, 0] = 1j * kappa / s * sh
        M[:, 1, 1] = ch + 1j * delta / s * sh
        return M
    if shift:
        P = np.zeros((wl.size, 2, 2), dtype=complex)
        P[:, 0, 0], P[:, 1, 1] = np.exp(-1j * shift), np.exp(1j * shift)
        M = section(L / 2) @ P @ section(L / 2)
    else:
        M = section(L)
    t = 1 / M[:, 1, 1]
    r = M[:, 0, 1] / M[:, 1, 1]
    return r, t


def dbr_reflectance(N, n_hi, n_lo, n_in=1.0, n_sub=1.0, wl=None, wl0=None):
    """Reflectance of a quarter-wave stack with N pairs (H L)^N between n_in and n_sub.
    At the design wavelength: R = [(1 − b)/(1 + b)]², b = (n_in/n_sub)(n_lo/n_hi)^{2N}.
    If wl and wl0 are given, returns the spectrum (characteristic matrix method)."""
    if wl is None:
        b = (n_in / n_sub) * (n_lo / n_hi) ** (2 * N)
        return ((1 - b) / (1 + b)) ** 2
    wl = np.atleast_1d(np.asarray(wl, dtype=float))
    R = np.empty(wl.size)
    for i, lam in enumerate(wl):
        M = np.eye(2, dtype=complex)
        for nn in [n_hi, n_lo] * N:
            dl = 2 * np.pi * nn * (wl0 / (4 * nn)) / lam
            M = M @ np.array([[np.cos(dl), 1j * np.sin(dl) / nn], [1j * nn * np.sin(dl), np.cos(dl)]])
        B, C = M @ np.array([1, n_sub])
        r = (n_in * B - C) / (n_in * B + C)
        R[i] = abs(r) ** 2
    return R


# ---------------------------------------------------------------------------
# Optical amplifiers (slides 152-158)
# ---------------------------------------------------------------------------
def fp_amplifier_gain(phi, Gs, R1, R2=None):
    """Gain of a Fabry–Perot amplifier with single-pass gain Gs and facet reflectances R1, R2 at
    single-pass phase phi: G = (1−R1)(1−R2)Gs / [(1 − √(R1R2)Gs)² + 4√(R1R2)Gs sin²φ]."""
    R2 = R1 if R2 is None else R2
    rr = np.sqrt(R1 * R2) * Gs
    return (1 - R1) * (1 - R2) * Gs / ((1 - rr) ** 2 + 4 * rr * np.sin(phi) ** 2)


def saturated_gain(G0, Pin, Psat):
    """Gain of a homogeneously broadened travelling-wave amplifier with input power Pin:
    solves G = G0 exp[−(G − 1) Pin / Psat] (Psat: saturation power of the gain medium)."""
    from scipy.optimize import brentq
    Pin = np.atleast_1d(np.asarray(Pin, dtype=float))
    out = np.array([brentq(lambda G: np.log(G) - np.log(G0) + (G - 1) * p / Psat, 1.0 + 1e-15, G0)
                    if p > 0 else G0 for p in Pin])
    return out if out.size > 1 else out[0]


# ---------------------------------------------------------------------------
# Beams and resonators (slide 75 + going further)
# ---------------------------------------------------------------------------
def hermite_gauss(x, y, m, n, w=1.0):
    """Field of the TEM_mn Hermite–Gauss mode at the waist (unnormalised)."""
    from scipy.special import eval_hermite
    X, Y = np.sqrt(2) * np.asarray(x) / w, np.sqrt(2) * np.asarray(y) / w
    return eval_hermite(m, X) * eval_hermite(n, Y) * np.exp(-(X**2 + Y**2) / 2)


def resonator_g(L, R1, R2):
    """Stability parameters g_i = 1 − L/R_i (R = ∞ for a flat mirror); stable if 0 ≤ g1 g2 ≤ 1."""
    return 1 - L / R1, 1 - L / R2


def transverse_mode_freqs(L, R1, R2, q, m, n, n_index=1.0):
    """Resonance frequencies ν_qmn = (c/2nL)[q + (m + n + 1) arccos(±√(g1g2))/π] of a two-mirror
    resonator (sign of g1 for the square root)."""
    g1, g2 = resonator_g(L, R1, R2)
    gouy = np.arccos(np.sign(g1) * np.sqrt(g1 * g2))
    return c0 / (2 * n_index * L) * (q + (m + n + 1) * gouy / np.pi)


def waist_from_resonator(L, R1, R2, wl):
    """Waist radius of the fundamental Gaussian mode of a stable two-mirror resonator."""
    g1, g2 = resonator_g(L, R1, R2)
    if np.isclose(g1, g2):                      # symmetric cavity (avoids 0/0 in the confocal case)
        return np.sqrt(wl * L / np.pi * np.sqrt((1 + g1) / (4 * (1 - g1))))
    return np.sqrt(wl * L / np.pi * np.sqrt(g1 * g2 * (1 - g1 * g2) / (g1 + g2 - 2 * g1 * g2) ** 2))


# ---------------------------------------------------------------------------
# Laser linewidth (going further)
# ---------------------------------------------------------------------------
def schawlow_townes(P, nu0, dnu_c, nsp=1.0, alpha_H=0.0):
    """Modified Schawlow–Townes linewidth Δν = π h ν0 Δν_c² n_sp (1 + α_H²) / P, where Δν_c = 1/(2π τ_ph)
    is the cold-cavity linewidth and α_H the Henry linewidth-enhancement factor."""
    return np.pi * h * nu0 * dnu_c**2 * nsp * (1 + alpha_H**2) / P
