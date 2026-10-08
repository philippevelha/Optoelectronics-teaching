"""
semilib.py -- small, transparent toolbox for the Chapter 3 companion
(Semiconductor science and light-emitting diodes).

Units: SI throughout (m, m^-3, m^2 V^-1 s^-1, s, A), EXCEPT energies, which are in eV.
The slides quote concentrations in cm^-3: multiply by CM3 (= 1e6) to get m^-3,
and mobilities in cm^2 V^-1 s^-1: multiply by CM2 (= 1e-4).

Contents
--------
- plotting style            : use_style()
- constants, units          : kT(), photon_energy_eV(), wavelength_from_eV()
- material data             : MATERIALS (Table 3.1), VARSHNI, varshni()
- band statistics           : dos_3d(), fermi_dirac(), fermi_half(), eff_dos(), n_i()
- doping                    : carriers(), fermi_level(), conductivity(), einstein_D()
- pn junction               : built_in_voltage(), depletion(), junction_profile()
- diode currents            : diode_currents(), diode_iv_rs()
- small signal              : c_dep(), r_d(), c_diff()
- LED physics               : led_spectrum(), led_linewidth(), alloy_Eg(), vegard(), wavelength_rgb()
- quantum wells             : qw_infinite(), qw_finite()
- LED efficiencies          : extraction_ratio(), V_photopic(), luminous_flux()
- LED dynamics              : led_response(), led_pulse()
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
eps0 = 8.8541878128e-12
CM3 = 1e6          # 1 cm^-3 = 1e6 m^-3
CM2 = 1e-4         # 1 cm^2 V^-1 s^-1 = 1e-4 m^2 V^-1 s^-1

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


def nm(x): return x * 1e-9
def um(x): return x * 1e-6


def kT(T=300.0):
    """Thermal energy k_B T in eV (0.02585 eV at 300 K)."""
    return kB * T / qe


def photon_energy_eV(wl):
    return h * c0 / wl / qe


def wavelength_from_eV(E):
    """λ = hc/E, E in eV, result in metres (1.24/E µm)."""
    return h * c0 / (np.asarray(E) * qe)


# ---------------------------------------------------------------------------
# Material data: Table 3.1 of the slides (300 K), SI units
# a (m), Eg (eV), chi electron affinity (eV), Nc, Nv, ni (m^-3), eps_r,
# mu_e, mu_h (m^2/Vs), direct bandgap?
# ---------------------------------------------------------------------------
def _m(a, Eg, chi, Nc, Nv, ni, er, mue, muh, direct):
    return dict(a=a * 1e-9, Eg=Eg, chi=chi, Nc=Nc * CM3, Nv=Nv * CM3, ni=ni * CM3, eps_r=er,
                mu_e=mue * CM2, mu_h=muh * CM2, direct=direct)


MATERIALS = {
    "Ge":   _m(0.5650, 0.66, 4.13, 1.04e19, 6.0e18, 2.3e13, 16.0, 3900, 1900, False),
    "Si":   _m(0.5431, 1.11, 4.05, 2.8e19, 1.2e19, 1.0e10, 11.8, 1450, 490, False),
    "InP":  _m(0.5868, 1.35, 4.50, 5.2e17, 1.1e19, 3.0e7, 12.6, 4600, 150, True),
    "GaAs": _m(0.5653, 1.42, 4.07, 4.7e17, 7.0e18, 2.1e6, 13.0, 8500, 400, True),
    "AlAs": _m(0.5661, 2.17, 3.50, 1.5e19, 1.7e19, 10, 10.1, 200, 100, False),
}
# NB: the slide prints Nv(Ge) = 6.0e19 cm^-3; Kasap's book and sqrt(NcNv)exp(-Eg/2kT) = ni
# both require 6.0e18 cm^-3, which is used here.

# Varshni parameters Eg(T) = Eg0 - A T^2/(B + T)   (Eg0 eV, A eV/K, B K)
VARSHNI = {
    "GaAs": (1.519, 5.41e-4, 204.0),      # slide 107
    "Si":   (1.170, 4.73e-4, 636.0),
    "Ge":   (0.7437, 4.774e-4, 235.0),
    "InP":  (1.421, 4.9e-4, 327.0),
    "GaN":  (3.507, 9.09e-4, 830.0),
}


def varshni(T, Eg0, A, B):
    """Bandgap vs temperature (eV)."""
    T = np.asarray(T, dtype=float)
    return Eg0 - A * T**2 / (B + T)


# ---------------------------------------------------------------------------
# Band statistics
# ---------------------------------------------------------------------------
def dos_3d(E, m_eff, Eedge=0.0):
    """Density of states per unit energy per unit volume, in m^-3 eV^-1.
    g(E) = 8√2 π m*^{3/2} (E − E_edge)^{1/2} / h^3  (E in eV)."""
    dE = np.clip(np.asarray(E, dtype=float) - Eedge, 0, None) * qe
    m = m_eff * me
    return 8 * np.sqrt(2) * np.pi * m**1.5 * np.sqrt(dE) / h**3 * qe


def fermi_dirac(E, EF, T=300.0):
    """Occupation probability f(E) = 1/(1 + exp[(E − EF)/kT]), energies in eV."""
    x = (np.asarray(E, dtype=float) - EF) / kT(T)
    return 0.5 * (1 - np.tanh(x / 2))           # overflow-free form


def fermi_half(eta):
    """Normalised Fermi–Dirac integral F_{1/2}(η) = (2/√π) ∫ √x/(1+e^{x−η}) dx.
    Tends to e^η when η << −1 (Boltzmann/non-degenerate limit)."""
    x = np.linspace(0, 60 + max(0.0, float(np.max(eta))), 6000)
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    f = np.sqrt(x)[None, :] * 0.5 * (1 - np.tanh((x[None, :] - eta[:, None]) / 2))
    out = 2 / np.sqrt(np.pi) * np.trapezoid(f, x, axis=1)
    return out if out.size > 1 else out[0]


def eff_dos(m_eff, T=300.0):
    """Effective density of states Nc (or Nv) = 2 (2π m* kT / h²)^{3/2}, in m^-3."""
    return 2 * (2 * np.pi * m_eff * me * kB * T / h**2) ** 1.5


def n_i(mat="Si", T=300.0, use_varshni=True):
    """Intrinsic concentration vs T (m^-3), anchored to the Table 3.1 value at 300 K:
    n_i(T) = n_i(300) (T/300)^{3/2} exp[Eg(300)/2k·300 − Eg(T)/2kT]  (Nc, Nv ∝ T^{3/2});
    the bandgap follows Varshni if parameters exist (shifted to match Table 3.1 at 300 K)."""
    p = MATERIALS[mat]
    T = np.asarray(T, dtype=float)
    Eg = p["Eg"]
    if use_varshni and mat in VARSHNI:
        Eg = varshni(T, *VARSHNI[mat]) - varshni(300, *VARSHNI[mat]) + p["Eg"]
    return p["ni"] * (T / 300) ** 1.5 * np.exp(p["Eg"] / (2 * kT(300)) - Eg / (2 * kT(T)))


# ---------------------------------------------------------------------------
# Doping
# ---------------------------------------------------------------------------
def carriers(Nd, Na, ni):
    """Electron and hole concentrations from charge neutrality n + Na = p + Nd and np = ni²
    (all dopants ionised, non-degenerate). Works for intrinsic, n, p and compensated material."""
    D = (np.asarray(Nd, dtype=float) - np.asarray(Na, dtype=float)) / 2
    root = np.sqrt(D**2 + ni**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(D >= 0, D + root, ni**2 / (root - D))   # avoids cancellation
    return n, ni**2 / n


def fermi_level(n, ni, T=300.0):
    """E_F − E_Fi = kT ln(n/ni) in eV (positive: n-type, negative: p-type)."""
    return kT(T) * np.log(np.asarray(n) / ni)


def conductivity(n, p, mu_e, mu_h):
    """σ = e n μe + e p μh (S/m)."""
    return qe * (n * mu_e + p * mu_h)


def einstein_D(mu, T=300.0):
    """Diffusion coefficient D = μ kT/e (m²/s)."""
    return mu * kT(T)


# ---------------------------------------------------------------------------
# pn junction electrostatics (abrupt junction, depletion approximation)
# ---------------------------------------------------------------------------
def built_in_voltage(Na, Nd, ni, T=300.0):
    """Vo = (kT/e) ln(Na Nd / ni²)  (V)."""
    return kT(T) * np.log(Na * Nd / ni**2)


def depletion(Na, Nd, eps_r, Vo, V=0.0):
    """Depletion width W, its parts Wp, Wn and the peak field Eo (V/m) at applied bias V
    (V > 0 forward, V < 0 reverse; requires V < Vo)."""
    eps = eps_r * eps0
    W = np.sqrt(2 * eps * (Na + Nd) * (Vo - V) / (qe * Na * Nd))
    Wn, Wp = W * Na / (Na + Nd), W * Nd / (Na + Nd)
    Eo = qe * Nd * Wn / eps
    return dict(W=W, Wn=Wn, Wp=Wp, Eo=Eo)


def junction_profile(Na, Nd, eps_r, Vo, V=0.0, npts=801, span=1.6):
    """Net space charge ρ(x) (C/m³), field E(x) (V/m) and potential V(x) (V) across an abrupt
    junction with the metallurgical junction at x = 0 (p side x < 0). Field is negative (points −x)."""
    d = depletion(Na, Nd, eps_r, Vo, V)
    eps = eps_r * eps0
    x = np.linspace(-span * d["W"], span * d["W"], npts)
    rho = np.where((x > -d["Wp"]) & (x < 0), -qe * Na, 0.0) + np.where((x >= 0) & (x < d["Wn"]), qe * Nd, 0.0)
    E = np.where((x > -d["Wp"]) & (x < 0), -qe * Na * (x + d["Wp"]) / eps, 0.0) + \
        np.where((x >= 0) & (x < d["Wn"]), -qe * Nd * (d["Wn"] - x) / eps, 0.0)
    Vx = np.where(x <= -d["Wp"], 0.0,
         np.where(x < 0, qe * Na * (x + d["Wp"]) ** 2 / (2 * eps),
         np.where(x < d["Wn"], (Vo - V) - qe * Nd * (d["Wn"] - x) ** 2 / (2 * eps), Vo - V)))
    return x, rho, E, Vx


# ---------------------------------------------------------------------------
# Diode currents
# ---------------------------------------------------------------------------
def diode_currents(V, A, Na, Nd, mat="GaAs", tau_e=None, tau_h=None, B=None, tau_g=None,
                   ln=None, lp=None, T=300.0, mu_e=None, mu_h=None):
    """Diffusion (Shockley) and SCL recombination/generation currents of an abrupt pn junction.

    Lifetimes: give tau_e, tau_h directly, or a direct-recombination coefficient B (m³/s), in
    which case tau_e = 1/(B Na) and tau_h = 1/(B Nd) (weak injection, slide 76).
    ln, lp: lengths of the neutral regions; if given and shorter than the diffusion length, the
    short-diode expression is used. Returns a dict with I_diff, I_rec, I (A) and the parameters."""
    p = MATERIALS[mat]
    ni, Vt = n_i(mat, T), kT(T)
    mu_e = p["mu_e"] if mu_e is None else mu_e
    mu_h = p["mu_h"] if mu_h is None else mu_h
    if B is not None:
        tau_e, tau_h = 1 / (B * Na), 1 / (B * Nd)
    De, Dh = einstein_D(mu_e, T), einstein_D(mu_h, T)
    Le, Lh = np.sqrt(De * tau_e), np.sqrt(Dh * tau_h)
    le = Le if (lp is None or lp > Le) else lp       # effective length on the p side
    lh = Lh if (ln is None or ln > Lh) else ln
    Iso = A * qe * ni**2 * (Dh / (lh * Nd) + De / (le * Na))
    Vo = built_in_voltage(Na, Nd, ni, T)
    V = np.asarray(V, dtype=float)
    d = depletion(Na, Nd, p["eps_r"], Vo, np.minimum(V, Vo - 1e-3))
    Iro = A * qe * ni / 2 * (d["Wp"] / tau_e + d["Wn"] / tau_h)
    I_diff = Iso * np.expm1(V / Vt)
    I_rec = np.where(V >= 0, Iro * np.expm1(V / (2 * Vt)), 0.0)
    tg = tau_g if tau_g is not None else (tau_e + tau_h) / 2
    I_gen = np.where(V < 0, -A * qe * ni * d["W"] / tg, 0.0)
    return dict(I_diff=I_diff, I_rec=I_rec + I_gen, I=I_diff + I_rec + I_gen, Iso=Iso, Iro=Iro,
                Vo=Vo, W=d["W"], De=De, Dh=Dh, Le=Le, Lh=Lh, tau_e=tau_e, tau_h=tau_h, ni=ni)


def diode_iv_rs(V, Io, eta=1.0, Rs=0.0, T=300.0):
    """Current of a diode with series resistance, I = Io[exp((V − I Rs)/(η Vt)) − 1],
    solved exactly with the Lambert W function."""
    from scipy.special import lambertw
    V = np.asarray(V, dtype=float)
    nVt = eta * kT(T)
    if Rs == 0:
        return Io * np.expm1(V / nVt)
    arg = Io * Rs / nVt * np.exp((V + Io * Rs) / nVt)
    return nVt / Rs * np.real(lambertw(arg)) - Io


# ---------------------------------------------------------------------------
# Small-signal elements
# ---------------------------------------------------------------------------
def c_dep(A, Na, Nd, eps_r, Vo, V=0.0):
    """Depletion capacitance εA/W (F)."""
    return eps_r * eps0 * A / depletion(Na, Nd, eps_r, Vo, V)["W"]


def r_d(I, eta=1.0, T=300.0):
    """Dynamic resistance η kT/(e I) (Ω)."""
    return eta * kT(T) / I


def c_diff(I, tau, T=300.0, kind="long"):
    """Diffusion capacitance: τh I/(2 Vt) (long diode, ac) or τt I/Vt (short diode)."""
    return tau * I / kT(T) / (2 if kind == "long" else 1)


# ---------------------------------------------------------------------------
# LED emission
# ---------------------------------------------------------------------------
def led_spectrum(E, Eg, T=300.0):
    """Spontaneous emission shape ∝ √(E − Eg) exp[−(E − Eg)/kT] (E, Eg in eV), peak normalised to 1.
    Peak at Eg + kT/2, FWHM ≈ 1.8 kT."""
    x = np.clip((np.asarray(E, dtype=float) - Eg) / kT(T), 0, None)
    return np.sqrt(x) * np.exp(-x) / (np.sqrt(0.5) * np.exp(-0.5))


def led_linewidth(wl0, T=300.0, m=3.0):
    """Δλ ≈ λ0² m kT/(hc), with Δ(hν) ≈ m kT (m = 3 on the slides, 1.8 for the ideal shape)."""
    return wl0**2 * m * kB * T / (h * c0)


# Bandgaps of common LED alloys at 300 K (eV); x is the fraction of the first element named
ALLOYS = {
    "AlxGa1-xAs (x<0.45)": lambda x: 1.424 + 1.247 * x,
    "GaAs1-yPy (y<0.45)":  lambda y: 1.424 + 1.150 * y + 0.176 * y**2,
    "InxGa1-xN":           lambda x: 3.42 * (1 - x) + 0.77 * x - 1.43 * x * (1 - x),
    "In1-xGaxAs":          lambda x: 0.354 * (1 - x) + 1.424 * x - 0.477 * x * (1 - x),
    "AlxGa1-xN":           lambda x: 3.42 * (1 - x) + 6.05 * x - 1.0 * x * (1 - x),
}


def alloy_Eg(name, x):
    return ALLOYS[name](np.asarray(x, dtype=float))


# Lattice constants of binaries (nm); zinc blende except the nitrides (wurtzite a-axis)
LATTICE = {"GaAs": 0.56533, "AlAs": 0.56611, "InAs": 0.60583, "GaP": 0.54505, "InP": 0.58687,
           "AlP": 0.54672, "GaSb": 0.60959, "InSb": 0.64794, "AlSb": 0.61355, "Si": 0.54310, "Ge": 0.56579,
           "GaN": 0.3189, "InN": 0.3545, "AlN": 0.3112}


def vegard(a1, a2, x):
    """Vegard's law: lattice constant of A(x)B(1−x) = x a1 + (1 − x) a2."""
    return x * a1 + (1 - x) * a2


def wavelength_rgb(wl_nm):
    """Approximate display colour (r, g, b in 0..1) of a monochromatic wavelength (Bruton's algorithm).
    Indicative only: monochromatic colours lie outside the sRGB gamut."""
    w = float(wl_nm)
    if 380 <= w < 440:   r, g, b = (440 - w) / 60, 0.0, 1.0
    elif 440 <= w < 490: r, g, b = 0.0, (w - 440) / 50, 1.0
    elif 490 <= w < 510: r, g, b = 0.0, 1.0, (510 - w) / 20
    elif 510 <= w < 580: r, g, b = (w - 510) / 70, 1.0, 0.0
    elif 580 <= w < 645: r, g, b = 1.0, (645 - w) / 65, 0.0
    elif 645 <= w <= 780: r, g, b = 1.0, 0.0, 0.0
    else: return (0.35, 0.35, 0.35)                  # UV or IR: grey
    f = 0.3 + 0.7 * (w - 380) / 40 if w < 420 else (0.3 + 0.7 * (780 - w) / 80 if w > 700 else 1.0)
    return tuple((f * c) ** 0.8 for c in (r, g, b))


# ---------------------------------------------------------------------------
# Quantum wells
# ---------------------------------------------------------------------------
def qw_infinite(n, d, m_eff):
    """Levels of an infinite well, ΔE_n = h² n²/(8 m* d²), in eV."""
    return h**2 * np.asarray(n) ** 2 / (8 * m_eff * me * d**2) / qe


def qw_finite(d, V0, m_w, m_b=None):
    """Bound levels (eV above the well bottom) of a finite square well of width d and depth V0 (eV).
    BenDaniel–Duke boundary conditions (ψ and ψ'/m* continuous); m_b defaults to m_w.
    Even states: (k/m_w) tan(kd/2) = κ/m_b ; odd states: −(k/m_w) cot(kd/2) = κ/m_b."""
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


def qw_wavefunction(x, E, d, V0, m_w, m_b=None, parity=0):
    """Unnormalised envelope ψ(x) of a finite-well state of energy E (eV), well centred at x = 0."""
    m_b = m_w if m_b is None else m_b
    k = np.sqrt(2 * m_w * me * E * qe) / hbar
    K = np.sqrt(2 * m_b * me * (V0 - E) * qe) / hbar
    inside = np.abs(x) <= d / 2
    if parity == 0:
        psi = np.where(inside, np.cos(k * x), np.cos(k * d / 2) * np.exp(-K * (np.abs(x) - d / 2)))
    else:
        psi = np.where(inside, np.sin(k * x), np.sign(x) * np.sin(k * d / 2) * np.exp(-K * (np.abs(x) - d / 2)))
    return psi / np.max(np.abs(psi))


# ---------------------------------------------------------------------------
# Extraction, efficiencies, photometry
# ---------------------------------------------------------------------------
def extraction_ratio(ns, na=1.0, exact=False):
    """Fraction of isotropically emitted light escaping through ONE flat face.
    Slide 122: (1/2)(1 − cos θc) × T(normal). With exact=True the Fresnel transmittance
    (unpolarised) is averaged over the escape cone instead."""
    tc = np.arcsin(min(1.0, na / ns))           # no TIR if the ambient index is higher
    if not exact:
        T = 4 * ns * na / (ns + na) ** 2
        return 0.5 * (1 - np.cos(tc)) * T
    th = np.linspace(0, tc, 4000)[:-1]
    ct = np.sqrt(1 - (ns / na * np.sin(th)) ** 2)
    rs = (ns * np.cos(th) - na * ct) / (ns * np.cos(th) + na * ct)
    rp = (na * np.cos(th) - ns * ct) / (na * np.cos(th) + ns * ct)
    T = 1 - 0.5 * (rs**2 + rp**2)
    return 0.5 * np.trapezoid(T * np.sin(th), th)


# CIE 1924 photopic luminous efficiency V(λ), 380–780 nm in 10 nm steps
_V_WL = np.arange(380, 781, 10)
_V = np.array([3.9e-5, 1.2e-4, 3.96e-4, 1.21e-3, 4.0e-3, 1.16e-2, 2.3e-2, 3.8e-2, 6.0e-2, 9.098e-2,
               0.13902, 0.20802, 0.323, 0.503, 0.71, 0.862, 0.954, 0.99495, 0.995, 0.952, 0.87,
               0.757, 0.631, 0.503, 0.381, 0.265, 0.175, 0.107, 0.061, 0.032, 0.017, 8.21e-3,
               4.102e-3, 2.091e-3, 1.047e-3, 5.2e-4, 2.49e-4, 1.2e-4, 6.0e-5, 3.0e-5, 1.5e-5])


def V_photopic(wl):
    """Photopic luminous efficiency V(λ) (CIE 1924), wl in metres; log-interpolated, 0 outside."""
    x = np.asarray(wl, dtype=float) * 1e9
    v = np.exp(np.interp(x, _V_WL, np.log(_V), left=-np.inf, right=-np.inf))
    return v


def luminous_flux(P, wl=None, spectrum=None):
    """Φv = 683 lm/W × ∫ P(λ) V(λ) dλ. Give a power P at one wavelength wl, or a spectrum
    (wl array, spectral power density array) normalised so that its integral equals P."""
    if spectrum is None:
        return 683.0 * P * V_photopic(wl)
    w, S = spectrum
    S = S / np.trapezoid(S, w) * P
    return 683.0 * np.trapezoid(S * V_photopic(w), w)


# ---------------------------------------------------------------------------
# LED dynamics
# ---------------------------------------------------------------------------
def led_response(f, tau):
    """Optical power modulation response Po(f)/Po(0) = 1/√(1 + (2πfτ)²)."""
    return 1 / np.sqrt(1 + (2 * np.pi * np.asarray(f) * tau) ** 2)


def led_pulse(t, tau, t_on=0.0, t_off=np.inf):
    """Normalised light output for a current step on at t_on, off at t_off (first-order response)."""
    t = np.asarray(t, dtype=float)
    rise = np.where(t >= t_on, 1 - np.exp(-(t - t_on) / tau), 0.0)
    at_off = 1 - np.exp(-(t_off - t_on) / tau) if np.isfinite(t_off) else 1.0
    return np.where(t >= t_off, at_off * np.exp(-(t - t_off) / tau), rise)
