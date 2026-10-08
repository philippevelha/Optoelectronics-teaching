"""
fiberlib.py -- small, transparent toolbox for the Chapter 2 companion
(Dielectric Waveguides and Optical Fibers).

Every function is short on purpose: students should be able to read the
physics straight from the code. SI units everywhere unless the name says
otherwise (e.g. ``D_ps_nm_km``).

Contents
--------
- plotting style                      : use_style()
- planar (slab) waveguide, TE/TM     : slab_modes(), slab_beta(), slab_field()
- step-index fiber, LP modes         : LP_b(), LP_all(), LP_field()
- glass dispersion (Sellmeier)        : n_silica(), n_doped(), group_index()
- dispersion                          : D_material(), D_total_SMF(), ...
- graded index                        : sigma_intermodal_grin()
- attenuation, bending                : rayleigh_alpha(), attenuation_model()
- communications                      : Q_to_BER(), imdd_link()
- gratings & interferometers          : fbg_reflectance(), fabry_perot_T()
"""
from __future__ import annotations

import numpy as np
from scipy import special as sp
from scipy.optimize import brentq
from scipy.special import erfc

c0 = 299_792_458.0          # speed of light in vacuum [m/s]
h = 6.62607015e-34          # Planck [J s]
kB = 1.380649e-23           # Boltzmann [J/K]
q = 1.602176634e-19         # elementary charge [C]

# ---------------------------------------------------------------------------
# Plot style (validated categorical palette, recessive grid)
# ---------------------------------------------------------------------------
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def use_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "figure.figsize": (7.5, 4.2),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
    })


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def V_number(a, wl, n1, n2):
    """V = (2 pi a / lambda) * sqrt(n1^2 - n2^2); a = half-thickness or core radius."""
    return 2 * np.pi * a / wl * np.sqrt(n1**2 - n2**2)


def NA(n1, n2, n0=1.0):
    return np.sqrt(n1**2 - n2**2) / n0


def Delta(n1, n2):
    """Normalized index difference (n1^2 - n2^2)/(2 n1^2) ~ (n1-n2)/n1."""
    return (n1**2 - n2**2) / (2 * n1**2)


def dB(x):
    return 10 * np.log10(x)


def dBm_to_W(p):
    return 1e-3 * 10 ** (np.asarray(p) / 10)


def W_to_dBm(p):
    return 10 * np.log10(np.asarray(p) / 1e-3)


def _roots(f, x0, x1, n=4000):
    """All simple roots of f on (x0, x1) via sign changes + Brent."""
    x = np.linspace(x0, x1, n)
    y = np.array([f(xi) for xi in x])
    out = []
    for i in range(n - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] * y[i + 1] < 0:
            out.append(brentq(f, x[i], x[i + 1], xtol=1e-14, rtol=1e-14))
    return out


# ---------------------------------------------------------------------------
# Planar (symmetric slab) waveguide -- ray picture, as in the slides
#   (2 pi n1 (2a) / lambda) cos(theta) - phi = m pi
# ---------------------------------------------------------------------------
def tir_phase(theta, n1, n2, pol="TE"):
    """Phase change phi on TIR (one reflection), theta measured from the normal.
    tan(phi/2) = sqrt(sin^2 - (n2/n1)^2) / cos            (TE)
    tan(phi/2) = sqrt(sin^2 - (n2/n1)^2) / (n^2 cos)      (TM)"""
    n = n2 / n1
    s = np.sqrt(np.maximum(np.sin(theta) ** 2 - n**2, 0.0))
    if pol == "TE":
        return 2 * np.arctan2(s, np.cos(theta))
    return 2 * np.arctan2(s, n**2 * np.cos(theta))      # TM


def slab_modes(d, wl, n1, n2, pol="TE"):
    """Allowed incidence angles theta_m (rad) of a slab of thickness d = 2a.
    Returns array ordered m = 0, 1, 2, ..."""
    k1 = 2 * np.pi * n1 / wl
    thc = np.arcsin(n2 / n1)
    out = []
    m = 0
    while True:
        f = lambda th: k1 * d * np.cos(th) - tir_phase(th, n1, n2, pol) - np.pi * m
        a, b = thc + 1e-12, np.pi / 2 - 1e-12
        if f(a) * f(b) > 0:
            break
        out.append(brentq(f, a, b, xtol=1e-15))
        m += 1
    return np.array(out)


def slab_penetration(theta, wl, n1, n2):
    """Penetration depth delta = 1/alpha_cladding of the evanescent field."""
    alpha = 2 * np.pi * n2 / wl * np.sqrt((n1 / n2) ** 2 * np.sin(theta) ** 2 - 1)
    return 1 / alpha


def slab_beta(omega, d, n1, n2, m, pol="TE"):
    """Propagation constant beta_m(omega) for a non-dispersive slab (nan below cutoff)."""
    wl = 2 * np.pi * c0 / omega
    th = slab_modes(d, wl, n1, n2, pol)
    if len(th) <= m:
        return np.nan
    return 2 * np.pi * n1 / wl * np.sin(th[m])


def slab_field(y, d, wl, n1, n2, theta, m):
    """Transverse TE field E_m(y) (normalized, |y| may exceed d/2)."""
    a = d / 2
    ky = 2 * np.pi * n1 / wl * np.cos(theta)
    gam = 1 / slab_penetration(theta, wl, n1, n2)
    even = (m % 2 == 0)
    core = np.cos(ky * y) if even else np.sin(ky * y)
    edge = np.cos(ky * a) if even else np.sin(ky * a)
    clad = edge * np.exp(-gam * (np.abs(y) - a)) * (1 if even else np.sign(y))
    E = np.where(np.abs(y) <= a, core, clad)
    return E / np.max(np.abs(E))


# ---------------------------------------------------------------------------
# Step-index fiber, weakly guiding LP modes
#   u J_{l-1}(u) / J_l(u) = - w K_{l-1}(w) / K_l(w),   u^2 + w^2 = V^2
#   b = w^2 / V^2 = (n_eff^2 - n2^2) / (n1^2 - n2^2)
# ---------------------------------------------------------------------------
def _lp_char(u, V, l):
    w = np.sqrt(max(V * V - u * u, 0.0))
    if w == 0:
        return np.nan
    # pole-free form:  u J_{l-1}(u) + w [K_{l-1}(w)/K_l(w)] J_l(u) = 0
    ratio = sp.kve(abs(l - 1), w) / sp.kve(l, w)
    return u * sp.jv(l - 1, u) + w * ratio * sp.jv(l, u)


def LP_b(V, l, m=None):
    """Normalized propagation constants b of LP_lm (m = 1, 2, ...).
    With m=None returns all guided b for that l (descending)."""
    if V <= 0:
        return [] if m is None else np.nan
    us = _roots(lambda u: _lp_char(u, V, l), 1e-9, V * (1 - 1e-12), n=max(400, int(80 * V)))
    bs = sorted([1 - (u / V) ** 2 for u in us], reverse=True)
    if m is None:
        return bs
    return bs[m - 1] if len(bs) >= m else np.nan


def LP_all(V):
    """List of (l, m, b) for every guided LP mode, sorted by b."""
    out = []
    l = 0
    while True:
        bs = LP_b(V, l)
        if not bs:
            break
        out += [(l, i + 1, b) for i, b in enumerate(bs)]
        l += 1
    return sorted(out, key=lambda t: -t[2])


def LP_count(V):
    """Number of guided modes including 2 polarizations and cos/sin (l>0) degeneracy."""
    return sum(2 if l == 0 else 4 for l, m, b in LP_all(V))


def LP_field(r_over_a, V, b, l):
    """Radial field profile (continuous at r = a, = 1 at the interface)."""
    u = V * np.sqrt(1 - b)
    w = V * np.sqrt(b)
    r = np.asarray(r_over_a, float)
    core = sp.jv(l, u * r) / sp.jv(l, u)
    clad = sp.kv(l, w * np.maximum(r, 1e-12)) / sp.kv(l, w)
    return np.where(r <= 1, core, clad)


def b_approx(V):
    """Kasap/Rudolph-Neumann LP01 approximation, valid for 1.5 < V < 2.5."""
    return (1.1428 - 0.996 / V) ** 2


def mfd_marcuse(a, V):
    """Mode field diameter 2w (Marcuse)."""
    return 2 * a * (0.65 + 1.619 * V**-1.5 + 2.879 * V**-6)


def mfd_exact_gaussian(a, V):
    """MFD from the best Gaussian fit (max overlap) to the exact LP01 field."""
    b = LP_b(V, 0, 1)
    r = np.linspace(0, 6, 3000)
    E = LP_field(r, V, b, 0)
    def neg_overlap(w):
        G = np.exp(-(r / w) ** 2)
        return -np.trapezoid(E * G * r, r) ** 2 / (np.trapezoid(E**2 * r, r) * np.trapezoid(G**2 * r, r))
    from scipy.optimize import minimize_scalar
    w = minimize_scalar(neg_overlap, bounds=(0.3, 5), method="bounded").x
    return 2 * a * w


def mfd_petermann2(a, V):
    """Petermann-II MFD (the ITU definition): 2w = 2 sqrt(2 int E^2 r dr / int (E')^2 r dr)."""
    b = LP_b(V, 0, 1)
    r = np.linspace(1e-6, 12, 20000)
    E = LP_field(r, V, b, 0)
    dE = np.gradient(E, r)
    return 2 * a * np.sqrt(2 * np.trapezoid(E**2 * r, r) / np.trapezoid(dE**2 * r, r))


# ---------------------------------------------------------------------------
# Glass refractive index (Sellmeier, wavelength in metres)
#   SiO2: Malitson (1965); GeO2: Fleming (1984). Doped glass: linear mixing
#   of the Sellmeier coefficients with the GeO2 mole fraction x.
# ---------------------------------------------------------------------------
_SIO2 = (np.array([0.6961663, 0.4079426, 0.8974794]),
         np.array([0.0684043, 0.1162414, 9.896161]))          # B, lambda_i [um]
_GEO2 = (np.array([0.80686642, 0.71815848, 0.85416831]),
         np.array([0.068972606, 0.15396605, 11.841931]))


def n_doped(wl, x=0.0):
    """Index of (SiO2)_{1-x}(GeO2)_x at vacuum wavelength wl [m]."""
    B = (1 - x) * _SIO2[0] + x * _GEO2[0]
    L = (1 - x) * _SIO2[1] + x * _GEO2[1]
    um2 = (np.asarray(wl) * 1e6) ** 2
    n2 = 1 + sum(Bi * um2 / (um2 - Li**2) for Bi, Li in zip(B, L))
    return np.sqrt(n2)


def n_silica(wl):
    return n_doped(wl, 0.0)


def _deriv(f, wl, k=1, h=1e-9):
    if k == 1:
        return (f(wl + h) - f(wl - h)) / (2 * h)
    return (f(wl + h) - 2 * f(wl) + f(wl - h)) / h**2


def group_index(wl, x=0.0):
    """N_g = n - lambda dn/dlambda."""
    return n_doped(wl, x) - wl * _deriv(lambda l: n_doped(l, x), wl)


def D_material(wl, x=0.0):
    """Material dispersion D_m = -(lambda/c) d^2n/dlambda^2  [s/m^2]."""
    return -(wl / c0) * _deriv(lambda l: n_doped(l, x), wl, k=2, h=5e-9)


PS_NM_KM = 1e-12 / (1e-9 * 1e3)     # 1 ps/(nm km) in s/m^2


# ---------------------------------------------------------------------------
# Single-mode fiber: exact (weak guidance) group delay and chromatic dispersion
# ---------------------------------------------------------------------------
def neff_smf(wl, a, x_core, x_clad=0.0):
    n1, n2 = n_doped(wl, x_core), n_doped(wl, x_clad)
    V = V_number(a, wl, n1, n2)
    b = LP_b(V, 0, 1)
    return np.sqrt(n2**2 + b * (n1**2 - n2**2))


def group_delay_smf(wl, a, x_core, x_clad=0.0, h=0.5e-9):
    """tau/L = (1/c)(n_eff - lambda dn_eff/dlambda) [s/m]."""
    f = lambda l: neff_smf(l, a, x_core, x_clad)
    return (f(wl) - wl * (f(wl + h) - f(wl - h)) / (2 * h)) / c0


def D_total_smf(wl, a, x_core, x_clad=0.0, h=2e-9):
    """Chromatic dispersion D = d(tau/L)/dlambda [s/m^2], material + waveguide + profile."""
    return (group_delay_smf(wl + h, a, x_core, x_clad) - group_delay_smf(wl - h, a, x_core, x_clad)) / (2 * h)


def D_waveguide_approx(wl, a, n2, Dlt):
    """Kasap/Ghatak waveguide dispersion with V d^2(Vb)/dV^2 ~ 0.080 + 0.549(2.834 - V)^2."""
    V = 2 * np.pi * a / wl * n2 * np.sqrt(2 * Dlt)
    return -(n2 * Dlt / (c0 * wl)) * (0.080 + 0.549 * (2.834 - V) ** 2)


# ---------------------------------------------------------------------------
# Graded-index fiber: Olshansky-Keck rms intermodal spread
# ---------------------------------------------------------------------------
def sigma_intermodal_grin(gamma, n1, Dlt, L=1e3, eps=0.0):
    """rms intermodal pulse spread for a power-law profile (Olshansky & Keck 1976)."""
    g = np.asarray(gamma, float)
    C1 = (g - 2 - eps) / (g + 2)
    C2 = (3 * g - 2 - 2 * eps) / (2 * (g + 2))
    pref = L * n1 * Dlt / (2 * c0) * g / (g + 1) * np.sqrt((g + 2) / (3 * g + 2))
    br = (C1**2 + 4 * C1 * C2 * Dlt * (g + 1) / (2 * g + 1)
          + 16 * Dlt**2 * C2**2 * (g + 1) ** 2 / ((5 * g + 2) * (3 * g + 2)))
    return pref * np.sqrt(br)


# ---------------------------------------------------------------------------
# Attenuation
# ---------------------------------------------------------------------------
def rayleigh_alpha(wl, n, betaT, Tf):
    """Rayleigh attenuation [1/m] of a frozen-in density fluctuation glass."""
    return 8 * np.pi**3 / (3 * wl**4) * (n**2 - 1) ** 2 * betaT * kB * Tf


def attenuation_model(wl, AR=0.90, OH_peak=0.3, x_ge=0.0):
    """Very simple silica fiber loss spectrum [dB/km] (wl in m):
    Rayleigh A_R/lambda^4 + IR lattice tail + UV Urbach tail + OH overtone at 1383 nm."""
    um = np.asarray(wl) * 1e6
    ray = AR / um**4
    ir = 7.81e11 * np.exp(-48.48 / um)
    uv = 1.542e-2 * (x_ge + 0.02) / (46.6 * (x_ge + 0.02) + 60) * np.exp(4.63 / um)
    oh = OH_peak * np.exp(-0.5 * ((um - 1.383) / 0.012) ** 2) \
        + 0.08 * OH_peak * np.exp(-0.5 * ((um - 1.24) / 0.015) ** 2)
    return ray + ir + uv + oh


# ---------------------------------------------------------------------------
# Communications helpers
# ---------------------------------------------------------------------------
def Q_to_BER(Q):
    return 0.5 * erfc(np.asarray(Q) / np.sqrt(2))


def prbs(order=7, n=None, seed=0b1111111):
    """PRBS-7 (x^7 + x^6 + 1) bit sequence."""
    taps = {7: (7, 6), 9: (9, 5), 11: (11, 9), 15: (15, 14)}[order]
    state = seed & ((1 << order) - 1) or 1
    n = n or (2**order - 1)
    out = np.empty(n, dtype=int)
    for i in range(n):
        bit = ((state >> (taps[0] - 1)) ^ (state >> (taps[1] - 1))) & 1
        out[i] = state & 1
        state = ((state << 1) | bit) & ((1 << order) - 1)
    return out


def imdd_link(bits, Rb=10e9, L_km=0.0, D_ps_nm_km=17.0, alpha_dB_km=0.2,
              P_launch_dBm=3.0, wl=1550e-9, sps=32, rise=0.3, ER_dB=15.0,
              fmt="NRZ", R=0.9, RL=50.0, T=300.0, Fn_dB=3.0, Bel=None,
              rng=None, noise=True):
    """Minimal intensity-modulation / direct-detection link (numpy only).

    Chirp-free transmitter -> linear fiber (loss + group-velocity dispersion,
    applied in the frequency domain) -> PIN photodiode with shot + thermal noise
    -> 4th-order-like Gaussian electrical filter.
    Returns t [s], transmitted power [W], received current [A]."""
    rng = np.random.default_rng(1) if rng is None else rng
    Ts = 1 / Rb
    fs = sps * Rb
    N = len(bits) * sps
    t = np.arange(N) / fs
    # --- drive waveform
    if fmt.upper() == "RZ":
        frac = (np.arange(sps) < sps // 2).astype(float)
        drive = (np.repeat(bits, sps) * np.tile(frac, len(bits))).astype(float)
    else:
        drive = np.repeat(bits, sps).astype(float)
    # finite rise time: Gaussian smoothing with 10-90% rise = rise*Ts
    sig_t = rise * Ts / 2.563
    f = np.fft.fftfreq(N, 1 / fs)
    drive = np.real(np.fft.ifft(np.fft.fft(drive) * np.exp(-0.5 * (2 * np.pi * f * sig_t) ** 2)))
    drive = np.clip(drive, 0, 1)
    # --- optical power with finite extinction ratio, mean = P_launch
    er = 10 ** (-ER_dB / 10)
    Pshape = er + (1 - er) * drive
    P_tx = Pshape / Pshape.mean() * dBm_to_W(P_launch_dBm)
    E = np.sqrt(P_tx).astype(complex)                # chirp-free field
    # --- fiber
    D = D_ps_nm_km * PS_NM_KM
    beta2 = -D * wl**2 / (2 * np.pi * c0)
    L = L_km * 1e3
    H = np.exp(1j * beta2 / 2 * (2 * np.pi * f) ** 2 * L)
    E = np.fft.ifft(np.fft.fft(E) * H) * 10 ** (-alpha_dB_km * L_km / 20)
    P_rx = np.abs(E) ** 2
    # --- receiver
    Bel = 0.7 * Rb if Bel is None else Bel
    I = R * P_rx
    if noise:
        sig_sh2 = 2 * q * I * Bel
        sig_th2 = 4 * kB * T * Bel / RL * 10 ** (Fn_dB / 10)
        # white noise with variance per sample such that after filtering it has the
        # right variance in the bandwidth Bel
        I = I + rng.standard_normal(N) * np.sqrt((sig_sh2 + sig_th2) * fs / (2 * Bel))
    Hrx = np.exp(-np.log(2) / 2 * (f / Bel) ** 2)          # Gaussian filter, |H(Bel)|^2 = 1/2
    I = np.real(np.fft.ifft(np.fft.fft(I) * Hrx))
    return t, P_tx, I


def eye_metrics(I, bits, sps):
    """Sample at the center of each bit; return Q, BER estimate and the samples."""
    # find best sampling phase (max eye opening)
    best = None
    for ph in range(sps):
        s = I[ph::sps][: len(bits)]
        one, zero = s[bits[: len(s)] == 1], s[bits[: len(s)] == 0]
        Q = (one.mean() - zero.mean()) / (one.std() + zero.std() + 1e-30)
        if best is None or Q > best[0]:
            best = (Q, ph)
    Q, ph = best
    return Q, Q_to_BER(Q), ph


# ---------------------------------------------------------------------------
# Gratings and interferometers
# ---------------------------------------------------------------------------
def fbg_reflectance(wl, lamB, neff, dn, L):
    """Uniform FBG reflectance (coupled-mode theory, Erdogan 1997).
    kappa = pi dn / lambda, detuning sigma = 2 pi neff (1/lambda - 1/lambda_B)."""
    wl = np.asarray(wl, float)
    kap = np.pi * dn / wl
    sig = 2 * np.pi * neff * (1 / wl - 1 / lamB)
    s = np.sqrt((kap**2 - sig**2).astype(complex))
    num = kap**2 * np.sinh(s * L) ** 2
    den = kap**2 * np.cosh(s * L) ** 2 - sig**2
    return np.real(num / den)


def fbg_bandwidth(lamB, neff, dn, L):
    """Full width between the first zeros: lambda_B^2/(pi n L) sqrt((kappa L)^2 + pi^2)."""
    kL = np.pi * dn / lamB * L
    return lamB**2 / (np.pi * neff * L) * np.sqrt(kL**2 + np.pi**2)


def fabry_perot_T(nu, d, R, n=1.0):
    """Airy transmission of a lossless FP with intensity reflectance R."""
    F = 4 * R / (1 - R) ** 2
    delta = 4 * np.pi * n * d * nu / c0
    return 1 / (1 + F * np.sin(delta / 2) ** 2)
