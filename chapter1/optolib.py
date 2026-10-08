"""
optolib.py -- small, transparent toolbox for the Chapter 1 companion
(The wave nature of light).

Convention: the slides (Kasap) use the ENGINEERING convention
    E ∝ exp[j(ωt − kz)],   N = n − jK,   ε_r = ε_r' − jε_r''.
Every function below follows it. SI units unless the name says otherwise.

Contents
--------
- plotting style                 : use_style()
- plane waves, units             : photon_energy_eV(), dlam_from_dnu()
- Gaussian beams                 : gaussian_w(), q_from_w0(), w_from_q(), abcd(), free(), lens()
- refractive index               : n_silica(), lorentz_eps(), drude_eps(), kramers_kronig()
- interfaces                     : fresnel(), penetration_depth(), brewster()
- thin films                     : tmm(), qw_stack(), bragg_R_max(), bragg_bandwidth()
- resonators                     : fp_summary(), airy_T()
- coherence                      : coherence_length(), interferogram()
- diffraction                    : grating_orders(), asm_propagate()
"""
from __future__ import annotations

import numpy as np

c0 = 299_792_458.0
eps0 = 8.8541878128e-12
mu0 = 1.25663706212e-6
h = 6.62607015e-34
qe = 1.602176634e-19

# ---------------------------------------------------------------------------
# Plot style (validated categorical palette, recessive grid) -- same as Chapter 2
# ---------------------------------------------------------------------------
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
def dB(x): return 10 * np.log10(x)


def photon_energy_eV(wl):
    return h * c0 / wl / qe


def dlam_from_dnu(dnu, wl):
    """Spectral width in wavelength for a width dnu in frequency: Δλ = λ²Δν/c."""
    return wl**2 * dnu / c0


# ---------------------------------------------------------------------------
# Gaussian beams
# ---------------------------------------------------------------------------
def gaussian_w(z, w0, wl, M2=1.0, n=1.0):
    """Beam radius w(z) and Rayleigh range z0 = π w0² n / (M² λ)."""
    z0 = np.pi * w0**2 * n / (M2 * wl)
    return w0 * np.sqrt(1 + (z / z0) ** 2), z0


def q_from_w0(w0, wl, n=1.0, z=0.0):
    """Complex beam parameter at distance z from a waist (q = z + j z0)."""
    return z + 1j * np.pi * w0**2 * n / wl


def w_from_q(q, wl, n=1.0):
    return np.sqrt(-wl / (np.pi * n * np.imag(1 / q)))


def abcd(q, M):
    A, B, C, D = np.asarray(M).ravel()
    return (A * q + B) / (C * q + D)


def free(d): return np.array([[1, d], [0, 1]])
def lens(f): return np.array([[1, 0], [-1 / f, 1]])


# ---------------------------------------------------------------------------
# Refractive index models
# ---------------------------------------------------------------------------
_B = np.array([0.6961663, 0.4079426, 0.8974794])
_L = np.array([0.0684043, 0.1162414, 9.896161])      # µm (Malitson 1965)


def n_silica(wl):
    """Fused silica (Sellmeier, Malitson), wl in metres."""
    l2 = (np.asarray(wl) * 1e6)[..., None] ** 2
    return np.sqrt(1 + np.sum(_B * l2 / (l2 - _L**2), axis=-1))


def group_index_silica(wl, h=1e-10):
    return n_silica(wl) - wl * (n_silica(wl + h) - n_silica(wl - h)) / (2 * h)


def lorentz_eps(w, w0, wp, gamma, eps_inf=1.0):
    """ε_r(ω) = ε∞ + ωp²/(ω0² − ω² + jγω)  -> returns ε' − jε'' (engineering convention)."""
    return eps_inf + wp**2 / (w0**2 - w**2 + 1j * gamma * w)


def drude_eps(w, wp, gamma):
    """Free-electron (Drude) model: Lorentz with ω0 = 0."""
    return lorentz_eps(w, 0.0, wp, gamma)


def n_K(eps):
    """Complex index N = n − jK from ε_r (branch with n > 0, K ≥ 0)."""
    N = np.sqrt(np.asarray(eps, dtype=complex))
    N = np.where(N.real < 0, -N, N)
    return N.real, -N.imag


def kramers_kronig(w, eps2):
    """ε'(ω) − 1 = (2/π) P∫ ω' ε''(ω') / (ω'² − ω²) dω', evaluated at grid midpoints."""
    wm = 0.5 * (w[1:] + w[:-1])
    dw = w[1] - w[0]
    return wm, 1 + np.array([2 / np.pi * np.sum(w * eps2 / (w**2 - wi**2)) * dw for wi in wm])


# ---------------------------------------------------------------------------
# Interfaces: Fresnel coefficients (Kasap convention)
# ---------------------------------------------------------------------------
def _root(x):
    """√x on the branch with Im ≤ 0: the decaying (evanescent) solution for e^{jωt}."""
    r = np.sqrt(np.asarray(x, dtype=complex))
    return np.where(r.imag > 0, np.conj(r), r)


def fresnel(n1, n2, th):
    """Amplitude coefficients (r⊥, r∥, t⊥, t∥) for light going from n1 to n2 at angle th (rad).
    Kasap/slide convention: r∥ = r⊥ at normal incidence; r∥ + n t∥ = 1, r⊥ + 1 = t⊥."""
    n = n2 / n1
    ci = np.cos(th)
    s = _root(n**2 - np.sin(th) ** 2)
    r_perp = (ci - s) / (ci + s)
    r_par = (s - n**2 * ci) / (s + n**2 * ci)
    t_perp = 2 * ci / (ci + s)
    t_par = 2 * n * ci / (s + n**2 * ci)
    return r_perp, r_par, t_perp, t_par


def critical_angle(n1, n2):
    return np.arcsin(n2 / n1)


def brewster(n1, n2):
    return np.arctan(n2 / n1)


def penetration_depth(n1, n2, th, wl):
    """1/α2 of the evanescent wave beyond the critical angle."""
    a2 = 2 * np.pi * n2 / wl * np.sqrt((n1 / n2) ** 2 * np.sin(th) ** 2 - 1)
    return 1 / a2


def lateral_displacement(L, n, th, n0=1.0):
    """Beam shift through a plate of thickness L and index n (slides 95–98)."""
    st = n0 * np.sin(th) / n
    tt = np.arcsin(st)
    return L * np.sin(th - tt) / np.cos(tt)


# ---------------------------------------------------------------------------
# Thin films: characteristic-matrix (transfer-matrix) method
# ---------------------------------------------------------------------------
def tmm(n_list, d_list, wl, th0=0.0, pol="s"):
    """Reflectance R, transmittance T and complex r of a planar stack.
    n_list: [n_incident, n_1, ..., n_m, n_substrate] (complex allowed, N = n − jK)
    d_list: thicknesses of the m inner layers (m)
    wl: vacuum wavelength (scalar or array); th0: incidence angle in medium 0 (rad)."""
    wl = np.atleast_1d(wl).astype(float)
    n = [np.asarray(x, dtype=complex) for x in n_list]
    s0 = n[0] * np.sin(th0)
    ncos = [_root(ni**2 - s0**2) for ni in n]
    eta = [nc if pol == "s" else ni**2 / nc for ni, nc in zip(n, ncos)]
    M = np.broadcast_to(np.eye(2, dtype=complex), (wl.size, 2, 2)).copy()
    for nc, et, d in zip(ncos[1:-1], eta[1:-1], d_list):
        dl = 2 * np.pi * nc * d / wl
        Mj = np.empty_like(M)
        Mj[:, 0, 0] = np.cos(dl); Mj[:, 0, 1] = 1j * np.sin(dl) / et
        Mj[:, 1, 0] = 1j * et * np.sin(dl); Mj[:, 1, 1] = np.cos(dl)
        M = M @ Mj
    B = M[:, 0, 0] + M[:, 0, 1] * eta[-1]
    C = M[:, 1, 0] + M[:, 1, 1] * eta[-1]
    r = (eta[0] * B - C) / (eta[0] * B + C)
    T = 4 * eta[0].real * eta[-1].real / np.abs(eta[0] * B + C) ** 2
    return np.abs(r) ** 2, T, r


def qw_stack(nH, nL, N, wl0, n0=1.0, ns=1.47, first="H"):
    """Index and thickness lists of a quarter-wave stack n0 | (H L)^N | ns."""
    pair = [nH, nL] if first == "H" else [nL, nH]
    return [n0] + pair * N + [ns], [wl0 / (4 * x) for x in pair * N]


def bragg_R_max(n0, n1, n2, ns, N):
    """Peak reflectance of N quarter-wave pairs (n1 next to n0), Kasap's formula."""
    x = (n0 * n2 ** (2 * N) - ns * n1 ** (2 * N)) / (n0 * n2 ** (2 * N) + ns * n1 ** (2 * N))
    return x**2


def bragg_bandwidth(nH, nL, wl0):
    """Stop-band width of an infinite quarter-wave stack."""
    return 4 * wl0 / np.pi * np.arcsin((nH - nL) / (nH + nL))


# ---------------------------------------------------------------------------
# Fabry–Perot resonator
# ---------------------------------------------------------------------------
def fp_summary(L, n, R, wl_target):
    """Mode number, mode wavelength, FSR, finesse, linewidth and Q (slides 196–209)."""
    m = round(2 * n * L / wl_target)
    lm = 2 * n * L / m
    nu = c0 / lm
    fsr = c0 / (2 * n * L)
    F = np.pi * np.sqrt(R) / (1 - R)
    dnu = fsr / F
    return dict(m=m, lam_m_nm=lm * 1e9, nu_THz=nu / 1e12, FSR_GHz=fsr / 1e9,
                FSR_nm=lm**2 * fsr / c0 * 1e9, finesse=F, dnu_GHz=dnu / 1e9,
                dlam_nm=lm**2 * dnu / c0 * 1e9, Q=m * F)


def airy_T(wl, L, R, n=1.0):
    """Transmission of a lossless FP with intensity reflectance R."""
    return (1 - R) ** 2 / ((1 - R) ** 2 + 4 * R * np.sin(2 * np.pi * n * L / wl) ** 2)


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------
def coherence_length(wl=None, dlam=None, dnu=None):
    """Slide convention: Δt = 1/Δν, l_c = c Δt = λ²/Δλ."""
    if dnu is None:
        dnu = c0 * dlam / wl**2
    return c0 / dnu, 1 / dnu, dnu


def interferogram(opd, wl0, dlam, n=4001):
    """Michelson output vs optical path difference for a Gaussian spectrum of FWHM dlam."""
    k0 = 2 * np.pi / wl0
    dk = 2 * np.pi * dlam / wl0**2
    k = np.linspace(k0 - 3 * dk, k0 + 3 * dk, n)
    S = np.exp(-4 * np.log(2) * ((k - k0) / dk) ** 2)
    S /= S.sum()
    return (S[None, :] * (1 + np.cos(k[None, :] * opd[:, None]))).sum(axis=1)


# ---------------------------------------------------------------------------
# Diffraction
# ---------------------------------------------------------------------------
def grating_orders(d, wl, thi_deg, mmax=6):
    """Propagating orders of d(sinθm − sinθi) = mλ: {m: θm in degrees}."""
    out = {}
    for m in range(-mmax, mmax + 1):
        s = np.sin(np.radians(thi_deg)) + m * wl / d
        if abs(s) <= 1:
            out[m] = float(np.degrees(np.arcsin(s)))
    return out


def asm_propagate(U, dx, wl, z, pad=2):
    """Band-limited angular-spectrum propagation (e^{jωt} convention), zero-padded."""
    Ny, Nx = U.shape
    P = np.zeros((pad * Ny, pad * Nx), complex)
    P[:Ny, :Nx] = U
    P = np.roll(P, (pad * Ny // 2 - Ny // 2, pad * Nx // 2 - Nx // 2), axis=(0, 1))
    fx = np.fft.fftfreq(pad * Nx, dx)
    FX, FY = np.meshgrid(fx, fx)
    arg = 1 / wl**2 - FX**2 - FY**2
    H = np.exp(-2j * np.pi * z * np.sqrt(np.maximum(arg, 0))) * (arg > 0)
    flim = 1 / (wl * np.sqrt((2 * z / (pad * Nx * dx)) ** 2 + 1))
    H *= (np.abs(FX) < flim) & (np.abs(FY) < flim)
    out = np.fft.ifft2(np.fft.fft2(P) * H)
    out = np.roll(out, (Ny // 2 - pad * Ny // 2, Nx // 2 - pad * Nx // 2), axis=(0, 1))
    return out[:Ny, :Nx]
