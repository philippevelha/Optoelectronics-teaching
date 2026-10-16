#!/usr/bin/env python3
"""
1D drift-diffusion simulator of a GaAs p-n junction (LED) under forward bias.

Unknowns (all in units of kT/q): electrostatic potential psi and quasi-Fermi
potentials phi_n, phi_p, with
    n = ni exp(psi - phi_n),   p = ni exp(phi_p - psi)
Equations:
    Poisson:        eps*Vt*psi'' / q + (p - n + Nd - Na) = 0
    electrons:      dGn/dx =  R,   Gn = Jn/q = -mu_n Vt n dphi_n/dx
    holes:          dGp/dx = -R,   Gp = Jp/q = -mu_p Vt p dphi_p/dx
    R = (np - ni^2) [ B + Cn n + Cp p + 1/(tau_p (n+n1) + tau_n (p+p1)) ]
         radiative      Auger            SRH (trap at Et - Ei,
                                         n1 = ni e^{Et/kT}, p1 = ni e^{-Et/kT})

Optional double heterostructure (--dh-d > 0): a p-GaAs active layer |x| < d/2
between AlxGa1-xAs claddings, with Delta Eg = 1.247 x split 67:33 between
Delta Ec and Delta Ev.  The offsets enter through band parameters,
    n = ni exp(psi + th_n - phi_n),   p = ni exp(phi_p - psi + th_p),
    th_n = -Delta Ec/kT,  th_p = -Delta Ev/kT  in the claddings (0 in the layer),
with the same Nc, Nv, mobilities and permittivity everywhere.  Each offset is
spread over one mesh cell.

Finite volumes on a non-uniform mesh, log-mean (exact exponential) carrier
densities at cell faces, damped Newton with a sparse coloured finite-difference
Jacobian, and voltage continuation.  The junction sits at x = 0, p side x < 0.
pn_led_sim.js is a line-by-line port used by the interactive panels of the book.

Usage:  python3 pn_led_sim.py [--Na 1e18 --Nd 1e17 --B 1e-10 --tau 1e-8 ...]
        python3 pn_led_sim.py --dh-d 100 --xal 0.3 --nact 1e17 --tau 1e-7   (heterostructure)
"""
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

q = 1.602176634e-19      # C
kB = 8.617333262e-5      # eV/K
eps0 = 8.8541878128e-14  # F/cm


def expm1_over_x(a):
    """(e^a - 1)/a, safe at a -> 0."""
    a = np.clip(a, -100.0, 100.0)
    out = np.empty_like(a)
    small = np.abs(a) < 1e-8
    out[small] = 1.0 + a[small] / 2.0
    out[~small] = np.expm1(a[~small]) / a[~small]
    return out


def graded(L, h0, flat, r, hmax):
    """Distances from the junction: fine flat spacing, then geometric grading."""
    d = [0.0]
    h = h0
    while d[-1] < L - 1e-15:
        if d[-1] >= flat:
            h = min(h * r, hmax)
        d.append(d[-1] + h)
    d = np.array(d)
    d[-1] = L
    if d[-1] - d[-2] < 0.3 * (d[-2] - d[-3]):
        d = np.delete(d, -2)
    return d


class PNDiode:
    def __init__(self, Na=1e18, Nd=1e17, Lp=15e-4, Ln=15e-4, T=300.0,
                 ni=2.1e6, Eg=1.424, epsr=12.9, mun=4000.0, mup=200.0,
                 tau_n=1e-8, tau_p=1e-8, B=1e-10, Cn=1e-30, Cp=1e-30, Et=0.0,
                 dh=None, mesh=(1e-7, 3e-5, 1.05, 5e-6)):
        """dh = dict(d=..., dEc=..., dEv=..., Nact=...) adds a double heterostructure:
        active layer |x| < d/2 (cm) with net doping Nact (> 0 donors, < 0 acceptors),
        claddings with the conduction band raised by dEc and the valence band
        lowered by dEv (eV)."""
        self.Na, self.Nd, self.ni, self.Eg = Na, Nd, ni, Eg
        self.mun, self.mup = mun, mup
        self.tau_n, self.tau_p = tau_n, tau_p
        self.B, self.Cn, self.Cp, self.Et, self.dh = B, Cn, Cp, Et, dh
        self.Vt = kB * T
        self.eps_q = epsr * eps0 / q
        self.epsr = epsr

        h0, flat, r, hmax = mesh                      # cm
        if dh:
            flat = max(flat, dh["d"] / 2 + 2e-5)
        left = graded(Lp, h0, flat, r, hmax)
        right = graded(Ln, h0, flat, r, hmax)
        self.x = np.concatenate([-left[::-1], right[1:]])
        self.N = len(self.x)
        self.h = np.diff(self.x)
        w = np.zeros(self.N)
        w[:-1] += self.h / 2
        w[1:] += self.h / 2
        self.w = w
        near = lambda a: int(np.argmin(np.abs(self.x - a)))
        self.thn = np.zeros(self.N)
        self.thp = np.zeros(self.N)
        if not dh:
            C = np.where(self.x > 0, Nd, -Na).astype(float)
            C[near(0.0)] = 0.5 * (Nd - Na)
        else:
            iL, iR = near(-dh["d"] / 2), near(dh["d"] / 2)
            idx = np.arange(self.N)
            inside = (idx > iL) & (idx < iR)
            C = np.where(inside, dh["Nact"], np.where(self.x < 0, -Na, Nd)).astype(float)
            tn, tp = -dh["dEc"] / self.Vt, -dh["dEv"] / self.Vt
            self.thn[~inside], self.thp[~inside] = tn, tp
            for k, Cout in ((iL, -Na), (iR, Nd)):     # interface nodes: average of both sides
                C[k] = 0.5 * (Cout + dh["Nact"])
                self.thn[k], self.thp[k] = 0.5 * tn, 0.5 * tp
            self.iL, self.iR = iL, iR
        self.C = C
        # local intrinsic density and SRH trap densities (trap at Et above the local midgap)
        nie = ni * np.exp((self.thn + self.thp) / 2)
        self.nie2 = nie**2
        self.n1 = nie * np.exp(Et / self.Vt)
        self.p1 = nie * np.exp(-Et / self.Vt)
        psi0 = self.psi_eq()
        self.psiL0, self.psiR0 = psi0[0], psi0[-1]
        self.Vbi = self.Vt * (self.psiR0 - self.psiL0)
        self.Cref = max(Na, Nd, abs(dh["Nact"]) if dh else 0.0)
        self.t0 = 1e-9

    def psi_eq(self):
        """Equilibrium, charge-neutral potential at every node."""
        return np.arcsinh(self.C / (2 * np.sqrt(self.nie2))) - (self.thn - self.thp) / 2

    # ---------------------------------------------------------------- physics
    def carriers(self, psi, fn, fp):
        n = self.ni * np.exp(np.clip(psi + self.thn - fn, -100, 100))
        p = self.ni * np.exp(np.clip(fp - psi + self.thp, -100, 100))
        return n, p

    def recomb(self, n, p, fn, fp):
        npm = self.nie2 * np.expm1(np.clip(fp - fn, -100, 100))   # n p - ni^2
        Rrad = self.B * npm
        Raug = (self.Cn * n + self.Cp * p) * npm
        Rsrh = npm / (self.tau_p * (n + self.n1) + self.tau_n * (p + self.p1))
        return Rrad, Raug, Rsrh

    def fluxes(self, psi, fn, fp, n, p):
        dpsi, dfn, dfp = np.diff(psi), np.diff(fn), np.diff(fp)
        dpn = dpsi + np.diff(self.thn)     # effective potentials of the band edges
        dpp = dpsi - np.diff(self.thp)
        nm = n[:-1] * expm1_over_x(dpn - dfn)
        pm = p[:-1] * expm1_over_x(dfp - dpp)
        Gn = -self.mun * self.Vt * nm * dfn / self.h
        Gp = -self.mup * self.Vt * pm * dfp / self.h
        return Gn, Gp, dpsi

    def residual(self, u, Vs):
        N = self.N
        psi, fn, fp = u[0::3], u[1::3], u[2::3]
        n, p = self.carriers(psi, fn, fp)
        Gn, Gp, dpsi = self.fluxes(psi, fn, fp, n, p)
        D = self.eps_q * self.Vt * dpsi / self.h
        R = sum(self.recomb(n, p, fn, fp))
        w = self.w
        P = np.zeros(N)
        En = np.zeros(N)
        Ep = np.zeros(N)
        P[1:-1] = ((D[1:] - D[:-1]) / w[1:-1] + (p - n + self.C)[1:-1]) / self.Cref
        En[1:-1] = ((Gn[1:] - Gn[:-1]) / w[1:-1] - R[1:-1]) * self.t0 / self.Cref
        Ep[1:-1] = ((Gp[1:] - Gp[:-1]) / w[1:-1] + R[1:-1]) * self.t0 / self.Cref
        # ohmic contacts: p side (anode) at +V, n side (cathode) at 0
        P[0] = psi[0] - (self.psiL0 + Vs)
        P[-1] = psi[-1] - self.psiR0
        En[0], Ep[0] = fn[0] - Vs, fp[0] - Vs
        En[-1], Ep[-1] = fn[-1], fp[-1]
        F = np.empty(3 * N)
        F[0::3], F[1::3], F[2::3] = P, En, Ep
        return F

    # ----------------------------------------------------------------- solver
    def jacobian(self, u, Vs, F0):
        N, eps = self.N, 1e-6
        rows, cols, vals = [], [], []
        for c in range(3):
            idx = np.arange(c, N, 3)
            for k in range(3):
                du = np.zeros_like(u)
                du[3 * idx + k] = eps
                dF = (self.residual(u + du, Vs) - F0) / eps
                for off in (-1, 0, 1):
                    jn = idx + off
                    m = (jn >= 0) & (jn < N)
                    jn, jp = jn[m], idx[m]
                    for r in range(3):
                        rows.append(3 * jn + r)
                        cols.append(3 * jp + k)
                        vals.append(dF[3 * jn + r])
        return sp.csr_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(3 * N, 3 * N))

    def solve(self, u, V, tol=1e-9, maxit=80, dmax=2.0):
        Vs = V / self.Vt
        for it in range(maxit):
            F0 = self.residual(u, Vs)
            J = self.jacobian(u, Vs, F0)
            rmax = abs(J).max(axis=1).toarray().ravel()
            rmax[rmax == 0] = 1.0
            Dr = sp.diags(1.0 / rmax)
            du = spla.spsolve((Dr @ J).tocsc(), -(Dr @ F0))
            m = np.max(np.abs(du))
            lam = min(1.0, dmax / m)
            u = u + lam * du
            if m < tol:
                return u, True, it + 1
        return u, False, maxit

    def initial_guess(self):
        u = np.zeros(3 * self.N)
        u[0::3] = self.psi_eq()
        return u

    # --------------------------------------------------------------- analysis
    def analyse(self, u, V):
        psi, fn, fp = u[0::3], u[1::3], u[2::3]
        n, p = self.carriers(psi, fn, fp)
        Gn, Gp, _ = self.fluxes(psi, fn, fp, n, p)
        Jint = q * (Gn + Gp)
        J = np.median(Jint)
        spread = np.max(np.abs(Jint - J)) / max(abs(J), 1e-300)
        Rr, Ra, Rs = self.recomb(n, p, fn, fp)
        rho = np.abs(p - n + self.C)
        mask = rho > 0.1 * np.abs(self.C)
        idx = np.where(mask)[0]
        lo, hi = (idx.min(), idx.max()) if len(idx) else (0, 0)
        regions = {"p": slice(0, lo), "scl": slice(lo, hi + 1), "n": slice(hi + 1, self.N)}
        if self.dh:
            regions["act"] = slice(self.iL, self.iR + 1)   # active layer, interface nodes included
        out = dict(V=V, J=J, spread=spread, lo=lo, hi=hi,
                   W=self.x[hi] - self.x[lo])
        for name, R in (("rad", Rr), ("aug", Ra), ("srh", Rs)):
            for reg, sl in regions.items():
                out[f"{name}_{reg}"] = q * np.sum(R[sl] * self.w[sl])
        out["profile"] = dict(x=self.x.copy(), psi=psi.copy(), fn=fn.copy(), fp=fp.copy(),
                              n=n, p=p, Rrad=Rr, Raug=Ra, Rsrh=Rs, lo=lo, hi=hi,
                              thn=self.thn, thp=self.thp)
        return out


def main():
    plt.switch_backend("Agg")      # command line: write the figure to a file, no window
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Na", type=float, default=1e18, help="acceptor doping, p side (cm^-3)")
    ap.add_argument("--Nd", type=float, default=1e17, help="donor doping, n side (cm^-3)")
    ap.add_argument("--B", type=float, default=1e-10, help="radiative coefficient (cm^3/s)")
    ap.add_argument("--tau", type=float, default=1e-8, help="SRH lifetime tau_n = tau_p (s)")
    ap.add_argument("--tau-n", type=float, default=None, help="SRH electron lifetime (s), overrides --tau")
    ap.add_argument("--tau-p", type=float, default=None, help="SRH hole lifetime (s), overrides --tau")
    ap.add_argument("--Et", type=float, default=0.0, help="SRH trap level Et - Ei (eV)")
    ap.add_argument("--C", type=float, default=1e-30, help="Auger coefficient Cn = Cp (cm^6/s)")
    ap.add_argument("--dh-d", type=float, default=0.0,
                    help="active-layer thickness (nm) of a double heterostructure; 0 = homojunction")
    ap.add_argument("--xal", type=float, default=0.3, help="Al fraction x of the AlxGa1-xAs claddings")
    ap.add_argument("--nact", type=float, default=1e17, help="acceptor doping of the p-GaAs active layer (cm^-3)")
    ap.add_argument("--Lp", type=float, default=15.0, help="p-side length (um)")
    ap.add_argument("--Ln", type=float, default=15.0, help="n-side length (um)")
    ap.add_argument("--vmax", type=float, default=1.25)
    ap.add_argument("--vstep", type=float, default=0.025)
    ap.add_argument("--out", default="pn_led")
    a = ap.parse_args()

    dh = None
    if a.dh_d > 0:
        dEg = 1.247 * a.xal
        dh = dict(d=a.dh_d * 1e-7, dEc=0.67 * dEg, dEv=0.33 * dEg, Nact=-a.nact)
    tau_n = a.tau if a.tau_n is None else a.tau_n
    tau_p = a.tau if a.tau_p is None else a.tau_p
    dev = PNDiode(Na=a.Na, Nd=a.Nd, Lp=a.Lp * 1e-4, Ln=a.Ln * 1e-4, B=a.B, tau_n=tau_n, tau_p=tau_p,
                  Cn=a.C, Cp=a.C, Et=a.Et, dh=dh)
    print(f"mesh: {dev.N} nodes,  Vbi = {dev.Vbi:.3f} V,  Vt = {dev.Vt*1e3:.2f} mV")
    if dh:
        print(f"double heterostructure: d = {a.dh_d:.0f} nm, x = {a.xal:.2f}, "
              f"dEc = {dh['dEc']*1e3:.0f} meV, dEv = {dh['dEv']*1e3:.0f} meV, active Na = {a.nact:.0e} cm^-3")

    u, ok, it = dev.solve(dev.initial_guess(), 0.0, maxit=400)
    print(f"equilibrium: converged={ok} in {it} Newton iterations")
    if not ok:
        raise SystemExit("equilibrium did not converge")

    grid = np.arange(0.0, a.vmax + 1e-12, a.vstep)
    keep = [0.6, 0.9, 1.1]
    results, profiles = [], {}
    for V in grid:
        if V > 0:
            u, ok, it = dev.solve(u, V)
            if not ok:
                print(f"Newton failed at V = {V:.3f} V; stopping the ramp here")
                break
        res = dev.analyse(u, V)
        results.append(res)
        for kv in keep:
            if abs(V - kv) < a.vstep / 2:
                profiles[kv] = res["profile"]
    Vg = np.array([r["V"] for r in results])
    J = np.array([r["J"] for r in results])

    def col(k):
        return np.array([r[k] for r in results])

    # split each channel into an inner region (SCL, or the active layer of a DH) and the rest
    inner, IN, OUT = ("act", "active layer", "claddings") if dh else ("scl", "SCL", "neutral regions")
    rad_tot = col("rad_p") + col("rad_scl") + col("rad_n")
    srh_tot = col("srh_p") + col("srh_scl") + col("srh_n")
    rad_in, srh_in = col(f"rad_{inner}"), col(f"srh_{inner}")
    rad_out, srh_out = rad_tot - rad_in, srh_tot - srh_in
    aug_all = col("aug_p") + col("aug_scl") + col("aug_n")
    leak = J - (rad_tot + srh_tot + aug_all)

    with np.errstate(divide="ignore", invalid="ignore"):
        nid = 1.0 / (dev.Vt * np.gradient(np.log(np.maximum(J, 1e-300)), Vg))

    # ---- text summary
    print(f"\n  V[V]   J[A/cm2]   {'rad-in-' + IN.split()[0] + '/rad-total':>22}   IQE=Jrad/J   SCL width[nm]   ideality")
    for target in (0.5, 0.8, 1.0, 1.2):
        i = int(np.argmin(np.abs(Vg - target)))
        print(f"  {Vg[i]:.3f}  {J[i]:9.3e}   {rad_in[i]/rad_tot[i]:22.3%}   {rad_tot[i]/J[i]:10.3%}   "
              f"{results[i]['W']*1e7:11.1f}   {nid[i]:7.2f}")
    print(f"\nmax relative spread of J(x) (current continuity check): "
          f"{max(r['spread'] for r in results[1:]):.2e}")

    tag_in, tag_out = ("active", "claddings") if dh else ("SCL", "neutral")
    np.savetxt(a.out + "_iv.csv",
               np.column_stack([Vg, J, rad_in, rad_out, srh_in, srh_out, aug_all, nid]),
               delimiter=",", header=f"V,J_total,J_rad_{tag_in},J_rad_{tag_out},J_SRH_{tag_in},J_SRH_{tag_out},"
                                     f"J_Auger,ideality",
               comments="")

    # ---- figure
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    cols = {0.6: "tab:blue", 0.9: "tab:green", 1.1: "tab:red"}

    def shade(axis, pr):
        """Shade the active layer of a DH, or the SCL of a homojunction."""
        if dh:
            axis.axvspan(-a.dh_d * 5e-4, a.dh_d * 5e-4, color="tab:green", alpha=0.12, zorder=0)
        else:
            axis.axvspan(pr["x"][pr["lo"]] * 1e4, pr["x"][pr["hi"]] * 1e4, color="0.85", zorder=0)

    # (a) band diagram at ~1.0 V
    kb = min(profiles, key=lambda k: abs(k - 1.0)) if profiles else None
    if kb is not None:
        pr = profiles[kb]
        xs = pr["x"] * 1e4
        ax[0, 0].plot(xs, -dev.Vt * (pr["psi"] + pr["thn"]) + dev.Eg / 2, "k", label="$E_c$")
        ax[0, 0].plot(xs, -dev.Vt * (pr["psi"] - pr["thp"]) - dev.Eg / 2, "k", label="$E_v$")
        ax[0, 0].plot(xs, -dev.Vt * pr["fn"], "tab:blue", label="$F_n$")
        ax[0, 0].plot(xs, -dev.Vt * pr["fp"], "tab:red", label="$F_p$")
        shade(ax[0, 0], pr)
        xl = max(0.6, a.dh_d * 1e-3)
        ax[0, 0].set_xlim(-xl, xl)
        ax[0, 0].set_xlabel("x [µm]  (junction at 0)")
        ax[0, 0].set_ylabel("energy [eV]")
        ax[0, 0].set_title(f"Band diagram at V = {kb:.2f} V (shaded: {IN})")
        ax[0, 0].legend(ncol=4, loc="upper right")

    # (b) recombination profiles
    for kv, pr in profiles.items():
        xs = pr["x"] * 1e4
        ax[0, 1].semilogy(xs, np.maximum(pr["Rrad"], 1e-3), color=cols[kv], label=f"radiative, {kv} V")
        ax[0, 1].semilogy(xs, np.maximum(pr["Rsrh"], 1e-3), "--", color=cols[kv], label=f"SRH, {kv} V")
    if 0.9 in profiles:
        shade(ax[0, 1], profiles[0.9])
    ax[0, 1].set_xlim(-3, 3)
    ax[0, 1].set_ylim(1e10, None)
    ax[0, 1].set_xlabel("x [µm]")
    ax[0, 1].set_ylabel("R(x) [cm$^{-3}$ s$^{-1}$]")
    ax[0, 1].set_title("Recombination rate profiles (solid: B·n·p, dashed: SRH)")
    ax[0, 1].legend(fontsize=7, ncol=2)

    # (c) J-V decomposition
    pos = lambda y: np.where(y > 0, y, np.nan)
    sel = Vg >= 0.3
    ax[1, 0].semilogy(Vg[sel], pos(J)[sel], "k", lw=2.2, label="total J")
    ax[1, 0].semilogy(Vg[sel], pos(srh_in)[sel], "--", color="tab:orange", label=f"SRH in {IN}")
    ax[1, 0].semilogy(Vg[sel], pos(srh_out)[sel], "--", color="tab:brown", label=f"SRH in {OUT}")
    ax[1, 0].semilogy(Vg[sel], pos(rad_in)[sel], color="tab:green", label=f"radiative in {IN}")
    ax[1, 0].semilogy(Vg[sel], pos(rad_out)[sel], color="tab:purple", label=f"radiative in {OUT}")
    ax[1, 0].semilogy(Vg[sel], pos(aug_all)[sel], ":", color="tab:pink", label="Auger")
    ax[1, 0].semilogy(Vg[sel], pos(leak)[sel], ":", color="0.4", label="contact leakage (remainder)")
    ax[1, 0].set_xlabel("V [V]")
    ax[1, 0].set_ylabel("current density [A/cm$^2$]")
    ax[1, 0].set_title("Where the current recombines")
    ax[1, 0].legend(fontsize=8)

    # (d) ideality factor
    s2 = Vg >= 0.4
    ax[1, 1].plot(Vg[s2], nid[s2], "k", lw=2)
    ax[1, 1].axhline(1, color="0.6", ls=":")
    ax[1, 1].axhline(2, color="0.6", ls=":")
    ax[1, 1].set_ylim(0.5, 3)
    ax[1, 1].set_xlabel("V [V]")
    ax[1, 1].set_ylabel("local ideality factor  $(1/V_t)\\,dV/d\\ln J$")
    ax[1, 1].set_title("n = 2: recombination in SCL, n = 1: diffusion / neutral regions")

    kind = f"AlGaAs/GaAs DH (d = {a.dh_d:.0f} nm, x = {a.xal:.2f})" if dh else "GaAs p-n LED junction"
    fig.suptitle(f"{kind}: Na={a.Na:.0e}, Nd={a.Nd:.0e} cm$^{{-3}}$, B={a.B:.0e} cm$^3$/s, "
                 f"$\\tau_{{SRH}}$={tau_n*1e9:.3g}/{tau_p*1e9:.3g} ns", y=0.995)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=130)
    print(f"saved {a.out}.png and {a.out}_iv.csv")


if __name__ == "__main__":
    main()
