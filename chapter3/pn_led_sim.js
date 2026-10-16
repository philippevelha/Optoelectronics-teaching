// 1D drift-diffusion simulator of a GaAs p-n junction (LED) under forward bias.
// JavaScript port of pn_led_sim.py, used by the interactive explorer of pn-led-sim.qmd.
//
// Unknowns (units of kT/q): potential psi and quasi-Fermi potentials phi_n, phi_p,
//     n = ni exp(psi - phi_n),   p = ni exp(phi_p - psi)
// R = (np - ni^2) [ B + Cn n + Cp p + 1/(tau_p (n + n1) + tau_n (p + p1)) ]
//       radiative     Auger             SRH (trap at Et - Ei, n1 = ni e^{Et/kT}, p1 = ni e^{-Et/kT})
// Same discretisation as the Python version (finite volumes, Scharfetter-Gummel-like
// log-mean face densities, damped Newton with a coloured finite-difference Jacobian).
// The linear system is block tridiagonal, solved here by banded Gaussian elimination
// with partial pivoting.  Et = 0 reproduces pn_led_sim.py exactly.
//
// Optional double heterostructure (dh = {d, dEc, dEv, Nact}): the layer |x| < d/2 is the narrow-gap
// active layer (doping Nact, > 0 donors, < 0 acceptors); the claddings outside have their conduction
// band raised by dEc and valence band lowered by dEv (eV).  This enters through band parameters
//     n = ni exp(psi + thn - phi_n),   p = ni exp(phi_p - psi + thp),   thn = -dEc/kT, thp = -dEv/kT
// in the claddings (0 in the active layer), with the same Nc, Nv, mobilities and permittivity everywhere.
// The offset is spread over one mesh cell (2 nm), so carriers cross it by drift-diffusion.

export const q = 1.602176634e-19;     // C
export const kB = 8.617333262e-5;     // eV/K
const eps0 = 8.8541878128e-14;        // F/cm
const clip = (a) => (a > 100 ? 100 : a < -100 ? -100 : a);

function expm1OverX(a) {
  a = clip(a);
  return Math.abs(a) < 1e-8 ? 1 + a / 2 : Math.expm1(a) / a;
}

function graded(L, h0, flat, r, hmax) {
  const d = [0];
  let h = h0;
  while (d[d.length - 1] < L - 1e-15) {
    if (d[d.length - 1] >= flat) h = Math.min(h * r, hmax);
    d.push(d[d.length - 1] + h);
  }
  const m = d.length;
  d[m - 1] = L;
  if (d[m - 1] - d[m - 2] < 0.3 * (d[m - 2] - d[m - 3])) d.splice(m - 2, 1);
  return d;
}

export class PNDiode {
  constructor({Na = 1e18, Nd = 1e17, Lp = 15e-4, Ln = 15e-4, T = 300, ni = 2.1e6, Eg = 1.424,
               epsr = 12.9, mun = 4000, mup = 200, tau_n = 1e-8, tau_p = 1e-8, Et = 0,
               B = 1e-10, Cn = 1e-30, Cp = 1e-30, mesh = [1e-7, 3e-5, 1.05, 5e-6], dh = null} = {}) {
    Object.assign(this, {Na, Nd, ni, Eg, mun, mup, tau_n, tau_p, B, Cn, Cp, Et, dh});
    this.Vt = kB * T;
    this.eps_q = epsr * eps0 / q;

    let [h0, flat, r, hmax] = mesh;
    if (dh) flat = Math.max(flat, dh.d / 2 + 2e-5);
    const left = graded(Lp, h0, flat, r, hmax), right = graded(Ln, h0, flat, r, hmax);
    const x = [...left.slice().reverse().map((d) => -d), ...right.slice(1)];
    const N = (this.N = x.length);
    this.x = Float64Array.from(x);
    this.h = new Float64Array(N - 1);
    for (let i = 0; i < N - 1; i++) this.h[i] = x[i + 1] - x[i];
    this.w = new Float64Array(N);
    for (let i = 0; i < N - 1; i++) { this.w[i] += this.h[i] / 2; this.w[i + 1] += this.h[i] / 2; }
    this.C = new Float64Array(N);
    this.thn = new Float64Array(N);
    this.thp = new Float64Array(N);
    const near = (a) => { let k = 0; for (let i = 0; i < N; i++) if (Math.abs(x[i] - a) < Math.abs(x[k] - a)) k = i; return k; };
    if (!dh) {
      for (let i = 0; i < N; i++) this.C[i] = x[i] > 0 ? Nd : -Na;
      this.C[near(0)] = 0.5 * (Nd - Na);
    } else {
      const tn = -dh.dEc / this.Vt, tp = -dh.dEv / this.Vt, iL = near(-dh.d / 2), iR = near(dh.d / 2);
      for (let i = 0; i < N; i++) {
        const inside = i > iL && i < iR;
        this.C[i] = inside ? dh.Nact : x[i] < 0 ? -Na : Nd;
        this.thn[i] = inside ? 0 : tn;
        this.thp[i] = inside ? 0 : tp;
      }
      for (const [k, Cout] of [[iL, -Na], [iR, Nd]]) {     // interface nodes: average of both sides
        this.C[k] = 0.5 * (Cout + dh.Nact);
        this.thn[k] = 0.5 * tn;
        this.thp[k] = 0.5 * tp;
      }
      this.iL = iL;
      this.iR = iR;
    }
    // local intrinsic density and SRH trap densities (trap at Et above the local midgap)
    this.nie2 = new Float64Array(N);
    this.n1 = new Float64Array(N);
    this.p1 = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const nie = ni * Math.exp((this.thn[i] + this.thp[i]) / 2);
      this.nie2[i] = nie * nie;
      this.n1[i] = nie * Math.exp(Et / this.Vt);
      this.p1[i] = nie * Math.exp(-Et / this.Vt);
    }
    this.psiL0 = this.psiEq(0);
    this.psiR0 = this.psiEq(N - 1);
    this.Vbi = this.Vt * (this.psiR0 - this.psiL0);
    this.Cref = Math.max(Na, Nd, dh ? Math.abs(dh.Nact) : 0);
    this.t0 = 1e-9;
  }

  // equilibrium, charge-neutral potential at node i
  psiEq(i) {
    const nie = Math.sqrt(this.nie2[i]);
    return Math.asinh(this.C[i] / (2 * nie)) - (this.thn[i] - this.thp[i]) / 2;
  }

  // ------------------------------------------------------------------ physics
  carriers(u) {
    const N = this.N, n = new Float64Array(N), p = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      n[i] = this.ni * Math.exp(clip(u[3 * i] + this.thn[i] - u[3 * i + 1]));
      p[i] = this.ni * Math.exp(clip(u[3 * i + 2] - u[3 * i] + this.thp[i]));
    }
    return {n, p};
  }

  // radiative, Auger and SRH rates at node i (cm^-3 s^-1)
  recombAt(i, n, p, fn, fp) {
    const npm = this.nie2[i] * Math.expm1(clip(fp - fn));   // n p - ni^2
    return [this.B * npm,
            (this.Cn * n + this.Cp * p) * npm,
            npm / (this.tau_p * (n + this.n1[i]) + this.tau_n * (p + this.p1[i]))];
  }

  // particle fluxes Gn = Jn/q, Gp = Jp/q on the N-1 faces
  fluxes(u, n, p) {
    const N = this.N, Gn = new Float64Array(N - 1), Gp = new Float64Array(N - 1);
    const kn = this.mun * this.Vt, kp = this.mup * this.Vt;
    for (let i = 0; i < N - 1; i++) {
      const dpsi = u[3 * i + 3] - u[3 * i], dfn = u[3 * i + 4] - u[3 * i + 1], dfp = u[3 * i + 5] - u[3 * i + 2];
      const dpn = dpsi + this.thn[i + 1] - this.thn[i], dpp = dpsi - this.thp[i + 1] + this.thp[i];
      Gn[i] = -kn * n[i] * expm1OverX(dpn - dfn) * dfn / this.h[i];
      Gp[i] = -kp * p[i] * expm1OverX(dfp - dpp) * dfp / this.h[i];
    }
    return {Gn, Gp};
  }

  residual(u, Vs) {
    const N = this.N, F = new Float64Array(3 * N);
    const {n, p} = this.carriers(u);
    const {Gn, Gp} = this.fluxes(u, n, p);
    const s = this.t0 / this.Cref, eV = this.eps_q * this.Vt;
    for (let i = 1; i < N - 1; i++) {
      const D1 = eV * (u[3 * i + 3] - u[3 * i]) / this.h[i];
      const D0 = eV * (u[3 * i] - u[3 * i - 3]) / this.h[i - 1];
      const [Rr, Ra, Rs] = this.recombAt(i, n[i], p[i], u[3 * i + 1], u[3 * i + 2]);
      const R = Rr + Ra + Rs, w = this.w[i];
      F[3 * i] = ((D1 - D0) / w + p[i] - n[i] + this.C[i]) / this.Cref;
      F[3 * i + 1] = ((Gn[i] - Gn[i - 1]) / w - R) * s;
      F[3 * i + 2] = ((Gp[i] - Gp[i - 1]) / w + R) * s;
    }
    // ohmic contacts: p side (anode) at +V, n side (cathode) at 0
    const L = 3 * (N - 1);
    F[0] = u[0] - (this.psiL0 + Vs); F[1] = u[1] - Vs; F[2] = u[2] - Vs;
    F[L] = u[L] - this.psiR0; F[L + 1] = u[L + 1]; F[L + 2] = u[L + 2];
    return F;
  }

  // ------------------------------------------------------------------- solver
  // Newton step: coloured FD Jacobian in band storage, row scaling, banded elimination
  // with partial pivoting.  Unknowns are interleaved, so the bandwidth is 5 on each side.
  newtonStep(u, Vs, F0) {
    const N = this.N, M = 3 * N, kl = 5, ku = 5, W = 2 * kl + ku + 1, eps = 1e-6;
    const A = new Float64Array(M * W);           // row i, column j  ->  A[i*W + j - i + kl]
    const du = new Float64Array(M);
    for (let c = 0; c < 3; c++) {
      for (let k = 0; k < 3; k++) {
        const up = Float64Array.from(u);
        for (let j = c; j < N; j += 3) up[3 * j + k] += eps;
        const F1 = this.residual(up, Vs);
        for (let j = c; j < N; j += 3) {
          const col = 3 * j + k;
          for (let i = Math.max(j - 1, 0); i <= Math.min(j + 1, N - 1); i++)
            for (let r = 0; r < 3; r++) {
              const row = 3 * i + r;
              A[row * W + col - row + kl] = (F1[row] - F0[row]) / eps;
            }
        }
      }
    }
    const b = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      let m = 0;
      for (let t = 0; t < W; t++) m = Math.max(m, Math.abs(A[i * W + t]));
      if (m === 0) m = 1;
      for (let t = 0; t < W; t++) A[i * W + t] /= m;
      b[i] = -F0[i] / m;
    }
    const at = (i, j) => i * W + j - i + kl;
    for (let k = 0; k < M; k++) {
      const last = Math.min(k + kl, M - 1), jmax = Math.min(k + ku + kl, M - 1);
      let piv = k;
      for (let i = k + 1; i <= last; i++) if (Math.abs(A[at(i, k)]) > Math.abs(A[at(piv, k)])) piv = i;
      if (piv !== k) {
        for (let j = k; j <= jmax; j++) {
          const t = A[at(k, j)]; A[at(k, j)] = A[at(piv, j)]; A[at(piv, j)] = t;
        }
        const t = b[k]; b[k] = b[piv]; b[piv] = t;
      }
      const akk = A[at(k, k)];
      for (let i = k + 1; i <= last; i++) {
        const l = A[at(i, k)] / akk;
        if (l === 0) continue;
        for (let j = k; j <= jmax; j++) A[at(i, j)] -= l * A[at(k, j)];
        b[i] -= l * b[k];
      }
    }
    for (let i = M - 1; i >= 0; i--) {
      let s = b[i];
      const jmax = Math.min(i + ku + kl, M - 1);
      for (let j = i + 1; j <= jmax; j++) s -= A[at(i, j)] * du[j];
      du[i] = s / A[at(i, i)];
    }
    return du;
  }

  solve(u, V, {tol = 1e-9, maxit = 80, dmax = 2.0} = {}) {
    const Vs = V / this.Vt;
    u = Float64Array.from(u);
    for (let it = 0; it < maxit; it++) {
      const du = this.newtonStep(u, Vs, this.residual(u, Vs));
      let m = 0;
      for (let i = 0; i < du.length; i++) m = Math.max(m, Math.abs(du[i]));
      if (!Number.isFinite(m)) return {u, ok: false, it};
      const lam = Math.min(1, dmax / m);
      for (let i = 0; i < du.length; i++) u[i] += lam * du[i];
      if (m < tol) return {u, ok: true, it: it + 1};
    }
    return {u, ok: false, it: maxit};
  }

  initialGuess() {
    const u = new Float64Array(3 * this.N);
    for (let i = 0; i < this.N; i++) u[3 * i] = this.psiEq(i);
    return u;
  }

  // ----------------------------------------------------------------- analysis
  analyse(u, V) {
    const N = this.N, {n, p} = this.carriers(u), {Gn, Gp} = this.fluxes(u, n, p);
    const Jint = Array.from(Gn, (g, i) => q * (g + Gp[i])).sort((a, b) => a - b);
    const J = Jint[Math.floor((N - 1) / 2)];
    const Rrad = new Float64Array(N), Raug = new Float64Array(N), Rsrh = new Float64Array(N);
    let lo = -1, hi = -1;
    for (let i = 0; i < N; i++) {
      [Rrad[i], Raug[i], Rsrh[i]] = this.recombAt(i, n[i], p[i], u[3 * i + 1], u[3 * i + 2]);
      if (Math.abs(p[i] - n[i] + this.C[i]) > 0.1 * Math.abs(this.C[i])) {
        if (lo < 0) lo = i;
        hi = i;
      }
    }
    if (lo < 0) lo = hi = 0;
    const out = {V, J, lo, hi, W: this.x[hi] - this.x[lo]};
    const regions = [["p", 0, lo], ["scl", lo, hi + 1], ["n", hi + 1, N]];
    if (this.dh) regions.push(["act", this.iL, this.iR + 1]);   // active layer, interface nodes included
    for (const [name, R] of [["rad", Rrad], ["aug", Raug], ["srh", Rsrh]])
      for (const [reg, a, b] of regions) {
        let s = 0;
        for (let i = a; i < b; i++) s += R[i] * this.w[i];
        out[`${name}_${reg}`] = q * s;
      }
    const psi = new Float64Array(N), fn = new Float64Array(N), fp = new Float64Array(N);
    for (let i = 0; i < N; i++) { psi[i] = u[3 * i]; fn[i] = u[3 * i + 1]; fp[i] = u[3 * i + 2]; }
    out.profile = {x: this.x, psi, fn, fp, n, p, Rrad, Raug, Rsrh, lo, hi, thn: this.thn, thp: this.thp};
    return out;
  }
}
