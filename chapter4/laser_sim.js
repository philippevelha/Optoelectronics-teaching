// laser_sim.js -- browser ports of the laserlib.py solvers used by the live panels of the
// Chapter 4 companion. SI units, photon energies in eV. Same models and parameters as laserlib.py.

export const c0 = 299792458, h = 6.62607015e-34, qe = 1.602176634e-19, kB = 1.380649e-23, me = 9.1093837015e-31;
export const kT = (T = 300) => kB * T / qe;
export const eV = (wl) => h * c0 / wl / qe;

// ---------------------------------------------------------------------------------------------
// Er3+ cross sections (model of laserlib.er_cross_sections)
// ---------------------------------------------------------------------------------------------
const ER_EM = [[1530.5, 4.2, 3.30], [1535.5, 7.0, 1.20], [1548.0, 11.0, 1.70], [1560.0, 14.0, 1.30],
               [1585.0, 22.0, 0.45], [1508.0, 12.0, 0.45], [1485.0, 16.0, 0.30]];
const erRaw = (w) => ER_EM.reduce((s, [c, wd, a]) => s + a * Math.exp(-0.5 * ((w - c) / wd) ** 2), 0);
const ER_SCALE = 3.2 / erRaw(1550);
const ER_EPS = eV(1550e-9) - kT(295) * Math.log(2.4 / 3.2);

export function erCross(wl) {            // wl in m -> [sigma_ab, sigma_em] in m^2
  const w = wl * 1e9;
  if (w > 1400 && w < 1700) {
    const em = ER_SCALE * erRaw(w) * 1e-25;
    return [em * Math.exp((eV(wl) - ER_EPS) / kT(295)), em];
  }
  return [2.2e-25 * Math.exp(-0.5 * ((w - 978) / 7) ** 2), 0];
}

// Two-level EDFA, forward pumping, RK4 along z (laserlib.edfa_propagate)
export function edfaPropagate({L, Pp0, Ps0, wls = 1550e-9, wlp = 980e-9, N0 = 1e25, tau = 10e-3, core = 5e-6,
                               Gp = 0.7, Gs = 0.7, nz = 400}) {
  const A = Math.PI * core * core / 4, [abs_, ems] = erCross(wls), [abp, emp] = erCross(wlp);
  const hvs = h * c0 / wls, hvp = h * c0 / wlp, dz = L / nz;
  const inv = (Pp, Ps) => { const fp = Gp * Pp / (hvp * A), fs = Gs * Ps / (hvs * A);
    return (abp * fp + abs_ * fs) / ((abp + emp) * fp + (abs_ + ems) * fs + 1 / tau); };
  const der = (Pp, Ps) => { const x = inv(Pp, Ps);
    return [Gp * (emp * x - abp * (1 - x)) * N0 * Pp, Gs * (ems * x - abs_ * (1 - x)) * N0 * Ps]; };
  let Pp = Pp0, Ps = Ps0; const out = [{z: 0, Pp, Ps, x: inv(Pp, Ps)}];
  for (let i = 0; i < nz; i++) {
    const k1 = der(Pp, Ps), k2 = der(Pp + dz / 2 * k1[0], Ps + dz / 2 * k1[1]);
    const k3 = der(Pp + dz / 2 * k2[0], Ps + dz / 2 * k2[1]), k4 = der(Pp + dz * k3[0], Ps + dz * k3[1]);
    Pp = Math.max(Pp + dz / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 0);
    Ps = Math.max(Ps + dz / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 0);
    out.push({z: (i + 1) * dz, Pp, Ps, x: inv(Pp, Ps)});
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Complex numbers and the coupled-mode grating (laserlib.grating_rt)
// ---------------------------------------------------------------------------------------------
const C = (re, im = 0) => ({re, im});
const add = (a, b) => C(a.re + b.re, a.im + b.im), sub = (a, b) => C(a.re - b.re, a.im - b.im);
const mul = (a, b) => C(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
const div = (a, b) => { const d = b.re * b.re + b.im * b.im; return C((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d); };
const csqrt = (a) => { const r = Math.hypot(a.re, a.im), re = Math.sqrt((r + a.re) / 2), im = Math.sign(a.im || 1) * Math.sqrt(Math.max(r - a.re, 0) / 2); return C(re, im); };
const cexp = (a) => { const e = Math.exp(a.re); return C(e * Math.cos(a.im), e * Math.sin(a.im)); };
const ccosh = (a) => { const p = cexp(a), m = cexp(C(-a.re, -a.im)); return C((p.re + m.re) / 2, (p.im + m.im) / 2); };
const csinh = (a) => { const p = cexp(a), m = cexp(C(-a.re, -a.im)); return C((p.re - m.re) / 2, (p.im - m.im) / 2); };
export const cabs2 = (a) => a.re * a.re + a.im * a.im;
const J = C(0, 1);
const mm = (A, B) => [[add(mul(A[0][0], B[0][0]), mul(A[0][1], B[1][0])), add(mul(A[0][0], B[0][1]), mul(A[0][1], B[1][1]))],
                      [add(mul(A[1][0], B[0][0]), mul(A[1][1], B[1][0])), add(mul(A[1][0], B[0][1]), mul(A[1][1], B[1][1]))]];

export function gratingRT(wl, period, neff, dn, L, g = 0, shift = 0) {
  const delta = C(2 * Math.PI * neff / wl - Math.PI / period, g / 2), kappa = C(Math.PI * dn / wl);
  const section = (Ls) => {
    const s = csqrt(sub(mul(kappa, kappa), mul(delta, delta))), sL = mul(s, C(Ls));
    const ch = ccosh(sL), sh = csinh(sL), ds = div(mul(J, delta), s), ks = div(mul(J, kappa), s);
    return [[sub(ch, mul(ds, sh)), mul(C(-1), mul(ks, sh))], [mul(ks, sh), add(ch, mul(ds, sh))]];
  };
  let M;
  if (shift) { const P = [[cexp(C(0, -shift)), C(0)], [C(0), cexp(C(0, shift))]]; M = mm(mm(section(L / 2), P), section(L / 2)); }
  else M = section(L);
  return {r: div(M[0][1], M[1][1]), t: div(C(1), M[1][1])};
}

// Quarter-wave stack reflectance spectrum (characteristic matrices), normal incidence
export function dbrSpectrum(wl, N, nH, nL, nIn, nSub, wl0) {
  let M = [[C(1), C(0)], [C(0), C(1)]];
  for (let i = 0; i < 2 * N; i++) {
    const n = i % 2 === 0 ? nH : nL, d = 2 * Math.PI * (wl0 / 4) / wl;  // n*thickness = wl0/4
    M = mm(M, [[C(Math.cos(d)), C(0, Math.sin(d) / n)], [C(0, n * Math.sin(d)), C(Math.cos(d))]]);
  }
  const B = add(M[0][0], mul(M[0][1], C(nSub))), Cc = add(M[1][0], mul(M[1][1], C(nSub)));
  return cabs2(div(sub(mul(C(nIn), B), Cc), add(mul(C(nIn), B), Cc)));
}

// ---------------------------------------------------------------------------------------------
// ODE helper (fixed-step RK4) and laser rate equations
// ---------------------------------------------------------------------------------------------
export function rk4(f, y0, t0, t1, n, every = 1) {
  const dt = (t1 - t0) / n, out = [{t: t0, y: y0.slice()}]; let y = y0.slice();
  for (let i = 0; i < n; i++) {
    const t = t0 + i * dt, k1 = f(t, y), k2 = f(t + dt / 2, y.map((v, j) => v + dt / 2 * k1[j]));
    const k3 = f(t + dt / 2, y.map((v, j) => v + dt / 2 * k2[j])), k4 = f(t + dt, y.map((v, j) => v + dt * k3[j]));
    y = y.map((v, j) => Math.max(v + dt / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]), 0));
    if ((i + 1) % every === 0) out.push({t: t + dt, y: y.slice()});
  }
  return out;
}

// Laser-diode rate equations of slide 124 with spontaneous coupling beta; current waveform I(t)
export function ldSimulate({I, Ith, tau, tauph, V, tEnd, n = 40000, every = 20, beta = 1e-4, y0}) {
  const nth = Ith * tau / (qe * V), Cg = 1 / (nth * tauph);
  const f = (t, [N, P]) => [I(t) / (qe * V) - N / tau - Cg * N * P, Cg * N * P - P / tauph + beta * N / tau];
  return rk4(f, y0 || [0, 0], 0, tEnd, n, every).map(({t, y}) => ({t, n: y[0], Nph: y[1]}));
}

// ---------------------------------------------------------------------------------------------
// Semiconductor free-carrier gain (laserlib.gain_spectrum_bulk, no broadening)
// ---------------------------------------------------------------------------------------------
export function fermiHalf(eta) {
  const N = 800, xmax = 60 + Math.max(0, eta); let s = 0;
  for (let i = 1; i <= N; i++) { const x = (i - 0.5) * xmax / N; s += Math.sqrt(x) / (1 + Math.exp(Math.min(x - eta, 700))); }
  return 2 / Math.sqrt(Math.PI) * s * xmax / N;
}
export function quasiFermi(n, m, T = 300) {     // eV from band edge, into the band
  const Nc = 2 * (2 * Math.PI * m * me * kB * T / h ** 2) ** 1.5;
  let a = -30, b = 200;
  for (let i = 0; i < 70; i++) { const c = (a + b) / 2; if (Nc * fermiHalf(c) < n) a = c; else b = c; }
  return (a + b) / 2 * kT(T);
}
export function gainBulk(E, {Eg, mE, mH, Fn, Fp, T = 300, alpha0 = 1.2e6}) {
  const x = Math.max(E - Eg, 0), mr = mE * mH / (mE + mH), e2 = x * mr / mE, e1 = x * mr / mH;
  const fc = 1 / (1 + Math.exp((e2 - Fn) / kT(T))), holes = 1 / (1 + Math.exp((e1 - Fp) / kT(T)));
  return alpha0 * Math.sqrt(x / 0.1) * (fc + holes - 1);
}

// ---------------------------------------------------------------------------------------------
// Radix-2 FFT (in place, arrays of re and im), used by the holography panel
// ---------------------------------------------------------------------------------------------
export function fft(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = 2 * Math.PI / len * (inverse ? 1 : -1), wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

// Angular-spectrum propagation of a 1D field (re, im arrays, sample spacing dx) over distance z
export function propagate1D(re, im, dx, wl, z) {
  const n = re.length, R = Float64Array.from(re), I = Float64Array.from(im);
  fft(R, I);
  for (let k = 0; k < n; k++) {
    const fx = (k < n / 2 ? k : k - n) / (n * dx), arg = 1 / (wl * wl) - fx * fx;
    if (arg <= 0) { R[k] = 0; I[k] = 0; continue; }
    const ph = 2 * Math.PI * z * Math.sqrt(arg), c = Math.cos(ph), s = Math.sin(ph);
    const r = R[k] * c - I[k] * s; I[k] = R[k] * s + I[k] * c; R[k] = r;
  }
  fft(R, I, true);
  return [R, I];
}
