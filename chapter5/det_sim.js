// det_sim.js -- browser ports of the detlib.py functions used by the live panels of the
// Chapter 5 companion. SI units, photon and band energies in eV. Same models and data as detlib.py.

export const c0 = 299792458, h = 6.62607015e-34, qe = 1.602176634e-19, kB = 1.380649e-23, eps0 = 8.8541878128e-12;
export const kT = (T = 300) => kB * T / qe;
export const eV = (wl) => h * c0 / wl / qe;
export const responsivity = (eta, wl) => eta * qe * wl / (h * c0);
export const qeff = (R, wl) => R * h * c0 / (qe * wl);

// ---------------------------------------------------------------------------------------------
// Absorption coefficients (λ in µm, log10 α in m^-1), same tables as detlib.MATERIALS
// ---------------------------------------------------------------------------------------------
export const MATERIALS = {
  "Si": {Eg: 1.12, n: 3.6, table: [[0.30, 8.26], [0.35, 8.02], [0.40, 6.98], [0.45, 6.41], [0.50, 6.05], [0.55, 5.81], [0.60, 5.62],
    [0.65, 5.45], [0.70, 5.28], [0.75, 5.11], [0.80, 4.93], [0.85, 4.73], [0.90, 4.49], [0.95, 4.20], [1.00, 3.81], [1.05, 3.21],
    [1.10, 2.54], [1.15, 1.43], [1.20, 0.34]]},
  "Ge": {Eg: 0.66, n: 4.0, table: [[0.35, 8.00], [0.40, 7.73], [0.50, 7.60], [0.55, 7.44], [0.60, 7.19], [0.70, 6.73], [0.80, 6.48],
    [0.90, 6.29], [1.00, 6.15], [1.20, 6.05], [1.30, 5.98], [1.40, 5.88], [1.50, 5.70], [1.52, 5.45], [1.54, 4.70], [1.56, 4.30],
    [1.60, 4.05], [1.70, 3.62], [1.80, 3.20], [1.87, 2.0]]},
  "GaAs": {Eg: 1.42, n: 3.6, table: [[0.40, 7.10], [0.50, 6.90], [0.57, 6.73], [0.60, 6.57], [0.70, 6.07], [0.80, 5.78], [0.85, 5.57],
    [0.87, 5.28], [0.875, 4.70], [0.88, 3.79], [0.89, 3.12], [0.90, 2.0]]},
  "InP": {Eg: 1.35, n: 3.2, table: [[0.40, 7.0], [0.50, 6.70], [0.60, 6.45], [0.70, 6.30], [0.80, 6.15], [0.90, 6.00], [0.91, 5.53],
    [0.92, 5.11], [0.94, 4.29], [0.96, 3.04], [0.97, 2.0]]},
  "In0.7Ga0.3As0.64P0.36": {Eg: 0.89, n: 3.4, table: [[0.80, 6.80], [1.00, 6.60], [1.145, 6.44], [1.25, 6.03], [1.30, 5.94],
    [1.35, 5.61], [1.38, 5.30], [1.39, 4.50], [1.40, 3.80], [1.41, 3.25], [1.42, 2.0]]},
  "In0.53Ga0.47As": {Eg: 0.75, n: 3.6, table: [[0.80, 6.50], [1.00, 6.30], [1.20, 6.15], [1.30, 6.08], [1.40, 6.00], [1.50, 5.90],
    [1.55, 5.85], [1.60, 5.72], [1.63, 5.50], [1.65, 5.10], [1.67, 4.30], [1.69, 3.20], [1.72, 2.0]]}
};

export function alpha(mat, wl) {          // m^-1 at wl (m); 0 beyond the table
  const t = MATERIALS[mat].table, x = wl * 1e6;
  if (x <= t[0][0]) return 10 ** t[0][1];
  if (x >= t[t.length - 1][0]) return 0;
  let i = 1; while (t[i][0] < x) i++;
  const [x0, y0] = t[i - 1], [x1, y1] = t[i];
  return 10 ** (y0 + (y1 - y0) * (x - x0) / (x1 - x0));
}

export const fresnelT = (n, n0 = 1) => 1 - ((n - n0) / (n + n0)) ** 2;

// I_ph/P_o (A/W) of a p+n / pin lit from the p side (slide 144)
export function structureR(wl, a, lp, W, Le = 0, Lh = 0, T = 1, etai = 1) {
  const start = Math.max(lp - Le, 0);
  return responsivity(etai * T, wl) * (Math.exp(-a * start) - Math.exp(-a * (lp + W + Lh)));
}

// ---------------------------------------------------------------------------------------------
// Transport: Caughey–Thomas drift velocity (Si, 300 K)
// ---------------------------------------------------------------------------------------------
const DRIFT = {e: {mu: 0.135, vsat: 1.07e5, beta: 1.11}, h: {mu: 0.048, vsat: 8.37e4, beta: 1.21}};
export function driftVelocity(E, carrier = "e") {
  const p = DRIFT[carrier], v0 = p.mu * Math.abs(E);
  return v0 / (1 + (v0 / p.vsat) ** p.beta) ** (1 / p.beta);
}

// ---------------------------------------------------------------------------------------------
// Shockley–Ramo: pin impulse and frequency response, generation exp(−αx) over the i-layer
// ---------------------------------------------------------------------------------------------
function profile(W, a, nx) {
  const x = [], g = []; let s = 0;
  for (let i = 0; i < nx; i++) { const xi = (i + 0.5) * W / nx, gi = a > 0 ? Math.exp(-a * xi) : 1; x.push(xi); g.push(gi); s += gi; }
  return {x, g: g.map(v => v / s)};
}

export function pinImpulse(ts, W, a, ve, vh, nx = 300) {     // current per EHP (A), arrays ie, ih
  const {x, g} = profile(W, a, nx);
  return ts.map(t => {
    let ie = 0, ih = 0;
    for (let i = 0; i < nx; i++) { if (t >= 0 && t < (W - x[i]) / ve) ie += g[i]; if (t >= 0 && t < x[i] / vh) ih += g[i]; }
    return {t, ie: ie * qe * ve / W, ih: ih * qe * vh / W};
  });
}

export function transitMag(f, W, a, ve, vh, nx = 200) {        // |H(f)| normalised
  const {x, g} = profile(W, a, nx), w = 2 * Math.PI * Math.max(f, 1e-6);
  let re = 0, im = 0, q = 0;
  for (let i = 0; i < nx; i++) {
    const Te = (W - x[i]) / ve, Th = x[i] / vh;
    // (v/W)(1 − e^{−jωT})/(jω) = (v/W)[sin ωT − j(1 − cos ωT)]/ω
    re += g[i] * (ve / W * Math.sin(w * Te) + vh / W * Math.sin(w * Th)) / w;
    im -= g[i] * (ve / W * (1 - Math.cos(w * Te)) + vh / W * (1 - Math.cos(w * Th))) / w;
    q += g[i] * (ve * Te + vh * Th) / W;
  }
  return Math.hypot(re, im) / q;
}

export function f3dB(fun, fmax, n = 2000) {           // first f where fun(f) < 1/√2
  for (let i = 1; i <= n; i++) { const f = fmax * i / n; if (fun(f) < Math.SQRT1_2) return f; }
  return NaN;
}

// ---------------------------------------------------------------------------------------------
// Avalanche multiplication
// ---------------------------------------------------------------------------------------------
export const IONIZATION = {Si: {e: [0.740e6, 1.16e6], h: [0.725e6, 2.20e6]}, InP: {e: [1.12e7, 3.11e6], h: [4.79e6, 2.55e6]}};
export const chynoweth = (E, mat = "Si", c = "e") => IONIZATION[mat][c][0] * Math.exp(-IONIZATION[mat][c][1] / E); // cm^-1, V/cm
export function apdGainUniform(ae, k, w) {           // valid for any k; Infinity beyond breakdown
  const M = Math.abs(k - 1) < 1e-9 ? 1 / (1 - ae * w) : (1 - k) / (Math.exp(-(1 - k) * ae * w) - k);
  return M >= 1 ? M : Infinity;
}
export function apdGainProfile(xs, ae, ah) {           // electrons injected at xs[0]
  let d = 0, I = 0;
  for (let i = 1; i < xs.length; i++) {
    const dx = xs[i] - xs[i - 1], f0 = ae[i - 1] * Math.exp(-d);
    d += 0.5 * (ae[i] - ah[i] + ae[i - 1] - ah[i - 1]) * dx;
    I += 0.5 * (f0 + ae[i] * Math.exp(-d)) * dx;
  }
  return I < 1 ? 1 / (1 - I) : Infinity;
}
export const mcintyre = (M, k) => k * M + (1 - k) * (2 - 1 / M);

// ---------------------------------------------------------------------------------------------
// Special functions
// ---------------------------------------------------------------------------------------------
export function erfc(x) {                     // Numerical Recipes erfcc, |error| < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
export const berQ = (Q) => 0.5 * erfc(Q / Math.SQRT2);

export function lambertWexp(L) {              // W(e^L) by Newton on ln w + w = L, no overflow
  let w = L < 1 ? Math.exp(L) : L - Math.log(L);
  if (w <= 0) w = 1e-300;
  for (let i = 0; i < 60; i++) {
    const dw = (Math.log(w) + w - L) / (1 / w + 1);
    w = Math.max(w - dw, w / 10);
    if (Math.abs(dw) < 1e-13 * (1 + w)) break;
  }
  return w;
}

// ---------------------------------------------------------------------------------------------
// Solar cell, one-diode model (slide 128): I = Io[exp((V − IRs)/nkT) − 1] + (V − IRs)/Rp − Iph
// ---------------------------------------------------------------------------------------------
export function solarI(V, Iph, Io, n = 1, Rs = 0, Rp = Infinity, T = 300) {
  const Vt = n * kT(T), G = isFinite(Rp) ? 1 / Rp : 0;
  if (Rs <= 0) return Io * Math.expm1(V / Vt) + G * V - Iph;
  const a = 1 + Rs * G, rhs = V + Rs * (Iph + Io);
  const Vd = rhs / a - Vt * lambertWexp(Math.log(Rs * Io / (a * Vt)) + rhs / (a * Vt));
  return (V - Vd) / Rs;
}
export function bisect(f, a, b, n = 100) {
  let fa = f(a);
  for (let i = 0; i < n; i++) { const m = 0.5 * (a + b), fm = f(m); if ((fm > 0) === (fa > 0)) { a = m; fa = fm; } else b = m; }
  return 0.5 * (a + b);
}
export function solarSummary(Iph, Io, n = 1, Rs = 0, Rp = Infinity, T = 300) {
  const Voc = bisect(V => solarI(V, Iph, Io, n, Rs, Rp, T), 0, 5 * n * kT(T) * Math.log(Iph / Io + 1) + 1e-3);
  const Isc = -solarI(0, Iph, Io, n, Rs, Rp, T);
  let Pm = 0, Vm = 0;
  for (let i = 0; i <= 2000; i++) { const V = Voc * i / 2000, P = -V * solarI(V, Iph, Io, n, Rs, Rp, T); if (P > Pm) { Pm = P; Vm = V; } }
  return {Voc, Isc, Vm, Im: Pm / Vm, Pm, FF: Pm / (Voc * Isc)};
}

// Detailed balance (Shockley–Queisser) under black-body sunlight
const fluxBB = (E, T) => 2 * Math.PI / (h ** 3 * c0 ** 2) * (E * qe) ** 2 / Math.expm1(E / kT(T)) * qe;   // m^-2 s^-1 eV^-1
function integ(fun, a, b, n = 800) { let s = 0; const d = (b - a) / n; for (let i = 0; i < n; i++) s += fun(a + (i + 0.5) * d); return s * d; }
export function shockleyQueisser(Eg, Tsun = 5778, Tcell = 300, X = 1) {
  const f = (6.957e8 / 1.496e11) ** 2 * X, sigma = 5.670374419e-8;
  const Jsc = qe * f * integ(E => fluxBB(E, Tsun), Eg, Eg + 40 * kT(Tsun));
  const J0 = qe * integ(E => fluxBB(E, Tcell), Eg, Eg + 40 * kT(Tcell));
  const Pin = f * sigma * Tsun ** 4, Vt = kT(Tcell), Voc = Vt * Math.log(Jsc / J0 + 1);
  let Pm = 0, Vm = 0;
  for (let i = 0; i <= 1000; i++) { const V = Voc * i / 1000, P = V * (Jsc - J0 * Math.expm1(V / Vt)); if (P > Pm) { Pm = P; Vm = V; } }
  return {Jsc, J0, Voc, Vm, Pin, eff: Pm / Pin, FF: Pm / (Voc * Jsc)};
}

// ---------------------------------------------------------------------------------------------
// Random numbers (seeded) for noise and image simulations
// ---------------------------------------------------------------------------------------------
export function rng(seed = 1) {
  let s = seed >>> 0;
  const u = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = () => { let a = 0; while (a === 0) a = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * u()); };
  const poisson = (lam) => {
    if (lam < 30) { const L = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= u(); } while (p > L); return k - 1; }
    return Math.max(0, Math.round(lam + Math.sqrt(lam) * gauss()));
  };
  return {u, gauss, poisson};
}
