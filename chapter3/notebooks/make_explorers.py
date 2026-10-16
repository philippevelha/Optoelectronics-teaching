"""Builds interactive_explorers.ipynb (ipywidgets versions of the website panels).
Run:  python make_explorers.py"""
import nbformat as nbf

from build_notebooks import setup_code, colab_badge, USER, REPO, SUB

SIM_RAW = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/{SUB}/pn_led_sim.py"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Chapter 3: interactive explorers (ipywidgets)

Python versions of the live panels of the companion website. Run all cells, then move the sliders.
Each explorer is self-contained and uses `semilib.py` (SI units, energies in eV).""")
md(colab_badge("interactive_explorers"))

code(setup_code() + """import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
import semilib as sl
sl.use_style()
CM3, CM2 = sl.CM3, sl.CM2
%matplotlib inline""")

md("## 1. Doping, carriers and Fermi level")
code("""@interact(mat=Dropdown(options=['Si', 'Ge', 'GaAs', 'InP']), lNd=FloatSlider(value=16, min=0, max=19, step=0.1, description='log Nd'),
          lNa=FloatSlider(value=0, min=0, max=19, step=0.1, description='log Na'), T=IntSlider(value=300, min=150, max=700, step=10))
def doping(mat, lNd, lNa, T):
    p = sl.MATERIALS[mat]
    ni = sl.n_i(mat, T)
    Nd, Na = (10**lNd*CM3 if lNd > 0 else 0), (10**lNa*CM3 if lNa > 0 else 0)
    n, h = sl.carriers(Nd, Na, ni)
    Nc, Nv = p['Nc']*(T/300)**1.5, p['Nv']*(T/300)**1.5
    EFi = p['Eg']/2 - 0.5*sl.kT(T)*np.log(Nc/Nv); EF = EFi + sl.fermi_level(n, ni, T)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axhspan(p['Eg'], p['Eg'] + 0.3, color=sl.PALETTE[0], alpha=0.15); ax.axhspan(-0.3, 0, color=sl.PALETTE[1], alpha=0.15)
    for y, c, ls, lab in ((p['Eg'], sl.INK, '-', 'E_c'), (0, sl.INK, '-', 'E_v'), (EFi, sl.INK2, ':', 'E_Fi'), (EF, sl.PALETTE[6], '--', 'E_F')):
        ax.axhline(y, color=c, ls=ls); ax.text(1.01, y, lab, transform=ax.get_yaxis_transform(), va='center')
    ax.set(xticks=[], ylabel='E − E_v (eV)', ylim=(-0.3, p['Eg'] + 0.3),
           title=f'n = {n/CM3:.2e}, p = {h/CM3:.2e} cm⁻³, σ = {sl.conductivity(n, h, p["mu_e"], p["mu_h"])/100:.2e} S/cm')
    plt.show()""")

md("## 2. pn junction: charge, field and bands")
code("""@interact(mat=Dropdown(options=['Si', 'GaAs', 'Ge']), lNa=FloatSlider(value=17, min=14, max=19, step=0.1, description='log Na'),
          lNd=FloatSlider(value=16, min=14, max=19, step=0.1, description='log Nd'), V=FloatSlider(value=0, min=-10, max=1, step=0.05))
def junction(mat, lNa, lNd, V):
    p = sl.MATERIALS[mat]; Na, Nd = 10**lNa*CM3, 10**lNd*CM3
    Vo = sl.built_in_voltage(Na, Nd, p['ni']); V = min(V, Vo - 0.05)
    x, rho, E, Vx = sl.junction_profile(Na, Nd, p['eps_r'], Vo, V)
    d = sl.depletion(Na, Nd, p['eps_r'], Vo, V)
    EcP = p['Eg']/2 + sl.kT()*np.log(Na/p['ni'])
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].plot(x*1e6, rho/sl.qe/CM3); ax[0].set(xlabel='x (µm)', ylabel='ρ/e (cm⁻³)')
    ax[1].plot(x*1e6, E/1e5, color=sl.PALETTE[1]); ax[1].set(xlabel='x (µm)', ylabel='E (kV/cm)')
    ax[2].plot(x*1e6, EcP - Vx, label='E_c'); ax[2].plot(x*1e6, EcP - p['Eg'] - Vx, label='E_v')
    ax[2].set(xlabel='x (µm)', ylabel='energy (eV)'); ax[2].legend(fontsize=8)
    fig.suptitle(f'Vo = {Vo:.3f} V, W = {d["W"]*1e6:.3f} µm, |Eo| = {d["Eo"]/1e5:.1f} kV/cm'); plt.tight_layout(); plt.show()""")

md("## 3. Diode I–V: diffusion and recombination")
code("""@interact(mat=Dropdown(options=['GaAs', 'Si', 'Ge'], value='GaAs'), lN=FloatSlider(value=17, min=15, max=19, step=0.1, description='log Na=Nd'),
          ltau=FloatSlider(value=-7.3, min=-10, max=-4, step=0.1, description='log τ (s)'))
def iv(mat, lN, ltau):
    N, tau = 10**lN*CM3, 10**ltau
    Vo = sl.built_in_voltage(N, N, sl.MATERIALS[mat]['ni'])
    V = np.linspace(0.02, Vo - 0.1, 400)
    d = sl.diode_currents(V, 1e-6, N, N, mat, tau_e=tau, tau_h=tau)
    plt.figure(figsize=(7, 3.6))
    plt.semilogy(V, d['I'], color=sl.INK, label='total'); plt.semilogy(V, d['I_diff'], '--', label='diffusion (η = 1)')
    plt.semilogy(V, d['I_rec'], '--', label='recombination (η = 2)')
    plt.ylim(1e-13, 1); plt.xlabel('V (V)'); plt.ylabel('I (A), A = 1 mm²'); plt.legend(fontsize=8); plt.show()""")

md("## 4. LED spectrum versus temperature")
code("""@interact(TC=IntSlider(value=25, min=-60, max=150, description='T (°C)'), sigma=IntSlider(value=0, min=0, max=40, description='σ (meV)'))
def spectrum(TC, sigma):
    E = np.linspace(1.3, 1.7, 3000)
    def S(TC):
        T = TC + 273.15; s = sl.led_spectrum(E, sl.varshni(T, *sl.VARSHNI['GaAs']), T)
        if sigma: g = np.exp(-((E - E.mean())**2)/(2*(sigma/1000)**2)); s = np.convolve(s, g/g.sum(), mode='same')
        return s
    ref = S(25).max(); lam = sl.wavelength_from_eV(E)*1e9
    plt.figure(figsize=(7, 3.4)); plt.plot(lam, S(25)/ref, '--', color=sl.INK2, label='25 °C'); plt.plot(lam, S(TC)/ref, label=f'{TC} °C')
    plt.xlim(800, 940); plt.xlabel('λ (nm)'); plt.ylabel('relative intensity'); plt.legend(); plt.show()""")

md("## 5. Finite quantum well")
code("""@interact(d_nm=FloatSlider(value=12, min=1, max=30, step=0.5, description='d (nm)'), V0=FloatSlider(value=0.3, min=0.02, max=0.6, step=0.01, description='ΔEc (eV)'),
          m=FloatSlider(value=0.067, min=0.02, max=0.5, step=0.001, description='m*/me'))
def well(d_nm, V0, m):
    d = d_nm*1e-9; x = np.linspace(-1.5*d, 1.5*d, 800); lev = sl.qw_finite(d, V0, m)
    plt.figure(figsize=(7, 3.8)); plt.plot(x*1e9, np.where(abs(x) <= d/2, 0, V0), color=sl.INK)
    for n, E in enumerate(lev):
        plt.plot(x*1e9, E + 0.3*V0/max(1, len(lev))*sl.qw_wavefunction(x, E, d, V0, m, parity=n % 2), label=f'E{n+1} = {E*1e3:.1f} meV')
    for E in sl.qw_infinite(np.arange(1, 4), d, m): plt.hlines(E, -d/2*1e9, d/2*1e9, color=sl.PALETTE[7], ls='--', lw=1)
    plt.ylim(-0.02, 1.25*V0); plt.xlabel('x (nm)'); plt.ylabel('E (eV)'); plt.legend(fontsize=8); plt.show()""")

md("## 6. LED efficiency chain")
code("""@interact(lam=IntSlider(value=625, min=380, max=1000, description='λ (nm)'), I=IntSlider(value=350, min=1, max=1000, description='I (mA)'),
          V=FloatSlider(value=2.2, min=1, max=4.5, step=0.01), iqe=FloatSlider(value=0.85, min=0.05, max=1, step=0.01),
          ns=FloatSlider(value=3.3, min=1.5, max=4, step=0.01), na=FloatSlider(value=1.5, min=1, max=1.8, step=0.01))
def chain(lam, I, V, iqe, ns, na):
    hv = 1239.84/lam; ee = sl.extraction_ratio(ns, na, exact=True); Po = iqe*ee*hv*I*1e-3
    phi = sl.luminous_flux(Po, lam*1e-9)
    print(f'extraction {ee*100:.1f} %, EQE {iqe*ee*100:.1f} %, Po = {Po*1e3:.1f} mW, PCE = {Po/(I*V*1e-3)*100:.1f} %')
    print(f'V(λ) = {float(sl.V_photopic(lam*1e-9)):.3g}, Φv = {float(phi):.1f} lm, efficacy = {float(phi)/(I*V*1e-3):.1f} lm/W')""")

md("## 7. Modulation and rise time")
code("""@interact(tau_ns=FloatSlider(value=10, min=0.5, max=50, step=0.5, description='τ (ns)'), B=IntSlider(value=50, min=5, max=500, description='Mb/s'),
          k=FloatSlider(value=0, min=0, max=1.5, step=0.05, description='speed-up k'))
def modulation(tau_ns, B, k):
    bits = [0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,1]; Tb = 1e3/B; tau = tau_ns
    t = np.arange(0, 16*Tb, min(Tb, tau)/200); b = np.array(bits)[np.minimum(15, (t//Tb).astype(int))]
    I = b.astype(float); last, ttr, step = 0, -1e9, 0
    for i, bi in enumerate(b):
        if bi != last: step, ttr, last = bi - last, t[i], bi
        I[i] += k*step*np.exp(-(t[i] - ttr)/tau)
    n = np.zeros_like(t)
    for i in range(1, t.size): n[i] = n[i-1] + (t[i] - t[i-1])*(I[i-1] - n[i-1])/tau
    plt.figure(figsize=(8, 3.2)); plt.step(t, b, ':', where='post', color=sl.INK2); plt.plot(t, n)
    plt.title(f'fc = {1e3/(2*np.pi*tau):.1f} MHz, rise time = {2.2*tau:.1f} ns, bit slot = {Tb:.1f} ns')
    plt.xlabel('t (ns)'); plt.ylabel('light output'); plt.show()""")

md("""## 8. Inside a forward-biased LED: drift–diffusion simulation

The full semiconductor equations of a GaAs pn junction, solved by `pn_led_sim.py` (see the chapter *Inside a forward-biased
LED*). Changing a device parameter re-runs the voltage sweep (1–2 s, when you release the slider). The voltage and window
sliders only re-plot. Set a channel's checkbox off to remove it.""")
code(f"""import sys, functools
if 'google.colab' in sys.modules:
    import urllib.request
    urllib.request.urlretrieve('{SIM_RAW}', 'pn_led_sim.py')
from ipywidgets import Checkbox
from pn_led_sim import PNDiode

DD_MESH = (2e-7, 3e-5, 1.1, 1e-5)    # twice coarser than the script default: same J within 0.1 %
DD_COL = dict(rad='#1f9e6a', srh='#eb6834', aug='#7b4fd0', n='#2a78d6', p='#d63a5a', ref='#8a8986')

def dd_sweep(dev, vmax, step=0.025):
    \"\"\"Equilibrium, then a voltage ramp; returns the list of analysed bias points.\"\"\"
    u, ok, _ = dev.solve(dev.initial_guess(), 0.0, maxit=400)
    out = [dev.analyse(u, 0.0)]
    for V in np.arange(step, vmax + 1e-9, step):
        u, ok, _ = dev.solve(u, V)
        if not ok:
            print(f'Newton did not converge at V = {{V:.3f}} V; sweep stopped there')
            break
        out.append(dev.analyse(u, V))
    return out

def dd_bands(ax, dev, pr, xs):
    ax.plot(xs, -dev.Vt*(pr['psi'] + pr['thn']) + dev.Eg/2, color=sl.INK, label='E_c, E_v')
    ax.plot(xs, -dev.Vt*(pr['psi'] - pr['thp']) - dev.Eg/2, color=sl.INK)
    ax.plot(xs, -dev.Vt*pr['fn'], '--', color=DD_COL['n'], label='E_Fn')
    ax.plot(xs, -dev.Vt*pr['fp'], '--', color=DD_COL['p'], label='E_Fp')
    ax.set(ylabel='energy (eV)'); ax.legend(fontsize=8, ncol=3, loc='best')

@functools.lru_cache(maxsize=8)
def homo_sweep(lNa, lNd, lB, ltn, ltp, Et, lC, rad, srh, aug):
    dev = PNDiode(Na=10**lNa, Nd=10**lNd, Et=Et, mesh=DD_MESH,
                  B=10**lB if rad else 0.0, Cn=10**lC if aug else 0.0, Cp=10**lC if aug else 0.0,
                  tau_n=10**ltn if srh else np.inf, tau_p=10**ltp if srh else np.inf)
    return dev, dd_sweep(dev, 1.3)

S = lambda v, lo, hi, st, d: FloatSlider(value=v, min=lo, max=hi, step=st, description=d, continuous_update=False)
@interact(lNa=S(18.5, 16, 19, 0.1, 'log Na'), lNd=S(17.5, 16, 19, 0.1, 'log Nd'), lB=S(-10, -12, -9, 0.1, 'log B'),
          ltn=S(-8, -12, -5, 0.1, 'log τn (s)'), ltp=S(-8, -12, -5, 0.1, 'log τp (s)'), Et=S(0, -0.6, 0.6, 0.02, 'Et − Ei (eV)'),
          lC=S(-30, -32, -27, 0.1, 'log Cn=Cp'), rad=Checkbox(value=True, description='radiative'),
          srh=Checkbox(value=True, description='SRH'), aug=Checkbox(value=True, description='Auger'),
          V=FloatSlider(value=1.0, min=0.025, max=1.3, step=0.025, description='V (V)'),
          xw=FloatSlider(value=10, min=0.2, max=15, step=0.1, description='x ± (µm)'))
def drift_diffusion(lNa, lNd, lB, ltn, ltp, Et, lC, rad, srh, aug, V, xw):
    dev, res = homo_sweep(lNa, lNd, lB, ltn, ltp, Et, lC, rad, srh, aug)
    k = min(int(round(V/0.025)), len(res) - 1); a = res[k]; pr = a['profile']; xs = pr['x']*1e4
    Vg = np.array([r['V'] for r in res]); J = np.array([r['J'] for r in res])
    col = lambda key: np.array([r[key] for r in res])
    rad_scl, rad_neu = col('rad_scl'), col('rad_p') + col('rad_n')
    srh_scl, srh_neu = col('srh_scl'), col('srh_p') + col('srh_n')
    aug_all = col('aug_p') + col('aug_scl') + col('aug_n')
    contact = J - rad_scl - rad_neu - srh_scl - srh_neu - aug_all
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = 1/(dev.Vt*np.gradient(np.log(np.maximum(J, 1e-300)), Vg))
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    for axx in ax[0, 0], ax[0, 1], ax[1, 0]:
        axx.axvspan(xs[pr['lo']], xs[pr['hi']], color=sl.INK2, alpha=0.12, lw=0); axx.set_xlim(-xw, xw); axx.set_xlabel('x (µm)')
    dd_bands(ax[0, 0], dev, pr, xs); ax[0, 0].set_title(f'bands and quasi-Fermi levels at V = {{a["V"]:.3f}} V (shaded: SCL)')
    ax[0, 1].semilogy(xs, pr['n'], color=DD_COL['n'], label='n'); ax[0, 1].semilogy(xs, pr['p'], color=DD_COL['p'], label='p')
    ax[0, 1].semilogy(xs, np.sqrt(pr['n']*pr['p']), ':', color=DD_COL['ref'], label='√(np)')
    top = max(pr['n'].max(), pr['p'].max())
    ax[0, 1].set(ylabel='concentration (cm⁻³)', title='carriers', ylim=(top/1e14, 3*top)); ax[0, 1].legend(fontsize=8)
    R = dict(rad=pr['Rrad'], srh=pr['Rsrh'], aug=pr['Raug']); tot = sum(R.values())
    floor = max(tot[np.abs(xs) <= xw].max()/1e7, 1e-30)
    for key, on, lab, ls in (('rad', rad, 'radiative B(np − nᵢ²)', '-'), ('srh', srh, 'SRH', '--'), ('aug', aug, 'Auger', ':')):
        if on:
            ax[1, 0].semilogy(xs, np.where(R[key] > floor, R[key], np.nan), ls, color=DD_COL[key], lw=2, label=lab)
    if rad:
        ax[1, 0].fill_between(xs, floor, np.maximum(R['rad'], floor), color=DD_COL['rad'], alpha=0.18, lw=0)
    ax[1, 0].set(ylim=(floor, None), ylabel='R(x) (cm⁻³ s⁻¹)', title='recombination per channel (green area: B·n·p)')
    ax[1, 0].legend(fontsize=8)
    sel = (Vg >= 0.2)
    pos = lambda y: np.where(y > 1e-14, y, np.nan)
    for y, lab, c, ls in ((J, 'total J', sl.INK, '-'), (rad_neu, 'radiative, neutral', DD_COL['rad'], '-'),
                          (rad_scl, 'radiative, SCL', DD_COL['rad'], ':'), (srh_neu, 'SRH, neutral', DD_COL['srh'], '-'),
                          (srh_scl, 'SRH, SCL', DD_COL['srh'], ':'), (aug_all, 'Auger', DD_COL['aug'], '-'),
                          (contact, 'contacts', DD_COL['ref'], '--')):
        ax[1, 1].semilogy(Vg[sel], pos(y)[sel], ls, color=c, lw=2.4 if lab == 'total J' else 1.6, label=lab)
    ax[1, 1].axvline(a['V'], color=sl.INK2, lw=1)
    ax[1, 1].set(xlabel='V (V)', ylabel='current density (A/cm²)', title='where the current recombines'); ax[1, 1].legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.show()
    radt = rad_scl[k] + rad_neu[k]
    i0 = int(np.argmin(np.abs(pr['x'])))
    print(f'V = {{a["V"]:.3f}} V (Vbi = {{dev.Vbi:.3f}} V), J = {{a["J"]:.3e}} A/cm², local η = {{eta[k]:.2f}}')
    print(f'IQE = {{100*radt/a["J"]:.1f}} %, light from the SCL = {{100*rad_scl[k]/radt if radt > 0 else 0:.1f}} % of the radiative total, '
          f'SCL width = {{a["W"]*1e7:.0f}} nm, n·p/nᵢ² at x = 0: {{pr["n"][i0]*pr["p"][i0]/dev.ni**2:.2e}}')""")

md("""## 9. Double heterostructure versus homojunction

A p-GaAs active layer of thickness d between p-Al$_x$Ga$_{1-x}$As ($N_a=3\\times10^{18}$ cm⁻³) and n-Al$_x$Ga$_{1-x}$As
($N_d=3\\times10^{17}$ cm⁻³) claddings, $\\Delta E_g = 1.247x$ eV split 67:33 between $\\Delta E_c$ and $\\Delta E_v$. The reference is
the same structure with $x=0$ (a homojunction). The two LEDs are compared at the same **current density**; the structure
sliders re-run both sweeps (3–4 s), the current and window sliders only re-plot. Run the cell of explorer 8 first.""")
code("""@functools.lru_cache(maxsize=8)
def dh_sweeps(x, d_nm, lNact, ltau):
    out = {}
    for key, xx in (('DH', x), ('ref', 0.0)):
        dEg = 1.247*xx
        dev = PNDiode(Na=3e18, Nd=3e17, tau_n=10**ltau, tau_p=10**ltau, mesh=DD_MESH,
                      dh=dict(d=d_nm*1e-7, dEc=0.67*dEg, dEv=0.33*dEg, Nact=-10**lNact))
        out[key] = (dev, [r for r in dd_sweep(dev, 1.5) if r['J'] > 0])
    return out

@interact(x=S(0.3, 0, 0.45, 0.01, 'Al fraction x'), d_nm=S(100, 10, 1000, 10, 'd (nm)'),
          lNact=S(17, 15, 19, 0.1, 'log Na (layer)'), ltau=S(-7, -9, -6, 0.1, 'log τ (s)'),
          lJ=FloatSlider(value=1.5, min=-2, max=3, step=0.1, description='log J (A/cm²)'),
          xw=FloatSlider(value=2, min=0.1, max=10, step=0.1, description='x ± (µm)'))
def heterostructure(x, d_nm, lNact, ltau, lJ, xw):
    sw = dh_sweeps(x, d_nm, lNact, ltau)
    iqe = lambda r: (r['rad_p'] + r['rad_n'] + r['rad_scl'])/r['J']
    pick = lambda res: min(res, key=lambda r: abs(np.log10(r['J']) - lJ))
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    for key, c, ls, lab in (('DH', DD_COL['rad'], '-', 'heterostructure'), ('ref', DD_COL['ref'], '--', 'reference (x = 0)')):
        res = sw[key][1]
        ax[0, 0].semilogx([r['J'] for r in res], [100*iqe(r) for r in res], ls, color=c, lw=2, label=lab)
    ax[0, 0].axvline(10**lJ, color=sl.INK2, lw=1)
    ax[0, 0].set(xlim=(1e-2, 1e3), ylim=(0, 100), xlabel='J (A/cm²)', ylabel='IQE (%)', title='IQE at equal current')
    ax[0, 0].legend(fontsize=8)
    dev, res = sw['DH']; a = pick(res); pr = a['profile']; xs = pr['x']*1e4
    devr, resr = sw['ref']; ar = pick(resr); prr = ar['profile']; xr = prr['x']*1e4
    for axx in ax[0, 1], ax[1, 0], ax[1, 1]:
        axx.axvspan(-d_nm*5e-4, d_nm*5e-4, color=DD_COL['rad'], alpha=0.12, lw=0); axx.set_xlim(-xw, xw); axx.set_xlabel('x (µm)')
    dd_bands(ax[0, 1], dev, pr, xs); ax[0, 1].set_title(f'DH bands, J = {a["J"]:.2g} A/cm² (shaded: active layer)')
    for key, ls in (('n', '-'), ('p', '-')):
        ax[1, 0].semilogy(xs, pr[key], ls, color=DD_COL[key], lw=2, label=f'{key}, heterostructure')
        ax[1, 0].semilogy(xr, prr[key], ':', color=DD_COL[key], lw=1.5, label=f'{key}, reference')
    top = max(pr['n'].max(), pr['p'].max())
    ax[1, 0].set(ylabel='concentration (cm⁻³)', title='carriers at the same current', ylim=(top/1e14, 3*top))
    ax[1, 0].legend(fontsize=7, ncol=2)
    m = np.abs(xs) <= xw; mr = np.abs(xr) <= xw
    top = max(pr['Rrad'][m].max(), pr['Rsrh'][m].max(), prr['Rrad'][mr].max()); floor = top/1e6
    clip = lambda y: np.where(y > floor, y, np.nan)
    ax[1, 1].fill_between(xs, floor, np.maximum(pr['Rrad'], floor), color=DD_COL['rad'], alpha=0.18, lw=0)
    ax[1, 1].semilogy(xs, clip(pr['Rrad']), color=DD_COL['rad'], lw=2, label='radiative')
    ax[1, 1].semilogy(xs, clip(pr['Rsrh']), '--', color=DD_COL['srh'], lw=2, label='SRH')
    ax[1, 1].semilogy(xs, clip(pr['Raug']), ':', color=DD_COL['aug'], lw=2, label='Auger')
    ax[1, 1].semilogy(xr, clip(prr['Rrad']), color=DD_COL['ref'], lw=2, label='radiative, reference')
    ax[1, 1].set(ylim=(floor, 2*top), ylabel='R(x) (cm⁻³ s⁻¹)', title='recombination at the same current'); ax[1, 1].legend(fontsize=7)
    plt.tight_layout(); plt.show()
    dEc = 0.67*1.247*x
    print(f'ΔEc = {1000*dEc:.0f} meV = {dEc/dev.Vt:.1f} kT, ΔEv = {1000*0.33*1.247*x:.0f} meV')
    for name, r in (('heterostructure', a), ('reference      ', ar)):
        radt = r['rad_p'] + r['rad_n'] + r['rad_scl']; act = r['rad_act'] + r['srh_act'] + r['aug_act']
        print(f'{name}: J = {r["J"]:.3g} A/cm² at V = {r["V"]:.3f} V, IQE = {100*radt/r["J"]:.1f} %, '
              f'light from the layer {100*r["rad_act"]/radt:.1f} %, recombination outside it {100*(1 - act/r["J"]):.1f} %')""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
nbf.write(nb, "interactive_explorers.ipynb")
print("written interactive_explorers.ipynb")
