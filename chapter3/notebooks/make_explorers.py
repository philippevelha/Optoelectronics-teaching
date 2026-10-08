"""Builds interactive_explorers.ipynb (ipywidgets versions of the website panels).
Run:  python make_explorers.py"""
import nbformat as nbf

from build_notebooks import setup_code, colab_badge

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

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
nbf.write(nb, "interactive_explorers.ipynb")
print("written interactive_explorers.ipynb")
