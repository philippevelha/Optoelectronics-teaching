"""Builds interactive_explorers.ipynb (ipywidgets versions of the website panels).
Run:  python make_explorers.py"""
import nbformat as nbf

from build_notebooks import setup_code, colab_badge

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Chapter 4: interactive explorers (ipywidgets)

Python versions of the live panels of the companion website. Run all cells, then move the sliders.
Each explorer is self-contained and uses `laserlib.py` (SI units, photon energies in eV).""")
md(colab_badge("interactive_explorers"))

code(setup_code() + """import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, FloatLogSlider, IntSlider, Dropdown, Checkbox
import laserlib as lz
lz.use_style()
%matplotlib inline""")

md("## 1. Black body: Planck versus Rayleigh–Jeans")
code("""@interact(T=IntSlider(value=3000, min=200, max=10000, step=50, description='T (K)'))
def blackbody(T):
    wl = np.logspace(-7, -4, 500)
    plt.figure(figsize=(7, 3.6))
    plt.loglog(wl*1e6, lz.planck_emittance(wl, T)*1e-6, label='Planck')
    plt.loglog(wl*1e6, 2*np.pi*lz.c0*lz.kB*T/wl**4*1e-6, '--', label='Rayleigh–Jeans')
    plt.axvline(2898/T, color=lz.INK2, ls=':')
    plt.ylim(1e-2, 1e9); plt.xlabel('wavelength (µm)'); plt.ylabel('M_λ (W m⁻² µm⁻¹)'); plt.legend(); plt.show()""")

md("## 2. Three- versus four-level lasers")
code("""@interact(lw=FloatSlider(value=0.3, min=-3, max=2, step=0.02, description='log Wpτ'),
          dth=FloatSlider(value=0.02, min=0.001, max=0.2, step=0.001, description='ΔN_th/N0'))
def levels(lw, dth):
    x = np.logspace(-3, 2, 300)
    plt.figure(figsize=(7, 3.6))
    plt.semilogx(x, (x - 1)/(x + 1), label='3-level'); plt.semilogx(x, x/(1 + x), label='4-level (τ10 → 0)')
    plt.axvline(10**lw, color=lz.INK2, ls=':'); plt.axhline(dth, color=lz.PALETTE[2], ls='--'); plt.axhline(0, color=lz.INK)
    plt.xlabel('W_p τ'); plt.ylabel('(N2 − N1)/N0'); plt.legend()
    plt.title(f'pump for ΔN_th: 3-level Wpτ = {(1 + dth)/(1 - dth):.3f}, 4-level Wpτ = {dth/(1 - dth):.4f}'); plt.show()""")

md("## 3. EDFA: gain spectrum versus inversion")
code("""@interact(x=FloatSlider(value=0.7, min=0, max=1, step=0.01, description='x = N2/N0'),
          N0=FloatSlider(value=1, min=0.1, max=3, step=0.05, description='N0 (1e19 cm⁻³)'),
          L=FloatSlider(value=3, min=0.5, max=20, step=0.1, description='L (m)'))
def edfa_spectrum(x, N0, L):
    wl = np.linspace(1450e-9, 1620e-9, 500)
    ab, em = lz.er_cross_sections(wl)
    g = (em*x - ab*(1 - x))*N0*1e25
    plt.figure(figsize=(7, 3.6)); plt.plot(wl*1e9, 4.343*g*L); plt.axhline(0, color=lz.INK)
    plt.axvspan(1530, 1565, color=lz.GRID, alpha=0.5)
    plt.xlabel('wavelength (nm)'); plt.ylabel('gain (dB)'); plt.show()""")

md("## 4. EDFA along the fiber (pump and signal rate equations)")
code("""@interact(Pp=FloatSlider(value=25, min=1, max=100, step=1, description='P_p (mW)'),
          Ps=FloatSlider(value=-30, min=-40, max=10, step=1, description='P_s,in (dBm)'),
          ws=FloatSlider(value=1550, min=1520, max=1600, step=1, description='λ_s (nm)'),
          L=FloatSlider(value=12, min=1, max=40, step=0.5, description='L (m)'))
def edfa_fiber(Pp, Ps, ws, L):
    r = lz.edfa_propagate(L, Pp*1e-3, 1e-3*10**(Ps/10), wl_s=ws*1e-9, nz=300)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(r['z'], lz.dB(r['Pp']/1e-3), label='pump'); ax[0].plot(r['z'], lz.dB(r['Ps']/1e-3), label='signal')
    ax[0].set(xlabel='z (m)', ylabel='power (dBm)'); ax[0].legend()
    ax[1].plot(r['z'], r['x']); ax[1].set(xlabel='z (m)', ylabel='inversion x')
    fig.suptitle(f"gain {lz.dB(r['Ps'][-1]/r['Ps'][0]):.1f} dB, PCE {(r['Ps'][-1] - r['Ps'][0])/(Pp*1e-3):.1%}"); plt.tight_layout(); plt.show()""")

md("## 5. Doppler gain curve and longitudinal modes")
code("""@interact(T=IntSlider(value=400, min=200, max=2000, step=10, description='T (K)'),
          L=FloatSlider(value=40, min=5, max=200, step=1, description='L (cm)'),
          thr=FloatSlider(value=0.5, min=0, max=0.99, step=0.01, description='threshold'),
          shift=FloatSlider(value=0, min=0, max=1, step=0.01, description='comb shift'))
def gas_modes(T, L, thr, shift):
    nu0 = lz.c0/632.8e-9; fw = lz.doppler_fwhm(nu0, T, 20.18); dnm = lz.c0/(2*L*1e-2)
    f = np.linspace(-1.6*fw, 1.6*fw, 800); G = lambda x: np.exp(-4*np.log(2)*(x/fw)**2)
    fm = (np.arange(-int(1.6*fw/dnm) - 1, int(1.6*fw/dnm) + 2) + shift)*dnm
    plt.figure(figsize=(7, 3.4)); plt.plot(f/1e9, G(f))
    for x in fm: plt.vlines(x/1e9, 0, G(x), color=lz.PALETTE[0] if G(x) > thr else lz.GRID, lw=2)
    plt.axhline(thr, color=lz.PALETTE[7], ls='--')
    plt.title(f'FWHM {fw/1e9:.2f} GHz, spacing {dnm/1e6:.0f} MHz, lasing modes: {np.sum(G(fm) > thr)}')
    plt.xlabel('ν − ν0 (GHz)'); plt.show()""")

md("## 6. Q-switched pulse")
code("""@interact(r=FloatSlider(value=3, min=1.05, max=10, step=0.05, description='N_i/N_th'),
          tc=FloatSlider(value=10, min=1, max=50, step=1, description='τ_c (ns)'))
def qswitch(r, tc):
    t, n, phi = lz.qswitch_pulse(r, tc*1e-9, t_end=60*tc*1e-9*max(1, 3/(r - 1)))
    plt.figure(figsize=(7, 3.4)); plt.plot(t*1e9, n, label='N/N_th'); plt.plot(t*1e9, phi, label='φ/N_th')
    plt.axhline(1, color=lz.INK2, ls=':'); plt.xlabel('time (ns)'); plt.legend()
    plt.title(f'extracted fraction η = {lz.qswitch_energy_fraction(r):.1%}'); plt.show()""")

md("## 7. Semiconductor gain spectrum")
code("""@interact(mat=Dropdown(options=list(lz.SEMI)), n=FloatSlider(value=2, min=0.5, max=5, step=0.05, description='n (1e18 cm⁻³)'),
          T=IntSlider(value=300, min=20, max=400, step=5, description='T (K)'))
def gain(mat, n, T):
    Eg = lz.SEMI[mat]['Eg']; E = np.linspace(Eg - 0.02, Eg + 0.3, 600)
    g = lz.gain_spectrum_bulk(E, n*1e24, mat, T)/100
    plt.figure(figsize=(7, 3.4)); plt.plot(E, g); plt.axhline(0, color=lz.INK)
    plt.axvline(lz.delta_F(n*1e24, mat, T), color=lz.PALETTE[7], ls='--', label='ΔF')
    plt.ylim(-15000, max(3000, 1.2*g.max())); plt.xlabel('hν (eV)'); plt.ylabel('g (cm⁻¹)'); plt.legend(); plt.show()""")

md("## 8. Laser diode L–I and efficiencies")
code("""@interact(lam=IntSlider(value=1310, min=400, max=1700, description='λ (nm)'), L=IntSlider(value=200, min=100, max=2000, step=10, description='L (µm)'),
          a_s=FloatSlider(value=20, min=1, max=50, description='α_s (cm⁻¹)'), R1=FloatSlider(value=0.33, min=0.01, max=0.99, step=0.01),
          R2=FloatSlider(value=0.33, min=0.01, max=0.99, step=0.01), Ith=FloatSlider(value=5, min=1, max=100, description='I_th (mA)'),
          Rs=FloatSlider(value=5, min=0, max=30, description='R_s (Ω)'))
def ld(lam, L, a_s, R1, R2, Ith, Rs):
    ee = lz.extraction_efficiency(a_s*100, L*1e-6, R1, R2); slope = ee*1239.84/lam
    I = np.linspace(0, 6*Ith, 300); P = slope*np.clip(I - Ith, 0, None); V = 1.0 + I*1e-3*Rs
    fig, ax = plt.subplots(figsize=(7, 3.4)); ax.plot(I, P, label='P_o (mW)')
    ax.plot(I[1:], 100*P[1:]/(I[1:]*V[1:]), label='PCE (%)'); ax.set_xlabel('I (mA)'); ax.legend()
    ax.set_title(f'η_EE = {ee:.1%}, slope = {slope:.3f} mW/mA'); plt.show()""")

md("## 9. Small-signal modulation response")
code("""@interact(I1=FloatSlider(value=15, min=2.5, max=100, step=0.5, description='I1 (mA)'), Ith=FloatSlider(value=2, min=1, max=50, step=0.5, description='I_th (mA)'),
          tau=FloatSlider(value=1, min=0.2, max=5, step=0.05, description='τ (ns)'), tph=FloatSlider(value=1.5, min=0.5, max=10, step=0.1, description='τ_ph (ps)'))
def response(I1, Ith, tau, tph):
    f = np.logspace(8, 11, 500)
    H = lz.ld_small_signal(f, I1*1e-3, Ith*1e-3, tau*1e-9, tph*1e-12)
    plt.figure(figsize=(7, 3.4)); plt.semilogx(f/1e9, 20*np.log10(abs(H))); plt.axhline(-3, color=lz.INK2, ls=':')
    plt.ylim(-20, 12); plt.xlabel('f (GHz)'); plt.ylabel('response (dB)')
    plt.title(f'f_r = {lz.relaxation_frequency(I1*1e-3, Ith*1e-3, tau*1e-9, tph*1e-12)/1e9:.2f} GHz'); plt.show()""")

md("## 10. DFB grating: uniform versus λ/4-shifted")
code("""@interact(kL=FloatSlider(value=2, min=0.2, max=5, step=0.05, description='κL'), gL=FloatSlider(value=1.2, min=0, max=4, step=0.01, description='gL'),
          qws=Checkbox(value=False, description='λ/4 shift'))
def dfb(kL, gL, qws):
    L = 400e-6; lB = 2*3.5*0.22e-6; wl = np.linspace(lB - 4e-9, lB + 4e-9, 2000)
    r, t = lz.grating_rt(wl, 0.22e-6, 3.5, kL*lB/(np.pi*L), L, g=gL/L, shift=np.pi/2 if qws else 0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(wl*1e9, abs(r)**2); ax[0].set(xlabel='λ (nm)', ylabel='|r|²')
    ax[1].semilogy(wl*1e9, abs(t)**2); ax[1].set(xlabel='λ (nm)', ylabel='|t|²'); plt.tight_layout(); plt.show()""")

md("## 11. Optical resonator: stability and mode size")
code("""@interact(L=FloatSlider(value=40, min=1, max=200, step=1, description='L (cm)'), R1=FloatSlider(value=1e4, min=5, max=1e4, step=1, description='R1 (cm)'),
          R2=FloatSlider(value=100, min=5, max=1e4, step=1, description='R2 (cm)'))
def resonator(L, R1, R2):
    g1, g2 = lz.resonator_g(L, R1, R2)
    if not (0 < g1*g2 < 1):
        print(f'g1 g2 = {g1*g2:.3f}: unstable'); return
    w0 = lz.waist_from_resonator(L*1e-2, R1*1e-2, R2*1e-2, 632.8e-9)
    gouy = np.arccos(np.sign(g1)*np.sqrt(g1*g2))/np.pi
    print(f'g1 g2 = {g1*g2:.3f}, waist w0 = {w0*1e3:.3f} mm, transverse-mode spacing = {gouy:.3f} × c/2L')""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["colab"] = {"provenance": []}
nbf.write(nb, "interactive_explorers.ipynb")
print("built interactive_explorers.ipynb")
