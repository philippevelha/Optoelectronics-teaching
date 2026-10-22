"""Builds interactive_explorers.ipynb (ipywidgets versions of the website panels).
Run:  python make_explorers.py"""
import nbformat as nbf

from build_notebooks import setup_code, colab_badge

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Chapter 5: interactive explorers (ipywidgets)

Python versions of the live panels of the companion website. Run all cells, then move the sliders.
Each explorer is self-contained and uses `detlib.py` (SI units, photon energies in eV).""")
md(colab_badge("interactive_explorers"))

code(setup_code() + """import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, FloatLogSlider, IntSlider, Dropdown, Checkbox, SelectMultiple
import detlib as dl
dl.use_style()
%matplotlib inline""")

md("## 1. Illuminated photodiode I–V")
code("""@interact(P=FloatSlider(value=20, min=0, max=100, step=0.5, description='P (µW)'),
          R=FloatSlider(value=0.5, min=0.05, max=1.2, step=0.01, description='R (A/W)'),
          lIo=FloatSlider(value=-10, min=-14, max=-6, step=0.1, description='log Io (A)'))
def iv(P, R, lIo):
    Io, Iph = 10**lIo, R*P*1e-6
    V = np.linspace(-2, max(0.2, dl.kT()*np.log1p(2*Iph/Io)), 600)
    plt.figure(figsize=(7, 3.6)); plt.plot(V, 1e6*dl.diode_current(V, Io), label='dark'); plt.plot(V, 1e6*dl.diode_current(V, Io, Iph), label='illuminated')
    plt.axhline(0, color=dl.INK2, lw=0.8); plt.axvline(0, color=dl.INK2, lw=0.8); plt.ylim(-1.3*max(Iph*1e6, 1), 0.8*max(Iph*1e6, 1))
    plt.xlabel('V (V)'); plt.ylabel('I (µA)'); plt.legend(); plt.title(f'V_oc = {dl.kT()*np.log1p(Iph/Io)*1e3:.0f} mV'); plt.show()""")

md("## 2. Shockley–Ramo: transit-time response of a drift region")
code("""@interact(W=FloatSlider(value=10, min=0.5, max=50, step=0.5, description='W (µm)'),
          la=FloatSlider(value=5, min=3, max=8, step=0.05, description='log α (m⁻¹)'),
          ve=FloatSlider(value=1e5, min=1e4, max=2e5, step=1e3, description='v_e (m/s)'),
          vh=FloatSlider(value=7e4, min=1e4, max=2e5, step=1e3, description='v_h (m/s)'))
def ramo(W, la, ve, vh):
    W = W*1e-6; a = 10**la; t = np.linspace(0, 1.1*W/min(ve, vh), 500)
    ie, ih = dl.pin_impulse_response(t, W, a, ve, vh)
    f = np.linspace(0, 2*max(ve, vh)/W, 800); H = dl.transit_response(f, W, a, ve, vh)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(t*1e12, ie/dl.qe*1e-12, label='electrons'); ax[0].plot(t*1e12, ih/dl.qe*1e-12, label='holes'); ax[0].plot(t*1e12, (ie + ih)/dl.qe*1e-12, 'k', label='total')
    ax[0].set(xlabel='t (ps)', ylabel='current (e/ps)'); ax[0].legend()
    ax[1].plot(f/1e9, 20*np.log10(abs(H))); ax[1].axhline(-3, ls=':', color=dl.INK2); ax[1].set(xlabel='f (GHz)', ylabel='|H| (dB)', ylim=(-15, 1))
    fig.suptitle(f'f3dB = {dl.f3dB(f, H)/1e9:.2f} GHz, absorbed {1 - np.exp(-a*W):.0%}'); plt.tight_layout(); plt.show()""")

md("## 3. Absorption spectra and penetration depth")
code("""@interact(mats=SelectMultiple(options=list(dl.MATERIALS), value=('Si', 'Ge', 'GaAs', 'In0.53Ga0.47As'), description='materials'),
          wl=IntSlider(value=850, min=300, max=1850, step=5, description='λ (nm)'), d=FloatSlider(value=5, min=0.1, max=100, step=0.1, description='d (µm)'))
def absorption(mats, wl, d):
    x = np.linspace(0.3e-6, 1.9e-6, 800); plt.figure(figsize=(7, 3.6))
    for m in mats:
        a = dl.alpha(m, x); plt.semilogy(x[a > 0]*1e6, a[a > 0], label=f"{m}: {1 - np.exp(-dl.alpha(m, wl*1e-9)*d*1e-6):.0%} absorbed in {d} µm")
    plt.axvline(wl/1000, color=dl.INK2, ls='--'); plt.ylim(1e3, 1e8); plt.xlabel('λ (µm)'); plt.ylabel('α (m⁻¹)'); plt.legend(fontsize=8); plt.show()""")

md("## 4. Responsivity from the device structure")
code("""@interact(mat=Dropdown(options=list(dl.MATERIALS), value='Si'), lp=FloatSlider(value=0.5, min=0.01, max=3, step=0.01, description='ℓp (µm)'),
          Le=FloatSlider(value=0.1, min=0, max=3, step=0.01, description='Le (µm)'), W=FloatSlider(value=30, min=0.5, max=200, step=0.5, description='W (µm)'),
          Lh=FloatSlider(value=10, min=0, max=200, step=0.5, description='Lh (µm)'), ar=Checkbox(value=False, description='AR coating'))
def resp(mat, lp, Le, W, Lh, ar):
    m = dl.MATERIALS[mat]; lg = dl.cutoff_wavelength(m['Eg']); wl = np.linspace(0.3e-6, 1.05*lg, 700)
    T = 1.0 if ar else dl.fresnel_T(m['n'])
    R = dl.photocurrent_structure(1, wl, dl.alpha(mat, wl), lp*1e-6, W*1e-6, Le*1e-6, Lh*1e-6, T=T)
    lx, Rx, ex = dl.max_qe_point(wl, R)
    plt.figure(figsize=(7, 3.6)); plt.plot(wl*1e9, R, label='structure'); plt.plot(wl*1e9, dl.responsivity(1, wl)*(wl < lg), label='ideal')
    plt.plot([0, 1.5*lx*1e9], [0, 1.5*Rx], '--', color=dl.INK2); plt.plot(lx*1e9, Rx, 'o')
    plt.title(f'peak {R.max():.2f} A/W, max QE {ex:.0%} at {lx*1e9:.0f} nm'); plt.xlabel('λ (nm)'); plt.ylabel('R (A/W)'); plt.legend(); plt.show()""")

md("## 5. Si pin: transit time versus RC")
code("""@interact(wl=IntSlider(value=850, min=400, max=1050, step=5, description='λ (nm)'), Vr=FloatSlider(value=20, min=1, max=200, description='Vr (V)'),
          D=FloatLogSlider(value=300, base=10, min=1, max=3.5, description='⌀ (µm)'), RL=FloatLogSlider(value=50, base=10, min=1, max=4, description='R_L (Ω)'))
def pin(wl, Vr, D, RL):
    a = dl.alpha('Si', wl*1e-9); A = np.pi*(D*1e-6/2)**2; Ws = np.logspace(np.log10(0.5e-6), -4, 40)
    res = np.array([dl.pin_bandwidth(W, A, RL, a, dl.drift_velocity(Vr/W, 'e'), dl.drift_velocity(Vr/W, 'h')) for W in Ws])
    fig, ax = plt.subplots(figsize=(7, 3.6)); ax.loglog(Ws*1e6, res[:, 0]/1e9, 'k', label='total'); ax.loglog(Ws*1e6, res[:, 1]/1e9, '--', label='transit')
    ax.loglog(Ws*1e6, res[:, 2]/1e9, '--', label='RC'); ax.set(xlabel='W (µm)', ylabel='f3dB (GHz)'); ax.legend()
    i = np.nanargmax(res[:, 0]); ax.set_title(f'optimum W = {Ws[i]*1e6:.1f} µm, {res[i, 0]/1e9:.2f} GHz, R = {dl.pin_photocurrent(1, wl*1e-9, a, Ws[i]):.2f} A/W'); plt.show()""")

md("## 6. Avalanche gain versus field")
code("""@interact(mat=Dropdown(options=['Si', 'InP']), inj=Dropdown(options=['electrons', 'holes']), w=FloatSlider(value=0.5, min=0.1, max=3, step=0.05, description='w (µm)'))
def apd(mat, inj, w):
    E = np.linspace(2e5, 8e5, 3000); ae, ah = dl.chynoweth(E, mat, 'e'), dl.chynoweth(E, mat, 'h')
    a1, a2 = (ae, ah) if inj == 'electrons' else (ah, ae)
    M = dl.apd_gain_uniform(a1, a2/a1, w*1e-4)
    plt.figure(figsize=(7, 3.6)); plt.semilogy(E/1e5, M, label='both carriers'); plt.semilogy(E/1e5, np.exp(a1*w*1e-4), label='injected carrier only')
    Ebr = E[np.argmax(~np.isfinite(M))] if (~np.isfinite(M)).any() else np.nan
    plt.ylim(1, 1e4); plt.xlabel('E (10⁵ V/cm)'); plt.ylabel('M'); plt.legend(); plt.title(f'breakdown at {Ebr/1e5:.2f}×10⁵ V/cm, V ≈ {Ebr*w*1e-4:.1f} V'); plt.show()""")

md("## 7. Receiver SNR")
code("""@interact(lP=FloatSlider(value=-8.2, min=-12, max=-3, step=0.05, description='log P (W)'), R=FloatSlider(value=0.85, min=0.1, max=1.1, step=0.01),
          lId=FloatSlider(value=-8.7, min=-12, max=-5, step=0.05, description='log Id (A)'), lB=FloatSlider(value=6, min=0, max=10.5, step=0.1, description='log B (Hz)'),
          lRL=FloatSlider(value=4, min=1, max=9, step=0.05, description='log R_L (Ω)'))
def receiver(lP, R, lId, lB, lRL):
    P = np.logspace(-13, -3, 300); B, RL, Id = 10**lB, 10**lRL, 10**lId
    plt.figure(figsize=(7, 3.6))
    plt.semilogx(P, 10*np.log10(dl.snr(R*P, Id, B, RL)), 'k', lw=2.5, label='total')
    plt.semilogx(P, 10*np.log10((R*P)**2/(4*dl.kB*300*B/RL)), label='thermal only'); plt.semilogx(P, 10*np.log10(R*P/(2*dl.qe*B)), label='quantum only')
    plt.axvline(10**lP, ls='--', color=dl.INK2); plt.ylim(-30, 80); plt.xlabel('P (W)'); plt.ylabel('SNR (dB)'); plt.legend()
    plt.title(f'SNR at P = {10**lP:.2e} W: {10*np.log10(dl.snr(R*10**lP, Id, B, RL)):.1f} dB'); plt.show()""")

md("## 8. APD optimum gain")
code("""@interact(lP=FloatSlider(value=-7.7, min=-10, max=-5, step=0.05, description='log P (W)'), lRL=FloatSlider(value=3, min=1, max=6, step=0.05, description='log R_L (Ω)'),
          k=FloatSlider(value=0.5, min=0, max=1, step=0.01), lB=FloatSlider(value=8.85, min=6, max=10.5, step=0.05, description='log B (Hz)'))
def apd_noise(lP, lRL, k, lB):
    M = np.logspace(0, 3, 400); s = dl.apd_snr(10**lP, 0.8, M, 1e-8, 10**lB, k=k, RL=10**lRL)
    plt.figure(figsize=(7, 3.6)); plt.semilogx(M, 10*np.log10(s)); i = np.argmax(s); plt.plot(M[i], 10*np.log10(s[i]), 'o')
    plt.xlabel('M'); plt.ylabel('SNR (dB)'); plt.title(f'M_opt = {M[i]:.1f}, gain over pin {10*np.log10(s[i]/s[0]):.1f} dB'); plt.show()""")

md("## 9. Solar cell and load")
code("""@interact(G=FloatSlider(value=900, min=50, max=1500, step=10, description='W/m²'), n=FloatSlider(value=2, min=1, max=2.5, step=0.05, description='η'),
          lJo=FloatSlider(value=-6.66, min=-14, max=-4, step=0.01, description='log Jo (A/cm²)'), Rs=FloatSlider(value=0, min=0, max=5, step=0.05, description='Rs (Ω)'),
          RL=FloatSlider(value=16, min=1, max=200, step=0.5, description='R load (Ω)'))
def solar(G, n, lJo, Rs, RL):
    Iph = 39.4e-3*G/1000; Io = 10**lJo; m = dl.solar_mpp(Iph, Io, n, Rs)
    V = np.linspace(0, m['Voc']*1.02, 400); V0, I0 = dl.solar_operating_point(RL, Iph, Io, n, Rs)
    plt.figure(figsize=(7, 3.6)); plt.plot(V, 1e3*dl.solar_iv(V, Iph, Io, n, Rs)); plt.plot(V, -1e3*V/RL, '--'); plt.plot(V0, 1e3*I0, 'o')
    plt.ylim(-1.15*Iph*1e3, 0.2*Iph*1e3); plt.xlabel('V (V)'); plt.ylabel('I (mA)')
    plt.title(f"FF = {m['FF']:.1%}, P(R) = {-V0*I0*1e3:.2f} mW, P_max = {m['Pm']*1e3:.2f} mW, η = {-V0*I0/(G*1e-4):.1%}"); plt.show()""")

md("## 10. Shockley–Queisser limit")
code("""@interact(Ts=IntSlider(value=5778, min=3000, max=7000, step=10, description='T_sun (K)'), Tc=IntSlider(value=300, min=200, max=400, description='T_cell (K)'),
          lX=FloatSlider(value=0, min=0, max=4.66, step=0.01, description='log X'))
def sq(Ts, Tc, lX):
    Eg = np.linspace(0.5, 3, 150); e = [100*dl.shockley_queisser(E, Ts, Tc, 10**lX)['eff'] for E in Eg]
    plt.figure(figsize=(7, 3.6)); plt.plot(Eg, e); i = int(np.argmax(e)); plt.plot(Eg[i], e[i], 'o')
    plt.xlabel('E_g (eV)'); plt.ylabel('efficiency limit (%)'); plt.title(f'max {e[i]:.1f} % at {Eg[i]:.2f} eV'); plt.show()""")

md("## 11. Simulated exposure of an image sensor")
code("""@interact(lph=FloatSlider(value=2.5, min=0, max=5, step=0.05, description='log photons'), rn=FloatSlider(value=5, min=0, max=50, step=0.5, description='read noise'),
          fw=IntSlider(value=20000, min=500, max=100000, step=500, description='full well'))
def exposure(lph, rn, fw):
    y, x = np.mgrid[0:160, 0:160]; scene = 0.05 + 0.25*x/160
    scene[(x//8 % 2 == 0) & (y > 10) & (y < 50)] = (0.3 + 0.7*x/160)[(x//8 % 2 == 0) & (y > 10) & (y < 50)]
    scene[(x - 110)**2 + (y - 105)**2 < 900] = 1.0
    img = dl.simulate_exposure(scene, 10**lph, read_noise=rn, full_well=fw, rng=1)
    plt.figure(figsize=(4.5, 4.5)); plt.imshow(np.sqrt(img/img.max()), cmap='gray'); plt.axis('off')
    S = 0.6*10**lph*(0.05 + 0.25*135/160); plt.title(f'background SNR ≈ {S/np.sqrt(S + rn**2):.1f}'); plt.show()""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["colab"] = {"provenance": []}
nbf.write(nb, "interactive_explorers.ipynb")
