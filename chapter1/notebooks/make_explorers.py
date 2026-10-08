"""Builds interactive_explorers.ipynb (ipywidgets versions of the website panels).
Run:  python make_explorers.py"""
import nbformat as nbf

from build_notebooks import setup_code, colab_badge

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Chapter 1: interactive explorers (ipywidgets)

Python versions of the live panels of the companion website. Run all cells, then move the sliders.
Each explorer is self-contained and uses `optolib.py`.""")
md(colab_badge("interactive_explorers"))

code(setup_code() + """import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, RadioButtons
import optolib as ol
ol.use_style()
%matplotlib inline""")

md("## 1. Gaussian beam caustic")
code("""@interact(w0_mm=FloatSlider(value=0.5, min=0.01, max=3, step=0.01, description='w₀ (mm)'),
          lam_um=FloatSlider(value=0.633, min=0.4, max=10.6, step=0.001, description='λ (µm)'),
          M2=FloatSlider(value=1, min=1, max=10, step=0.1, description='M²'),
          zmax=Dropdown(options=[0.1, 1, 10, 100], value=10, description='range (m)'))
def gauss(w0_mm, lam_um, M2, zmax):
    z = np.linspace(-zmax, zmax, 600)
    w, z0 = ol.gaussian_w(z, w0_mm*1e-3, lam_um*1e-6, M2)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.fill_between(z, -w*1e3, w*1e3, alpha=0.25, lw=0); ax.plot(z, w*1e3, color=ol.PALETTE[0]); ax.plot(z, -w*1e3, color=ol.PALETTE[0])
    ax.set(xlabel='z (m)', ylabel='w (mm)', title=f'z₀ = {z0:.3g} m, θ = {M2*lam_um*1e-6/(np.pi*w0_mm*1e-3)*1e3:.3g} mrad')
    plt.show()""")

md("## 2. Fresnel coefficients")
code("""@interact(n1=FloatSlider(value=1.44, min=1, max=4, step=0.01), n2=FloatSlider(value=1.0, min=1, max=4, step=0.01),
          show=RadioButtons(options=['|r|', 'R', 'phase']))
def fres(n1, n2, show):
    th = np.radians(np.linspace(0, 89.9, 900))
    rs, rp, ts, tp = ol.fresnel(n1, n2, th)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    if show == '|r|':   ax.plot(np.degrees(th), abs(rs), label='|r⊥|'); ax.plot(np.degrees(th), abs(rp), label='|r∥|')
    elif show == 'R':   ax.plot(np.degrees(th), abs(rs)**2, label='R⊥'); ax.plot(np.degrees(th), abs(rp)**2, label='R∥')
    else:               ax.plot(np.degrees(th), np.degrees(np.angle(rs)), label='φ⊥'); ax.plot(np.degrees(th), np.degrees(np.angle(rp)), label='φ∥')
    ax.axvline(np.degrees(ol.brewster(n1, n2)), ls=':', c=ol.PALETTE[6])
    if n1 > n2: ax.axvline(np.degrees(ol.critical_angle(n1, n2)), ls=':', c=ol.PALETTE[7])
    ax.set(xlabel='θᵢ (deg)'); ax.legend(); plt.show()""")

md("## 3. Thin film on a substrate")
code("""@interact(n2=FloatSlider(value=1.9, min=1, max=4, step=0.01, description='film n₂'),
          n3=FloatSlider(value=3.5, min=1, max=4, step=0.01, description='substrate n₃'),
          d=IntSlider(value=92, min=0, max=600, step=1, description='d (nm)'))
def film(n2, n3, d):
    lam = np.linspace(300e-9, 1200e-9, 900)
    R = ol.tmm([1, n2, n3], [d*1e-9], lam)[0]
    plt.figure(figsize=(8, 3.2)); plt.plot(lam*1e9, R, label='with film')
    plt.axhline(((1 - n3)/(1 + n3))**2, color=ol.INK2, ls='--', label='bare substrate')
    plt.xlabel('λ (nm)'); plt.ylabel('R'); plt.legend(); plt.show()""")

md("## 4. Bragg mirror")
code("""@interact(nH=FloatSlider(value=2.49, min=1.3, max=4, step=0.01), nL=FloatSlider(value=1.55, min=1, max=3.5, step=0.01),
          N=IntSlider(value=4, min=1, max=30), ns=FloatSlider(value=1.47, min=1, max=4, step=0.01))
def bragg(nH, nL, N, ns):
    lam0 = 850e-9; lam = np.linspace(400e-9, 1400e-9, 1500)
    R = ol.tmm(*ol.qw_stack(nH, nL, N, lam0, ns=ns), lam)[0]
    plt.figure(figsize=(8, 3.2)); plt.plot(lam*1e9, R)
    plt.title(f'R_max = {100*ol.bragg_R_max(1, nH, nL, ns, N):.2f} %, Δλ ≈ {ol.bragg_bandwidth(nH, nL, lam0)*1e9:.0f} nm')
    plt.xlabel('λ (nm)'); plt.ylabel('R'); plt.show()""")

md("## 5. Fabry–Perot cavity")
code("""@interact(R=FloatSlider(value=0.9, min=0.04, max=0.995, step=0.005), L_um=IntSlider(value=100, min=1, max=1000),
          n=FloatSlider(value=1.0, min=1, max=4, step=0.01))
def fp(R, L_um, n):
    s = ol.fp_summary(L_um*1e-6, n, R, 900e-9)
    lam = np.linspace(900e-9 - 1.5*s['FSR_nm']*1e-9, 900e-9 + 1.5*s['FSR_nm']*1e-9, 3000)
    plt.figure(figsize=(8, 3.2)); plt.plot(lam*1e9, ol.airy_T(lam, L_um*1e-6, R, n))
    plt.title(f"FSR = {s['FSR_GHz']:.1f} GHz, finesse = {s['finesse']:.1f}, Q = {s['Q']:.3g}")
    plt.xlabel('λ (nm)'); plt.ylabel('T'); plt.show()""")

md("## 6. Diffraction patterns")
code("""from scipy.special import j1
@interact(shape=RadioButtons(options=['slit', 'circular', 'N slits']), a_um=IntSlider(value=100, min=1, max=500),
          lam_nm=IntSlider(value=532, min=400, max=1600), N=IntSlider(value=5, min=2, max=20), d_um=IntSlider(value=300, min=2, max=1000))
def diff(shape, a_um, lam_nm, N, d_um):
    lam, a, d = lam_nm*1e-9, a_um*1e-6, d_um*1e-6
    th = np.linspace(-min(1, 6*lam/a), min(1, 6*lam/a), 4000) if shape != 'N slits' else np.linspace(-3*lam/a, 3*lam/a, 6000)
    s = np.sin(th)
    if shape == 'circular':
        g = np.pi*a*s/lam + 1e-12; I = (2*j1(g)/g)**2
    else:
        I = np.sinc(a*s/lam)**2
        if shape == 'N slits':
            ph = np.pi*d*s/lam
            I = I*np.where(abs(np.sin(ph)) < 1e-9, N**2, (np.sin(N*ph)/np.sin(ph + 1e-15))**2)/N**2
    plt.figure(figsize=(8, 3.2)); plt.plot(th*1e3, I); plt.xlabel('θ (mrad)'); plt.ylabel('relative intensity'); plt.show()""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
nbf.write(nb, "interactive_explorers.ipynb")
print("written interactive_explorers.ipynb")
