"""Builds interactive_explorers.ipynb (ipywidgets versions of the website panels).
Run:  python make_explorers.py"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Chapter 2: interactive explorers (ipywidgets)

Python versions of the live panels of the companion website. Run all cells, then move the sliders.
Each explorer is self-contained and uses `fiberlib.py` from the parent folder.""")

from build_notebooks import setup_code, colab_badge
md(colab_badge("interactive_explorers"))

code(setup_code() + """import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, FloatLogSlider, Checkbox, RadioButtons
import fiberlib as fl
fl.use_style()
%matplotlib inline""")

md("## 1. Planar waveguide: graphical mode solution")
code("""@interact(V=FloatSlider(value=3.15, min=0.2, max=12, step=0.05))
def slab(V):
    u = np.linspace(1e-4, 12.5, 6000)
    fig, ax = plt.subplots(figsize=(6, 5))
    for k in range(9):
        uu = u[(u > k*np.pi/2) & (u < (k+1)*np.pi/2)]
        w = uu*np.tan(uu) if k % 2 == 0 else -uu/np.tan(uu)
        ok = (w >= 0) & (w < 13)
        ax.plot(uu[ok], w[ok], color=fl.PALETTE[k % 2])
    t = np.linspace(0, np.pi/2, 200); ax.plot(V*np.cos(t), V*np.sin(t), '--', color=fl.INK2)
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 12.5); ax.set_aspect('equal')
    ax.set_xlabel('u'); ax.set_ylabel('w'); ax.set_title(f'V = {V:.2f}: {int(2*V/np.pi)+1} TE mode(s)')
    plt.show()""")

md("## 2. Step-index fiber designer and LP modes")
code("""@interact(n1=FloatSlider(value=1.4532, min=1.44, max=1.50, step=0.0001, readout_format='.4f'),
          n2=FloatSlider(value=1.4485, min=1.44, max=1.50, step=0.0001, readout_format='.4f'),
          d_um=FloatSlider(value=8.2, min=2, max=100, step=0.1, description='2a (µm)'),
          lam_um=FloatSlider(value=1.31, min=0.6, max=1.7, step=0.01, description='λ (µm)'))
def designer(n1, n2, d_um, lam_um):
    if n1 <= n2:
        print('n1 must be > n2'); return
    a, lam = d_um*1e-6/2, lam_um*1e-6
    V = fl.V_number(a, lam, n1, n2); NAv = fl.NA(n1, n2)
    print(f'Δ = {100*fl.Delta(n1, n2):.3f} %   NA = {NAv:.4f}   2α_max = {2*np.degrees(np.arcsin(NAv)):.1f}°')
    print(f'V = {V:.3f}   λc = {2*np.pi*a*NAv/2.405*1e9:.0f} nm   ' + ('single-mode' if V < 2.405 else f'multimode, M ≈ {V**2/2:.0f}'))
    if 0.8 < V < 3: print(f'MFD: Marcuse {fl.mfd_marcuse(a, V)*1e6:.2f} µm, Petermann II {fl.mfd_petermann2(a, V)*1e6:.2f} µm')
    if V < 15:
        modes = fl.LP_all(V)
        n = min(len(modes), 10)
        fig, axs = plt.subplots(1, n, figsize=(1.8*n, 2))
        axs = np.atleast_1d(axs)
        x = np.linspace(-1.6, 1.6, 161); X, Y = np.meshgrid(x, x)
        for ax, (l, m, b) in zip(axs, modes[:n]):
            E = fl.LP_field(np.hypot(X, Y), V, b, l)*np.cos(l*np.arctan2(Y, X))
            ax.imshow(E/np.abs(E).max(), cmap='RdBu_r', vmin=-1, vmax=1); ax.axis('off')
            ax.set_title(f'LP{l}{m}\\nb={b:.2f}', fontsize=8)
        plt.show()""")

md("## 3. Chromatic dispersion engineering (exact model)")
code("""@interact(a_um=FloatSlider(value=4.1, min=1.75, max=5.5, step=0.05, description='a (µm)'),
          x=FloatSlider(value=0.035, min=0.02, max=0.15, step=0.005, description='GeO₂ frac.'))
def dispersion(a_um, x):
    lam = np.linspace(1.15, 1.70, 30)*1e-6
    D = [fl.D_total_smf(l, a_um*1e-6, x)/fl.PS_NM_KM for l in lam]
    Dm = fl.D_material(lam, x)/fl.PS_NM_KM
    fig, ax = plt.subplots()
    ax.plot(lam*1e9, Dm, label='material (core)'); ax.plot(lam*1e9, D, color=fl.INK, label='total (exact)')
    ax.axhline(0, color=fl.INK2, lw=0.8); ax.axvspan(1530, 1565, color=fl.PALETTE[2], alpha=0.1)
    V = fl.V_number(a_um*1e-6, 1.55e-6, fl.n_doped(1.55e-6, x), fl.n_silica(1.55e-6))
    ax.set_title(f'V(1550 nm) = {V:.2f}'); ax.set_xlabel('λ (nm)'); ax.set_ylabel('D (ps/nm/km)'); ax.legend()
    ax.set_ylim(-40, 35); plt.show()""")

md("## 4. IM/DD link simulator: eye diagram and BER")
code("""@interact(L=IntSlider(value=60, min=0, max=200, step=5, description='L (km)'),
          Rb=Dropdown(options=[2.5, 5, 10, 25, 40], value=10, description='Gb/s'),
          D=FloatSlider(value=17, min=-20, max=20, step=0.5, description='D'),
          P=FloatSlider(value=3, min=-10, max=15, step=0.5, description='P (dBm)'),
          fmt=RadioButtons(options=['NRZ', 'RZ']), amp=Checkbox(value=False, description='no loss'))
def link(L, Rb, D, P, fmt, amp):
    bits, sps = fl.prbs(9, 1500), 32
    t, Ptx, I = fl.imdd_link(bits, Rb=Rb*1e9, L_km=L, D_ps_nm_km=D, P_launch_dBm=P, sps=sps, fmt=fmt,
                             alpha_dB_km=0 if amp else 0.2)
    Q, ber, ph = fl.eye_metrics(I, bits, sps)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for k in range(10, 300):
        s = I[ph + k*sps: ph + k*sps + 2*sps]
        if len(s) == 2*sps: ax.plot(np.arange(2*sps)/sps, s*1e3, color=fl.PALETTE[0], lw=0.4, alpha=0.3)
    ax.set_xlabel('time (bits)'); ax.set_ylabel('I (mA)')
    ax.set_title(f'Q = {Q:.2f}, BER ≈ {ber:.1e}'); plt.show()""")

md("## 5. Fiber Bragg grating")
code("""@interact(dn=FloatSlider(value=1, min=0.1, max=20, step=0.1, description='Δn ×1e-4'),
          L_mm=FloatSlider(value=10, min=0.5, max=50, step=0.5, description='L (mm)'),
          strain=IntSlider(value=0, min=-2000, max=2000, step=50, description='µε'),
          dT=IntSlider(value=0, min=-100, max=200, step=5, description='ΔT (K)'))
def fbg(dn, L_mm, strain, dT):
    lB = 1550e-9*(1 + 0.78*strain*1e-6 + 7.25e-6*dT)
    wl = np.linspace(1547e-9, 1553e-9, 3000)
    R = fl.fbg_reflectance(wl, lB, 1.455, dn*1e-4, L_mm*1e-3)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(wl*1e9, R); ax.axvline(1550, color=fl.INK2, ls=':')
    ax.set_xlabel('λ (nm)'); ax.set_ylabel('R')
    ax.set_title(f'κL = {np.pi*dn*1e-4/1550e-9*L_mm*1e-3:.2f}, R_max = {R.max():.3f}, shift = {(lB-1550e-9)*1e12:.0f} pm')
    plt.show()""")

md("## 6. OTDR trace")
code("""@interact(tau_ns=Dropdown(options=[10, 100, 1000, 10000], value=100, description='pulse (ns)'),
          N=Dropdown(options=[1, 64, 4096, 262144], value=4096, description='averages'))
def otdr(tau_ns, N):
    rng = np.random.default_rng()
    dz = 2.0; z = np.arange(0, 34e3, dz); vg = fl.c0/1.468; W = vg*tau_ns*1e-9/2
    a, as_ = 0.33/4.343/1e3, 0.3/4.343/1e3
    loss = 0.1*(z >= 8e3) + 0.6*(z >= 15e3) + 0.4*(z >= 22e3)
    raw = 1e-3*as_*np.exp(-2*a*z)*10**(-loss/5)*(z < 30e3)*dz
    raw[int(22e3/dz)] += 1e-4*10**(-0.7/5); raw[int(30e3/dz)] += 0.035*10**(-1.1/5)*np.exp(-2*a*30e3)
    k = max(1, int(W/dz)); P = 0.02*np.convolve(raw, np.ones(k), mode='full')[:z.size]
    P = P + rng.normal(0, 3e-12/np.sqrt(N), z.size)
    fig, ax = plt.subplots()
    ax.plot(z/1e3, 5*np.log10(np.maximum(P, 1e-16)/1e-3), lw=0.8)
    ax.set_xlabel('distance (km)'); ax.set_ylabel('dB (one-way)'); ax.set_title(f'resolution ≈ {W:.1f} m')
    plt.show()""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
nbf.write(nb, "interactive_explorers.ipynb")
print("written interactive_explorers.ipynb")
