# Chapter 4 companion: Stimulated Emission Devices — Optical Amplifiers and Lasers

Interactive companion to Chapter 4 of the Optoelectronics course (University of Trento).

**Website:** https://philippevelha.github.io/Optoelectronics-teaching/chapter4/
**Run in Colab:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippevelha/Optoelectronics-teaching/blob/main/chapter4/notebooks/interactive_explorers.ipynb)

| Folder / file | Content |
|---|---|
| `*.qmd` | the chapters (Quarto) |
| `laserlib.py` | the physics toolbox: black body, Einstein coefficients, level systems, line shapes, EDFA, Fabry–Perot cavity, Q-switching, mode-locking, semiconductor gain, laser-diode rate equations, quantum wells, gratings, DBR mirrors, amplifiers, resonators, linewidth |
| `laser_sim.js` | browser ports of the heavier `laserlib.py` solvers (EDFA, coupled-mode gratings, rate equations, gain, FFT), used by the live panels |
| `notebooks/` | one notebook per chapter + `interactive_explorers.ipynb` (ipywidgets) |
| `_freeze/` | cached computation results (keeps re-rendering fast) |

## Updating

```bash
pip install -r requirements.txt
cd notebooks && python build_notebooks.py && cd ..
git add -A && git commit -m "..." && git push
bash ../publish.sh
```
