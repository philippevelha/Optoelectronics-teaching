# Chapter 5 companion: Photodetectors and Image Sensors

Interactive companion to Chapter 5 of the Optoelectronics course (University of Trento).

**Website:** https://philippevelha.github.io/Optoelectronics-teaching/chapter5/
**Run in Colab:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippevelha/Optoelectronics-teaching/blob/main/chapter5/notebooks/interactive_explorers.ipynb)

| Folder / file | Content |
|---|---|
| `*.qmd` | the chapters (Quarto) |
| `detlib.py` | the physics toolbox: absorption data, QE and responsivity, drift and diffusion, Shockley–Ramo responses, pin bandwidth, avalanche gain and excess noise, Schottky/MSM/phototransistor, photoconductors, load line and RC, noise/SNR/NEP/D*, receiver sensitivity and BER, solar cells and the Shockley–Queisser limit, image sensors |
| `det_sim.js` | browser ports of the `detlib.py` functions used by the live panels |
| `notebooks/` | one notebook per chapter + `interactive_explorers.ipynb` (ipywidgets) |
| `_freeze/` | cached computation results (keeps re-rendering fast) |

## Updating

```bash
pip install -r requirements.txt
cd notebooks && python build_notebooks.py && cd ..
git add -A && git commit -m "..." && git push
bash ../publish.sh
```
