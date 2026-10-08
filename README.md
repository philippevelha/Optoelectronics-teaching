# Chapter 2 companion: Dielectric Waveguides and Optical Fibers

Interactive companion to Chapter 2 of the Optoelectronics course (University of Trento).

**Website:** https://philippevelha.github.io/optoelectronics-ch2/
**Run in Colab:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippevelha/optoelectronics-ch2/blob/main/notebooks/interactive_explorers.ipynb)

| Folder / file | Content |
|---|---|
| `*.qmd` | the chapters (Quarto) |
| `fiberlib.py` | the physics toolbox: modes, dispersion, loss, link simulation, gratings |
| `notebooks/` | one notebook per chapter + `interactive_explorers.ipynb` (ipywidgets) |
| `_freeze/` | cached computation results (keeps re-rendering fast) |

## Updating

```bash
pip install -r requirements.txt
cd notebooks && python build_notebooks.py && cd ..
git add -A && git commit -m "..." && git push
quarto publish gh-pages
```
