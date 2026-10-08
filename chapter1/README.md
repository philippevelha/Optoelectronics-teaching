# Chapter 1 companion: The Wave Nature of Light

Interactive companion to Chapter 1 of the Optoelectronics course (University of Trento).

**Website:** https://philippevelha.github.io/Optoelectronics-teaching/chapter1/
**Run in Colab:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippevelha/Optoelectronics-teaching/blob/main/chapter1/notebooks/interactive_explorers.ipynb)

| Folder / file | Content |
|---|---|
| `*.qmd` | the chapters (Quarto) |
| `optolib.py` | the physics toolbox: Gaussian beams, Fresnel, thin films, Bragg mirrors, Fabry–Perot, diffraction |
| `notebooks/` | one notebook per chapter + `interactive_explorers.ipynb` (ipywidgets) |
| `_freeze/` | cached computation results (keeps re-rendering fast) |

## Updating

```bash
pip install -r requirements.txt
cd notebooks && python build_notebooks.py && cd ..
git add -A && git commit -m "..." && git push
bash ../publish.sh
```
