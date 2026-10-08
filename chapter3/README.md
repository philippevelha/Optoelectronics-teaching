# Chapter 3 companion: Semiconductor Science and Light-Emitting Diodes

Interactive companion to Chapter 3 of the Optoelectronics course (University of Trento).

**Website:** https://philippevelha.github.io/Optoelectronics-teaching/chapter3/
**Run in Colab:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippevelha/Optoelectronics-teaching/blob/main/chapter3/notebooks/interactive_explorers.ipynb)

| Folder / file | Content |
|---|---|
| `*.qmd` | the chapters (Quarto) |
| `semilib.py` | the physics toolbox: carrier statistics, doping, pn junction, diode currents, LED spectrum, quantum wells, photometry |
| `notebooks/` | one notebook per chapter + `interactive_explorers.ipynb` (ipywidgets) |
| `_freeze/` | cached computation results (keeps re-rendering fast) |

## Updating

```bash
pip install -r requirements.txt
cd notebooks && python build_notebooks.py && cd ..
git add -A && git commit -m "..." && git push
bash ../publish.sh
```
