# OptoElectronics companion of the course "Basics of OptoElectronics"

Interactive companions to the chapters of the course (University of Trento).

**Website:** https://philippevelha.github.io/Optoelectronics-teaching/

| Chapter | Website | Run in Colab |
|---|---|---|
| 1. The wave nature of light | [open](https://philippevelha.github.io/Optoelectronics-teaching/chapter1/) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippevelha/Optoelectronics-teaching/blob/main/chapter1/notebooks/interactive_explorers.ipynb) |
| 2. Dielectric waveguides and optical fibers | [open](https://philippevelha.github.io/Optoelectronics-teaching/chapter2/) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippevelha/Optoelectronics-teaching/blob/main/chapter2/notebooks/interactive_explorers.ipynb) |
| 3. Semiconductor science and light-emitting diodes | [open](https://philippevelha.github.io/Optoelectronics-teaching/chapter3/) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philippevelha/Optoelectronics-teaching/blob/main/chapter3/notebooks/interactive_explorers.ipynb) |

## Structure

- `chapter1/`: Quarto book for Chapter 1 (chapters `*.qmd`, `optolib.py`, `notebooks/`)
- `chapter2/`: Quarto book for Chapter 2 (chapters `*.qmd`, `fiberlib.py`, `notebooks/`)
- `chapter3/`: Quarto book for Chapter 3 (chapters `*.qmd`, `semilib.py`, `notebooks/`)
- `site/index.html`: landing page of the website, listing the chapters
- `publish.sh`: renders each chapter and updates the website (`gh-pages` branch)

## Updating the website

```bash
pip install -r chapter2/requirements.txt
(cd chapter2/notebooks && python build_notebooks.py)   # if chapters changed
git add -A && git commit -m "..." && git push
bash publish.sh
```
