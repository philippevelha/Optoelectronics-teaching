"""Regenerate every notebook in this folder from the Quarto chapters.

    cd notebooks
    python build_notebooks.py

- converts each chapter (.qmd) with `quarto convert`,
- replaces the first code cell's header with a setup that works both locally
  (imports ../fiberlib.py) and in Google Colab (downloads fiberlib.py from GitHub),
- removes Quarto-only syntax (shortcodes, Colab badge line),
- rebuilds interactive_explorers.ipynb (make_explorers.py).
The GitHub user/repo are read from ../_variables.yml.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
cfg = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf8"))
USER, REPO, SUB = cfg["github_user"], cfg["repo"], cfg["subdir"]
RAW = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/{SUB}/fiberlib.py"

CHAPTERS = "planar fibers dispersion grin attenuation bitrate wdm special-fbg sensors ideas".split()
NEEDS_OPTIC = {"bitrate"}          # chapters that use OptiCommPy


def setup_code(extra_pip=()):
    pip = ""
    if extra_pip:
        pkgs = ", ".join(repr(p) for p in extra_pip)
        pip = (f"    import subprocess\n"
               f"    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', {pkgs}])\n")
    return (
        "# --- setup: works locally and in Google Colab ---------------------------\n"
        "import sys\n"
        "if 'google.colab' in sys.modules:\n"
        "    import urllib.request\n"
        f"    urllib.request.urlretrieve('{RAW}', 'fiberlib.py')\n"
        f"{pip}"
        "else:\n"
        "    sys.path.insert(0, '..')                 # fiberlib.py is in the parent folder\n"
        "def ojs_define(**kw): pass                   # only used by the Quarto website\n"
        "# ----------------------------------------------------------------------\n"
    )


def colab_badge(name):
    url = f"https://colab.research.google.com/github/{USER}/{REPO}/blob/main/{SUB}/notebooks/{name}.ipynb"
    return f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def clean_markdown(src):
    src = re.sub(r"\n?\[!\[Open in Colab\].*?\n", "\n", src)          # badge line from the .qmd
    src = re.sub(r"\{\{<\s*var\s+(\w+)\s*>\}\}", lambda m: str(cfg.get(m.group(1), "")), src)
    return src


def build_chapter(ch):
    out = HERE / f"{ch}.ipynb"
    subprocess.run(["quarto", "convert", str(ROOT / f"{ch}.qmd"), "-o", str(out)], check=True,
                   capture_output=True)
    nb = json.loads(out.read_text(encoding="utf8"))
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            c["source"] = clean_markdown("".join(c["source"]))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            body = "".join(c["source"]).replace("#| include: false\n", "")
            c["source"] = setup_code(["OptiCommPy"] if ch in NEEDS_OPTIC else ()) + body
            break
    nb["cells"].insert(1, {"cell_type": "markdown", "metadata": {}, "source":
        f"{colab_badge(ch)}\n\n"
        "> Generated from the Quarto companion. The browser sliders of the website are not included; "
        "see `interactive_explorers.ipynb` for ipywidgets versions."})
    nb.setdefault("metadata", {})["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                                   "name": "python3"}
    nb["metadata"]["colab"] = {"provenance": []}
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf8")
    print("built", out.name)


if __name__ == "__main__":
    for ch in CHAPTERS:
        build_chapter(ch)
    subprocess.run([sys.executable, str(HERE / "make_explorers.py")], check=True, cwd=HERE)
