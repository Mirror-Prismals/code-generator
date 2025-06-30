#!/usr/bin/env python3
# synthetic_ipynb.py · v0.1.1
"""
Generate synthetic Jupyter notebooks with random markdown and code cells.

Major features
--------------
* Deterministic output with --seed
* Configurable number of cells
* Plugin architecture for markdown and code cell generators
* Uses nbformat to build a valid .ipynb file
* --out to save directly to disk

Dependencies
------------
pip install nbformat

Usage
-----
python synthetic_ipynb.py 10           # 10 cells
python synthetic_ipynb.py 15 --seed 42 --out fake.ipynb
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

__version__ = "0.1.1"

@dataclass(frozen=True)
class IpynbConfig:
    cells: int = 10
    seed: Optional[int] = None

GeneratorFn = Callable[[Dict], nbformat.NotebookNode]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner

# ──────────────────────────────────────────────────────────────
# Markdown cell generators
# ──────────────────────────────────────────────────────────────

@register("md_heading")
def gen_md_heading(state: Dict):
    rng = state["rng"]
    level = rng.randint(1, 3)
    text = " ".join(rng.choice(state["lorem"]) for _ in range(rng.randint(2,5))).title()
    return new_markdown_cell(f"{'#'*level} {text}")

@register("md_paragraph")
def gen_md_paragraph(state: Dict):
    rng = state["rng"]
    sentences = []
    for _ in range(rng.randint(1,3)):
        words = [rng.choice(state["lorem"]) for _ in range(rng.randint(4,10))]
        sentences.append(" ".join(words).capitalize() + ".")
    return new_markdown_cell(" ".join(sentences))

@register("md_list")
def gen_md_list(state: Dict):
    rng = state["rng"]
    items = [f"- {rng.choice(state['lorem']).capitalize()}" for _ in range(rng.randint(3,6))]
    return new_markdown_cell("\n".join(items))

# ──────────────────────────────────────────────────────────────
# Code cell generators
# ──────────────────────────────────────────────────────────────

@register("code_import")
def gen_code_import(state: Dict):
    rng = state["rng"]
    pkg = rng.choice(["math","random","datetime","sys","os"])
    return new_code_cell(f"import {pkg}")

@register("code_print")
def gen_code_print(state: Dict):
    rng = state["rng"]
    msg = " ".join(rng.choice(state["lorem"]) for _ in range(rng.randint(2,5)))
    return new_code_cell(f"print({msg!r})")

@register("code_loop")
def gen_code_loop(state: Dict):
    rng = state["rng"]
    var = rng.choice(["i","j","k","n"])
    count = rng.randint(3,7)
    return new_code_cell(f"for {var} in range({count}):\n    print({var})")

@register("code_func")
def gen_code_func(state: Dict):
    rng = state["rng"]
    fname = rng.choice(state["lorem"])[:5]
    return new_code_cell(
        f"def {fname}():\n"
        f"    '''Auto-generated'''\n"
        f"    return {rng.randint(0,100)}\n\n"
        f"print({fname}())"
    )

@register("code_plot")
def gen_code_plot(state: Dict):
    return new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1,2,3],[4,5,6])\n"
        "plt.show()"
    )

def build_ipynb(cfg: IpynbConfig) -> nbformat.NotebookNode:
    rng = random.Random(cfg.seed)
    state = {
        "rng": rng,
        "lorem": [
            "lorem","ipsum","dolor","sit","amet","consectetur",
            "adipiscing","elit","sed","do","eiusmod","tempor"
        ]
    }
    notebook = new_notebook()
    kinds, weights = zip(*[
        ("md_heading",    0.15),
        ("md_paragraph",  0.25),
        ("md_list",       0.10),
        ("code_import",   0.10),
        ("code_print",    0.15),
        ("code_loop",     0.10),
        ("code_func",     0.10),
        ("code_plot",     0.05),
    ])
    cells: List[nbformat.NotebookNode] = []
    for _ in range(cfg.cells):
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        cells.append(_REGISTRY[kind](state))
    notebook["cells"] = cells
    return notebook

def _cli():
    p = argparse.ArgumentParser(description="Generate a synthetic .ipynb notebook.")
    p.add_argument("cells", nargs="?", type=int, default=10, help="Number of cells")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Output .ipynb file")
    args = p.parse_args()

    cfg = IpynbConfig(cells=args.cells, seed=args.seed)
    nb = build_ipynb(cfg)
    out = args.out or Path("synthetic.ipynb")
    with out.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"✔ Saved synthetic notebook to {out}")

if __name__ == "__main__":
    _cli()
