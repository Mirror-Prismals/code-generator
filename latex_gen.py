#!/usr/bin/env python3
# synthetic_latex.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—LaTeX documents.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_latex.py 100
python synthetic_latex.py 200 --seed 42 --out paper.tex
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

__version__ = "0.1.0"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class LatexConfig:
    loc: int = 100                # approx. number of lines in body
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":     0.05,
        "section":     0.10,
        "subsection":  0.10,
        "paragraph":   0.30,
        "itemize":     0.10,
        "enumerate":   0.10,
        "equation":    0.10,
        "table":       0.10,
        "figure":      0.05,
    })

# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

GeneratorFn = Callable[[Dict], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

LOREM = [
    "lorem","ipsum","dolor","sit","amet","consectetur",
    "adipiscing","elit","sed","do","eiusmod","tempor",
    "incididunt","ut","labore","et","dolore","magna","aliqua"
]

def random_sentence(rng: random.Random, min_w=5, max_w=12) -> str:
    n = rng.randint(min_w, max_w)
    words = [rng.choice(LOREM) for _ in range(n)]
    s = " ".join(words).capitalize() + "."
    return s

def random_paragraph(rng: random.Random, min_s=2, max_s=5) -> str:
    return " ".join(random_sentence(rng) for _ in range(rng.randint(min_s, max_s)))

def random_title(rng: random.Random, words=3) -> str:
    return " ".join(rng.choice(LOREM).capitalize() for _ in range(words))

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["TODO", "FIXME", "NOTE", "HACK"]
    return f"% {rng.choice(tags)}: {random_sentence(rng,3,6)}\n"

@register("section")
def gen_section(state: Dict) -> str:
    rng = state["rng"]
    title = random_title(rng, words=rng.randint(1,4))
    return f"\\section{{{title}}}\n\n"

@register("subsection")
def gen_subsection(state: Dict) -> str:
    rng = state["rng"]
    title = random_title(rng, words=rng.randint(1,3))
    return f"\\subsection{{{title}}}\n\n"

@register("paragraph")
def gen_paragraph(state: Dict) -> str:
    rng = state["rng"]
    return random_paragraph(rng) + "\n\n"

@register("itemize")
def gen_itemize(state: Dict) -> str:
    rng = state["rng"]
    count = rng.randint(3,6)
    items = "".join(f"  \\item {random_sentence(rng,3,6)}\n" for _ in range(count))
    return "\\begin{itemize}\n" + items + "\\end{itemize}\n\n"

@register("enumerate")
def gen_enumerate(state: Dict) -> str:
    rng = state["rng"]
    count = rng.randint(3,6)
    items = "".join(f"  \\item {random_sentence(rng,3,6)}\n" for _ in range(count))
    return "\\begin{enumerate}\n" + items + "\\end{enumerate}\n\n"

@register("equation")
def gen_equation(state: Dict) -> str:
    rng = state["rng"]
    a, b, c = rng.randint(1,9), rng.randint(1,9), rng.randint(1,9)
    expr = f"{a}x + {b} = {c}"
    return "\\begin{equation}\n" + f"  {expr}\n" + "\\end{equation}\n\n"

@register("table")
def gen_table(state: Dict) -> str:
    rng = state["rng"]
    cols = rng.randint(2,4)
    rows = rng.randint(2,4)
    # header
    headers = " & ".join(random_title(rng,1) for _ in range(cols)) + " \\\\"
    sep = " & ".join(r" \hline " for _ in range(cols)) + " \\\\"
    body = ""
    for _ in range(rows):
        row = " & ".join(str(rng.randint(0,100)) for _ in range(cols)) + " \\\\"
        body += row + "\n"
    return (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        "  \\begin{tabular}{|" + "c|"*cols + "}\n"
        "  \\hline\n"
        f"  {headers}\n"
        "  \\hline\n"
        f"{body}"
        "  \\hline\n"
        "  \\end{tabular}\n"
        "  \\caption{Data table}\n"
        "\\end{table}\n\n"
    )

@register("figure")
def gen_figure(state: Dict) -> str:
    rng = state["rng"]
    caption = random_sentence(rng,3,6)[:-1]  # drop final period
    return (
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  % Placeholder for figure\n"
        "  \\fbox{\\rule[0pt]{5cm}{3cm}}\n"
        f"  \\caption{{{caption}}}\n"
        "\\end{figure}\n\n"
    )

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_latex(cfg: LatexConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {"rng": rng}
    parts: List[str] = [
        "% Auto-generated LaTeX document\n",
        "\\documentclass{article}\n",
        "\\usepackage{amsmath,graphicx,booktabs}\n",
        "\\begin{document}\n\n"
    ]
    lines = sum(p.count("\n") for p in parts)
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc + 4:  # +4 for the header/footer lines
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if not snippet:
            continue
        parts.append(snippet)
        lines += snippet.count("\n")

    parts.append("\\end{document}\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic LaTeX document.")
    p.add_argument("loc", nargs="?", type=int, default=100, help="Approx. body line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .tex")
    args = p.parse_args()

    cfg = LatexConfig(loc=args.loc, seed=args.seed)
    code = build_latex(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated LaTeX to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
