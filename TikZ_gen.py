#!/usr/bin/env python3
# synthetic_tikz.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—TikZ/LaTeX files with random shapes and annotations.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new drawing snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_tikz.py 50            # ~50 lines of TikZ code
python synthetic_tikz.py 100 --seed 42 --out drawing.tex
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

__version__ = "0.1.0"

@dataclass(frozen=True, slots=True)
class TikzConfig:
    loc: int = 50                    # approximate number of lines inside tikzpicture
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":     0.05,
        "grid":        0.05,
        "line":        0.20,
        "arrow":       0.15,
        "circle":      0.15,
        "rectangle":   0.10,
        "ellipse":     0.10,
        "node":        0.20,
    })

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

def rand_coord(rng: random.Random, low: int = -3, high: int = 3) -> float:
    return round(rng.uniform(low, high), 2)

def rand_point(rng: random.Random) -> str:
    x = rand_coord(rng)
    y = rand_coord(rng)
    return f"({x},{y})"

def rand_color(rng: random.Random) -> str:
    # choose a known TikZ color
    return rng.choice(["red","blue","green","black","orange","purple","cyan"])

def rand_label(rng: random.Random) -> str:
    # single letter or word
    return rng.choice(["A","B","C","D","E","Node","P","Q","R","X","Y","Z"])

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    texts = ["grid","axes","shapes","diagram","example","demo"]
    return f"% {rng.choice(texts)}\n"

@register("grid")
def gen_grid(state: Dict) -> str:
    rng = state["rng"]
    step = rng.randint(1,2)
    extent = rng.randint(2,5)
    return f"  \\draw[step={step}cm,gray,very thin] (-{extent},-{extent}) grid ({extent},{extent});\n"

@register("line")
def gen_line(state: Dict) -> str:
    rng = state["rng"]
    p1 = rand_point(rng)
    p2 = rand_point(rng)
    col = rand_color(rng)
    width = rng.choice(["thin","thick","ultra thick"])
    return f"  \\draw[{col},{width}] {p1} -- {p2};\n"

@register("arrow")
def gen_arrow(state: Dict) -> str:
    rng = state["rng"]
    p1 = rand_point(rng)
    p2 = rand_point(rng)
    col = rand_color(rng)
    style = rng.choice(["->","<-","<->"])
    return f"  \\draw[{col},{style}] {p1} to {p2};\n"

@register("circle")
def gen_circle(state: Dict) -> str:
    rng = state["rng"]
    center = rand_point(rng)
    r = round(rng.uniform(0.5,2.0),2)
    col = rand_color(rng)
    fill = "" if rng.random()<0.7 else f",fill={col}!20"
    return f"  \\draw[{col}{fill}] {center} circle ({r}cm);\n"

@register("rectangle")
def gen_rectangle(state: Dict) -> str:
    rng = state["rng"]
    p1 = rand_point(rng)
    p2 = rand_point(rng)
    col = rand_color(rng)
    fill = "" if rng.random()<0.7 else f",fill={col}!20"
    return f"  \\draw[{col}{fill}] {p1} rectangle {p2};\n"

@register("ellipse")
def gen_ellipse(state: Dict) -> str:
    rng = state["rng"]
    center = rand_point(rng)
    rx = round(rng.uniform(0.5,2.0),2)
    ry = round(rng.uniform(0.5,2.0),2)
    col = rand_color(rng)
    return f"  \\draw[{col}] {center} ellipse ({rx}cm and {ry}cm);\n"

@register("node")
def gen_node(state: Dict) -> str:
    rng = state["rng"]
    pos = rand_point(rng)
    label = rand_label(rng)
    opts = "" if rng.random()<0.5 else "[draw,circle]"
    return f"  \\node{opts} at {pos} {{{label}}};\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_tikz(cfg: TikzConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {"rng": rng}
    parts: List[str] = [
        "% Auto-generated TikZ drawing\n",
        "\\documentclass[tikz,border=2mm]{standalone}\n",
        "\\usepackage{tikz}\n",
        "\\begin{document}\n",
        "\\begin{tikzpicture}\n"
    ]
    lines = sum(p.count("\n") for p in parts)
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc + 4:  # +4 for begin/end lines
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if not snippet:
            continue
        parts.append(snippet)
        lines += snippet.count("\n")

    parts.append("\\end{tikzpicture}\n")
    parts.append("\\end{document}\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic TikZ/LaTeX code.")
    p.add_argument("loc", nargs="?", type=int, default=50, help="Approx. drawing lines")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .tex")
    args = p.parse_args()

    cfg = TikzConfig(loc=args.loc, seed=args.seed)
    code = build_tikz(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated TikZ to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
