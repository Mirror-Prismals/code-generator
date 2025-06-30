#!/usr/bin/env python3
# synthetic_r.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—R source files.

Major features
--------------
* Deterministic output with --seed
* Approximate line-count control (--loc)
* Plugin architecture for snippet generators
* Random comments, library() calls, variable assignments (vectors, scalars,
  lists, data frames), user-defined functions, if/else blocks, for-loops,
  apply() family calls, and basic ggplot2 usage
* Auto-runs one randomly generated function at the bottom of the script
* --out to save directly to disk

Usage
-----
python synthetic_r.py 150
python synthetic_r.py 250 --seed 42 --loc 220 --out fake.R
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.0"

# ───────────────────────────────────────────────────── datatypes & helpers ─────
@dataclass(frozen=True)
class RConfig:
    loc:  int = 150
    seed: int | None = None
    out:  Path | None = None

class NameGen:
    """Generate fresh lower-snake_case R identifiers."""
    def __init__(self, rng: random.Random):
        self.rng  = rng
        self.used: set[str] = set()

    def fresh(self, length: int = 6) -> str:
        alpha = "abcdefghijklmnopqrstuvwxyz"
        for _ in range(1000):
            name = "".join(self.rng.choice(alpha) for _ in range(length))
            if name not in self.used:
                self.used.add(name)
                return name
        raise RuntimeError("Identifier space exhausted")

@dataclass
class RState:
    rng:  random.Random
    names: NameGen
    funcs: List[str]

# ───────────────────────────────────────────── registry / decorator / typedef ──
GeneratorFn = Callable[[RState], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def deco(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return deco

# ───────────────────────────────────────────────────────── snippet generators ──
@register("comment")
def gen_comment(st: RState) -> str:
    txt = "".join(st.rng.choice("abcdefghijklmnopqrstuvwxyz ")
                  for _ in range(st.rng.randint(10,30))).strip()
    return f"# {txt}\n"

@register("library")
def gen_library(st: RState) -> str:
    lib = st.rng.choice(["ggplot2", "dplyr", "tidyr", "stringr"])
    return f"library({lib})\n"

@register("assign_scalar")
def gen_scalar(st: RState) -> str:
    name = st.names.fresh()
    val  = st.rng.randint(0, 100)
    return f"{name} <- {val}\n"

@register("assign_vector")
def gen_vector(st: RState) -> str:
    name = st.names.fresh()
    vals = ", ".join(str(st.rng.randint(0, 50)) for _ in range(st.rng.randint(3,6)))
    return f"{name} <- c({vals})\n"

@register("assign_list")
def gen_list(st: RState) -> str:
    name  = st.names.fresh()
    a, b  = st.rng.randint(0,9), st.rng.randint(10,19)
    return f"{name} <- list(a = {a}, b = {b})\n"

@register("data_frame")
def gen_df(st: RState) -> str:
    name = st.names.fresh()
    col1 = ", ".join(str(st.rng.randint(0,9)) for _ in range(5))
    col2 = ", ".join(f'\"{st.names.fresh(3)}\"' for _ in range(5))
    return (
        f"{name} <- data.frame(num = c({col1}), "
        f"txt = c({col2}))\n"
    )

@register("function")
def gen_function(st: RState) -> str:
    fname  = st.names.fresh()
    st.funcs.append(fname)
    param  = st.names.fresh()
    body   = st.rng.choice([
        f"return({param} * 2)",
        f"sum({param})",
        f"mean({param})",
        f"{param}[{param} %% 2 == 0]"
    ])
    return (
        f"{fname} <- function({param}) {{\n"
        f"  {body}\n"
        f"}}\n"
    )

@register("if_block")
def gen_if(st: RState) -> str:
    var = st.names.fresh()
    return (
        f"if ({var} > {st.rng.randint(10,50)}) {{\n"
        f"  print('large')\n"
        f"}} else {{\n"
        f"  print('small')\n"
        f"}}\n"
    )

@register("for_loop")
def gen_for(st: RState) -> str:
    idx = st.names.fresh(1)
    return (
        f"for ({idx} in 1:{st.rng.randint(3,7)}) {{\n"
        f"  print({idx})\n"
        f"}}\n"
    )

@register("apply")
def gen_apply(st: RState) -> str:
    vec = st.names.fresh()
    fn  = st.names.fresh()
    return f"{fn} <- lapply({vec}, function(x) x^2)\n"

@register("ggplot")
def gen_ggplot(st: RState) -> str:
    df  = st.names.fresh()
    return (
        f"ggplot({df}, aes(x = num, y = seq_along(num))) +\n"
        f"  geom_point()\n"
    )

# ─────────────────────────────────────────────────────────── builder function ──
def build_r(cfg: RConfig) -> str:
    rng   = random.Random(cfg.seed)
    st    = RState(rng=rng, names=NameGen(rng), funcs=[])
    parts: List[str] = []
    lines = 0

    kinds, weights = zip(*{
        "comment":        0.08,
        "library":        0.05,
        "assign_scalar":  0.10,
        "assign_vector":  0.10,
        "assign_list":    0.06,
        "data_frame":     0.08,
        "function":       0.18,
        "if_block":       0.08,
        "for_loop":       0.08,
        "apply":          0.06,
        "ggplot":         0.13,
    }.items())

    while lines < cfg.loc:
        kind   = rng.choices(kinds, weights=weights, k=1)[0]
        chunk  = _REGISTRY[kind](st)
        parts.append(chunk)
        lines += chunk.count("\n")

    # ensure at least one library for ggplot chunks
    if "ggplot(" in "".join(parts) and not any("library(ggplot2)" in p for p in parts):
        parts.insert(0, "library(ggplot2)\n")

    # call a random generated function, if any, at script end
    if st.funcs:
        fn = rng.choice(st.funcs)
        var = st.names.fresh()
        parts.append(f"{var} <- {fn}(1:5)\nprint({var})\n")

    return "".join(parts)

# ───────────────────────────────────────────────────────────────────── CLI ─────
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic R code.")
    p.add_argument("loc", nargs="?", type=int, default=150,
                   help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out",  type=Path, help="Path to save generated .R")
    args = p.parse_args()

    cfg = RConfig(loc=args.loc, seed=args.seed, out=args.out)
    code = build_r(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved synthetic R script to {cfg.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
