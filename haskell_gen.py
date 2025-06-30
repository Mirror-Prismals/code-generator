#!/usr/bin/env python3
# synthetic_haskell.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Haskell source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_haskell.py 200
python synthetic_haskell.py 300 --seed 42 --out Fake.hs
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

__version__ = "0.1.0"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class HsConfig:
    loc: int = 200
    seed: Optional[int] = None
    modules: Sequence[str] = ("Data.List", "Data.Maybe", "Control.Monad", "System.IO")
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":     0.08,
        "import":      0.05,
        "value":       0.30,
        "function":    0.30,
        "data_type":   0.15,
        "deriving":    0.12,
    })
    max_data: Optional[int] = None

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

class ImportManager:
    def __init__(self, mods: Sequence[str]) -> None:
        self._mods = list(dict.fromkeys(mods))

    def add(self, mod: str) -> None:
        if mod not in self._mods:
            self._mods.append(mod)

    def render(self) -> str:
        return "".join(f"import {m}\n" for m in self._mods) + "\n"


class NameGenerator:
    HS_KEYWORDS = {
        "case","class","data","default","deriving","do","else","if","import",
        "in","infix","infixl","infixr","instance","let","module","newtype",
        "of","then","type","where","foreign","export","ccall","stdcall"
    }

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.reserved = set(self.HS_KEYWORDS)

    def fresh(self, *, min_len: int = 3, max_len: int = 8, capital: bool = False) -> str:
        letters = "abcdefghijklmnopqrstuvwxyz"
        for _ in range(10_000):
            length = self.rng.randint(min_len, max_len)
            name = "".join(self.rng.choice(letters) for _ in range(length))
            if capital:
                name = name.capitalize()
            if name not in self.reserved:
                self.reserved.add(name)
                return name
        raise RuntimeError("Identifier space exhausted")

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
# Literal generation
# ──────────────────────────────────────────────────────────────

def gen_literal(state: Dict, depth: int = 0) -> str:
    rng = state["rng"]
    if depth >= 2 or rng.random() < 0.4:
        t = rng.random()
        if t < 0.4:
            return str(rng.randint(0, 999))
        elif t < 0.7:
            s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 6)))
            return f"\"{s}\""
        else:
            return rng.choice(["True", "False"])
    left = gen_literal(state, depth + 1)
    right = gen_literal(state, depth + 1)
    op = rng.choice(["+", "-", "*", "++"])
    return f"({left} {op} {right})"

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["TODO", "FIXME", "NOTE", "HACK"]
    actions = ["refactor", "optimize", "handle edge case", "clean up"]
    return f"-- {rng.choice(tags)}: {rng.choice(actions)}\n"

@register("import")
def gen_import(state: Dict) -> str:
    rng = state["rng"]
    mods = state["modules_all"]
    imp = rng.choice(mods)
    state["imports"].add(imp)
    return ""  # imports only rendered at top

@register("value")
def gen_value(state: Dict) -> str:
    rng = state["rng"]
    names = state["names"]
    symbols = state["symbols"]

    name = names.fresh()
    symbols["values"].add(name)
    lit = gen_literal(state)
    return f"{name} = {lit}\n"

@register("function")
def gen_function(state: Dict) -> str:
    rng = state["rng"]
    names = state["names"]
    symbols = state["symbols"]

    fname = names.fresh()
    symbols["functions"].add(fname)

    # parameters
    n = rng.randint(0, 3)
    params = [names.fresh() for _ in range(n)]
    body = gen_literal(state)

    sig = f"{fname} {' '.join(params)} = {body}\n"
    return sig

@register("data_type")
def gen_data(state: Dict) -> str:
    cfg = state["cfg"]
    rng = state["rng"]
    names = state["names"]
    symbols = state["symbols"]

    if cfg.max_data is not None and len(symbols["data"]) >= cfg.max_data:
        return ""
    tname = names.fresh(capital=True)
    symbols["data"].add(tname)

    # constructors
    n = rng.randint(1, 3)
    ctors = []
    for _ in range(n):
        cname = names.fresh(capital=True)
        arity = rng.randint(0, 2)
        if arity:
            args = " ".join(["Int"] * arity)
            ctors.append(f"{cname} {args}")
        else:
            ctors.append(cname)
    return f"data {tname} = {' | '.join(ctors)}\n"

@register("deriving")
def gen_deriving(state: Dict) -> str:
    rng = state["rng"]
    symbols = state["symbols"]
    if not symbols["data"]:
        return ""
    t = rng.choice(tuple(symbols["data"]))
    cls = rng.choice(["Show", "Eq", "Ord"])
    return f"  deriving ({cls})\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_haskell(cfg: HsConfig) -> str:
    rng = random.Random(cfg.seed)
    names = NameGenerator(rng)
    imports = ImportManager(cfg.modules)
    symbols = {"values": set(), "functions": set(), "data": set()}

    state = {
        "cfg": cfg,
        "rng": rng,
        "names": names,
        "imports": imports,
        "symbols": symbols,
        "modules_all": cfg.modules,
    }

    parts: List[str] = [
        "module Main where\n\n",
        imports.render(),
    ]
    lines = sum(p.count("\n") for p in parts)
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if chunk:
            parts.append(chunk)
            lines += chunk.count("\n")

    # main
    parts.append("\nmain :: IO ()\n")
    parts.append("main = print ")
    if symbols["functions"]:
        parts.append(rng.choice(tuple(symbols["functions"])))
    elif symbols["values"]:
        parts.append(rng.choice(tuple(symbols["values"])))
    else:
        parts.append("()")
    parts.append("\n")

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Haskell source file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed for deterministic output")
    p.add_argument("--max-data", type=int, help="Maximum data types to generate")
    p.add_argument("--out", type=Path, help="Path to save generated code")
    args = p.parse_args()

    cfg = HsConfig(
        loc=args.loc,
        seed=args.seed,
        max_data=args.max_data,
    )
    code = build_haskell(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated Haskell to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
