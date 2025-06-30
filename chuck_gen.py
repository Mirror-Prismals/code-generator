#!/usr/bin/env python3
# synthetic_chuck.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—ChucK source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_chuck.py 200
python synthetic_chuck.py 300 --seed 42 --out fake.ck
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
class ChucKConfig:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":   0.10,
        "var_decl":  0.25,
        "function":  0.20,
        "spork":     0.15,
        "ugens":     0.20,
        "time_loop": 0.10,
    })
    max_functions: Optional[int] = None


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

class NameGenerator:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.reserved = set()

    def fresh(self, prefix: str = "x", min_len: int = 3, max_len: int = 8) -> str:
        for _ in range(10_000):
            length = self.rng.randint(min_len, max_len)
            name = prefix + "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
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
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    if rng.random() < 0.5:
        tag = rng.choice(["TODO", "FIXME", "NOTE", "HACK"])
        txt = rng.choice(["tweak this", "optimize flow", "check timing", "refactor later"])
        return f"// {tag}: {txt}\n"
    else:
        tag = rng.choice(["temporary", "legacy", "placeholder", "wip"])
        return f"/* {tag} */\n"

@register("var_decl")
def gen_var_decl(state: Dict) -> str:
    rng    = state["rng"]
    names  = state["names"]
    symbols= state["symbols"]

    ctype  = rng.choice(["int", "float", "string"])
    name   = names.fresh(prefix=ctype[0])
    symbols["variables"].add(name)

    # literal
    if ctype == "int":
        lit = str(rng.randint(0, 127))
    elif ctype == "float":
        lit = f"{rng.uniform(0,1):.3f}"
    else:
        txt = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3,6)))
        lit = f'"{txt}"'

    return f"{ctype} {name} = {lit};\n"

@register("function")
def gen_function(state: Dict) -> str:
    cfg      = state["cfg"]
    rng      = state["rng"]
    names    = state["names"]
    symbols  = state["symbols"]

    if cfg.max_functions is not None and len(symbols["functions"]) >= cfg.max_functions:
        return ""
    fname    = names.fresh(prefix="fun")
    symbols["functions"].add(fname)

    # parameters
    n_params = rng.randint(0, 2)
    params   = []
    for _ in range(n_params):
        ptype = rng.choice(["int", "float"])
        pname = names.fresh(prefix=ptype[0])
        symbols["variables"].add(pname)
        params.append(f"{ptype} {pname}")

    header = f"fun void {fname}({', '.join(params)}) {{\n"
    body: List[str] = []
    # optional UGen inside function
    if rng.random() < 0.5:
        body.append("    SinOsc osc => dac;\n")
        body.append(f"    {rng.choice(['440', '880', '660'])} => osc.freq;\n")
        body.append("    1::second => now;\n")
    else:
        body.append("    // empty function body\n")
    body.append("}\n\n")
    return header + "".join(body)

@register("spork")
def gen_spork(state: Dict) -> str:
    rng     = state["rng"]
    symbols = state["symbols"]

    if not symbols["functions"]:
        return ""
    fn = rng.choice(tuple(symbols["functions"]))
    return f"spork ~ {fn}();\n"

@register("ugens")
def gen_ugens(state: Dict) -> str:
    rng    = state["rng"]
    names  = state["names"]

    ug = rng.choice(["SinOsc", "SawOsc", "TriOsc", "Noise"])
    name = names.fresh(prefix="u")
    lines = [
        f"{ug} {name} => dac;\n",
        f"{rng.uniform(0.1, 0.9):.2f} => {name}.gain;\n",
        f"{rng.choice(['440','220','330'])} => {name}.freq;\n"
    ]
    return "".join(lines)

@register("time_loop")
def gen_time_loop(state: Dict) -> str:
    rng = state["rng"]
    count = rng.randint(2,5)
    dt = rng.choice(["::second", "::ms", "::minute"])
    step = rng.randint(100, 1000) if dt == "::ms" else rng.randint(1,5)
    body = [
        "for (0 => int i; i < %d; i++) {\n" % count,
        f"    {step}{dt} => now;\n",
        "}\n"
    ]
    return "".join(body)


# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_chuck(cfg: ChucKConfig) -> str:
    rng      = random.Random(cfg.seed)
    names    = NameGenerator(rng)
    symbols  = {"functions": set(), "variables": set()}
    state    = {"cfg": cfg, "rng": rng, "names": names, "symbols": symbols}

    parts: List[str] = [
        "// Auto-generated ChucK program – do not edit\n\n"
    ]
    lines = sum(p.count("\n") for p in parts)
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind  = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    # optionally spork a main time‐loop
    parts.append("\n// end of generated code\n")
    return "".join(parts)


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic ChucK source file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed for deterministic output")
    p.add_argument("--max-funcs", type=int, help="Maximum functions to generate")
    p.add_argument("--out", type=Path, help="Path to save generated code")
    args = p.parse_args()

    cfg = ChucKConfig(
        loc=args.loc,
        seed=args.seed,
        max_functions=args.max_funcs,
    )
    code = build_chuck(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated ChucK to {args.out}")
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    _cli()
