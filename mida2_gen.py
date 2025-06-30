#!/usr/bin/env python3
# synthetic_mida2.py · v0.1.1
"""
Generate synthetic—yet syntactically valid—Mida 2 code files,
with procedural Frog Level puzzles.

Usage
-----
python synthetic_mida2.py 200 --seed 42 --out fake.mida2
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

__version__ = "0.1.1"

@dataclass(frozen=True, slots=True)
class Mida2Config:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":      0.05,
        "frog_level":   0.10,
        "static_mix":   0.15,
        "network":      0.10,
        "api":          0.10,
        "bloodline":    0.05,
        "memory":       0.05,
        "general":      0.10,
        "oop":          0.10,
        "smiler":       0.05,
        "choreo":       0.10,
        "dawgchain":    0.05,
        "sharkdown":    0.05,
    })

# ──────────────────────────────────────────────────────────────
# Plugin registry
# ──────────────────────────────────────────────────────────────

GeneratorFn = Callable[[Dict], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str):
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner

# ──────────────────────────────────────────────────────────────
# Frog‐level: now fully procedural!
# ──────────────────────────────────────────────────────────────

@register("frog_level")
def gen_frog(state: Dict) -> str:
    rng = state["rng"]
    # choose dimensions
    rows = rng.randint(3, 5)
    cols = rng.randint(5, 8)
    # plan a Manhattan path: rights (R) and downs (D)
    moves = ["R"] * (cols - 1) + ["D"] * (rows - 1)
    rng.shuffle(moves)
    # walk the path from (0,0) to (rows-1, cols-1)
    path = [(0, 0)]
    r, c = 0, 0
    for m in moves:
        if m == "R":
            c += 1
        else:
            r += 1
        path.append((r, c))
    # build a map of row -> list of (col, symbol)
    grid: Dict[int, List[tuple[int, str]]] = {}
    for idx, (pr, pc) in enumerate(path):
        if idx == 0:
            sym = "S"
        elif idx == len(path) - 1:
            sym = "G"
        else:
            r0, c0 = path[idx - 1]
            sym = "=" if pr == r0 else "||"
        grid.setdefault(pr, []).append((pc, sym))
    # render it
    lines: List[str] = ["|| Frog Level (procedural) |>"]
    for row in range(rows):
        cells = grid.get(row, [])
        cells.sort(key=lambda x: x[0])
        row_str = "".join(f"[{sym}]" for _, sym in cells)
        lines.append("```mida")
        lines.append(row_str or "")
        lines.append("```")
        lines.append("")  # blank line
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────
# Other generators (unchanged)
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["TODO", "FIXME", "NOTE", "HACK"]
    acts = ["refine", "test", "debug", "polish"]
    return f"|| {rng.choice(tags)}: {rng.choice(acts)} |>\n"

@register("static_mix")
def gen_static_mix(state: Dict) -> str:
    rng = state["rng"]
    inst = rng.choice(["Kick","Snare","Bass","Guitar","Piano"]) + f" {rng.randint(1,4)}"
    return (
        f"'{inst}' <~*^\n"
        f"  \\ = -{rng.randint(1,8)}db\n"
        f"  p = {rng.randint(0,50)}>\n"
        f"  hc = {rng.choice(['5khz','10khz','12khz'])}\n"
        f"  lc = {rng.choice(['50hz','80hz','100hz'])}\n"
    )

@register("network")
def gen_network(state: Dict) -> str:
    return (
        "(|::::|): 255.255.255.255\n"
        "|00|\n"
        "|00|\n\n"
        "@rx -> /\\/ {01,02}\n"
        "@tx -> <> {01,02}\n"
        "(|::::|): <~*^ 255.255.255.255\n"
    )

@register("api")
def gen_api(state: Dict) -> str:
    return (
        "here\n"
        "'}'|'{'\"/\n"
        "F0|/|\\OF\n"
        ".v 9hyper-daemon0\n"
    )

@register("bloodline")
def gen_blood(state: Dict) -> str:
    return ":: ::\n*C4 E4 G4*\n"

@register("memory")
def gen_memory(state: Dict) -> str:
    return (
        ",< L/L\n"
        "  /* experimental code */\n"
        "  p(\"risky\")\n"
        ".>\n"
    )

@register("general")
def gen_general(state: Dict) -> str:
    rng = state["rng"]
    n = rng.randint(1,5)
    return (
        f"p(\"Count to {n}\")\n"
        f"range {n} {{ p(\"{rng.choice(['go','run','jump'])}\") }}\n"
    )

@register("oop")
def gen_oop(state: Dict) -> str:
    rng = state["rng"]
    cname = f"\"Cls{rng.randint(1,99)}\""
    var  = chr(rng.randint(65,90))
    return (
        f"* ^ {cname}{var}; h int 32 {{ x, y }}\n"
        f"> *\n"
        f"  §f (dx, dy) {{ x += dx; y += dy }}\n"
        f"  §f () <~*^ ^ {{ x * x + y * y }}\n"
        f");\n"
    )

@register("smiler")
def gen_smiler(state: Dict) -> str:
    return "\\*C4\\~E3 G4 - - .\\*\n"

@register("choreo")
def gen_choreo(state: Dict) -> str:
    return (
        "dancerA = 0,0\n"
        "*dancerA_hip---dancerA_ankleR*\n"
        "dancerA_hip: from 0,0 to 1,0, mag=0.5, az=45, el=0\n"
    )

@register("dawgchain")
def gen_chain(state: Dict) -> str:
    return "[Start]-->[Middle]-->[End]-->\n"

@register("sharkdown")
def gen_shark(state: Dict) -> str:
    return "\\(\\*\\| \\_ \\^\\| \\_ \\*\\| \\)\\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_mida2(cfg: Mida2Config) -> str:
    rng    = random.Random(cfg.seed)
    state  = {"cfg": cfg, "rng": rng}
    parts  = ["// Auto-generated Mida 2 – do not edit\n\n"]
    lines  = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())
    while lines < cfg.loc:
        kind  = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")
    return "".join(parts)

def _cli():
    p = argparse.ArgumentParser(description="Generate synthetic Mida 2 code.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. lines")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Write to file")
    args = p.parse_args()

    cfg  = Mida2Config(loc=args.loc, seed=args.seed)
    code = build_mida2(cfg)
    if args.out:
        args.out.parent.mkdir(exist_ok=True, parents=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
