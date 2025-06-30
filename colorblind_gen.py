#!/usr/bin/env python3
# synthetic_colorblind.py · v0.1.0
"""
Generate synthetic color‐blindness simulation statements (protanopia, deuteranopia, tritanopia).

Major features
--------------
* Deterministic output with --seed
* Configurable number of statements
* Natural‐language templates describing how a given color is perceived
* Extended palette of colors with approximate mappings
* --out to save directly to disk

Usage
-----
python synthetic_colorblind.py 10
python synthetic_colorblind.py 20 --seed 42 --out colorblind.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

__version__ = "0.1.0"

@dataclass(frozen=True)
class CBConfig:
    count: int = 10
    seed: int | None = None
    out: Path | None = None

# Palette of base colors
COLORS = [
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown",
    "grey", "black", "white", "cyan", "magenta", "lime", "maroon",
    "navy", "olive", "teal", "aqua"
]

# Approximate perceived colors for each deficiency
PROTANOPIA: Dict[str,str] = {
    "red":"dark brown", "green":"olive",    "blue":"blue",
    "yellow":"tan",     "orange":"brown",   "purple":"blue",
    "pink":"grey",      "brown":"brown",    "grey":"grey",
    "black":"black",    "white":"off-white","cyan":"cyan",
    "magenta":"grey",   "lime":"olive",     "maroon":"brown",
    "navy":"grey",      "olive":"olive",    "teal":"teal",
    "aqua":"teal"
}
DEUTERANOPIA: Dict[str,str] = {
    "red":"brown",           "green":"dark brown", "blue":"blue",
    "yellow":"brownish-yellow","orange":"brown",  "purple":"blue",
    "pink":"grey",           "brown":"brown",      "grey":"grey",
    "black":"black",         "white":"off-white",  "cyan":"cyan",
    "magenta":"grey",        "lime":"brownish-green","maroon":"brown",
    "navy":"grey",           "olive":"dark olive","teal":"teal",
    "aqua":"teal"
}
TRITANOPIA: Dict[str,str] = {
    "red":"red",              "green":"greenish-blue", "blue":"green",
    "yellow":"pinkish",       "orange":"reddish",      "purple":"red",
    "pink":"rose",            "brown":"brown",         "grey":"grey",
    "black":"black",          "white":"off-white",     "cyan":"blue-green",
    "magenta":"pink",         "lime":"teal",           "maroon":"red-brown",
    "navy":"blue",            "olive":"yellowish",     "teal":"green",
    "aqua":"green"
}

# Sentence templates
TEMPLATES = [
    "{color} looks like {p} for someone with protanopia, {d} for deuteranopia, and {t} for tritanopia.",
    "Under protanopia {color} appears {p}; under deuteranopia it appears {d}; under tritanopia it appears {t}.",
    "Someone with protanopia sees {color} as {p}, with deuteranopia as {d}, and with tritanopia as {t}.",
    "For {color}: protanopia → {p}, deuteranopia → {d}, tritanopia → {t}.",
]

def generate_statement(rng: random.Random) -> str:
    color = rng.choice(COLORS)
    p = PROTANOPIA[color]
    d = DEUTERANOPIA[color]
    t = TRITANOPIA[color]
    tmpl = rng.choice(TEMPLATES)
    return tmpl.format(color=color, p=p, d=d, t=t)

def build_statements(cfg: CBConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    return [generate_statement(rng) for _ in range(cfg.count)]

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate colorblindness simulation statements.")
    p.add_argument("count", nargs="?", type=int, default=10, help="Number of statements")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--out", type=Path, help="Path to save output")
    args = p.parse_args()

    cfg = CBConfig(count=args.count, seed=args.seed, out=args.out)
    lines = build_statements(cfg)
    output = "\n".join(lines) + "\n"

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(output, encoding="utf-8")
        print(f"✔ Saved {cfg.count} statements to {cfg.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    _cli()
