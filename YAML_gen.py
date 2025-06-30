#!/usr/bin/env python3
# synthetic_yaml.py · v0.1.0
"""
Generate synthetic—yet well-formed—YAML files.

Major features
--------------
* Deterministic output with --seed
* Approximate line-count control (--lines)
* Nested mappings & sequences up to a configurable depth
* Random booleans, ints, floats, strings
* Optional anchors (&foo) and aliases (*foo) for repeated values
* --out to save directly to disk

Usage
-----
python synthetic_yaml.py 120
python synthetic_yaml.py 250 --seed 42 --depth 4 --out fake.yaml
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Optional

__version__ = "0.1.0"

# ───────────────────────────────────────────────────────── configuration
@dataclass(frozen=True)
class YamlConfig:
    lines:  int = 120     # rough target number of lines
    depth:  int = 3       # max nesting depth
    seed:   Optional[int] = None
    out:    Optional[Path] = None

# ───────────────────────────────────────────────────────── helpers
LETTERS = "abcdefghijklmnopqrstuvwxyz"

def rand_word(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(length))

def random_scalar(rng: random.Random) -> Union[int, float, str, bool]:
    choice = rng.random()
    if choice < 0.25:
        return rng.randint(0, 999)
    if choice < 0.50:
        return round(rng.uniform(0, 999), 2)
    if choice < 0.75:
        return rng.choice([True, False])
    return rand_word(rng, rng.randint(3, 8))

# ───────────────────────────────────────────────────────── generator
def gen_node(
    rng: random.Random,
    cfg: YamlConfig,
    anchors: Dict[str, str],
    depth: int,
    indent: int
) -> List[str]:
    IND = "  " * indent
    lines: List[str] = []

    # Decide node type
    if depth >= cfg.depth or rng.random() < 0.3:
        # scalar leaf
        val = random_scalar(rng)
        lines.append(f"{val}")
        return lines

    node_type = rng.choice(["map", "seq"])
    if node_type == "map":
        n_keys = rng.randint(1, 3)
        for _ in range(n_keys):
            key = rand_word(rng)
            # 10 % chance to emit an alias to a previous anchor
            if anchors and rng.random() < 0.1:
                alias = rng.choice(list(anchors.values()))
                lines.append(f"{IND}{key}: *{alias}")
                continue
            # 15 % chance to anchor this mapping
            anchor_name = None
            if rng.random() < 0.15:
                anchor_name = rand_word(rng, 4)
                anchors[key] = anchor_name
            prefix = f"{IND}{key}:" + (f" &{anchor_name}" if anchor_name else "")
            lines.append(prefix)
            child = gen_node(rng, cfg, anchors, depth + 1, indent + 1)
            # prepend indent to child lines
            for ln in child:
                lines.append(f"{IND}  {ln}")
    else:  # sequence
        n_items = rng.randint(2, 4)
        for _ in range(n_items):
            lines.append(f"{IND}- ",)
            child = gen_node(rng, cfg, anchors, depth + 1, indent + 1)
            # first line of child continues current “- ”
            first, *rest = child
            lines[-1] += first
            for ln in rest:
                lines.append(f"{IND}  {ln}")
    return lines

def build_yaml(cfg: YamlConfig) -> str:
    rng = random.Random(cfg.seed)
    anchors: Dict[str, str] = {}
    doc: List[str] = ["---"]  # start-of-document marker
    total_lines = 1

    while total_lines < cfg.lines:
        top_key = rand_word(rng)
        doc.append(f"{top_key}:")
        body = gen_node(rng, cfg, anchors, depth=1, indent=1)
        doc.extend(body)
        total_lines = len(doc)

    doc.append("...")  # end-of-document marker
    return "\n".join(doc) + "\n"

# ───────────────────────────────────────────────────────── CLI
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic YAML file.")
    p.add_argument("lines",  nargs="?", type=int, default=120,
                   help="Approximate number of lines")
    p.add_argument("--depth", type=int, default=3, help="Maximum nesting depth")
    p.add_argument("--seed",  type=int, help="Random seed")
    p.add_argument("--out",   type=Path, help="Path to save generated .yaml")
    args = p.parse_args()

    cfg = YamlConfig(lines=args.lines, depth=args.depth,
                     seed=args.seed, out=args.out)
    yaml_text = build_yaml(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(yaml_text, encoding="utf-8")
        print(f"✔ Saved synthetic YAML to {cfg.out}")
    else:
        sys.stdout.write(yaml_text)

if __name__ == "__main__":
    _cli()
