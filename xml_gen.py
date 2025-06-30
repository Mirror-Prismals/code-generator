#!/usr/bin/env python3
# synthetic_xml.py · v0.1.0
"""
Generate synthetic—yet well-formed—XML documents.

Major features
--------------
* Deterministic output with --seed
* Configurable number of top-level children
* Configurable max nesting depth and max children per element
* Configurable max attributes per element
* Random tag names, attribute names/values, and text content
* --out to save directly to disk

Usage
-----
python synthetic_xml.py 5            # 5 top-level children, defaults for others
python synthetic_xml.py 8 --depth 4 --breadth 3 --max-attrs 2 --seed 42 --out data.xml
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__version__ = "0.1.0"

@dataclass(frozen=True)
class XmlConfig:
    children: int = 5          # number of direct children under <root>
    depth: int = 3             # maximum nesting depth
    breadth: int = 3           # max children per element
    max_attributes: int = 2    # max attributes per element
    seed: Optional[int] = None
    out: Path | None = None

def random_tag(rng: random.Random) -> str:
    length = rng.randint(3, 8)
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))

def random_attr(rng: random.Random) -> tuple[str, str]:
    name = random_tag(rng)
    # simple value: 1–3 words
    words = rng.randint(1, 3)
    value = " ".join(random_tag(rng) for _ in range(words))
    return name, value

def random_text(rng: random.Random) -> str:
    # simple text node: 2–5 words
    words = rng.randint(2, 5)
    return " ".join(random_tag(rng) for _ in range(words))

def generate_element(rng: random.Random, cfg: XmlConfig, depth: int, indent: int = 0) -> str:
    """Recursively generate an XML element string."""
    tag = random_tag(rng)
    # attributes
    attrs = []
    for _ in range(rng.randint(0, cfg.max_attributes)):
        k, v = random_attr(rng)
        attrs.append(f'{k}="{v}"')
    attr_str = " " + " ".join(attrs) if attrs else ""
    pad = "  " * indent

    # decide if leaf or with children/text
    if depth >= cfg.depth or rng.random() < 0.3:
        # leaf: include text content
        text = random_text(rng)
        return f"{pad}<{tag}{attr_str}>{text}</{tag}>\n"
    else:
        # non-leaf: generate children
        count = rng.randint(1, cfg.breadth)
        inner = ""
        # optional mixed text
        if rng.random() < 0.3:
            inner += f"{pad}  {random_text(rng)}\n"
        for _ in range(count):
            inner += generate_element(rng, cfg, depth + 1, indent + 1)
        return f"{pad}<{tag}{attr_str}>\n{inner}{pad}</{tag}>\n"

def build_xml(cfg: XmlConfig) -> str:
    rng = random.Random(cfg.seed)
    parts = ['<?xml version="1.0" encoding="UTF-8"?>\n', "<root>\n"]
    for _ in range(cfg.children):
        parts.append(generate_element(rng, cfg, depth=0, indent=1))
    parts.append("</root>\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic XML document.")
    p.add_argument("children", nargs="?", type=int, default=5,
                   help="Number of top-level <root> children")
    p.add_argument("--depth", type=int, default=3, help="Maximum nesting depth")
    p.add_argument("--breadth", type=int, default=3, help="Max children per element")
    p.add_argument("--max-attrs", type=int, default=2, help="Max attributes per element")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated XML")
    args = p.parse_args()

    cfg = XmlConfig(
        children=args.children,
        depth=args.depth,
        breadth=args.breadth,
        max_attributes=args.max_attrs,
        seed=args.seed,
        out=args.out,
    )
    xml = build_xml(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(xml, encoding="utf-8")
        print(f"✔ Saved synthetic XML to {cfg.out}")
    else:
        sys.stdout.write(xml)

if __name__ == "__main__":
    _cli()
