#!/usr/bin/env python3
# synthetic_html.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—HTML files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new element generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_html.py 200
python synthetic_html.py 300 --seed 42 --out fake.html
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
class HtmlConfig:
    loc: int = 200                # approx. number of lines
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":   0.05,
        "heading":   0.15,
        "paragraph": 0.30,
        "list":      0.10,
        "link":      0.10,
        "image":     0.10,
        "table":     0.10,
        "div":       0.10,
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

LOREM_WORDS = [
    "lorem","ipsum","dolor","sit","amet","consectetur",
    "adipiscing","elit","sed","do","eiusmod","tempor",
    "incididunt","ut","labore","et","dolore"
]

def random_text(rng: random.Random, min_w: int = 3, max_w: int = 10) -> str:
    n = rng.randint(min_w, max_w)
    words = [rng.choice(LOREM_WORDS) for _ in range(n)]
    sentence = " ".join(words).capitalize() + "."
    return sentence

def random_id(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(length))

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    text = random_text(rng, 2, 6)
    return f"    <!-- {text} -->\n"

@register("heading")
def gen_heading(state: Dict) -> str:
    rng = state["rng"]
    level = rng.choice([1, 2, 3])
    text = random_text(rng, 2, 5)[:-1]  # drop period
    return f"    <h{level}>{text}</h{level}>\n"

@register("paragraph")
def gen_paragraph(state: Dict) -> str:
    rng = state["rng"]
    text = random_text(rng, 5, 15)
    return f"    <p>{text}</p>\n"

@register("list")
def gen_list(state: Dict) -> str:
    rng = state["rng"]
    count = rng.randint(2, 5)
    items = "".join(f"        <li>{random_text(rng,3,8)[:-1]}</li>\n" for _ in range(count))
    return "    <ul>\n" + items + "    </ul>\n"

@register("link")
def gen_link(state: Dict) -> str:
    rng = state["rng"]
    href = f"https://example.com/{random_id(rng)}"
    text = random_text(rng, 1, 3)[:-1]
    return f'    <a href="{href}">{text}</a>\n'

@register("image")
def gen_image(state: Dict) -> str:
    rng = state["rng"]
    src = f"https://example.com/{random_id(rng)}.png"
    alt = random_text(rng, 1, 4)[:-1]
    return f'    <img src="{src}" alt="{alt}" />\n'

@register("table")
def gen_table(state: Dict) -> str:
    rng = state["rng"]
    rows = rng.randint(1, 3)
    cols = rng.randint(2, 4)
    # header
    header_cells = "".join(f"<th>Hdr{c+1}</th>" for c in range(cols))
    header = f"        <tr>{header_cells}</tr>\n"
    # body
    body = ""
    for r in range(rows):
        cells = "".join(f"<td>{random_text(rng,1,3)[:-1]}</td>" for _ in range(cols))
        body += f"        <tr>{cells}</tr>\n"
    return "    <table>\n" + header + body + "    </table>\n"

@register("div")
def gen_div(state: Dict) -> str:
    rng = state["rng"]
    cid = random_id(rng)
    # inner content: either paragraph or link
    inner = rng.choice(["paragraph", "link"])
    chunk = _REGISTRY[inner](state).strip()
    return f'    <div id="div-{cid}">{chunk}</div>\n'

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_html(cfg: HtmlConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {"cfg": cfg, "rng": rng}

    parts: List[str] = [
        "<!DOCTYPE html>\n",
        "<html lang=\"en\">\n",
        "<head>\n",
        "    <meta charset=\"UTF-8\">\n",
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        "    <title>Auto-generated Page</title>\n",
        "</head>\n",
        "<body>\n",
    ]
    lines = sum(p.count("\n") for p in parts)
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        parts.append(snippet)
        lines += snippet.count("\n")

    parts.append("</body>\n</html>\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic HTML file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed for deterministic output")
    p.add_argument("--out", type=Path, help="Path to save generated HTML")
    args = p.parse_args()

    cfg = HtmlConfig(loc=args.loc, seed=args.seed)
    html = build_html(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(html, encoding="utf-8")
        print(f"✔ Saved generated HTML to {args.out}")
    else:
        sys.stdout.write(html)

if __name__ == "__main__":
    _cli()
