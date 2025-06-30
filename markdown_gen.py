#!/usr/bin/env python3
# synthetic_md.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Markdown files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new element generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_md.py 200
python synthetic_md.py 300 --seed 42 --out fake.md
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
class MdConfig:
    loc: int = 200  # approximate number of lines
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "heading":      0.10,
        "paragraph":    0.30,
        "list":         0.15,
        "code_block":   0.10,
        "image":        0.05,
        "table":        0.10,
        "blockquote":   0.05,
        "link":         0.10,
        "hr":           0.05,
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

def random_sentence(rng: random.Random, min_w=4, max_w=10) -> str:
    n = rng.randint(min_w, max_w)
    words = [rng.choice(LOREM) for _ in range(n)]
    s = " ".join(words).capitalize() + "."
    return s

def random_paragraph(rng: random.Random, min_s=2, max_s=5) -> str:
    return " ".join(random_sentence(rng) for _ in range(rng.randint(min_s, max_s)))

def random_code_line(rng: random.Random) -> str:
    samples = [
        "print('Hello, world!')",
        "for i in range(10):",
        "console.log('test');",
        "<div class=\"foo\"></div>",
        "def foo(bar):",
        "return x * y;",
        "let x = 42;",
        "if (x > 0) {",
        "    x--;",
        "}",
    ]
    return rng.choice(samples)

def random_url(rng: random.Random) -> str:
    domains = ["example.com", "test.org", "demo.net"]
    path = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3,8)))
    return f"https://{rng.choice(domains)}/{path}"

def random_text(rng: random.Random, min_w=1, max_w=4) -> str:
    words = [rng.choice(LOREM) for _ in range(rng.randint(min_w, max_w))]
    return " ".join(words)

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("heading")
def gen_heading(state: Dict) -> str:
    rng = state["rng"]
    level = rng.randint(1, 3)
    text = random_text(rng, 2, 5).title()
    return f'{"#"*level} {text}\n\n'

@register("paragraph")
def gen_paragraph(state: Dict) -> str:
    rng = state["rng"]
    return random_paragraph(rng) + "\n\n"

@register("list")
def gen_list(state: Dict) -> str:
    rng = state["rng"]
    count = rng.randint(2, 5)
    ordered = rng.random() < 0.5
    lines = []
    for i in range(1, count+1):
        bullet = f"{i}." if ordered else "-"
        lines.append(f"{bullet} {random_text(rng, 2, 6)}")
    return "\n".join(lines) + "\n\n"

@register("code_block")
def gen_code_block(state: Dict) -> str:
    rng = state["rng"]
    lang = rng.choice(["python","javascript","html","bash",""])
    lines = rng.randint(2, 5)
    body = "\n".join(f"    {random_code_line(rng)}" for _ in range(lines))
    return f"```{lang}\n{body}\n```\n\n"

@register("image")
def gen_image(state: Dict) -> str:
    rng = state["rng"]
    alt = random_text(rng, 1, 4).capitalize()
    return f"![{alt}]({random_url(rng)}/image.png)\n\n"

@register("table")
def gen_table(state: Dict) -> str:
    rng = state["rng"]
    cols = rng.randint(2, 4)
    # header
    headers = [random_text(rng,1,2).title() for _ in range(cols)]
    sep = ["---"] * cols
    rows = rng.randint(1, 3)
    body = []
    for _ in range(rows):
        row = [random_text(rng,1,3) for _ in range(cols)]
        body.append("| " + " | ".join(row) + " |")
    table = (
        "| " + " | ".join(headers) + " |\n" +
        "| " + " | ".join(sep) + " |\n" +
        "\n".join(body)
    )
    return table + "\n\n"

@register("blockquote")
def gen_blockquote(state: Dict) -> str:
    rng = state["rng"]
    return "> " + random_sentence(rng) + "\n\n"

@register("link")
def gen_link(state: Dict) -> str:
    rng = state["rng"]
    text = random_text(rng,1,3).title()
    return f"[{text}]({random_url(rng)})\n\n"

@register("hr")
def gen_hr(state: Dict) -> str:
    return "---\n\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_md(cfg: MdConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {"cfg": cfg, "rng": rng}
    parts: List[str] = ["<!-- Auto-generated markdown, do not edit -->\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Markdown file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated Markdown")
    args = p.parse_args()

    cfg = MdConfig(loc=args.loc, seed=args.seed)
    code = build_md(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated Markdown to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
