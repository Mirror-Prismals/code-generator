#!/usr/bin/env python3
# synthetic_css.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—CSS files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_css.py 200
python synthetic_css.py 300 --seed 42 --out fake.css
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
class CssConfig:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":     0.05,
        "selector":    0.50,
        "media_query": 0.10,
        "keyframes":   0.10,
        "import":      0.05,
        "font_face":   0.10,
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

HTML_ELEMENTS = ["div", "span", "p", "h1", "h2", "ul", "li", "a", "button"]
CSS_PROPERTIES = [
    "color", "background-color", "margin", "padding", "border", "font-size",
    "width", "height", "opacity", "transform", "display", "flex", "grid-template-columns"
]

def random_color(rng: random.Random) -> str:
    return "#{:02X}{:02X}{:02X}".format(rng.randint(0,255), rng.randint(0,255), rng.randint(0,255))

def random_size(rng: random.Random) -> str:
    unit = rng.choice(["px","em","rem","%"])
    return f"{rng.randint(1,100)}{unit}"

def random_selector(rng: random.Random) -> str:
    kind = rng.choice(["element","class","id"])
    if kind == "element":
        return rng.choice(HTML_ELEMENTS)
    elif kind == "class":
        return f".{rng.choice(HTML_ELEMENTS)}-{rng.randint(1,99)}"
    else:
        return f"#{rng.choice(HTML_ELEMENTS)}-{rng.randint(1,99)}"

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["TODO", "FIXME", "NOTE", "HACK"]
    text = rng.choice(["refactor", "optimize", "debug", "cleanup"])
    return f"/* {rng.choice(tags)}: {text} */\n"

@register("selector")
def gen_selector(state: Dict) -> str:
    rng = state["rng"]
    sel = random_selector(rng)
    n_props = rng.randint(1, 5)
    props = []
    for _ in range(n_props):
        prop = rng.choice(CSS_PROPERTIES)
        if prop in ("color", "background-color"):
            val = random_color(rng)
        elif prop == "opacity":
            val = f"{rng.random():.2f}"
        elif prop == "transform":
            val = f"rotate({rng.randint(0,360)}deg)"
        else:
            val = random_size(rng)
        props.append(f"    {prop}: {val};")
    return f"{sel} {{\n" + "\n".join(props) + "\n}}\n\n"

@register("media_query")
def gen_media_query(state: Dict) -> str:
    rng = state["rng"]
    width = rng.randint(300, 1200)
    # inside media, generate 1-2 selectors
    inner = "".join(_REGISTRY["selector"](state) for _ in range(rng.randint(1,2)))
    return f"@media screen and (max-width: {width}px) {{\n" + inner + "}\n\n"

@register("keyframes")
def gen_keyframes(state: Dict) -> str:
    rng = state["rng"]
    name = f"anim{rng.randint(1,99)}"
    frames = ["0%", "50%", "100%"]
    body = ""
    for f in frames:
        # choose 1 property
        prop = rng.choice(["opacity", "transform"])
        if prop == "opacity":
            val = f"{rng.random():.2f}"
        else:
            val = f"translateX({rng.randint(-50,50)}px)"
        body += f"  {f} {{ {prop}: {val}; }}\n"
    return f"@keyframes {name} {{\n{body}}}\n\n"

@register("import")
def gen_import(state: Dict) -> str:
    rng = state["rng"]
    url = f"https://cdn.example.com/{rng.choice(['lib','theme','reset'])}.css"
    return f"@import url('{url}');\n\n"

@register("font_face")
def gen_font_face(state: Dict) -> str:
    rng = state["rng"]
    name = f"Font{rng.randint(1,5)}"
    url = f"https://fonts.example.com/{name.lower()}.woff2"
    return (
        "@font-face {\n"
        f"    font-family: '{name}';\n"
        f"    src: url('{url}') format('woff2');\n"
        "}\n\n"
    )

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_css(cfg: CssConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {"cfg": cfg, "rng": rng}
    parts: List[str] = ["/* Auto-generated CSS – do not edit */\n\n"]
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
    p = argparse.ArgumentParser(description="Generate a synthetic CSS file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated CSS")
    args = p.parse_args()

    cfg = CssConfig(loc=args.loc, seed=args.seed)
    code = build_css(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated CSS to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
