#!/usr/bin/env python3
# synthetic_json.py · v0.1.1
"""
Generate synthetic—yet syntactically valid—JSON documents.

Major features
--------------
* Deterministic output with --seed
* Configurable number of root keys (approximate size)
* Configurable maximum nesting depth and children per object/array
* Plugin architecture for new value generators
* --out to save directly to disk

Usage
-----
python synthetic_json.py 20            # ~20 top-level keys
python synthetic_json.py 50 --seed 42 --max-depth 3 --max-children 5 --out data.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__version__ = "0.1.1"

@dataclass(frozen=True)
class JsonConfig:
    keys: int = 20
    seed: Optional[int] = None
    max_depth: int = 2
    max_children: int = 4
    weights: Dict[str, float] = field(default_factory=lambda: {
        "object":  0.2,
        "array":   0.2,
        "string":  0.25,
        "number":  0.2,
        "boolean": 0.1,
        "null":    0.05,
    })

GeneratorFn = Callable[["Context", int], Any]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner

class Context:
    def __init__(self, cfg: JsonConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def random_string(self, length: int = 8) -> str:
        return "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))

    def random_number(self) -> float:
        if self.rng.random() < 0.5:
            return self.rng.randint(0, 100)
        return round(self.rng.uniform(0, 100), 2)

@register("object")
def gen_object(ctx: Context, depth: int) -> Dict[str, Any]:
    if depth >= ctx.cfg.max_depth:
        return gen_string(ctx, depth)
    n = ctx.rng.randint(1, ctx.cfg.max_children)
    obj: Dict[str, Any] = {}
    for _ in range(n):
        key = ctx.random_string(ctx.rng.randint(3, 8))
        obj[key] = gen_value(ctx, depth + 1)
    return obj

@register("array")
def gen_array(ctx: Context, depth: int) -> List[Any]:
    if depth >= ctx.cfg.max_depth:
        return [gen_number(ctx, depth) for _ in range(ctx.rng.randint(1, ctx.cfg.max_children))]
    n = ctx.rng.randint(1, ctx.cfg.max_children)
    return [gen_value(ctx, depth + 1) for _ in range(n)]

@register("string")
def gen_string(ctx: Context, depth: int) -> str:
    return ctx.random_string(ctx.rng.randint(3, 12))

@register("number")
def gen_number(ctx: Context, depth: int) -> float:
    return ctx.random_number()

@register("boolean")
def gen_boolean(ctx: Context, depth: int) -> bool:
    return ctx.rng.choice([True, False])

@register("null")
def gen_null(ctx: Context, depth: int) -> None:
    return None

def gen_value(ctx: Context, depth: int) -> Any:
    kinds, weights = zip(*ctx.cfg.weights.items())
    kind = ctx.rng.choices(kinds, weights=weights, k=1)[0]
    if depth >= ctx.cfg.max_depth and kind in ("object", "array"):
        kind = ctx.rng.choice(["string", "number", "boolean", "null"])
    return _REGISTRY[kind](ctx, depth)

def build_json(cfg: JsonConfig) -> Any:
    ctx = Context(cfg)
    result: Dict[str, Any] = {}
    for _ in range(cfg.keys):
        key = ctx.random_string(ctx.rng.randint(3, 8))
        result[key] = gen_value(ctx, 0)
    return result

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic JSON file.")
    p.add_argument("keys", nargs="?", type=int, default=20, help="Number of top-level keys")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--max-depth", type=int, default=2, help="Maximum nesting depth")
    p.add_argument("--max-children", type=int, default=4, help="Max keys/items in objects/arrays")
    p.add_argument("--out", type=Path, help="Path to save generated JSON")
    args = p.parse_args()

    cfg = JsonConfig(
        keys=args.keys,
        seed=args.seed,
        max_depth=args.max_depth,
        max_children=args.max_children,
    )
    data = build_json(cfg)
    json_text = json.dumps(data, indent=2)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json_text, encoding="utf-8")
        print(f"✔ Saved generated JSON to {args.out}")
    else:
        sys.stdout.write(json_text + "\n")

if __name__ == "__main__":
    _cli()
