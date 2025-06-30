#!/usr/bin/env python3
# synthetic_mida.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Mida source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_mida.py 200
python synthetic_mida.py 300 --seed 42 --out fake.mida
"""
from __future__ import annotations
import argparse
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List

__version__ = "0.1.0"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    loc: int = 200
    seed: int | None = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment": 0.05,
        "audicle": 0.25,
        "lyricle": 0.10,
        "blawc": 0.10,
        "loop": 0.10,
        "bunker": 0.10,
        "assign": 0.05,
        "call": 0.05,
        "function": 0.10,
        "fence": 0.05,
        "mutability": 0.05,
        "print": 0.05,
    })
    max_literal_depth: int = 2
    macro_comments: List[str] = field(default_factory=lambda: [
        "TODO: harmonize this",
        "FIXME: adjust loop count",
        "NOTE: sync channels",
        "¯\\_(ツ)_/¯",
        "left as an exercise for the reader",
    ])

# ──────────────────────────────────────────────────────────────
# Name generator
# ──────────────────────────────────────────────────────────────

class NameGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: set[str] = set()

    def fresh(self, prefix: str = "x", length: int = 4) -> str:
        for _ in range(10000):
            name = prefix + "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
            if name not in self.used:
                self.used.add(name)
                return name
        raise RuntimeError("Identifier space exhausted")

# ──────────────────────────────────────────────────────────────
# Context must be defined before use
# ──────────────────────────────────────────────────────────────

@dataclass
class Context:
    cfg: Config
    rng: random.Random
    names: NameGenerator

    def comment(self) -> str:
        return f"|| {self.rng.choice(self.cfg.macro_comments)} |>\n"

    def random_pitch(self) -> str:
        note = self.rng.choice(["C", "D", "E", "F", "G", "A", "B"])
        accidental = self.rng.choice(["", "#", "b"])
        octave = self.rng.randint(1, 6)
        return f"{note}{accidental}{octave}"

    def literal(self, depth: int = 0) -> str:
        if depth >= self.cfg.max_literal_depth or self.rng.random() < 0.5:
            return str(self.rng.randint(0, 9))
        lhs = self.literal(depth + 1)
        rhs = self.literal(depth + 1)
        op = self.rng.choice(["+", "-", "*"])
        return f"({lhs} {op} {rhs})"

# ──────────────────────────────────────────────────────────────
# Registry & generators
# ──────────────────────────────────────────────────────────────

GeneratorFn = Callable[[Context], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner

@register("comment")
def gen_comment(ctx: Context) -> str:
    return ctx.comment()

@register("audicle")
def gen_audicle(ctx: Context) -> str:
    notes = [ctx.random_pitch() for _ in range(ctx.rng.randint(1, 4))]
    tokens = []
    for n in notes:
        tokens.append(n)
        for _ in range(ctx.rng.randint(0, 2)):
            tokens.append(ctx.rng.choice([".", "-"]))
    if ctx.rng.random() < 0.3 and len(notes) > 1:
        body = "~".join(notes)
    else:
        body = " ".join(tokens)
    return f"*{body}*\n"

@register("lyricle")
def gen_lyricle(ctx: Context) -> str:
    words = ["la", "hey", "oh", "love", "Mida", "beat", "groove"]
    seq = []
    for _ in range(ctx.rng.randint(2,5)):
        seq.append(ctx.rng.choice(words))
        seq.append(ctx.rng.choice([".", "-"]))
    body = " ".join(seq).strip()
    return f"\"{body}\" ||\n"

@register("blawc")
def gen_blawc(ctx: Context) -> str:
    parts = ["`~#\n'\n"]
    for _ in range(ctx.rng.randint(1,3)):
        parts.append(gen_audicle(ctx))
    parts.append("'\n")
    return "".join(parts)

@register("loop")
def gen_loop(ctx: Context) -> str:
    count = ctx.rng.randint(2, 5)
    aud = gen_audicle(ctx).strip()
    return f"{count} {{{aud}}}\n"

@register("bunker")
def gen_bunker(ctx: Context) -> str:
    symbols = ["*|", "|*", "^|", "|^", "_|", "|_", "{|", "|}"]
    pattern = [ctx.rng.choice(symbols) for _ in range(ctx.rng.randint(2,6))]
    return "(" + " ".join(pattern) + ")\n"

@register("assign")
def gen_assign(ctx: Context) -> str:
    name = ctx.names.fresh("aud")
    aud = gen_audicle(ctx).strip()
    return f"{name} ^*~> {aud}\n"

@register("call")
def gen_call(ctx: Context) -> str:
    name = ctx.names.fresh("aud")
    return f"{name} <~*^\n"

@register("function")
def gen_function(ctx: Context) -> str:
    fname = ctx.names.fresh("fn")
    a, b = ctx.names.fresh("a"), ctx.names.fresh("b")
    return (
        f"d int \"{fname}\"(int {a}, int {b})\n"
        f"  return {{ {a} + {b} }}\n\n"
        f"{fname}({ctx.rng.randint(1,5)}, {ctx.rng.randint(1,5)}) <~*^\n"
    )

@register("fence")
def gen_fence(ctx: Context) -> str:
    chef = ctx.names.fresh("chef", 5)
    chop = ctx.names.fresh("chop", 4)
    salad = ctx.names.fresh("salad", 4)
    return (
        f"d str \"{chef}\"(\n"
        f"  $f {chop} @ (*C4 - - -*) ^*~> {chop},\n"
        f"  $f {salad} @ (*D4 - - -*) ^*~> {salad}\n"
        f")\n\n"
        f"return {{{chop} |||| {salad}}}\n\n"
        f"{chef}({chop}, {salad}) <~*^\n"
    )

@register("mutability")
def gen_mutability(ctx: Context) -> str:
    name = ctx.names.fresh("rv")
    val = ctx.rng.randint(0,9)
    lines = [f"rv {name} = {val}\n"]
    if ctx.rng.random() < 0.5:
        lines.append(f"bash {name}\n")
    if ctx.rng.random() < 0.3:
        lines.append(f"wash {name}\n")
    if ctx.rng.random() < 0.2:
        lines.append(f"mint {name}\n")
    return "".join(lines)

@register("print")
def gen_print(ctx: Context) -> str:
    msg = ctx.names.fresh("msg", 6)
    return f'p("Hello from {msg}")\n'

# ──────────────────────────────────────────────────────────────
# Build pipeline
# ──────────────────────────────────────────────────────────────

def build_mida(cfg: Config) -> str:
    rng = random.Random(cfg.seed)
    ctx = Context(cfg=cfg, rng=rng, names=NameGenerator(rng))
    parts: List[str] = [";; Auto-generated Mida – do not edit\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](ctx)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    return "".join(parts)

# ──────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Mida file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate lines target")
    p.add_argument("--seed", type=int, help="Random seed for deterministic output")
    p.add_argument("--out", type=str, help="Path to save the generated Mida code")
    args = p.parse_args()

    cfg = Config(loc=args.loc, seed=args.seed)
    code = build_mida(cfg)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✔ Saved generated Mida to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
