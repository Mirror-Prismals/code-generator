#!/usr/bin/env python3
# synthetic_perl.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Perl source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_perl.py 200
python synthetic_perl.py 300 --seed 42 --out fake.pl
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
class PerlConfig:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":        0.10,
        "pragmas":        0.05,
        "var_decl":       0.20,
        "sub_def":        0.15,
        "conditional":    0.10,
        "loop":           0.10,
        "regex":          0.10,
        "print":          0.10,
        "array_op":       0.05,
        "hash_op":        0.05,
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

LETTERS = "abcdefghijklmnopqrstuvwxyz"
def fresh_name(rng: random.Random, length: int = 5) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(length))

def literal(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return str(rng.randint(0, 100))
    else:
        s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 8)))
        return f'"{s}"'

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["TODO", "FIXME", "NOTE", "HACK"]
    text = fresh_name(rng, rng.randint(3, 8))
    return f"# {rng.choice(tags)}: {text}\n"

@register("pragmas")
def gen_pragmas(state: Dict) -> str:
    return "use strict;\nuse warnings;\n\n"

@register("var_decl")
def gen_var_decl(state: Dict) -> str:
    rng = state["rng"]
    vtype = rng.choice(["$", "@", "%"])
    name = fresh_name(rng)
    if vtype == "$":
        val = literal(rng)
        return f"my $ {name} = {val};\n"
    elif vtype == "@":
        vals = ", ".join(literal(rng) for _ in range(rng.randint(2,5)))
        return f"my @ {name} = ({vals});\n"
    else:
        pairs = []
        for _ in range(rng.randint(1,4)):
            key = fresh_name(rng)
            val = literal(rng)
            pairs.append(f'"{key}" => {val}')
        return f"my % {name} = ({', '.join(pairs)});\n"

@register("sub_def")
def gen_sub_def(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    body = []
    # simple body: assign or print
    if rng.random() < 0.5:
        var = fresh_name(rng)
        val = literal(rng)
        body.append(f"    my $ {var} = {val};\n")
    else:
        body.append(f'    print "In {name}\\n";\n')
    return f"sub {name} {{\n{''.join(body)}}}\n\n"

@register("conditional")
def gen_conditional(state: Dict) -> str:
    rng = state["rng"]
    var = fresh_name(rng)
    val = literal(rng)
    then_stmt = f'    print "True branch\\n";\n'
    else_stmt = f'    print "False branch\\n";\n'
    return (
        f"if ($ {var} == {val}) {{\n"
        f"{then_stmt}"
        f"}} else {{\n"
        f"{else_stmt}"
        f"}}\n"
    )

@register("loop")
def gen_loop(state: Dict) -> str:
    rng = state["rng"]
    n = rng.randint(2, 6)
    var = fresh_name(rng)
    body = f'    print "{var}=$$\\n";\n'
    return (
        f"for (my $ {var} = 0; $ {var} < {n}; $ {var}++) {{\n"
        f"{body}"
        f"}}\n"
    )

@register("regex")
def gen_regex(state: Dict) -> str:
    rng = state["rng"]
    pat = fresh_name(rng, 3)
    return (
        f"my $str = {literal(rng)};\n"
        f"if ($str =~ /{pat}/) {{ print \"matched {pat}\\n\"; }}\n"
    )

@register("print")
def gen_print(state: Dict) -> str:
    rng = state["rng"]
    return f'print "Value: {literal(rng)}\\n";\n'

@register("array_op")
def gen_array_op(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    val = literal(rng)
    return f"push @{name}, {val};\n"

@register("hash_op")
def gen_hash_op(state: Dict) -> str:
    rng = state["rng"]
    name = fresh_name(rng)
    key = fresh_name(rng)
    val = literal(rng)
    return f"$ {name}{{\"{key}\"}} = {val};\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_perl(cfg: PerlConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {"cfg": cfg, "rng": rng}
    parts: List[str] = ["#!/usr/bin/perl\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        parts.append(snippet)
        lines += snippet.count("\n")

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Perl script.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated Perl")
    args = p.parse_args()

    cfg = PerlConfig(loc=args.loc, seed=args.seed)
    code = build_perl(cfg)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated Perl to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
