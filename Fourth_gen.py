#!/usr/bin/env python3
# synthetic_fourth.py · v0.1.1
"""
Generate synthetic—yet syntactically valid—Fourth (Forth‐like) source files.

Major features
--------------
* Deterministic output with --seed
* Approximate word count control (--words)
* Plugin architecture for snippet generators
* Random comments, literal pushes, arithmetic, stack ops, word definitions,
  conditionals, loops, and variable ops
* --out to save directly to disk

Usage
-----
python synthetic_fourth.py 200
python synthetic_fourth.py 300 --seed 42 --words 250 --out fake.fth
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.1"

@dataclass(frozen=True)
class FourthConfig:
    words: int = 200            # approximate number of space-separated tokens
    seed: int | None = None
    out: Path | None = None

class NameGen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used = set()
    def fresh(self, prefix: str = "") -> str:
        for _ in range(1000):
            name = prefix + "".join(
                self.rng.choice("abcdefghijklmnopqrstuvwxyz")
                for _ in range(self.rng.randint(3,6))
            )
            if name not in self.used and not name.isdigit():
                self.used.add(name)
                return name
        raise RuntimeError("Identifier space exhausted")

@dataclass
class FourthState:
    rng: random.Random
    names: NameGen
    defined: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)

# Now that FourthState exists, we can define our GeneratorFn and registry
GeneratorFn = Callable[[FourthState], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str):
    """Decorator to register a snippet generator under a given key."""
    def decorator(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return decorator

@register("comment")
def gen_comment(state: FourthState) -> str:
    # Fourth uses parentheses for comments: ( this is a comment )
    text = "".join(
        state.rng.choice("abcdefghijklmnopqrstuvwxyz ")
        for _ in range(state.rng.randint(5,20))
    ).strip()
    return f"( {text} ) "

@register("push")
def gen_push(state: FourthState) -> str:
    # push integer literal
    return f"{state.rng.randint(0, 99)} "

@register("arith")
def gen_arith(state: FourthState) -> str:
    return state.rng.choice(["+ ", "- ", "* ", "/ ", "mod "])

@register("stack")
def gen_stack(state: FourthState) -> str:
    return state.rng.choice(["dup ", "drop ", "swap ", "over "])

@register("variable")
def gen_variable(state: FourthState) -> str:
    # define a variable and push its name
    name = state.names.fresh()
    state.variables.append(name)
    return f"VARIABLE {name} "

@register("var_fetch")
def gen_var_fetch(state: FourthState) -> str:
    if not state.variables:
        return ""
    name = state.rng.choice(state.variables)
    return f"{name} @ "

@register("var_store")
def gen_var_store(state: FourthState) -> str:
    if not state.variables:
        return ""
    name = state.rng.choice(state.variables)
    return f"{name} ! "

@register("define")
def gen_define(state: FourthState) -> str:
    # define a new word with a simple body
    word = state.names.fresh()
    body = ""
    for _ in range(state.rng.randint(3,6)):
        kind = state.rng.choice(["push", "arith", "stack"])
        body += _REGISTRY[kind](state)
    state.defined.append(word)
    return f": {word} {body}; "

@register("call")
def gen_call(state: FourthState) -> str:
    # call a defined word
    if not state.defined:
        return ""
    word = state.rng.choice(state.defined)
    return f"{word} "

@register("if")
def gen_if(state: FourthState) -> str:
    # simple IF ... ELSE ... THEN
    return "0= IF 1 ELSE 2 THEN "

@register("loop")
def gen_loop(state: FourthState) -> str:
    # simple DO ... LOOP
    return "1 5 DO I . LOOP "

def build_fourth(cfg: FourthConfig) -> str:
    rng = random.Random(cfg.seed)
    state = FourthState(rng=rng, names=NameGen(rng))
    parts: List[str] = []
    count = 0
    kinds, weights = zip(*{
        "comment":   0.05,
        "push":      0.20,
        "arith":     0.15,
        "stack":     0.10,
        "variable":  0.05,
        "var_fetch": 0.05,
        "var_store": 0.05,
        "define":    0.10,
        "call":      0.10,
        "if":        0.10,
        "loop":      0.05,
    }.items())

    while count < cfg.words:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        if not snippet:
            continue
        parts.append(snippet)
        count += len(snippet.strip().split())
    return "".join(parts).strip() + "\n"

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic Fourth code.")
    p.add_argument("words", nargs="?", type=int, default=200,
                   help="Approximate number of space-separated tokens")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--out", type=Path, help="Path to save generated .fth")
    args = p.parse_args()

    cfg = FourthConfig(words=args.words, seed=args.seed, out=args.out)
    code = build_fourth(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved synthetic Fourth code to {cfg.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
