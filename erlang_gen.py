#!/usr/bin/env python3
# synthetic_erlang.py · v0.1.1
"""
Generate synthetic—yet syntactically valid—Erlang source files.

Major features
--------------
* Deterministic output with --seed
* Approximate line-count control (--loc)
* Random comments, attributes, records, functions, case/if/receive blocks,
  list comprehensions, and a runnable main/0 entry point
* --out to save directly to disk
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.1"

# ───────────────────────────────────────────────────── datatypes & helpers ─────
@dataclass(frozen=True)
class ErlangConfig:
    loc: int = 150
    seed: int | None = None
    out: Path | None = None

class NameGen:
    """Generate fresh (lower-case) Erlang identifiers/atoms."""
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def fresh(self, length: int = 6, capital: bool = False) -> str:
        alpha = "abcdefghijklmnopqrstuvwxyz"
        for _ in range(1000):
            name = "".join(self.rng.choice(alpha) for _ in range(length))
            if name not in self.used:
                self.used.add(name)
                return name.capitalize() if capital else name
        raise RuntimeError("Identifier space exhausted")

@dataclass
class ErlangState:
    rng:   random.Random
    names: NameGen
    funcs: List[str]
    exports: List[str]

# ───────────────────────────────────────────── registry / decorator / typedef ──
GeneratorFn = Callable[[ErlangState], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def deco(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return deco

# ───────────────────────────────────────────────────────── snippet generators ──
@register("comment")
def gen_comment(st: ErlangState) -> str:
    txt = "".join(st.rng.choice("abcdefghijklmnopqrstuvwxyz ")
                  for _ in range(st.rng.randint(10,35))).strip()
    return f"% {txt}\n"

@register("compile")
def gen_compile(_: ErlangState) -> str:
    return "-compile(export_all).\n"

@register("export")
def gen_export(st: ErlangState) -> str:
    if st.exports:
        return ""
    # export already collected functions plus main/0
    all_fns = st.funcs + ["main"]
    sig = ", ".join(f"{fn}/0" for fn in all_fns)
    st.exports.extend(all_fns)
    return f"-export([{sig}]).\n"

@register("record")
def gen_record(st: ErlangState) -> str:
    name  = st.names.fresh(capital=True)
    field = st.names.fresh()
    return f"-record({name.lower()}, {{ {field} = 0 }}).\n"

@register("function")
def gen_function(st: ErlangState) -> str:
    name = st.names.fresh()
    st.funcs.append(name)
    body_choice = st.rng.choice(["arith", "listcomp", "case"])
    if body_choice == "arith":
        a, b = st.rng.randint(1,50), st.rng.randint(1,50)
        body = f"{a} + {b}"
    elif body_choice == "listcomp":
        var  = st.names.fresh(length=1)
        body = f"[{var}*2 || {var} <- lists:seq(1,{st.rng.randint(3,8)})]"
    else:  # case
        body = (
            "case random:uniform(2) of\n"
            "  1 -> ok;\n"
            "  _ -> error\n"
            "end"
        )
    return f"{name}() -> {body}.\n"

@register("receive")
def gen_receive(st: ErlangState) -> str:
    tag = st.names.fresh()
    return (
        "spawn(fun() ->\n"
        f"    receive {tag} -> ok after 1000 -> timeout end\n"
        "end),\n"
    )

@register("if")
def gen_if(st: ErlangState) -> str:
    cond = st.rng.choice(["1 == 1", "2 < 1", "true"])
    return f"if {cond} -> ok; true -> error end.\n"

# ─────────────────────────────────────────────────────── builder / CLI helpers ─
def build_erlang(cfg: ErlangConfig) -> str:
    rng   = random.Random(cfg.seed)
    st    = ErlangState(rng=rng, names=NameGen(rng), funcs=[], exports=[])
    parts: List[str] = ["-module(synthetic).\n"]
    lines = 1

    kinds, weights = zip(*{
        "comment":  0.10,
        "compile":  0.05,
        "export":   0.05,
        "record":   0.05,
        "function": 0.45,
        "receive":  0.10,
        "if":       0.20,
    }.items())

    while lines < cfg.loc:
        kind   = rng.choices(kinds, weights=weights, k=1)[0]
        chunk  = _REGISTRY[kind](st)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    # ensure compile/export lines are present exactly once
    if "-compile(" not in "".join(parts):
        parts.insert(1, _REGISTRY["compile"](st))
    if not st.exports:
        parts.insert(2, _REGISTRY["export"](st))

    # main/0 entry
    target = rng.choice(st.funcs) if st.funcs else "erlang:now"
    parts.append(f"main() -> io:format(\"~p~n\", [{target}()]).\n")

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic Erlang code.")
    p.add_argument("loc", nargs="?", type=int, default=150, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .erl")
    args = p.parse_args()

    cfg = ErlangConfig(loc=args.loc, seed=args.seed, out=args.out)
    code = build_erlang(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved synthetic Erlang module to {cfg.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
