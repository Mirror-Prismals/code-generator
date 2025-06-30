#!/usr/bin/env python3
# synthetic_elixir.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Elixir source files.

Major features
--------------
* Deterministic output with --seed
* Approximate line count control (--loc)
* Plugin architecture for snippet generators
* Random comments, module attributes, imports, function definitions,
  pattern matches, case/if expressions, comprehensions, and data structures
* Auto-generates an IO.inspect call at end
* --out to save directly to disk

Usage
-----
python synthetic_elixir.py 100
python synthetic_elixir.py 150 --seed 42 --loc 120 --out fake.ex
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.0"

@dataclass(frozen=True)
class ElixirConfig:
    loc: int = 100
    seed: int | None = None
    out: Path | None = None

GeneratorFn = Callable[["ElixirState"], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str):
    def decorator(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return decorator

class NameGen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used = set()
    def fresh(self, prefix: str = "") -> str:
        for _ in range(1000):
            name = prefix + "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(self.rng.randint(3,6)))
            if name not in self.used:
                self.used.add(name)
                return name
        raise RuntimeError("Identifier space exhausted")

@dataclass
class ElixirState:
    rng: random.Random
    names: NameGen
    funcs: List[str]

@register("comment")
def gen_comment(state: ElixirState) -> str:
    comments = ["TODO", "FIXME", "NOTE", "HACK", "OPTIMIZE", "REFACTOR"]
    return f"  # {state.rng.choice(comments)}: {state.names.fresh()}\n"

@register("moduledoc")
def gen_moduledoc(state: ElixirState) -> str:
    doc = "Auto-generated module"
    return f'  @moduledoc "{doc}"\n'

@register("import")
def gen_import(state: ElixirState) -> str:
    libs = ["Enum", "List", "Map"]
    lib = state.rng.choice(libs)
    return f"  import {lib}\n"

@register("alias")
def gen_alias(state: ElixirState) -> str:
    modules = ["String", "IO", "Agent"]
    mod = state.rng.choice(modules)
    return f"  alias {mod}\n"

@register("function")
def gen_function(state: ElixirState) -> str:
    name = state.names.fresh()
    state.funcs.append(name)
    arity = state.rng.randint(0, 2)
    params = ", ".join(state.names.fresh() for _ in range(arity))
    # simple body: sum params or literal
    if arity > 0:
        body = " + ".join(params.split(", "))
    else:
        body = str(state.rng.randint(0, 42))
    return (
        f"  def {name}({params}) do\n"
        f"    {body}\n"
        f"  end\n"
    )

@register("private_function")
def gen_private(state: ElixirState) -> str:
    name = state.names.fresh("_")
    arity = 1
    param = state.names.fresh()
    body = str(state.rng.choice(["nil", state.rng.randint(0,10)]))
    return (
        f"  defp {name}({param}) do\n"
        f"    {body}\n"
        f"  end\n"
    )

@register("if")
def gen_if(state: ElixirState) -> str:
    cond = state.rng.choice(["true", "false", "1 < 2", "length([]) == 0"])
    return (
        "  if " + cond + " do\n"
        "    :ok\n"
        "  else\n"
        "    :error\n"
        "  end\n"
    )

@register("case")
def gen_case(state: ElixirState) -> str:
    var = state.names.fresh()
    val = state.rng.choice([":ok", ":error", "42"])
    return (
        f"  case {val} do\n"
        "    :ok -> IO.puts(\"success\")\n"
        "    :error -> IO.puts(\"failure\")\n"
        "    _ -> IO.puts(\"other\")\n"
        "  end\n"
    )

@register("comprehension")
def gen_comprehension(state: ElixirState) -> str:
    var = state.names.fresh()
    return (
        f"  {var} = for x <- 1..{state.rng.randint(2,5)}, do: x * {state.rng.randint(2,5)}\n"
    )

@register("map_op")
def gen_map_op(state: ElixirState) -> str:
    key = state.names.fresh()
    val = state.rng.randint(0, 100)
    return (
        f"  map = %{{:{key} => {val}}}\n"
        f"  map[:{key}]\n"
    )

def build_elixir(cfg: ElixirConfig) -> str:
    rng = random.Random(cfg.seed)
    state = ElixirState(rng=rng, names=NameGen(rng), funcs=[])
    parts: List[str] = []
    parts.append("defmodule Synthetic do\n")
    lines = 1
    kinds, weights = zip(*{
        "comment":           0.10,
        "moduledoc":         0.05,
        "import":            0.05,
        "alias":             0.05,
        "function":          0.30,
        "private_function":  0.10,
        "if":                0.10,
        "case":              0.10,
        "comprehension":     0.10,
        "map_op":            0.05,
    }.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        snippet = _REGISTRY[kind](state)
        parts.append(snippet)
        lines += snippet.count("\n")

    parts.append("end\n\n")
    # call one of the generated functions
    if state.funcs:
        fn = rng.choice(state.funcs)
        parts.append(f"IO.inspect(Synthetic.{fn}({', '.join(['1' for _ in fn])}))\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic Elixir source.")
    p.add_argument("loc", nargs="?", type=int, default=100, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .ex")
    args = p.parse_args()

    cfg = ElixirConfig(loc=args.loc, seed=args.seed, out=args.out)
    code = build_elixir(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved synthetic Elixir to {cfg.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
