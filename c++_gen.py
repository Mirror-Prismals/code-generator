#!/usr/bin/env python3
# synthetic_cpp.py · v0.2.0
"""
Generate synthetic—yet syntactically valid—C++ source files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new statement generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_cpp.py 200
python synthetic_cpp.py 300 --seed 42 --out fake.cpp
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

__version__ = "0.2.0"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CppConfig:
    loc: int = 200
    seed: Optional[int] = None
    includes: Sequence[str] = ("<iostream>", "<string>", "<vector>")
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment": 0.1,
        "var_decl": 0.3,
        "function": 0.3,
        "class": 0.3,
    })
    max_functions: Optional[int] = None
    max_classes: Optional[int] = None


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

class IncludeManager:
    def __init__(self, headers: Sequence[str]) -> None:
        self._headers = list(dict.fromkeys(headers))

    def add(self, header: str) -> None:
        if header not in self._headers:
            self._headers.append(header)

    def render(self) -> str:
        return "".join(f"#include {hdr}\n" for hdr in self._headers) + "\n"


class NameGenerator:
    CPP_KEYWORDS = {
        "alignas","alignof","and","and_eq","asm","auto","bitand","bitor","bool",
        "break","case","catch","char","char16_t","char32_t","class","compl",
        "const","constexpr","const_cast","continue","decltype","default","delete",
        "do","double","dynamic_cast","else","enum","explicit","export","extern",
        "false","float","for","friend","goto","if","inline","int","long","mutable",
        "namespace","new","noexcept","not","not_eq","nullptr","operator","or",
        "or_eq","private","protected","public","register","reinterpret_cast",
        "return","short","signed","sizeof","static","static_assert","static_cast",
        "struct","switch","template","this","thread_local","throw","true","try",
        "typedef","typeid","typename","union","unsigned","using","virtual","void",
        "volatile","wchar_t","while","xor","xor_eq"
    }

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.reserved = set(self.CPP_KEYWORDS)

    def fresh(self, *, min_len: int = 3, max_len: int = 10, capital: bool = False) -> str:
        letters = "abcdefghijklmnopqrstuvwxyz"
        for _ in range(10_000):
            length = self.rng.randint(min_len, max_len)
            name = "".join(self.rng.choice(letters) for _ in range(length))
            if capital:
                name = name.capitalize()
            if name not in self.reserved:
                self.reserved.add(name)
                return name
        raise RuntimeError("Identifier space exhausted")


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
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    if rng.random() < 0.5:
        tag = rng.choice(["TODO", "FIXME", "NOTE", "HACK", "BUG"])
        text = rng.choice(["optimize this", "handle edge case", "refactor later", "remove debug"])
        return f"// {tag}: {text}\n"
    else:
        tag = rng.choice(["temporary", "legacy", "placeholder", "wip"])
        return f"/* {tag} */\n"

@register("var_decl")
def gen_var_decl(state: Dict) -> str:
    rng = state["rng"]
    names = state["names"]
    includes = state["includes"]
    symbols = state["symbols"]

    ctype = rng.choice(["int", "double", "auto"])
    name = names.fresh()
    symbols["variables"].add(name)

    # literal
    if rng.random() < 0.4:
        lit = str(rng.randint(0, 999))
    elif rng.random() < 0.7:
        lit = f"{rng.uniform(0,100):.2f}"
    else:
        txt = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3,8)))
        includes.add("<string>")
        lit = f'std::string("{txt}")'

    return f"{ctype} {name} = {lit};\n"

@register("function")
def gen_function(state: Dict) -> str:
    cfg      = state["cfg"]
    rng      = state["rng"]
    names    = state["names"]
    symbols  = state["symbols"]

    if cfg.max_functions is not None and len(symbols["functions"]) >= cfg.max_functions:
        return ""
    ret      = rng.choice(["int","double","void","auto"])
    fname    = names.fresh()
    symbols["functions"].add(fname)

    # params
    n = rng.randint(0,3)
    params = []
    for _ in range(n):
        ptype = rng.choice(["int","double","auto"])
        pname = names.fresh()
        symbols["variables"].add(pname)
        params.append(f"{ptype} {pname}")

    body = ["{\n"]
    if ret != "void":
        # return literal
        if rng.random() < 0.5:
            lit = str(rng.randint(0,999))
        else:
            lit = f"{rng.uniform(0,100):.2f}"
        body.append(f"    return {lit};\n")
    else:
        if rng.random() < 0.3:
            body.append("    // nothing\n")
    body.append("}\n\n")

    sig = f"{ret} {fname}({', '.join(params)}) "
    return sig + "".join(body)

@register("class")
def gen_class(state: Dict) -> str:
    cfg      = state["cfg"]
    rng      = state["rng"]
    names    = state["names"]
    symbols  = state["symbols"]

    if cfg.max_classes is not None and len(symbols["classes"]) >= cfg.max_classes:
        return ""
    cname    = names.fresh(capital=True)
    symbols["classes"].add(cname)

    # single private member
    mtype = rng.choice(["int","double"])
    mname = names.fresh()
    symbols["variables"].add(mname)

    parts = [
        f"class {cname} {{\npublic:\n",
        f"    {cname}() : {mname}({rng.randint(0,999)}) {{}}\n",
        f"    {mtype} get_{mname}() const {{ return {mname}; }}\n",
        "private:\n",
        f"    {mtype} {mname};\n",
        "};\n\n"
    ]
    return "".join(parts)


# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_cpp(cfg: CppConfig) -> str:
    rng      = random.Random(cfg.seed)
    names    = NameGenerator(rng)
    includes = IncludeManager(cfg.includes)
    symbols  = {"functions": set(), "classes": set(), "variables": set()}

    state = {
        "cfg": cfg,
        "rng": rng,
        "names": names,
        "includes": includes,
        "symbols": symbols,
    }

    parts = [
        "// Auto-generated C++ file – do not edit\n\n",
        includes.render(),
    ]
    lines = sum(p.count("\n") for p in parts)
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind  = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    # main
    parts.append("int main() {\n")
    if symbols["functions"]:
        fn = rng.choice(tuple(symbols["functions"]))
        parts.append(f"    std::cout << {fn}() << std::endl;\n")
    if symbols["classes"]:
        cls = rng.choice(tuple(symbols["classes"]))
        var = next(iter(symbols["variables"]))
        parts.append(f"    {cls} obj;\n")
        parts.append(f"    std::cout << obj.get_{var}() << std::endl;\n")
    parts.append("    return 0;\n")
    parts.append("}\n")

    return "".join(parts)


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic C++ source file.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed for deterministic output")
    p.add_argument("--max-funcs", type=int, help="Maximum functions")
    p.add_argument("--max-classes", type=int, help="Maximum classes")
    p.add_argument("--out", type=Path, help="Path to save generated code")
    args = p.parse_args()

    cfg = CppConfig(
        loc=args.loc,
        seed=args.seed,
        max_functions=args.max_funcs,
        max_classes=args.max_classes,
    )
    code = build_cpp(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated C++ to {args.out}")
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    _cli()
