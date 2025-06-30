#!/usr/bin/env python3
# synthetic_asm.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—x86-64 Intel-syntax assembly files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_asm.py 200
python synthetic_asm.py 300 --seed 42 --out fake.asm
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

__version__ = "0.1.0"

@dataclass(frozen=True, slots=True)
class AsmConfig:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":      0.10,
        "directive":    0.10,
        "data":         0.15,
        "label":        0.10,
        "instr":        0.30,
        "cond_jump":    0.10,
        "loop":         0.10,
    })

GeneratorFn = Callable[[Dict], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str):
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
    tags = ["NOTE", "TODO", "FIXME", "HACK"]
    text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(4,10)))
    return f"; {rng.choice(tags)}: {text}\n"

@register("directive")
def gen_directive(state: Dict) -> str:
    rng = state["rng"]
    choices = [
        "section .data",
        "section .text",
        "global _start",
        "extern printf",
    ]
    return rng.choice(choices) + "\n"

@register("data")
def gen_data(state: Dict) -> str:
    rng = state["rng"]
    n = state["data_count"]
    state["data_count"] += 1
    label = f"msg{n}"
    state["data_labels"].append(label)
    # pick between ascii string or quad
    if rng.random() < 0.6:
        s = "".join(rng.choice("Hello, world! ") for _ in range(rng.randint(5,15)))
        return f"{label}: db `{s}`,0x0A,0\n"
    else:
        val = rng.randint(0, 0xFFFFFFFF)
        return f"{label}: dq 0x{val:X}\n"

@register("label")
def gen_label(state: Dict) -> str:
    n = state["label_count"]
    state["label_count"] += 1
    name = f"L{n}"
    state["labels"].append(name)
    return f"{name}:\n"

@register("instr")
def gen_instr(state: Dict) -> str:
    rng = state["rng"]
    regs = state["registers"]
    # pick an instruction
    ins = rng.choice(["mov", "add", "sub", "xor", "push", "pop"])
    if ins in ("push","pop"):
        reg = rng.choice(regs)
        return f"    {ins} {reg}\n"
    dst = rng.choice(regs)
    # half the time use immediate, else reg-to-reg
    if rng.random() < 0.5:
        src = rng.choice(regs)
    else:
        src = str(rng.randint(0,255))
    return f"    {ins} {dst}, {src}\n"

@register("cond_jump")
def gen_cond_jump(state: Dict) -> str:
    rng = state["rng"]
    regs = state["registers"]
    if not state["labels"]:
        return ""  # no label to jump to yet
    lbl = rng.choice(state["labels"])
    # emit a cmp then jnz
    r1, r2 = rng.sample(regs, 2)
    return f"    cmp {r1}, {r2}\n    jne {lbl}\n"

@register("loop")
def gen_loop(state: Dict) -> str:
    rng = state["rng"]
    reg = "rcx"
    count = rng.randint(2, 10)
    n = state["loop_count"]
    state["loop_count"] += 1
    start = f"LP{n}"
    end   = f"LE{n}"
    body = (
        f"{start}:\n"
        f"    mov {reg}, {count}\n"
        f"{end}:\n"
        f"    dec {reg}\n"
        f"    jnz {end}\n"
    )
    return body

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_asm(cfg: AsmConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {
        "cfg": cfg,
        "rng": rng,
        "data_count": 0,
        "label_count": 0,
        "loop_count": 0,
        "data_labels": [],
        "labels": [],
        "registers": ["rax","rbx","rcx","rdx","rsi","rdi","r8","r9"],
    }
    parts: List[str] = ["; Auto-generated x86-64 assembly\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    # ensure we have an entry point
    parts.append("_start:\n")
    parts.append("    xor rdi, rdi\n")
    parts.append("    mov rax, 60\n")
    parts.append("    syscall\n")
    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic x86-64 assembly.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .asm")
    args = p.parse_args()

    cfg = AsmConfig(loc=args.loc, seed=args.seed)
    code = build_asm(cfg)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated assembly to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
