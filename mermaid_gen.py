#!/usr/bin/env python3
# synthetic_mermaid.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—Mermaid diagrams in Markdown.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new diagram generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_mermaid.py 200
python synthetic_mermaid.py 300 --seed 42 --out fake.md
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
class MermaidConfig:
    loc: int = 200               # approximate number of lines
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "flowchart":     0.25,
        "sequence":      0.20,
        "class":         0.15,
        "state":         0.15,
        "gantt":         0.15,
        "pie":           0.10,
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

def random_date(rng: random.Random) -> str:
    # fixed start for simplicity
    return "2020-01-01"

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("flowchart")
def gen_flowchart(state: Dict) -> str:
    rng = state["rng"]
    n = rng.randint(3, 6)
    nodes = [f"N{i}" for i in range(1, n+1)]
    # choose direction
    dir_ = rng.choice(["LR","TD"])
    lines = [f"```mermaid", f"flowchart {dir_}"]
    for i in range(n-1):
        a, b = nodes[i], nodes[i+1]
        lines.append(f"    {a}[\"{a}\"] --> {b}[\"{b}\"]")
    lines.append("```")
    return "\n".join(lines) + "\n\n"

@register("sequence")
def gen_sequence(state: Dict) -> str:
    rng = state["rng"]
    p = ["Alice","Bob","Carol","Dave"]
    count = rng.randint(2, 4)
    parties = rng.sample(p, k=count)
    lines = ["```mermaid", "sequenceDiagram"]
    for name in parties:
        lines.append(f"    participant {name}")
    # messages
    for _ in range(count):
        a, b = rng.sample(parties, 2)
        msg = rng.choice(["Hello","Ping","Status?","Ack"])
        lines.append(f"    {a}->>{b}: {msg}")
    lines.append("```")
    return "\n".join(lines) + "\n\n"

@register("class")
def gen_class(state: Dict) -> str:
    rng = state["rng"]
    count = rng.randint(2, 4)
    classes = [f"Class{i}" for i in range(1, count+1)]
    lines = ["```mermaid", "classDiagram"]
    for c in classes:
        lines.append(f"    class {c} {{")
        # random fields
        for _ in range(rng.randint(1, 3)):
            lines.append(f"        +field{rng.randint(1,5)}")
        lines.append("    }")
    # inheritance
    if count >= 2:
        lines.append(f"    {classes[0]} <|-- {classes[1]}")
    lines.append("```")
    return "\n".join(lines) + "\n\n"

@register("state")
def gen_state(state: Dict) -> str:
    rng = state["rng"]
    count = rng.randint(2, 4)
    states = [f"S{i}" for i in range(1, count+1)]
    lines = ["```mermaid", "stateDiagram-v2", "    [*] --> " + states[0]]
    for i in range(count-1):
        lines.append(f"    {states[i]} --> {states[i+1]} : evt{rng.randint(1,9)}")
    lines.append(f"    {states[-1]} --> [*]")
    lines.append("```")
    return "\n".join(lines) + "\n\n"

@register("gantt")
def gen_gantt(state: Dict) -> str:
    rng = state["rng"]
    tasks = [f"Task{rng.randint(1,5)}" for _ in range(rng.randint(2, 4))]
    lines = ["```mermaid", "gantt", "    title Project Timeline", "    dateFormat  YYYY-MM-DD"]
    for i, t in enumerate(tasks):
        start = random_date(rng)
        dur = rng.randint(5, 20)
        alias = f"a{i}"
        lines.append(f"    section Sec{i+1}")
        lines.append(f"    {t} :{alias}, {start}, {dur}d")
    lines.append("```")
    return "\n".join(lines) + "\n\n"

@register("pie")
def gen_pie(state: Dict) -> str:
    rng = state["rng"]
    entries = rng.randint(2, 5)
    total = 100
    parts = []
    # generate random parts summing to <=100
    remaining = total
    for i in range(entries-1):
        val = rng.randint(1, remaining - (entries - i - 1))
        parts.append(val)
        remaining -= val
    parts.append(remaining)
    lines = ["```mermaid", "pie title Distribution"]
    for idx, v in enumerate(parts, 1):
        lines.append(f"    \"Part{idx}\" : {v}")
    lines.append("```")
    return "\n".join(lines) + "\n\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_mermaid(cfg: MermaidConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {"rng": rng, "cfg": cfg}
    parts: List[str] = ["<!-- Auto-generated Mermaid diagrams -->\n\n"]
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
    p = argparse.ArgumentParser(description="Generate synthetic Mermaid Markdown.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approx. line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save output")
    args = p.parse_args()

    cfg = MermaidConfig(loc=args.loc, seed=args.seed)
    md = build_mermaid(cfg)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"✔ Saved to {args.out}")
    else:
        sys.stdout.write(md)

if __name__ == "__main__":
    _cli()
