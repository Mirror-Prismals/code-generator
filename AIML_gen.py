#!/usr/bin/env python3
# synthetic_aiml.py · v0.1.0
"""
Generate synthetic AIML files with random categories, patterns, and templates.

Major features
--------------
* Deterministic output with --seed
* Configurable number of categories
* Plugin architecture for pattern and template generators
* Random lexicon of greetings, questions, and statements
* --out to save directly to disk

Usage
-----
python synthetic_aiml.py 10           # 10 categories
python synthetic_aiml.py 20 --seed 42 --out bot.aiml
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

__version__ = "0.1.0"

@dataclass(frozen=True)
class AimlConfig:
    categories: int = 10
    seed: int | None = None
    out: Path | None = None

GeneratorFn = Callable[[random.Random], str]
pattern_generators: Dict[str, GeneratorFn] = {}
template_generators: Dict[str, GeneratorFn] = {}

def register_pattern(kind: str):
    def decorator(fn: GeneratorFn) -> GeneratorFn:
        pattern_generators[kind] = fn
        return fn
    return decorator

def register_template(kind: str):
    def decorator(fn: GeneratorFn) -> GeneratorFn:
        template_generators[kind] = fn
        return fn
    return decorator

# ──────────────────────────────────────────────────────────────
# Lexicon
# ──────────────────────────────────────────────────────────────
GREETINGS = ["HELLO", "HI", "HEY", "GOOD MORNING", "GOOD EVENING"]
QUESTIONS = [
    "HOW ARE YOU", "WHAT IS YOUR NAME", "WHERE ARE YOU FROM",
    "WHAT CAN YOU DO", "TELL ME A JOKE", "WHAT TIME IS IT"
]
STATEMENTS = [
    "I LIKE MUSIC", "I LOVE PYTHON", "I ENJOY READING",
    "AI IS FUN", "CHATBOTS ARE COOL", "I AM A ROBOT"
]
RESPONSES = [
    "NICE TO MEET YOU", "I AM FINE THANK YOU", "MY NAME IS BOT",
    "I AM FROM INTERNET", "WHY DID THE CHICKEN CROSS THE ROAD",
    "IT IS TIME TO LEARN", "TELL ME MORE", "I UNDERSTAND"
]

# ──────────────────────────────────────────────────────────────
# Pattern generators
# ──────────────────────────────────────────────────────────────
@register_pattern("greeting")
def gen_greeting_pattern(rng: random.Random) -> str:
    return rng.choice(GREETINGS)

@register_pattern("question")
def gen_question_pattern(rng: random.Random) -> str:
    return rng.choice(QUESTIONS) + "?"

@register_pattern("statement")
def gen_statement_pattern(rng: random.Random) -> str:
    return rng.choice(STATEMENTS)

@register_pattern("wildcard")
def gen_wildcard_pattern(rng: random.Random) -> str:
    base = rng.choice(QUESTIONS + STATEMENTS + GREETINGS)
    # insert wildcard at random position
    parts = base.split()
    idx = rng.randint(0, len(parts))
    parts.insert(idx, "*")
    return " ".join(parts)

# ──────────────────────────────────────────────────────────────
# Template generators
# ──────────────────────────────────────────────────────────────
@register_template("simple")
def gen_simple_template(rng: random.Random) -> str:
    return rng.choice(RESPONSES)

@register_template("random_fact")
def gen_random_fact_template(rng: random.Random) -> str:
    fact = rng.choice([
        "THE SKY IS BLUE", "WATER IS WET", "FIRE IS HOT",
        "SNOW IS COLD", "CATS SAY MEOW"
    ])
    return fact

@register_template("echo")
def gen_echo_template(rng: random.Random) -> str:
    return "<srai>INPUT</srai>"

@register_template("wildcard_response")
def gen_wildcard_response(rng: random.Random) -> str:
    return "I DID NOT UNDERSTAND <star/>"

# ──────────────────────────────────────────────────────────────
# Building AIML
# ──────────────────────────────────────────────────────────────
def build_category(rng: random.Random) -> str:
    # pick random pattern and template types
    p_kind = rng.choice(list(pattern_generators))
    t_kind = rng.choice(list(template_generators))
    pattern = pattern_generators[p_kind](rng)
    template = template_generators[t_kind](rng)
    # special-case echo to use star
    if t_kind == "echo":
        template = template.replace("INPUT", pattern)
    return (
        "  <category>\n"
        f"    <pattern>{pattern}</pattern>\n"
        f"    <template>{template}</template>\n"
        "  </category>\n"
    )

def build_aiml(cfg: AimlConfig) -> str:
    rng = random.Random(cfg.seed)
    parts: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        '<aiml version="2.0">\n'
    ]
    for _ in range(cfg.categories):
        parts.append(build_category(rng))
    parts.append("</aiml>\n")
    return "".join(parts)

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic AIML file.")
    p.add_argument("categories", nargs="?", type=int, default=10,
                   help="Number of AIML <category> entries")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--out", type=Path, help="Path to save generated .aiml")
    args = p.parse_args()

    cfg = AimlConfig(categories=args.categories, seed=args.seed, out=args.out)
    aiml = build_aiml(cfg)

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(aiml, encoding="utf-8")
        print(f"✔ Saved synthetic AIML to {cfg.out}")
    else:
        sys.stdout.write(aiml)

if __name__ == "__main__":
    _cli()
