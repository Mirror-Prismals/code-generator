#!/usr/bin/env python3
# synthetic_search_queries.py · v0.1.1
"""
Generate synthetic search-engine queries without mentioning real brands,
websites, or actual geographic places.

Major features
--------------
* Deterministic output with --seed
* Configurable number of queries (--count)
* Templates for informational, transactional, troubleshooting, comparison,
  time/weather, image/video, translation, etc.
* Generic vocabulary only (“smartphone”, “capital city”, “example error” …)
* --out to save one query per line
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

__version__ = "0.1.1"

# ────────────────────────────────────────── config
@dataclass(frozen=True)
class QueryConfig:
    count: int = 40
    seed:  int | None = None
    out:   Path | None = None

# ────────────────────────────────────────── vocabulary
WH_WORD    = ["how to", "what is", "why does", "when will", "where can I", "best way to"]
VERB       = ["reset", "install", "cook", "change", "delete", "buy", "repair", "download"]
TECH       = ["smartphone", "laptop", "wireless earbuds", "gaming console",
              "smartwatch", "tablet", "router", "digital camera"]
FOOD       = ["sourdough starter", "ramen broth", "cold brew", "tiramisu", "kimchi"]
PLACE_GEN  = ["capital city", "coastal town", "mountain village",
              "island resort", "historical site", "national park"]
E_COMM     = ["cheap", "discount", "second-hand", "free shipping", "best price"]
COMPARISON = ["vs", "versus", "compared to", "difference between"]
ERROR      = ["error code 123", "blue screen", "kernel panic", "network timeout",
              "null pointer exception"]

TEMPLATES = [
    "{WH} {VERB} {TECH}",
    "{WH} {VERB} {FOOD}",
    "weather tomorrow in {PLACE}",
    "{TECH} {ERROR} fix",
    "buy {E_COMM} {TECH}",
    "{TECH} {COMPARISON} {TECH2}",
    "flights {PLACE} to {PLACE2} {YEAR}",
    "best restaurants near {PLACE}",
    "video tutorial {VERB} {TECH}",
    "image of {PLACE} skyline {YEAR}",
    "{TECH} not turning on what to do",
    "how many calories in {FOOD}",
    "translate '{PLACE}' to spanish",
]

# ────────────────────────────────────────── helpers
def pick(rng: random.Random, items: List[str]) -> str:
    return rng.choice(items)

def build_query(rng: random.Random) -> str:
    template = pick(rng, TEMPLATES)
    subs = {
        "WH"        : pick(rng, WH_WORD),
        "VERB"      : pick(rng, VERB),
        "TECH"      : pick(rng, TECH),
        "TECH2"     : pick(rng, TECH),
        "FOOD"      : pick(rng, FOOD),
        "PLACE"     : pick(rng, PLACE_GEN),
        "PLACE2"    : pick(rng, PLACE_GEN),
        "E_COMM"    : pick(rng, E_COMM),
        "COMPARISON": pick(rng, COMPARISON),
        "ERROR"     : pick(rng, ERROR),
        "YEAR"      : str(rng.randint(2020, 2026)),
    }
    for key, val in subs.items():
        template = template.replace(f"{{{key}}}", val)
    return " ".join(template.split()).strip()

def build_queries(cfg: QueryConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    return [build_query(rng) for _ in range(cfg.count)]

# ────────────────────────────────────────── CLI
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic search queries.")
    p.add_argument("count", nargs="?", type=int, default=40,
                   help="Number of queries to generate")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--out", type=Path, help="Path to save queries (one per line)")
    args = p.parse_args()

    cfg = QueryConfig(count=args.count, seed=args.seed, out=args.out)
    queries = build_queries(cfg)
    output = "\n".join(queries) + "\n"

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(output, encoding="utf-8")
        print(f"✔ Saved {cfg.count} queries to {cfg.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    _cli()
