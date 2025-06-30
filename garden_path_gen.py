#!/usr/bin/env python3
# synthetic_garden_path.py · v0.1.1
"""
Generate synthetic “garden path” sentences—grammatical but temporarily misleading.

Major features
--------------
* Deterministic output with --seed
* Configurable number of sentences
* Several classic garden-path templates with randomized lexicon
* Expanded lexicon for richer variety
* --out to save directly to disk

Usage
-----
python synthetic_garden_path.py 10
python synthetic_garden_path.py 15 --seed 42 --out paths.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

__version__ = "0.1.1"

@dataclass(frozen=True)
class GPConfig:
    count: int = 10
    seed: int | None = None
    out: Path | None = None

# Words that can serve as both noun and verb (for classic ambiguity)
N2V = [
    "man", "race", "police", "flock", "sail", "pilot", "duck", "ship",
    "garden", "book", "fish", "park", "dust", "paint", "hammer",
    "seed", "water", "iron", "chain", "ring", "watch", "brush", "slide",
    "tip", "help", "look", "turn", "run", "ride"
]

# Generic nouns
NOUNS = [
    "horse", "barn", "flower", "story", "child", "driver", "crew",
    "soldier", "family", "mouse", "cat", "lamp", "table", "window",
    "road", "chair", "sky", "page", "song", "leaf", "branch", "stone",
    "hill", "valley", "forest", "ocean", "breeze", "wave", "bridge",
    "desert", "meadow", "river", "cloud", "field", "house", "garden",
    "brook", "trail", "canyon", "island", "shore"
]

# Generic verbs (past tense where appropriate)
VERBS = [
    "raced", "fell", "sent", "told", "heard", "liked", "accumulates",
    "sank", "married", "cried", "painted", "parked", "fished", "seeded",
    "watered", "hammered", "dusted", "walked", "watched", "turned",
    "helped", "looked", "slid", "jumped", "listened", "smiled",
    "laughed", "sang", "pondered", "strode", "tripped", "wandered",
    "glanced"
]

# Adjectives
ADJS = [
    "old", "complex", "lonely", "silent", "burnt", "broken", "golden",
    "empty", "gentle", "quiet", "ancient", "tranquil", "rustic", "vivid",
    "delicate", "serene", "whimsical", "mysterious", "radiant",
    "melancholy", "flickering", "distant", "hollow", "verdant", "azure",
    "silver", "fragile", "luminous", "shadowy"
]

# Time/place phrases
PLACES = [
    "weekends", "Sundays", "Saturdays", "mornings", "evenings",
    "Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays",
    "midnights", "afternoons", "sunrises", "sunsets", "dawns", "dusks"
]

# Garden-path templates
def tmpl1(rng: random.Random) -> str:
    # "The old man the boats."
    adj   = rng.choice(ADJS)
    n2v   = rng.choice(N2V)
    noun2 = rng.choice(NOUNS)
    return f"The {adj} {n2v} the {noun2}."

def tmpl2(rng: random.Random) -> str:
    # "The horse raced past the barn fell."
    noun1 = rng.choice(NOUNS)
    verb1 = rng.choice(VERBS)
    noun2 = rng.choice(NOUNS)
    verb2 = rng.choice(VERBS)
    return f"The {noun1} {verb1} past the {noun2} {verb2}."

def tmpl3(rng: random.Random) -> str:
    # "The man who hunts ducks out on Sundays."
    noun1 = rng.choice(NOUNS)
    verb1 = rng.choice(VERBS)
    n2v   = rng.choice(N2V)
    place = rng.choice(PLACES)
    return f"The {noun1} who {verb1} {n2v} out on {place}."

def tmpl4(rng: random.Random) -> str:
    # "Fat people eat accumulates."
    adj   = rng.choice(ADJS)
    noun  = rng.choice(NOUNS)
    verb  = rng.choice(VERBS)
    n2v   = rng.choice(N2V)
    return f"{adj.capitalize()} {noun} {verb} {n2v}."

TEMPLATES = [tmpl1, tmpl2, tmpl3, tmpl4]

def build_sentences(cfg: GPConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    sentences: List[str] = []
    for _ in range(cfg.count):
        fn = rng.choice(TEMPLATES)
        sentences.append(fn(rng))
    return sentences

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic garden‐path sentences.")
    p.add_argument("count", nargs="?", type=int, default=10, help="Number of sentences")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save sentences to")
    args = p.parse_args()

    cfg = GPConfig(count=args.count, seed=args.seed, out=args.out)
    sents = build_sentences(cfg)
    output = "\n".join(sents) + "\n"

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(output, encoding="utf-8")
        print(f"✔ Saved {cfg.count} garden‐path sentences to {cfg.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    _cli()
