#!/usr/bin/env python3
# synthetic_captions.py · v0.1.0
"""
Generate synthetic image captions (alt-text style).

Major features
--------------
* Deterministic output with --seed
* Configurable number of captions (--count)
* Variety of templates: single-subject, subject-with-action, multi-subject,
  setting/season, emotion, perspective, weather, time-of-day
* Vocabulary pools of adjectives, subjects, actions, places, times
* --out to save captions directly to disk (one per line)

Usage
-----
python synthetic_captions.py 50
python synthetic_captions.py 200 --seed 42 --out captions.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

__version__ = "0.1.0"

# ───────────────────────────────────────────────────────── configuration
@dataclass(frozen=True)
class CapConfig:
    count: int = 50
    seed:  int | None = None
    out:   Path | None = None

# ───────────────────────────────────────────────────────── vocabulary
ADJ       = ["playful", "sleepy", "curious", "happy", "lonely", "majestic",
             "tiny", "massive", "vibrant", "serene", "rustic", "ancient"]
SUBJECT   = ["cat", "dog", "child", "elderly man", "woman", "hiker",
             "cyclist", "bird", "horse", "chef", "robot", "artist"]
ACTIONS   = ["running", "jumping", "sleeping", "reading", "painting",
             "cooking", "gazing", "smiling", "playing", "dancing"]
PLACES    = ["on a beach", "in a forest", "beside a lake", "under neon lights",
             "at a bustling market", "on a snowy mountain", "inside a cozy cafe",
             "along a cobblestone street", "in a desert", "at a carnival"]
TIMES     = ["at sunrise", "at sunset", "during golden hour", "on a foggy morning",
             "under a starry sky", "in the afternoon", "at dusk", "after rain"]
WEATHER   = ["during a light snowfall", "in heavy rain", "on a windy day",
             "beneath clear skies", "under stormy clouds"]

TEMPLATES = [
    "{A} {S}.",                                                    # simple
    "A {ADJ} {SUBJ} {PLACE}.",
    "Two {ADJ} {SUBJ_PLURAL} {ACTION} {PLACE}.",
    "Close-up of a {ADJ} {SUBJ} {ACTION}.",
    "{A} {S} {PLACE} {TIME}.",
    "{A} {S} {WEATHER}.",
    "From above: {A} {S} {PLACE}.",
    "Silhouette of a {SUBJ} {ACTION} {TIME}.",
    "{A} {S} framed by {ADJ} trees {TIME}.",
]

def article(word: str) -> str:
    return "An" if word[0].lower() in "aeiou" else "A"

def plural(subj: str) -> str:
    # simple pluralization rules
    if subj.endswith("y") and subj[-2] not in "aeiou":
        return subj[:-1] + "ies"
    if subj.endswith(("s", "x", "z")):
        return subj + "es"
    if subj.endswith(("man")):
        return subj[:-3] + "men"
    return subj + "s"

# ───────────────────────────────────────────────────────── generator
def build_caption(rng: random.Random) -> str:
    adj   = rng.choice(ADJ)
    subj  = rng.choice(SUBJECT)
    place = rng.choice(PLACES)
    time  = rng.choice(TIMES)
    weather = rng.choice(WEATHER)
    action  = rng.choice(ACTIONS)

    substitutions = {
        "ADJ": adj,
        "SUBJ": subj,
        "SUBJ_PLURAL": plural(subj),
        "ACTION": action,
        "PLACE": place,
        "TIME": time,
        "WEATHER": weather,
        # helpers
        "A": article(adj),
        "S": f"{adj} {subj}",
    }

    template = rng.choice(TEMPLATES)
    for key, val in substitutions.items():
        template = template.replace(f"{{{key}}}", val)
    return template

def build_captions(cfg: CapConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    return [build_caption(rng) for _ in range(cfg.count)]

# ───────────────────────────────────────────────────────── CLI
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic image captions.")
    p.add_argument("count", nargs="?", type=int, default=50,
                   help="Number of captions to generate")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--out", type=Path, help="Path to save captions (one per line)")
    args = p.parse_args()

    cfg = CapConfig(count=args.count, seed=args.seed, out=args.out)
    captions = build_captions(cfg)
    output = "\n".join(captions) + "\n"

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(output, encoding="utf-8")
        print(f"✔ Saved {cfg.count} captions to {cfg.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    _cli()
