#!/usr/bin/env python3
# synthetic_haiku.py · v0.4.0
"""
Generate synthetic haikus (5-7-5 syllable structure) with a richly expanded hard-coded vocabulary.

Major features
--------------
* Deterministic output with --seed
* Configurable number of haikus
* Large static word→syllable dictionary (~80 base words)
* Automatic prefix/suffix combos (~2,500 total entries)
* Backtracking to fill each line to the exact syllable count
* --out to save directly to disk

Usage
-----
python synthetic_haiku.py 5
python synthetic_haiku.py 10 --seed 42 --out my_haikus.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__version__ = "0.4.0"

@dataclass(frozen=True)
class HaikuConfig:
    count: int = 5
    seed: Optional[int] = None
    out: Path | None = None

# ──────────────────────────────────────────────────────────────────────────────
# Base vocabulary: ~80 words with manually assigned syllable counts
# ──────────────────────────────────────────────────────────────────────────────
BASE_SYLLABLES: Dict[str,int] = {
    "autumn":2, "wind":1, "whisper":2, "sun":1, "light":1,
    "river":2, "shadow":2, "dance":1, "lone":1, "mountain":2,
    "echo":2, "across":2, "soft":1, "dream":1, "morning":2,
    "butterfly":3, "murmur":2, "harmony":3, "silence":2, "twilight":2,
    "gentle":2, "blossom":2, "moon":1, "raindrop":2, "petal":2,
    "memory":3, "still":1, "sunrise":2, "sunset":2, "dark":1,
    "night":1, "star":1, "bright":1, "quiet":2, "heart":1,
    "sorrow":2, "beauty":2, "evening":3, "floating":2, "crimson":2,
    "golden":2, "forest":2, "ocean":2, "wave":1, "tide":1,
    "pebble":2, "shore":1, "meadow":2, "breeze":1, "leaf":1,
    "garden":2, "rose":1, "violet":2, "daisy":2, "lily":2,
    "lotus":2, "sunflower":3, "dragonfly":3, "birdsong":2, "feather":2,
    "wing":1, "flight":1, "cloud":1, "rain":1, "mist":1,
    "fog":1, "dew":1, "pond":1, "lake":1, "stream":1,
    "brook":1, "water":2, "flame":1, "ember":2, "smoke":1,
    "stone":1, "rock":1, "canyon":2, "desert":2, "prairie":2,
    "earth":1, "ground":1, "root":1, "seed":1, "sprout":1,
    "bark":1, "branch":1, "trunk":1, "wood":1, "time":1,
    "moment":2, "second":2, "hour":1, "day":1, "dusk":1,
    "dawn":1, "mystery":3, "solitude":3, "peace":1, "chaos":2,
    "joy":1, "tear":1, "echoes":2, "whispers":2
}

# ──────────────────────────────────────────────────────────────────────────────
# Affixes with approximate syllable counts
# ──────────────────────────────────────────────────────────────────────────────
PREFIXES: Dict[str,int] = {
    "re":1, "un":1, "in":1, "de":1
}
SUFFIXES: Dict[str,int] = {
    "ness":1, "ly":1, "ing":1, "ment":1, "ion":2, "able":2
}

# Build the full syllable lookup by combining base words with affixes
SYLLABLES: Dict[str,int] = {}
for word, syl in BASE_SYLLABLES.items():
    # original word
    SYLLABLES[word] = syl
    # with prefixes
    for pre, ps in PREFIXES.items():
        key = pre + word
        SYLLABLES[key] = syl + ps
    # with suffixes
    for suf, ss in SUFFIXES.items():
        key = word + suf
        SYLLABLES[key] = syl + ss
    # prefix + suffix combos
    for pre, ps in PREFIXES.items():
        for suf, ss in SUFFIXES.items():
            key = pre + word + suf
            SYLLABLES[key] = syl + ps + ss

# ──────────────────────────────────────────────────────────────────────────────
# Haiku generation logic
# ──────────────────────────────────────────────────────────────────────────────
def generate_line(rng: random.Random, target: int) -> str:
    """
    Build a line of exactly `target` syllables by randomly selecting words.
    Backtracks on dead ends.
    """
    for _ in range(2000):
        words: List[str] = []
        rem = target
        while rem > 0:
            # pick words fitting in rem
            candidates = [w for w,s in SYLLABLES.items() if s <= rem]
            if not candidates:
                break
            w = rng.choice(candidates)
            words.append(w)
            rem -= SYLLABLES[w]
        if rem == 0:
            return " ".join(words)
    # fallback: join what we have
    return " ".join(words)

def generate_haiku(rng: random.Random) -> Tuple[str,str,str]:
    return (
        generate_line(rng, 5),
        generate_line(rng, 7),
        generate_line(rng, 5),
    )

def build_haikus(cfg: HaikuConfig) -> List[str]:
    rng = random.Random(cfg.seed)
    result: List[str] = []
    for _ in range(cfg.count):
        l1, l2, l3 = generate_haiku(rng)
        result.extend([l1, l2, l3, ""])
    return result

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic haikus.")
    p.add_argument("count", nargs="?", type=int, default=5, help="Number of haikus")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p.add_argument("--out", type=Path, help="Path to save haikus to")
    args = p.parse_args()

    cfg = HaikuConfig(count=args.count, seed=args.seed, out=args.out)
    haikus = build_haikus(cfg)
    text = "\n".join(haikus).rstrip() + "\n"

    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(text, encoding="utf-8")
        print(f"✔ Saved {cfg.count} haikus to {cfg.out}")
    else:
        sys.stdout.write(text)

if __name__ == "__main__":
    _cli()
