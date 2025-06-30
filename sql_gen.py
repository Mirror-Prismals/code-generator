#!/usr/bin/env python3
# synthetic_sql.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—SQL scripts.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new statement generators
* Tracks created tables/columns for realistic INSERT/SELECT/JOIN
* --out to save directly to disk

Usage
-----
python synthetic_sql.py 50
python synthetic_sql.py 100 --seed 42 --out fake.sql
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

__version__ = "0.1.0"

@dataclass(frozen=True)
class SqlConfig:
    loc: int = 50                 # approx. number of lines
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":       0.05,
        "create_table":  0.20,
        "insert":        0.25,
        "select":        0.25,
        "join":          0.15,
        "delete":        0.10,
    })

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

LETTERS = "abcdefghijklmnopqrstuvwxyz"
SQL_TYPES = ["INT", "VARCHAR(10)", "DATE", "BOOLEAN"]

def fresh_name(rng: random.Random, length: int = 6, prefix: str = "") -> str:
    name = "".join(rng.choice(LETTERS) for _ in range(length))
    return f"{prefix}{name}"

def random_value(rng: random.Random, col_type: str) -> str:
    if col_type == "INT":
        return str(rng.randint(0, 100))
    if col_type.startswith("VARCHAR"):
        s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 8)))
        return f"'{s}'"
    if col_type == "DATE":
        # simple ISO date
        y = rng.randint(2000, 2022)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        return f"'{y:04d}-{m:02d}-{d:02d}'"
    if col_type == "BOOLEAN":
        return rng.choice(["TRUE", "FALSE"])
    return "NULL"

# ──────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["-- TODO", "-- FIXME", "-- NOTE", "-- HACK"]
    text = fresh_name(rng, rng.randint(3,8))
    return f"{rng.choice(tags)}: {text}\n"

@register("create_table")
def gen_create_table(state: Dict) -> str:
    rng = state["rng"]
    # generate table name
    tbl = fresh_name(rng, rng.randint(4,8), prefix="t_")
    # always include id INT primary key
    cols: List[Tuple[str,str]] = [("id", "INT PRIMARY KEY")]
    # add 2-4 more columns
    for _ in range(rng.randint(2,4)):
        cname = fresh_name(rng, rng.randint(3,6))
        ctype = rng.choice(SQL_TYPES)
        cols.append((cname, ctype))
    # record
    state["tables"][tbl] = cols
    # build SQL
    cols_sql = ",\n    ".join(f"{n} {t}" for n,t in cols)
    return f"CREATE TABLE {tbl} (\n    {cols_sql}\n);\n"

@register("insert")
def gen_insert(state: Dict) -> str:
    rng = state["rng"]
    if not state["tables"]:
        return ""
    tbl = rng.choice(list(state["tables"].keys()))
    cols = state["tables"][tbl]
    names = [n for n,_ in cols if not n.endswith("id")]  # include all
    # values
    vals = [random_value(rng, t.split()[0]) for _,t in cols]
    names_sql = ", ".join(n for n,_ in cols)
    vals_sql = ", ".join(vals)
    return f"INSERT INTO {tbl} ({names_sql}) VALUES ({vals_sql});\n"

@register("select")
def gen_select(state: Dict) -> str:
    rng = state["rng"]
    if not state["tables"]:
        return ""
    tbl = rng.choice(list(state["tables"].keys()))
    cols = state["tables"][tbl]
    # choose 1-3 columns
    sel = rng.sample([n for n,_ in cols], k=min(len(cols), rng.randint(1,3)))
    sel_sql = ", ".join(sel)
    # optional WHERE
    where = ""
    if rng.random() < 0.5:
        col, ctype = rng.choice(cols)
        val = random_value(rng, ctype.split()[0])
        where = f" WHERE {col} = {val}"
    return f"SELECT {sel_sql} FROM {tbl}{where};\n"

@register("join")
def gen_join(state: Dict) -> str:
    rng = state["rng"]
    if len(state["tables"]) < 2:
        return ""
    t1, t2 = rng.sample(list(state["tables"].keys()), 2)
    cols1 = state["tables"][t1]
    cols2 = state["tables"][t2]
    # assume both have id
    sel1 = rng.choice([n for n,_ in cols1])
    sel2 = rng.choice([n for n,_ in cols2])
    return (
        f"SELECT a.{sel1}, b.{sel2}\n"
        f"FROM {t1} a\n"
        f"JOIN {t2} b ON a.id = b.id;\n"
    )

@register("delete")
def gen_delete(state: Dict) -> str:
    rng = state["rng"]
    if not state["tables"]:
        return ""
    tbl = rng.choice(list(state["tables"].keys()))
    cols = state["tables"][tbl]
    col, ctype = rng.choice(cols)
    val = random_value(rng, ctype.split()[0])
    return f"DELETE FROM {tbl} WHERE {col} = {val};\n"

# ──────────────────────────────────────────────────────────────
# Build & CLI
# ──────────────────────────────────────────────────────────────

def build_sql(cfg: SqlConfig) -> str:
    rng = random.Random(cfg.seed)
    state = {
        "rng": rng,
        "tables": {}  # tbl name -> list of (col,type)
    }
    parts: List[str] = ["-- Auto-generated SQL script\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        stmt = _REGISTRY[kind](state)
        if not stmt:
            continue
        parts.append(stmt)
        lines += stmt.count("\n")

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic SQL script.")
    p.add_argument("loc", nargs="?", type=int, default=50, help="Approx. number of lines")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .sql")
    args = p.parse_args()

    cfg = SqlConfig(loc=args.loc, seed=args.seed)
    code = build_sql(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated SQL to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
