#!/usr/bin/env python3
"""
Export AB-alternating chain texts to a wide CSV.

- Reuses the A–B–A–B chain extraction from the legacy convergence scripts.
- Filters to chains with length >= min length (default: 4).
- Writes a single CSV where each row is a chain and each column is a message (msg_1, msg_2, ...).

Run:
  python scripts/export_chain_texts.py \
    --posts-csv simulation2/posts.csv \
    --out-csv simulation2/chain_texts.csv \
    --min-chain-len 4

Notes:
- Input CSV must include columns: id, tweet, user_id, comment_to, thread_id
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------
# Text cleaning (consistent with repo)
# ---------------------------------
import re

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = s.replace("\n", " ")
    s = WS_RE.sub(" ", s)
    return s.strip()


# ---------------------------------
# Data loading and chain extraction
# ---------------------------------
@dataclass
class Utterance:
    id: int
    user_id: int
    parent_id: int
    thread_id: int
    text: str


def load_sim_posts(posts_csv: Path) -> Dict[int, Utterance]:
    df = pd.read_csv(posts_csv)
    need = {"id", "tweet", "user_id", "comment_to", "thread_id"}
    if not need.issubset(df.columns):
        missing = sorted(need - set(df.columns))
        raise ValueError(f"posts.csv missing required columns: {missing}")
    df["text"] = df["tweet"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0]
    out: Dict[int, Utterance] = {}
    for _, row in df.iterrows():
        try:
            uid = int(row["id"])  # utterance id
            user = int(row["user_id"]) if pd.notna(row["user_id"]) else -1
            parent = int(row["comment_to"]) if pd.notna(row["comment_to"]) else -1
            tid = int(row["thread_id"]) if pd.notna(row["thread_id"]) else -1
            txt = str(row["text"])
        except Exception:
            continue
        out[uid] = Utterance(id=uid, user_id=user, parent_id=parent, thread_id=tid, text=txt)
    return out


def build_children(utts: Dict[int, Utterance]) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {}
    for u in utts.values():
        children.setdefault(u.parent_id, []).append(u.id)
    for v in children.values():
        v.sort()
    return children


def root_ids(utts: Dict[int, Utterance]) -> List[int]:
    return sorted([u.id for u in utts.values() if u.parent_id < 0])


def enumerate_paths(children: Dict[int, List[int]], start_id: int) -> List[List[int]]:
    out: List[List[int]] = []
    stack: List[Tuple[int, List[int]]] = [(start_id, [start_id])]
    while stack:
        node, path = stack.pop()
        kids = children.get(node, [])
        if not kids:
            out.append(path)
        else:
            for c in kids:
                stack.append((c, path + [c]))
    return out


def segment_ab_alternating(path: List[int], utts: Dict[int, Utterance], min_len: int = 3) -> List[List[int]]:
    segs: List[List[int]] = []
    i = 0
    n = len(path)

    def uid(idx: int) -> int:
        return utts[path[idx]].user_id

    while i < n - 1:
        if uid(i) == uid(i + 1):
            i += 1
            continue
        a_user = uid(i)
        b_user = uid(i + 1)
        expected = a_user
        j = i + 2
        while j < n and uid(j) == expected:
            expected = b_user if expected == a_user else a_user
            j += 1
        seg = path[i:j]
        if len(seg) >= min_len:
            segs.append(seg)
        i = j - 1
    return segs


def extract_ab_chains(children: Dict[int, List[int]], roots: List[int], utts: Dict[int, Utterance], min_len: int = 3) -> List[List[int]]:
    chains: List[List[int]] = []
    for r in roots:
        for p in enumerate_paths(children, r):
            chains.extend(segment_ab_alternating(p, utts, min_len=min_len))
    return chains


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run(posts_csv: Path, out_csv: Path, min_chain_len: int) -> int:
    # Load posts and build chains
    utts = load_sim_posts(posts_csv)
    children = build_children(utts)
    roots = root_ids(utts)
    chains = extract_ab_chains(children, roots, utts, min_len=min_chain_len)

    if not chains:
        print("No AB-alternating chains found; nothing to write.")
        return 1

    # Map chains to text lists
    chain_texts: List[List[str]] = [[utts[uid].text for uid in ch] for ch in chains]

    # Prepare header up to max length
    max_len = max(len(ch) for ch in chain_texts)
    header = [f"msg_{i+1}" for i in range(max_len)]

    # Ensure output directory exists
    ensure_dir(out_csv.parent)

    # Write CSV with jagged rows padded by empty strings
    import csv

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for texts in chain_texts:
            row = texts + [""] * (max_len - len(texts))
            w.writerow(row)

    print(f"Wrote {len(chain_texts)} chains to {out_csv} (min_len>={min_chain_len})")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Export AB-alternating chain texts to a wide CSV")
    p.add_argument("--posts-csv", type=Path, default=Path("simulation2/posts.csv"))
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--min-chain-len", type=int, default=4)
    args = p.parse_args(argv)

    out_csv = args.out_csv or (args.posts_csv.parent / "chain_texts.csv")
    return run(args.posts_csv, out_csv, args.min_chain_len)


if __name__ == "__main__":
    raise SystemExit(main())
