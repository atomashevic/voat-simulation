#!/usr/bin/env python3
"""
Alternate Convergence Entropy Analysis using scripts/entropy.py

- Extracts A–B–A–B alternating chains from simulation reply threads.
- For each chain, computes legacy entropy for all ordered pairs (i -> j), j>i,
  up to a maximum lag distance using the formulation in scripts/entropy.py.
- Produces lag-vs-entropy plots and distributional comparisons for intra vs inter pairs.

Run:
  python scripts/convergence_entropy_chains_legacy.py \
    --posts-csv simulation/posts.csv \
    --outdir simulation/convergence_entropy_legacy \
    --sigma 0.3 --max-tokens 256 --max-lag 10 --min-chain-len 3

Notes:
- Uses the entropy() module from scripts/entropy.py (Normal on 1+max cosine similarities).
- Outputs are not bounded or in bits; also provides per-token normalized variant (H/T_x).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from entropy import entropy as LegacyEntropy


# -------------------------
# Text cleaning
# -------------------------
import re

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = s.replace("\n", " ")
    s = WS_RE.sub(" ", s)
    return s.strip()


# -------------------------
# Data loading and chain extraction
# -------------------------
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
        raise ValueError(f"posts.csv must include columns: {sorted(need)}")
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


# -------------------------
# Embeddings
# -------------------------
@dataclass
class TokenEmbed:
    token_vecs: np.ndarray  # (T, D) float32
    n_tokens: int


def make_model(model_name: str, device: str):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    mdl.to(device)
    return tok, mdl, device


def compute_token_embeddings(
    tokenizer,
    model,
    device: str,
    items: List[Tuple[str, str]],
    max_tokens: int = 256,
) -> Dict[str, TokenEmbed]:
    out: Dict[str, TokenEmbed] = {}
    batch_uids: List[str] = []
    batch_texts: List[str] = []

    def flush():
        if not batch_uids:
            return
        with torch.no_grad():
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out_m = model(**enc).last_hidden_state  # (B,T,D)
            for i, uid in enumerate(batch_uids):
                ids = enc["input_ids"][i]
                vecs = out_m[i]
                # remove specials
                special_ids = set([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id])
                keep = torch.ones_like(ids, dtype=torch.bool)
                for sid in special_ids:
                    if sid is not None:
                        keep = keep & (ids != sid)
                kept = vecs[keep]
                if kept.shape[0] == 0:
                    nonpad = ids != tokenizer.pad_token_id
                    kept = vecs[nonpad]
                out[uid] = TokenEmbed(
                    token_vecs=kept.detach().cpu().numpy().astype(np.float32),
                    n_tokens=int(kept.shape[0]),
                )
        batch_uids.clear()
        batch_texts.clear()

    for uid, txt in items:
        batch_uids.append(uid)
        batch_texts.append(txt)
        if len(batch_uids) >= 32:
            flush()
    flush()
    return out


# -------------------------
# Legacy entropy compute (scripts/entropy.py)
# -------------------------
def legacy_entropy_x_to_y(ex_np: np.ndarray, ey_np: np.ndarray, sigma: float = 0.3) -> float:
    """Compute legacy entropy for x->y using Normal(1, sigma) on 1 + max cosine.

    Returns a scalar (unnormalized) as produced by scripts/entropy.py (entropy.forward with dim=-1).
    """
    ex = torch.from_numpy(ex_np)
    ey = torch.from_numpy(ey_np)
    ent = LegacyEntropy(sigma=sigma, dim=-1)
    val = ent.forward(ex, ey, dim=-1)
    return float(val.detach().cpu().numpy())


# -------------------------
# Runner
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run(posts_csv: Path, outdir: Path, model_name: str, device: str, sigma: float, max_tokens: int, max_lag: int, min_chain_len: int) -> int:
    ensure_dir(outdir)

    utts = load_sim_posts(posts_csv)
    children = build_children(utts)
    roots = root_ids(utts)
    chains = extract_ab_chains(children, roots, utts, min_len=min_chain_len)
    if not chains:
        print("No A-B alternating chains found; aborting.")
        return 1

    # Save chains exploded
    chains_csv = outdir / "chains.csv"
    with chains_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain_id", "seq_idx", "id", "user_id", "parent_id", "thread_id", "chain_len", "user_a", "user_b"])
        cid = 0
        for ch in chains:
            ua = utts[ch[0]].user_id
            ub = utts[ch[1]].user_id
            clen = len(ch)
            for i, uid in enumerate(ch):
                u = utts[uid]
                w.writerow([cid, i, u.id, u.user_id, u.parent_id, u.thread_id, clen, ua, ub])
            cid += 1

    # Unique utterances used in pairs up to max_lag
    used: set[int] = set()
    for ch in chains:
        L = len(ch)
        for i in range(L - 1):
            for j in range(i + 1, min(L, i + max_lag + 1)):
                used.add(ch[i]); used.add(ch[j])
    items = [(str(uid), utts[uid].text) for uid in sorted(used)]

    # Embeddings
    tok, mdl, dev = make_model(model_name, device)
    embeds = compute_token_embeddings(tok, mdl, dev, items, max_tokens=max_tokens)

    # Pairs + legacy entropy
    pairs_csv = outdir / "pairs_all.csv"
    with pairs_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "chain_id","thread_id","user_a","user_b","i_idx","j_idx","lag",
            "id_x","id_y","user_x","user_y","pair_type","tokens_x","tokens_y",
            "H_legacy","H_legacy_per_token","sigma","model","max_tokens"
        ])
        cid = 0
        for ch in chains:
            ua = utts[ch[0]].user_id
            ub = utts[ch[1]].user_id
            L = len(ch)
            for i in range(L - 1):
                id_x = ch[i]
                ex = embeds.get(str(id_x))
                if ex is None or ex.n_tokens < 1:
                    continue
                max_j = min(L, i + max_lag + 1)
                for j in range(i + 1, max_j):
                    id_y = ch[j]
                    ey = embeds.get(str(id_y))
                    if ey is None or ey.n_tokens < 1:
                        continue
                    Hval = legacy_entropy_x_to_y(ex.token_vecs, ey.token_vecs, sigma=sigma)
                    Hpt = float(Hval / ex.n_tokens) if ex.n_tokens > 0 else 0.0
                    ux = utts[id_x].user_id
                    uy = utts[id_y].user_id
                    ptype = "intrapersonal" if ux == uy else "interpersonal"
                    w.writerow([
                        cid, utts[id_x].thread_id, ua, ub, i, j, (j - i),
                        id_x, id_y, ux, uy, ptype, ex.n_tokens, ey.n_tokens,
                        f"{Hval:.6f}", f"{Hpt:.6f}", f"{sigma}", model_name, max_tokens
                    ])
            cid += 1

    # Plots + summary
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        df = pd.read_csv(pairs_csv)

        # Lag vs H_legacy
        plt.figure(figsize=(7,5))
        sns.stripplot(data=df, x="lag", y="H_legacy", jitter=True, alpha=0.3, size=3)
        sns.pointplot(data=df, x="lag", y="H_legacy", errorbar="ci", color="red")
        plt.xlabel("Turn distance (lag)")
        plt.ylabel("Legacy entropy (scripts/entropy.py)")
        plt.title("Lag vs Legacy Convergence Entropy")
        plt.tight_layout()
        plt.savefig(outdir / "lag_vs_entropy_legacy.png", dpi=150)
        plt.close()

        # Lag vs H_legacy_per_token
        plt.figure(figsize=(7,5))
        sns.stripplot(data=df, x="lag", y="H_legacy_per_token", jitter=True, alpha=0.3, size=3)
        sns.pointplot(data=df, x="lag", y="H_legacy_per_token", errorbar="ci", color="red")
        plt.xlabel("Turn distance (lag)")
        plt.ylabel("Legacy entropy per token")
        plt.title("Lag vs Legacy Convergence Entropy (per token)")
        plt.tight_layout()
        plt.savefig(outdir / "lag_vs_entropy_legacy_per_token.png", dpi=150)
        plt.close()

        # Distributions by pair type
        plt.figure(figsize=(7,5))
        try:
            sns.kdeplot(data=df, x="H_legacy", hue="pair_type", common_norm=False, fill=True, alpha=0.3)
        except Exception:
            pass
        plt.title("Legacy entropy: intra vs inter (distribution)")
        plt.tight_layout()
        plt.savefig(outdir / "entropy_legacy_dist_inter_vs_intra.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7,5))
        try:
            sns.ecdfplot(data=df, x="H_legacy", hue="pair_type")
        except Exception:
            for subset, color in [("interpersonal","C0"),("intrapersonal","C1")]:
                dsub = df[df["pair_type"]==subset]["H_legacy"].dropna().sort_values()
                if dsub.empty: continue
                y = np.arange(1, len(dsub)+1)/len(dsub)
                plt.step(dsub.values, y, where="post", label=subset)
            plt.legend()
        plt.title("Legacy entropy: intra vs inter (ECDF)")
        plt.tight_layout()
        plt.savefig(outdir / "entropy_legacy_ecdf_inter_vs_intra.png", dpi=150)
        plt.close()

        # Overall distributions (no hue)
        plt.figure(figsize=(7,5))
        try:
            sns.kdeplot(data=df, x="H_legacy", fill=True, alpha=0.3)
        except Exception:
            pass
        plt.hist(df["H_legacy"], bins=50, alpha=0.3)
        plt.title("Distribution of legacy entropy")
        plt.tight_layout()
        plt.savefig(outdir / "entropy_legacy_dist.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7,5))
        try:
            sns.ecdfplot(data=df, x="H_legacy")
        except Exception:
            x = np.sort(df["H_legacy"].dropna().values)
            y = np.arange(1, len(x)+1)/len(x)
            plt.step(x, y, where="post")
        plt.title("ECDF of legacy entropy")
        plt.tight_layout()
        plt.savefig(outdir / "entropy_legacy_ecdf.png", dpi=150)
        plt.close()

        # Summary JSON
        def stats(s: pd.Series) -> Dict[str, float]:
            if s.empty:
                return {"count": 0}
            return {
                "count": int(s.shape[0]),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
                "p10": float(s.quantile(0.10)),
                "p50": float(s.quantile(0.50)),
                "p90": float(s.quantile(0.90)),
            }
        summary = {
            "overall": {
                "H_legacy": stats(df["H_legacy"].dropna()),
                "H_legacy_per_token": stats(df["H_legacy_per_token"].dropna()),
            },
            "by_lag": {}
        }
        for lag, grp in df.groupby("lag"):
            summary["by_lag"][int(lag)] = {
                "H_legacy": stats(grp["H_legacy"].dropna()),
                "H_legacy_per_token": stats(grp["H_legacy_per_token"].dropna()),
            }
        (outdir / "agg_stats.json").write_text(__import__("json").dumps(summary, indent=2))
    except Exception as e:
        print(f"Warning: plotting or summary failed: {e}")

    print(f"Wrote legacy outputs to {outdir}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convergence entropy analysis using scripts/entropy.py")
    p.add_argument("--posts-csv", type=Path, default=Path("simulation/posts.csv"))
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--model", type=str, default="bert-base-uncased")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-lag", type=int, default=10)
    p.add_argument("--min-chain-len", type=int, default=3)
    args = p.parse_args(argv)

    outdir = args.outdir or (args.posts_csv.parent / "convergence_entropy_legacy")
    return run(args.posts_csv, outdir, args.model, args.device, args.sigma, args.max_tokens, args.max_lag, args.min_chain_len)


if __name__ == "__main__":
    raise SystemExit(main())
