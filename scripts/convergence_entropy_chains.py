#!/usr/bin/env python3
"""
Convergence Entropy Analysis (AB chains, legacy entropy)

- Extracts A–B–A–B… alternating chains (two users, strict alternation) from simulation reply threads.
- For each chain, computes legacy-style convergence entropy for all ordered pairs (i -> j), j>i,
  up to a maximum lag using the formulation in scripts/entropy.py.
- Produces plots with x = turn distance (lag) and y = entropy (H and H/T_x), plus overall distributions and ECDFs.

Run:
  python scripts/convergence_entropy_chains.py --posts-csv simulation/posts.csv --outdir simulation/convergence_entropy

Notes:
- Uses Normal(μ=1, σ) on (1 + max cosine) per token as in scripts/entropy.py, then sums −exp(log_prob)*log_prob.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Text cleaning (consistent with repo)
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
# BERT token embeddings and sentence pooling
# -------------------------
import torch
from transformers import AutoModel, AutoTokenizer
from entropy import entropy as LegacyEntropy


@dataclass
class EmbedItem:
    token_ids: np.ndarray  # shape (T,)
    token_vecs: np.ndarray  # shape (T, D)
    sent_vec: np.ndarray  # shape (D,)
    n_tokens: int


def make_model(model_name: str, device: Optional[str] = None):
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model, device


def compute_embeddings(
    tokenizer,
    model,
    device: str,
    items: List[Tuple[str, str]],
    max_tokens: int = 256,
) -> Dict[str, EmbedItem]:
    """Compute token-level embeddings and mean-pooled sentence vectors for texts.

    items: list of (uid, text)
    Returns: uid -> EmbedItem
    """
    out: Dict[str, EmbedItem] = {}

    batch_uids: List[str] = []
    batch_texts: List[str] = []

    def flush_batch():
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
            outputs = model(**enc)
            hidden = outputs.last_hidden_state  # (B, T, D)
            # Build per-sample outputs excluding special tokens
            for i, uid in enumerate(batch_uids):
                input_ids = enc["input_ids"][i]  # (T,)
                vecs = hidden[i]  # (T, D)
                # Mask out special tokens by token ids
                special_ids = set([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id])
                keep_mask = torch.ones_like(input_ids, dtype=torch.bool)
                for sid in special_ids:
                    if sid is not None:
                        keep_mask = keep_mask & (input_ids != sid)
                kept_ids = input_ids[keep_mask]
                kept_vecs = vecs[keep_mask]
                if kept_vecs.shape[0] == 0:
                    # Fallback: keep at least non-pad tokens
                    nonpad_mask = input_ids != tokenizer.pad_token_id
                    kept_ids = input_ids[nonpad_mask]
                    kept_vecs = vecs[nonpad_mask]
                # Mean pool for sentence vector
                sent_vec = kept_vecs.mean(dim=0)
                # Move to CPU numpy
                out[uid] = EmbedItem(
                    token_ids=kept_ids.detach().cpu().numpy().astype(np.int64),
                    token_vecs=kept_vecs.detach().cpu().numpy().astype(np.float32),
                    sent_vec=sent_vec.detach().cpu().numpy().astype(np.float32),
                    n_tokens=int(kept_vecs.shape[0]),
                )
        batch_uids.clear()
        batch_texts.clear()

    # batching
    for uid, text in items:
        batch_uids.append(uid)
        batch_texts.append(text)
        if len(batch_uids) >= 32:  # batch size
            flush_batch()
    flush_batch()
    return out


def l2_normalize(a: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(a, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return a / denom


# -------------------------
# Legacy entropy via scripts/entropy.py
# -------------------------
def legacy_entropy_x_to_y(ex_np: np.ndarray, ey_np: np.ndarray, sigma: float = 0.3) -> float:
    """Compute legacy entropy for x->y using Normal(1, sigma) on 1 + max cosine similarities.

    Returns a scalar (unnormalized) as produced by scripts/entropy.py (entropy.forward with dim=-1).
    """
    ex = torch.from_numpy(ex_np)
    ey = torch.from_numpy(ey_np)
    ent = LegacyEntropy(sigma=sigma, dim=-1)
    val = ent.forward(ex, ey, dim=-1)
    return float(val.detach().cpu().numpy())


def sentence_distance(x_sent: np.ndarray, y_sent: np.ndarray) -> float:
    X = l2_normalize(x_sent.reshape(1, -1), axis=1)
    Y = l2_normalize(y_sent.reshape(1, -1), axis=1)
    sim = float(X @ Y.T)
    return 1.0 - sim


# -------------------------
# Chains extraction
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
    roots = [u.id for u in utts.values() if u.parent_id < 0]
    return sorted(roots)


def enumerate_paths(children: Dict[int, List[int]], start_id: int) -> List[List[int]]:
    """Return all root->leaf id paths starting at start_id."""
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
    """Segment a root->leaf path into maximal A-B-A-B... alternating chains with exactly two users.

    - Start only when two consecutive users differ.
    - Inside a chain, users must alternate perfectly; break otherwise.
    - Keep segments with length >= min_len.
    """
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
        expected = a_user  # user expected at index i+2
        j = i + 2
        while j < n and uid(j) == expected:
            expected = b_user if expected == a_user else a_user
            j += 1
        seg = path[i:j]
        if len(seg) >= min_len:
            segs.append(seg)
        i = j - 1  # allow overlapping starts at break point
    return segs


def extract_ab_chains(children: Dict[int, List[int]], roots: List[int], utts: Dict[int, Utterance], min_len: int = 3) -> List[List[int]]:
    chains: List[List[int]] = []
    for r in roots:
        for p in enumerate_paths(children, r):
            chains.extend(segment_ab_alternating(p, utts, min_len=min_len))
    return chains


# -------------------------
# Orchestration
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run(posts_csv: Path, outdir: Path, model_name: str, device: str, sigma: float, max_tokens: int, max_lag: int, min_chain_len: int) -> int:
    ensure_dir(outdir)
    cache_dir = outdir / "cache"
    ensure_dir(cache_dir)

    # 1) Load utterances and build paths
    utts = load_sim_posts(posts_csv)
    children = build_children(utts)
    roots = root_ids(utts)
    # 2) Build AB alternating chains (maximal segments)
    ab_chains: List[List[int]] = extract_ab_chains(children, roots, utts, min_len=min_chain_len)
    if not ab_chains:
        print("No A-B alternating chains found; aborting.")
        return 1

    # Save chains exploded to CSV
    chains_csv = outdir / "chains.csv"
    with chains_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain_id", "seq_idx", "id", "user_id", "parent_id", "thread_id", "chain_len", "user_a", "user_b"])
        cid = 0
        for p in ab_chains:
            clen = len(p)
            ua = utts[p[0]].user_id
            ub = utts[p[1]].user_id
            for i, uid in enumerate(p):
                u = utts[uid]
                w.writerow([cid, i, u.id, u.user_id, u.parent_id, u.thread_id, clen, ua, ub])
            cid += 1

    # 3) Prepare unique utterances used in pairs (all within-chain pairs up to max_lag)
    used_ids_set = set()
    for chain in ab_chains:
        L = len(chain)
        for i in range(L - 1):
            max_j = min(L, i + max_lag + 1)
            for j in range(i + 1, max_j):
                used_ids_set.add(chain[i])
                used_ids_set.add(chain[j])
    used_ids = sorted(used_ids_set)
    texts = [(str(uid), utts[uid].text) for uid in used_ids]

    # 4) Embed with BERT-only
    tokenizer, model, dev = make_model(model_name, device)
    emb: Dict[str, EmbedItem] = compute_embeddings(tokenizer, model, dev, texts, max_tokens=max_tokens)

    # 5) Compute pair metrics for all within-chain pairs up to max_lag
    pairs_csv = outdir / "pairs_all.csv"
    with pairs_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "chain_id",
                "thread_id",
                "user_a",
                "user_b",
                "i_idx",
                "j_idx",
                "lag",
                "id_x",
                "id_y",
                "user_x",
                "user_y",
                "pair_type",
                "tokens_x",
                "tokens_y",
                "H",
                "H_per_token",
                "sigma",
                "model",
                "max_tokens",
            ]
        )
        cid = 0
        for chain in ab_chains:
            ua = utts[chain[0]].user_id
            ub = utts[chain[1]].user_id
            L = len(chain)
            for i in range(L - 1):
                id_x = chain[i]
                ex = emb.get(str(id_x))
                if ex is None or ex.n_tokens < 1:
                    continue
                max_j = min(L, i + max_lag + 1)
                for j in range(i + 1, max_j):
                    id_y = chain[j]
                    ey = emb.get(str(id_y))
                    if ey is None or ey.n_tokens < 1:
                        continue
                    Hval = legacy_entropy_x_to_y(ex.token_vecs, ey.token_vecs, sigma=sigma)
                    Hpt = float(Hval / ex.n_tokens) if ex.n_tokens > 0 else 0.0
                    ux = utts[id_x].user_id
                    uy = utts[id_y].user_id
                    pair_type = "intrapersonal" if ux == uy else "interpersonal"
                    w.writerow(
                        [
                            cid,
                            utts[id_x].thread_id,
                            ua,
                            ub,
                            i,
                            j,
                            j - i,
                            id_x,
                            id_y,
                            ux,
                            uy,
                            pair_type,
                            ex.n_tokens,
                            ey.n_tokens,
                            f"{Hval:.6f}",
                            f"{Hpt:.6f}",
                            f"{sigma}",
                            model_name,
                            max_tokens,
                        ]
                    )
            cid += 1

    # 6) Aggregate + plots
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.read_csv(pairs_csv)
        # Consistent scientific style and palette
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            plt.style.use("seaborn-whitegrid")
        plt.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.6,
            "figure.dpi": 150,
        })

        # Ensure pair_type exists
        if "pair_type" not in df.columns:
            df["pair_type"] = np.where(df["user_x"] == df["user_y"], "intrapersonal", "interpersonal")

        # Lag vs Entropy (H): blue/orange lines by pair_type
        pal = {"interpersonal": "#1f77b4", "intrapersonal": "#ff7f0e"}
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for ptype in ("interpersonal", "intrapersonal"):
            grp = df[df["pair_type"] == ptype]
            if grp.empty:
                continue
            g = grp.groupby("lag")["H"].median().sort_index()
            ax.plot(g.index.values, g.values, label=ptype.capitalize(), color=pal.get(ptype, None))
        ax.set_xlabel("Turn distance (lag)")
        ax.set_ylabel("Convergence entropy H")
        ax.set_title("Lag vs Convergence Entropy (H)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "lag_vs_entropy.png")
        plt.close(fig)

        # Lag vs Entropy (H per token): blue/orange lines by pair_type
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for ptype in ("interpersonal", "intrapersonal"):
            grp = df[df["pair_type"] == ptype]
            if grp.empty:
                continue
            g = grp.groupby("lag")["H_per_token"].median().sort_index()
            ax.plot(g.index.values, g.values, label=ptype.capitalize(), color=pal.get(ptype, None))
        ax.set_xlabel("Turn distance (lag)")
        ax.set_ylabel("Convergence entropy per token (H/T_x)")
        ax.set_title("Lag vs Convergence Entropy (per token)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "lag_vs_entropy_per_token.png")
        plt.close(fig)

        # Distribution plots (H) — thick-line KDE, scientific style
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        try:
            sns.kdeplot(data=df, x="H", color="#1f77b4", lw=2.6, fill=False, ax=ax)
        except Exception:
            # fallback: simple histogram outline
            vals = df["H"].dropna().values
            ax.hist(vals, bins=50, histtype="step", color="#1f77b4")
        ax.set_xlabel("Convergence entropy H")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of convergence entropy (H)")
        fig.tight_layout()
        fig.savefig(outdir / "entropy_dist.png")
        plt.close(fig)

        # Distribution plots (H per token) — thick-line KDE
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        try:
            sns.kdeplot(data=df, x="H_per_token", color="#ff7f0e", lw=2.6, fill=False, ax=ax)
        except Exception:
            vals = df["H_per_token"].dropna().values
            ax.hist(vals, bins=50, histtype="step", color="#ff7f0e")
        ax.set_xlabel("Convergence entropy per token (H/T_x)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of convergence entropy per token (H/T_x)")
        fig.tight_layout()
        fig.savefig(outdir / "entropy_dist_per_token.png")
        plt.close(fig)

        # ECDFs (H and H per token)
        plt.figure(figsize=(7, 5))
        try:
            sns.ecdfplot(data=df, x="H")
        except Exception:
            x = np.sort(df["H"].dropna().values)
            y = np.arange(1, len(x) + 1) / len(x)
            plt.step(x, y, where="post")
        plt.title("ECDF of convergence entropy (H)")
        plt.tight_layout()
        plt.savefig(outdir / "entropy_ecdf.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        try:
            sns.ecdfplot(data=df, x="H_per_token")
        except Exception:
            x = np.sort(df["H_per_token"].dropna().values)
            y = np.arange(1, len(x) + 1) / len(x)
            plt.step(x, y, where="post")
        plt.title("ECDF of convergence entropy per token (H/T_x)")
        plt.tight_layout()
        plt.savefig(outdir / "entropy_ecdf_per_token.png", dpi=150)
        plt.close()

        # Distributional comparison: intra vs inter (H)
        if "pair_type" not in df.columns:
            df["pair_type"] = np.where(df["user_x"] == df["user_y"], "intrapersonal", "interpersonal")

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        try:
            sns.kdeplot(data=df, x="H", hue="pair_type", common_norm=False, fill=False, lw=2.6, palette=pal, ax=ax)
        except Exception:
            pass
        ax.set_xlabel("Convergence entropy H")
        ax.set_ylabel("Density")
        ax.set_title("H: intra vs inter (distribution)")
        fig.tight_layout()
        fig.savefig(outdir / "entropy_dist_inter_vs_intra.png")
        plt.close(fig)

        plt.figure(figsize=(7, 5))
        try:
            sns.ecdfplot(data=df, x="H", hue="pair_type")
        except Exception:
            # fallback simple ECDF by split
            for subset, color in [("interpersonal", "C0"), ("intrapersonal", "C1")]:
                dsub = df[df["pair_type"] == subset]["H"].dropna().sort_values()
                if dsub.empty:
                    continue
                y = np.arange(1, len(dsub) + 1) / len(dsub)
                plt.step(dsub.values, y, where="post", label=subset)
            plt.legend()
        plt.title("H: intra vs inter (ECDF)")
        plt.tight_layout()
        plt.savefig(outdir / "entropy_ecdf_inter_vs_intra.png", dpi=150)
        plt.close()

        # Distributional comparison: H per token
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        try:
            sns.kdeplot(data=df, x="H_per_token", hue="pair_type", common_norm=False, fill=False, lw=2.6, palette=pal, ax=ax)
        except Exception:
            pass
        ax.set_xlabel("Convergence entropy per token (H/T_x)")
        ax.set_ylabel("Density")
        ax.set_title("H/T_x: intra vs inter (distribution)")
        fig.tight_layout()
        fig.savefig(outdir / "entropy_per_token_dist_inter_vs_intra.png")
        plt.close(fig)

        plt.figure(figsize=(7, 5))
        try:
            sns.ecdfplot(data=df, x="H_per_token", hue="pair_type")
        except Exception:
            for subset, color in [("interpersonal", "C0"), ("intrapersonal", "C1")]:
                dsub = df[df["pair_type"] == subset]["H_per_token"].dropna().sort_values()
                if dsub.empty:
                    continue
                y = np.arange(1, len(dsub) + 1) / len(dsub)
                plt.step(dsub.values, y, where="post", label=subset)
            plt.legend()
        plt.title("H/T_x: intra vs inter (ECDF)")
        plt.tight_layout()
        plt.savefig(outdir / "entropy_per_token_ecdf_inter_vs_intra.png", dpi=150)
        plt.close()

        # New: point-level distributions colored by lag (intra vs inter)
        # H scatter by pair_type, colored by lag
        try:
            dplot = df.dropna(subset=["H", "lag"]).copy()
            if "pair_type" not in dplot.columns:
                dplot["pair_type"] = np.where(dplot["user_x"] == dplot["user_y"], "intrapersonal", "interpersonal")
            x_map = {"interpersonal": 0, "intrapersonal": 1}
            dplot = dplot[dplot["pair_type"].isin(x_map.keys())]
            if not dplot.empty:
                # Stable jitter for readability
                rng = np.random.default_rng(42)
                x_base = dplot["pair_type"].map(x_map).astype(float).values
                x_jit = rng.uniform(-0.2, 0.2, size=x_base.shape[0])
                x = x_base + x_jit
                y = dplot["H"].astype(float).values
                c = dplot["lag"].astype(float).values
                plt.figure(figsize=(8, 5))
                sc = plt.scatter(x, y, c=c, cmap="viridis", alpha=0.5, s=12, edgecolors="none")
                plt.xticks([0, 1], ["interpersonal", "intrapersonal"])
                plt.xlabel("Pair type")
                plt.ylabel("Convergence entropy H")
                plt.title("H by pair type, colored by lag")
                cbar = plt.colorbar(sc)
                cbar.set_label("Lag (turn distance)")
                plt.tight_layout()
                plt.savefig(outdir / "scatter_H_by_pair_type_colored_by_lag.png", dpi=150)
                plt.close()
        except Exception:
            pass

        # H_per_token scatter by pair_type, colored by lag
        try:
            dplot = df.dropna(subset=["H_per_token", "lag"]).copy()
            if "pair_type" not in dplot.columns:
                dplot["pair_type"] = np.where(dplot["user_x"] == dplot["user_y"], "intrapersonal", "interpersonal")
            x_map = {"interpersonal": 0, "intrapersonal": 1}
            dplot = dplot[dplot["pair_type"].isin(x_map.keys())]
            if not dplot.empty:
                rng = np.random.default_rng(42)
                x_base = dplot["pair_type"].map(x_map).astype(float).values
                x_jit = rng.uniform(-0.2, 0.2, size=x_base.shape[0])
                x = x_base + x_jit
                y = dplot["H_per_token"].astype(float).values
                c = dplot["lag"].astype(float).values
                plt.figure(figsize=(8, 5))
                sc = plt.scatter(x, y, c=c, cmap="viridis", alpha=0.5, s=12, edgecolors="none")
                plt.xticks([0, 1], ["interpersonal", "intrapersonal"])
                plt.xlabel("Pair type")
                plt.ylabel("Convergence entropy per token (H/T_x)")
                plt.title("H/T_x by pair type, colored by lag")
                cbar = plt.colorbar(sc)
                cbar.set_label("Lag (turn distance)")
                plt.tight_layout()
                plt.savefig(outdir / "scatter_H_per_token_by_pair_type_colored_by_lag.png", dpi=150)
                plt.close()
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: plotting failed: {e}")

    # 7) Summary JSON
    try:
        df = pd.read_csv(pairs_csv)
        def stats(series: pd.Series) -> Dict[str, float]:
            if series.empty:
                return {"count": 0}
            return {
                "count": int(series.shape[0]),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
                "p10": float(series.quantile(0.10)),
                "p50": float(series.quantile(0.50)),
                "p90": float(series.quantile(0.90)),
            }
        summary = {
            "overall": {
                "H": stats(df["H"].dropna()),
                "H_per_token": stats(df["H_per_token"].dropna()),
            },
            "by_lag": {},
        }
        for lag, grp in df.groupby("lag"):
            summary["by_lag"][int(lag)] = {
                "H": stats(grp["H"].dropna()),
                "H_per_token": stats(grp["H_per_token"].dropna()),
            }
        (outdir / "agg_stats.json").write_text(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"Warning: summary failed: {e}")

    print(f"Wrote outputs to {outdir}")
    return 0


def generate_plots(outdir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    pairs_csv = outdir / "pairs_all.csv"
    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs_all.csv not found in {outdir}")

    df = pd.read_csv(pairs_csv)

    # Consistent scientific style and palette
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.6,
        "figure.dpi": 150,
    })

    # Ensure pair_type exists
    if "pair_type" not in df.columns:
        df["pair_type"] = np.where(df["user_x"] == df["user_y"], "intrapersonal", "interpersonal")

    pal = {"interpersonal": "#1f77b4", "intrapersonal": "#ff7f0e"}

    # Lag vs Entropy (H): revert to dots + summary line
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    try:
        sns.stripplot(data=df, x="lag", y="H", jitter=True, alpha=0.35, size=3,
                      color="#1f77b4", ax=ax)
        sns.pointplot(data=df, x="lag", y="H", errorbar="ci", color="#ff7f0e",
                      markers="o", linestyles="-", ax=ax)
    except Exception:
        # Fallback: scatter of all points and median line
        sub = df.dropna(subset=["lag", "H"]).copy()
        ax.scatter(sub["lag"].values, sub["H"].values, s=8, alpha=0.35, color="#1f77b4")
        m = sub.groupby("lag")["H"].median().sort_index()
        ax.plot(m.index.values, m.values, color="#ff7f0e")
    ax.set_xlabel("Turn distance (lag)")
    ax.set_ylabel("Convergence entropy H")
    ax.set_title("Lag vs Convergence Entropy (H)")
    fig.tight_layout()
    fig.savefig(outdir / "lag_vs_entropy.png")
    plt.close(fig)

    # Lag vs Entropy (H per token): dots + summary line
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    try:
        sns.stripplot(data=df, x="lag", y="H_per_token", jitter=True, alpha=0.35, size=3,
                      color="#1f77b4", ax=ax)
        sns.pointplot(data=df, x="lag", y="H_per_token", errorbar="ci", color="#ff7f0e",
                      markers="o", linestyles="-", ax=ax)
    except Exception:
        sub = df.dropna(subset=["lag", "H_per_token"]).copy()
        ax.scatter(sub["lag"].values, sub["H_per_token"].values, s=8, alpha=0.35, color="#1f77b4")
        m = sub.groupby("lag")["H_per_token"].median().sort_index()
        ax.plot(m.index.values, m.values, color="#ff7f0e")
    ax.set_xlabel("Turn distance (lag)")
    ax.set_ylabel("Convergence entropy per token (H/T_x)")
    ax.set_title("Lag vs Convergence Entropy (per token)")
    fig.tight_layout()
    fig.savefig(outdir / "lag_vs_entropy_per_token.png")
    plt.close(fig)

    # Distribution plots (H)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    try:
        sns.kdeplot(data=df, x="H", color="#1f77b4", lw=2.6, fill=False, ax=ax)
    except Exception:
        vals = df["H"].dropna().values
        ax.hist(vals, bins=50, histtype="step", color="#1f77b4")
    ax.set_xlabel("Convergence entropy H")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of convergence entropy (H)")
    fig.tight_layout()
    fig.savefig(outdir / "entropy_dist.png")
    plt.close(fig)

    # Distribution plots (H per token)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    try:
        sns.kdeplot(data=df, x="H_per_token", color="#ff7f0e", lw=2.6, fill=False, ax=ax)
    except Exception:
        vals = df["H_per_token"].dropna().values
        ax.hist(vals, bins=50, histtype="step", color="#ff7f0e")
    ax.set_xlabel("Convergence entropy per token (H/T_x)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of convergence entropy per token (H/T_x)")
    fig.tight_layout()
    fig.savefig(outdir / "entropy_dist_per_token.png")
    plt.close(fig)

    # ECDFs (H and H per token)
    plt.figure(figsize=(7, 5))
    try:
        sns.ecdfplot(data=df, x="H")
    except Exception:
        x = np.sort(df["H"].dropna().values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post")
    plt.title("ECDF of convergence entropy (H)")
    plt.tight_layout()
    plt.savefig(outdir / "entropy_ecdf.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    try:
        sns.ecdfplot(data=df, x="H_per_token")
    except Exception:
        x = np.sort(df["H_per_token"].dropna().values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post")
    plt.title("ECDF of convergence entropy per token (H/T_x)")
    plt.tight_layout()
    plt.savefig(outdir / "entropy_ecdf_per_token.png", dpi=150)
    plt.close()

    # Distributional comparison: intra vs inter (H)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    try:
        sns.kdeplot(data=df, x="H", hue="pair_type", common_norm=False, fill=False, lw=2.6, palette=pal, ax=ax)
    except Exception:
        pass
    ax.set_xlabel("Convergence entropy H")
    ax.set_ylabel("Density")
    ax.set_title("H: intra vs inter (distribution)")
    fig.tight_layout()
    fig.savefig(outdir / "entropy_dist_inter_vs_intra.png")
    plt.close(fig)

    # Distributional comparison: H per token
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    try:
        sns.kdeplot(data=df, x="H_per_token", hue="pair_type", common_norm=False, fill=False, lw=2.6, palette=pal, ax=ax)
    except Exception:
        pass
    ax.set_xlabel("Convergence entropy per token (H/T_x)")
    ax.set_ylabel("Density")
    ax.set_title("H/T_x: intra vs inter (distribution)")
    fig.tight_layout()
    fig.savefig(outdir / "entropy_per_token_dist_inter_vs_intra.png")
    plt.close(fig)

    # ECDFs split by pair_type (H and H_per_token)
    plt.figure(figsize=(7, 5))
    try:
        sns.ecdfplot(data=df, x="H", hue="pair_type")
    except Exception:
        # fallback simple ECDF by split
        for subset, color in [("interpersonal", "C0"), ("intrapersonal", "C1")]:
            dsub = df[df["pair_type"] == subset]["H"].dropna().sort_values()
            if dsub.empty:
                continue
            y = np.arange(1, len(dsub) + 1) / len(dsub)
            plt.step(dsub.values, y, where="post", label=subset)
        plt.legend()
    plt.title("H: intra vs inter (ECDF)")
    plt.tight_layout()
    plt.savefig(outdir / "entropy_ecdf_inter_vs_intra.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    try:
        sns.ecdfplot(data=df, x="H_per_token", hue="pair_type")
    except Exception:
        for subset, color in [("interpersonal", "C0"), ("intrapersonal", "C1")]:
            dsub = df[df["pair_type"] == subset]["H_per_token"].dropna().sort_values()
            if dsub.empty:
                continue
            y = np.arange(1, len(dsub) + 1) / len(dsub)
            plt.step(dsub.values, y, where="post", label=subset)
        plt.legend()
    plt.title("H/T_x: intra vs inter (ECDF)")
    plt.tight_layout()
    plt.savefig(outdir / "entropy_per_token_ecdf_inter_vs_intra.png", dpi=150)
    plt.close()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convergence entropy analysis (BERT-only)")
    p.add_argument("--posts-csv", type=Path, default=Path("simulation/posts.csv"))
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--model", type=str, default="bert-base-uncased")
    p.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-lag", type=int, default=10, help="Compute pairs up to this turn distance (lag)")
    p.add_argument("--min-chain-len", type=int, default=3, help="Minimum length of AB-alternating chains to keep")
    # Replot mode (alias for common typo: --repolot)
    p.add_argument("--replot", dest="replot", action="store_true", help="Rebuild plots from existing pairs_all.csv in --outdir")
    p.add_argument("--repolot", dest="replot", action="store_true", help="Alias for --replot")
    args = p.parse_args(argv)

    outdir = args.outdir or (args.posts_csv.parent / "convergence_entropy")
    if args.replot:
        generate_plots(outdir)
        print(f"Rebuilt plots from {outdir / 'pairs_all.csv'}")
        return 0
    return run(args.posts_csv, outdir, args.model, args.device, args.sigma, args.max_tokens, args.max_lag, args.min_chain_len)


if __name__ == "__main__":
    raise SystemExit(main())
    plt.title("ECDF of convergence entropy per token (H/T_x)")
    plt.tight_layout()
    plt.savefig(outdir / "entropy_ecdf_per_token.png", dpi=150)
    plt.close()

    # Distributional comparison: intra vs inter (H)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    try:
        sns.kdeplot(data=df, x="H", hue="pair_type", common_norm=False, fill=False, lw=2.6, palette=pal, ax=ax)
    except Exception:
        pass
    ax.set_xlabel("Convergence entropy H")
    ax.set_ylabel("Density")
    ax.set_title("H: intra vs inter (distribution)")
    fig.tight_layout()
    fig.savefig(outdir / "entropy_dist_inter_vs_intra.png")
    plt.close(fig)

    # Distributional comparison: H per token
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    try:
        sns.kdeplot(data=df, x="H_per_token", hue="pair_type", common_norm=False, fill=False, lw=2.6, palette=pal, ax=ax)
    except Exception:
        pass
    ax.set_xlabel("Convergence entropy per token (H/T_x)")
    ax.set_ylabel("Density")
    ax.set_title("H/T_x: intra vs inter (distribution)")
    fig.tight_layout()
    fig.savefig(outdir / "entropy_per_token_dist_inter_vs_intra.png")
    plt.close(fig)

    # ECDFs split by pair_type (H and H_per_token)
    plt.figure(figsize=(7, 5))
    try:
        sns.ecdfplot(data=df, x="H", hue="pair_type")
    except Exception:
        for subset, color in [("interpersonal", "C0"), ("intrapersonal", "C1")]:
            dsub = df[df["pair_type"] == subset]["H"].dropna().sort_values()
            if dsub.empty:
                continue
            y = np.arange(1, len(dsub) + 1) / len(dsub)
            plt.step(dsub.values, y, where="post", label=subset)
        plt.legend()
    plt.title("H: intra vs inter (ECDF)")
    plt.tight_layout()
    plt.savefig(outdir / "entropy_ecdf_inter_vs_intra.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    try:
        sns.ecdfplot(data=df, x="H_per_token", hue="pair_type")
    except Exception:
        for subset, color in [("interpersonal", "C0"), ("intrapersonal", "C1")]:
            dsub = df[df["pair_type"] == subset]["H_per_token"].dropna().sort_values()
            if dsub.empty:
                continue
            y = np.arange(1, len(dsub) + 1) / len(dsub)
            plt.step(dsub.values, y, where="post", label=subset)
        plt.legend()
    plt.title("H/T_x: intra vs inter (ECDF)")
    plt.tight_layout()
    plt.savefig(outdir / "entropy_per_token_ecdf_inter_vs_intra.png", dpi=150)
    plt.close()
