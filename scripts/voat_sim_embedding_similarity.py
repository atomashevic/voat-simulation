#!/usr/bin/env python3
"""
Compute embedding-based similarity between simulation2 posts/comments and Voat posts/comments
from the full MADOC parquet. Reports average cosine similarity and counts above thresholds,
with optional t-SNE/UMAP visualizations.

Usage (pyenv):
  PYENV_VERSION=ysocial python scripts/voat_sim_embedding_similarity.py \
    --madoc-parquet MADOC/voat-technology/voat_technology_madoc.parquet \
    [--mode both|posts|comments] [--plot-tsne] [--plot-umap]

Embedding model defaults to the same MiniLM used elsewhere. The implementation
reuses the same embedding method patterns as scripts/sim_comments_to_voat_match.py
including instruction prefixes for E5/GTE/BGE.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# Text normalization utils
# -------------------------
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = s.replace("\n", " ")
    s = WS_RE.sub(" ", s)
    return s.strip()


# -------------------------
# Embedding utilities (mirroring sim_comments_to_voat_match.py)
# -------------------------
def make_embedding_model(
    device: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    local_path: Optional[str] = None,
    hf_offline: bool = False,
):
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            d = str(device).strip().lower()
            if d.startswith("cu"):
                d = "cuda"
            dev = d if d in {"cpu", "cuda", "mps"} else ("cuda" if torch.cuda.is_available() else "cpu")
        if hf_offline:
            import os

            os.environ["HF_HUB_OFFLINE"] = "1"
        target = local_path if local_path else model_name
        return SentenceTransformer(target, device=dev)
    except Exception as e:
        raise RuntimeError(
            "Failed to load SentenceTransformer; ensure torch and sentence-transformers are installed and model is available."
        ) from e


def encode_texts(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True)


def maybe_apply_instruction_prefix(model_name: str, texts: List[str], is_query: bool) -> List[str]:
    name = model_name.lower()
    if "e5" in name or "gte" in name or "bge" in name:
        return [("query: " if is_query else "passage: ") + t for t in texts]
    return texts


def l2_normalize(a: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(a, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return a / denom


# -------------------------
# Data loading
# -------------------------
def load_sim2_subset(posts_csv: Path, tox_csv: Path, want_comments: bool) -> pd.DataFrame:
    """Load simulation2 posts.csv merged with toxigen.csv and filter by is_comment flag.

    Returns columns: id, text (string)
    """
    posts = pd.read_csv(posts_csv)
    tox = pd.read_csv(tox_csv)
    if not {"id", "tweet"}.issubset(posts.columns):
        raise ValueError("simulation2 posts.csv must contain columns: id,tweet")
    if not {"id", "is_comment"}.issubset(tox.columns):
        raise ValueError("simulation2 toxigen.csv must contain columns: id,is_comment")
    df = posts.merge(tox[["id", "is_comment"]], on="id", how="left")
    df = df[df["is_comment"] == bool(want_comments)].copy()
    df["text"] = df["tweet"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0]
    return df[["id", "text"]].reset_index(drop=True)


def load_voat_subset(parquet_path: Path, want_comments: bool) -> pd.DataFrame:
    """Load Voat MADOC parquet and return only posts or comments.

    Returns columns: post_id, content (renamed to text)
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"MADOC parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=["post_id", "content", "interaction_type"])
    types = df["interaction_type"].astype(str).str.lower()
    if want_comments:
        mask = types.isin(["comment", "comments"])
    else:
        mask = types.isin(["post", "posts"])  # robust to naming in datasets
    df = df[mask].dropna(subset=["content"]).copy()
    df["text"] = df["content"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0]
    return df[["post_id", "text"]].reset_index(drop=True)


# -------------------------
# Similarity and reporting
# -------------------------
@dataclass
class SimilarityStats:
    mean_cosine: float
    median_cosine: float
    count_ge_high: int
    count_ge_mid_lt_high: int
    threshold_mid: float
    threshold_high: float
    n_queries: int


def nearest_by_cosine(corpus_embs: np.ndarray, query_embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sims = cosine_similarity(query_embs, corpus_embs)
    idx = np.asarray(sims.argmax(axis=1)).ravel()
    sc = sims.max(axis=1)
    return idx, sc


def compute_stats(best_scores: np.ndarray, thr_mid: float, thr_high: float) -> SimilarityStats:
    best_scores = np.asarray(best_scores).ravel()
    mean_cos = float(np.mean(best_scores)) if best_scores.size else 0.0
    med_cos = float(np.median(best_scores)) if best_scores.size else 0.0
    ge_high = int(np.sum(best_scores >= thr_high))
    ge_mid_lt_high = int(np.sum((best_scores >= thr_mid) & (best_scores < thr_high)))
    return SimilarityStats(
        mean_cosine=mean_cos,
        median_cosine=med_cos,
        count_ge_high=ge_high,
        count_ge_mid_lt_high=ge_mid_lt_high,
        threshold_mid=thr_mid,
        threshold_high=thr_high,
        n_queries=int(best_scores.size),
    )


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def maybe_plot_2d(
    out_prefix: Path,
    method: str,
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    labels: Tuple[str, str],
    max_corpus_points: int = 2000,
    tsne_perplexity: int = 30,
):
    """Save a 2D plot using t-SNE or UMAP for a subset of points.

    - labels: (label for queries, label for corpus)
    - max_corpus_points: sample this many from the corpus for readability
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np

        # Sample corpus if large, but keep all queries
        if corpus_embs.shape[0] > max_corpus_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(corpus_embs.shape[0], size=max_corpus_points, replace=False)
            c_plot = corpus_embs[idx]
        else:
            c_plot = corpus_embs
        q_plot = query_embs

        X = np.vstack([q_plot, c_plot])
        y = np.array([0] * len(q_plot) + [1] * len(c_plot))

        if method == "tsne":
            from sklearn.manifold import TSNE

            reducer = TSNE(
                n_components=2,
                learning_rate="auto",
                init="pca",
                perplexity=tsne_perplexity,
                metric="cosine",
                random_state=42,
            )
            Z = reducer.fit_transform(X)
            suffix = f"tsne_p{tsne_perplexity}"
        elif method == "umap":
            from umap import UMAP

            reducer = UMAP(n_components=2, metric="cosine", random_state=42)
            Z = reducer.fit_transform(X)
            suffix = "umap"
        else:
            return  # unsupported

        # Plot
        plt.figure(figsize=(8, 6))
        colors = ["tab:blue", "tab:orange"]
        labels_full = [labels[0], labels[1]]
        for cls, color, lab in [(0, colors[0], labels_full[0]), (1, colors[1], labels_full[1])]:
            pts = Z[y == cls]
            plt.scatter(pts[:, 0], pts[:, 1], s=8 if cls == 1 else 20, alpha=0.7 if cls == 1 else 0.9, label=lab, c=color)
        plt.legend()
        plt.title(f"{labels[0]} vs {labels[1]} ({suffix.upper()})")
        plt.tight_layout()
        out_path = out_prefix.with_name(out_prefix.name + f"_{suffix}.png")
        ensure_dir(out_path)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {suffix.upper()} plot to {out_path}")
    except Exception as e:
        print(f"Warning: failed to create {method} plot: {e}", file=sys.stderr)


# -------------------------
# Main
# -------------------------
def run_one(
    label: str,
    sim_df: pd.DataFrame,
    voat_df: pd.DataFrame,
    model_name: str,
    device: Optional[str],
    local_path: Optional[str],
    hf_offline: bool,
    thr_mid: float,
    thr_high: float,
    plot_tsne: bool,
    plot_umap: bool,
    plot_prefix: Path,
    tsne_perplexities: List[int],
    remove_top_k_tokens: int,
    plot_sample_size: int,
) -> Dict[str, object]:
    # Prepare texts
    q_texts = sim_df["text"].astype(str).tolist()
    d_texts = voat_df["text"].astype(str).tolist()

    # Remove top-K most frequent tokens (document frequency) across both corpora if requested
    if remove_top_k_tokens > 0:
        import collections
        token_re = re.compile(r"[A-Za-z][A-Za-z0-9_']+")
        df_counter = collections.Counter()

        def doc_tokens(doc: str) -> set:
            return set(token_re.findall(doc.lower()))

        for doc in q_texts + d_texts:
            for tok in doc_tokens(doc):
                df_counter[tok] += 1
        common = {w for w, _ in df_counter.most_common(remove_top_k_tokens)}

        def strip_common(doc: str) -> str:
            return " ".join([w for w in token_re.findall(doc.lower()) if w not in common])

        q_texts = [strip_common(t) for t in q_texts]
        d_texts = [strip_common(t) for t in d_texts]

    # Build model & encode with instruction prefix when needed
    model = make_embedding_model(device=device, model_name=model_name, local_path=local_path, hf_offline=hf_offline)
    q_texts_prep = maybe_apply_instruction_prefix(model_name, q_texts, is_query=True)
    d_texts_prep = maybe_apply_instruction_prefix(model_name, d_texts, is_query=False)
    query_embs = l2_normalize(encode_texts(model, q_texts_prep))
    corpus_embs = l2_normalize(encode_texts(model, d_texts_prep))

    # Nearest neighbor by cosine
    idx, scores = nearest_by_cosine(corpus_embs, query_embs)
    stats = compute_stats(scores, thr_mid, thr_high)

    # Optional plots
    if plot_tsne:
        for pval in tsne_perplexities:
            maybe_plot_2d(
                plot_prefix.with_name(plot_prefix.name + f"_{label}"),
                "tsne",
                query_embs,
                corpus_embs,
                labels=("Simulation", "Voat"),
                max_corpus_points=plot_sample_size,
                tsne_perplexity=pval,
            )
    if plot_umap:
        maybe_plot_2d(
            plot_prefix.with_name(plot_prefix.name + f"_{label}"),
            "umap",
            query_embs,
            corpus_embs,
            labels=("Simulation", "Voat"),
            max_corpus_points=plot_sample_size,
        )

    # Return summary
    return {
        "label": label,
        "n_sim2": int(len(sim_df)),
        "n_voat": int(len(voat_df)),
        "mean_cosine": stats.mean_cosine,
        "median_cosine": stats.median_cosine,
        "count_ge_high": stats.count_ge_high,
        "count_ge_mid_lt_high": stats.count_ge_mid_lt_high,
        "threshold_mid": stats.threshold_mid,
        "threshold_high": stats.threshold_high,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Embedding similarity between simulation2 and Voat (posts and/or comments)")
    p.add_argument("--sim2-posts", type=Path, default=Path("simulation3/posts.csv"))
    p.add_argument("--sim2-tox", type=Path, default=Path("simulation3/toxigen.csv"))
    p.add_argument("--madoc-parquet", type=Path, default=Path("MADOC/voat-technology/voat_technology_madoc.parquet"))
    p.add_argument("--mode", type=str, choices=["both", "posts", "comments"], default="both")
    p.add_argument("--device", type=str, default=None, help="Force device for embeddings: cuda/cpu")
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedding-local-path", type=str, default=None)
    p.add_argument("--hf-offline", action="store_true")
    p.add_argument("--threshold-high", type=float, default=0.8)
    p.add_argument("--threshold-mid", type=float, default=0.6)
    p.add_argument("--plot-tsne", action="store_true")
    p.add_argument("--tsne-perplexities", type=str, default="5,30,80", help="Comma-separated perplexities for t-SNE when plotting")
    p.add_argument("--remove-top-k-tokens", type=int, default=0, help="Remove top K most frequent tokens (document frequency) across corpora before embedding (0 disables)")
    p.add_argument("--plot-umap", action="store_true")
    p.add_argument("--plot-sample-size", type=int, default=2000, help="Max corpus points to include in plots")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--plot-prefix", type=Path, default=None)
    args = p.parse_args(argv)

    tsne_perplexities = [int(x) for x in str(args.tsne_perplexities).split(',') if x.strip()]

    # Compute default outputs under the simulation folder if not provided
    if args.out_json is None:
        args.out_json = args.sim2_posts.parent / "sim_voat_embedding_similarity.json"
    if args.plot_prefix is None:
        args.plot_prefix = args.sim2_posts.parent / "sim_voat_similarity"

    # Load data subsets according to mode
    results: List[Dict[str, object]] = []
    if args.mode in {"both", "comments"}:
        sim_comments = load_sim2_subset(args.sim2_posts, args.sim2_tox, want_comments=True)
        voat_comments = load_voat_subset(args.madoc_parquet, want_comments=True)
        if sim_comments.empty or voat_comments.empty:
            print("Warning: comments subset empty; skipping.", file=sys.stderr)
        else:
            res = run_one(
                label="comments",
                sim_df=sim_comments,
                voat_df=voat_comments,
                model_name=args.embedding_model,
                device=args.device,
                local_path=args.embedding_local_path,
                hf_offline=args.hf_offline,
                thr_mid=args.threshold_mid,
                thr_high=args.threshold_high,
                plot_tsne=args.plot_tsne,
                plot_umap=args.plot_umap,
                plot_prefix=args.plot_prefix,
                tsne_perplexities=tsne_perplexities,
                remove_top_k_tokens=args.remove_top_k_tokens,
                plot_sample_size=args.plot_sample_size,
            )
            results.append(res)

    if args.mode in {"both", "posts"}:
        sim_posts = load_sim2_subset(args.sim2_posts, args.sim2_tox, want_comments=False)
        voat_posts = load_voat_subset(args.madoc_parquet, want_comments=False)
        if sim_posts.empty or voat_posts.empty:
            print("Warning: posts subset empty; skipping.", file=sys.stderr)
        else:
            res = run_one(
                label="posts",
                sim_df=sim_posts,
                voat_df=voat_posts,
                model_name=args.embedding_model,
                device=args.device,
                local_path=args.embedding_local_path,
                hf_offline=args.hf_offline,
                thr_mid=args.threshold_mid,
                thr_high=args.threshold_high,
                plot_tsne=args.plot_tsne,
                plot_umap=args.plot_umap,
                plot_prefix=args.plot_prefix,
                tsne_perplexities=tsne_perplexities,
                remove_top_k_tokens=args.remove_top_k_tokens,
                plot_sample_size=args.plot_sample_size,
            )
            results.append(res)

    # Write summary JSON
    ensure_dir(args.out_json)
    payload = {
        "embedding_model": args.embedding_model,
        "madoc_parquet": str(args.madoc_parquet),
        "mode": args.mode,
        "thresholds": {"mid": args.threshold_mid, "high": args.threshold_high},
    "results": results,
    "tsne_perplexities": tsne_perplexities,
    "removed_top_k_tokens": args.remove_top_k_tokens,
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote summary to: {args.out_json}")

    # Console summary
    for r in results:
        print(
            f"[{r['label']}] n_sim2={r['n_sim2']:,} n_voat={r['n_voat']:,} "
            f"mean={r['mean_cosine']:.3f} median={r['median_cosine']:.3f} "
            f">={args.threshold_high} -> {r['count_ge_high']:,}; "
            f"{args.threshold_mid}–{args.threshold_high} -> {r['count_ge_mid_lt_high']:,}"
        )

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
