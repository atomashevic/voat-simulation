#!/usr/bin/env python3
"""
Find top toxic posts/comments in Voat sample_1 and their closest matches in simulation2
based on MiniLM embeddings, subject to a toxicity floor in simulation2.

Outputs a readable text report with 20 examples and similarity metrics.

Usage (GPU, defaults):
  PYENV_VERSION=ysocial python scripts/voat_toxic_match.py --device cuda --save-report

Key defaults:
- MADOC sample: MADOC/voat-technology/sample_1/voat_sample_1.parquet (columns: content, toxicity_toxigen)
- Simulation2: simulation2/posts.csv (tweet text) + simulation2/toxigen.csv (toxicity)
- Select top 20 Voat items by toxicity, then find best simulation2 match with toxicity >= 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = s.replace("\n", " ")
    s = WS_RE.sub(" ", s)
    return s.strip()


def load_voat_top_toxic(parquet_path: Path, topn: int) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"MADOC parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    # Ensure required columns
    for c in ("content", "toxicity_toxigen"):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in {parquet_path}")
    df = df.dropna(subset=["content", "toxicity_toxigen"]).copy()
    df["content"] = df["content"].astype(str).map(clean_text)
    df = df[df["content"].str.len() > 0]
    df = df.sort_values("toxicity_toxigen", ascending=False)
    return df.head(topn).reset_index(drop=True)


def load_sim2_candidates(posts_csv: Path, tox_csv: Path, tox_threshold: float) -> pd.DataFrame:
    if not posts_csv.exists():
        raise FileNotFoundError(f"simulation2 posts not found: {posts_csv}")
    if not tox_csv.exists():
        raise FileNotFoundError(f"simulation2 toxigen not found: {tox_csv}")
    posts = pd.read_csv(posts_csv)
    tox = pd.read_csv(tox_csv)
    if "id" not in posts.columns or "tweet" not in posts.columns:
        raise ValueError("Expected columns 'id' and 'tweet' in simulation2 posts.csv")
    if "id" not in tox.columns or "toxicity" not in tox.columns:
        raise ValueError("Expected columns 'id' and 'toxicity' in simulation2 toxigen.csv")
    df = posts.merge(tox[["id", "toxicity"]], on="id", how="left")
    df = df.dropna(subset=["tweet", "toxicity"]).copy()
    df["tweet"] = df["tweet"].astype(str).map(clean_text)
    df = df[(df["tweet"].str.len() > 0) & (df["toxicity"] >= tox_threshold)]
    return df.reset_index(drop=True)


def make_embedding_model(device: Optional[str] = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", local_path: Optional[str] = None, hf_offline: bool = False):
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        # Normalize/validate device string
        if device is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            d = str(device).strip().lower()
            if d.startswith("cu"):
                d = "cuda"
            allowed = {"cpu", "cuda", "mps"}
            dev = d if d in allowed else ("cuda" if torch.cuda.is_available() else "cpu")
        if hf_offline:
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
        target = local_path if local_path else model_name
        return SentenceTransformer(target, device=dev)
    except Exception as e:
        raise RuntimeError("Failed to load SentenceTransformer; ensure torch and sentence-transformers are installed and model is available.") from e


def encode_texts(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True)


def best_sim_match(query_vecs: np.ndarray, cand_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sim = cosine_similarity(query_vecs, cand_vecs)
    best_idx = sim.argmax(axis=1)
    best_sim = sim[np.arange(sim.shape[0]), best_idx]
    return best_idx, best_sim


def write_report(out_path: Path, voat_df: pd.DataFrame, voat_texts: List[str], voat_sims: np.ndarray, sim2_df: pd.DataFrame, sim2_indices: np.ndarray, sim_threshold: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("Voat vs Simulation2: Toxicity and Semantic Similarity Matches\n")
    lines.append(f"Criteria: simulation toxicity >= {sim_threshold:.2f} (floor), similarity >= threshold if stated in CLI\n")
    lines.append("")
    for i, row in voat_df.iterrows():
        vtox = float(row["toxicity_toxigen"]) if "toxicity_toxigen" in row else float("nan")
        vtxt = voat_texts[i]
        cand_idx = int(sim2_indices[i])
        cand = sim2_df.iloc[cand_idx]
        sim = float(voat_sims[i])
        lines.append(f"=== Example {i+1} ===")
        lines.append(f"Voat post_id: {row.get('post_id', 'NA')} | toxicity: {vtox:.3f}")
        lines.append(f"Voat text: {vtxt[:500]}{'...' if len(vtxt) > 500 else ''}")
        lines.append(f"-- Matched Simulation2 --")
        lines.append(f"sim2 id: {cand['id']} | toxicity: {cand['toxicity']:.3f} | similarity: {sim:.3f}")
        stxt = str(cand["tweet"]) if isinstance(cand["tweet"], str) else ""
        lines.append(f"sim2 text: {stxt[:500]}{'...' if len(stxt) > 500 else ''}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Match most toxic Voat sample_1 posts to similar toxic simulation2 posts")
    p.add_argument("--madoc-parquet", type=Path, default=Path("MADOC/voat-technology/sample_1/voat_sample_1.parquet"))
    p.add_argument("--sim2-posts", type=Path, default=Path("simulation2/posts.csv"))
    p.add_argument("--sim2-tox", type=Path, default=Path("simulation2/toxigen.csv"))
    p.add_argument("--topn", type=int, default=20, help="Number of most toxic Voat items to use")
    p.add_argument("--sim-tox-floor", type=float, default=0.1, help="Minimum toxicity in simulation2 candidates")
    p.add_argument("--sim-threshold", type=float, default=0.85, help="Similarity threshold considered 'very high'")
    p.add_argument("--device", type=str, default=None, help="Force device for embeddings: cuda/cpu")
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedding-local-path", type=str, default=None)
    p.add_argument("--hf-offline", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    # Load data
    voat_df = load_voat_top_toxic(args.madoc_parquet, args.topn)
    sim2_df = load_sim2_candidates(args.sim2_posts, args.sim2_tox, tox_threshold=args.sim_tox_floor)

    if sim2_df.empty:
        print("No simulation2 candidates meet toxicity floor.", file=sys.stderr)
        return 1

    # Embed
    model = make_embedding_model(device=args.device, model_name=args.embedding_model, local_path=args.embedding_local_path, hf_offline=args.hf_offline)
    voat_texts = voat_df["content"].astype(str).tolist()
    sim_texts = sim2_df["tweet"].astype(str).tolist()
    voat_vecs = encode_texts(model, voat_texts)
    sim_vecs = encode_texts(model, sim_texts)

    # Match
    best_idx, best_sim = best_sim_match(voat_vecs, sim_vecs)

    # Optional: we can filter by sim-threshold when reporting, but still show best even if < threshold
    out_path = args.out or (args.sim2_posts.parent / "voat_toxic_matches.txt")
    write_report(out_path, voat_df, voat_texts, best_sim, sim2_df, best_idx, args.sim_threshold)
    print(f"Wrote examples to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
