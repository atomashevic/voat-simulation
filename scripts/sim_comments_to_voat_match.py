#!/usr/bin/env python3
"""
Sample 20 toxic simulation comments and find most similar Voat comments (full MADOC parquet)
using embedding-based similarity only. Writes a readable text report.

Approaches implemented:
- embed:<model>: SentenceTransformer bi-encoder cosine per listed model
- cross-encoder (optional): rerank top-K from the first embed model

Usage (GPU autodetect):
  PYENV_VERSION=ysocial python scripts/sim_comments_to_voat_match.py

Options include --tox-threshold, --n-sample, --device, --embedding-models, and optional --cross-encoder reranking.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import sys

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


def make_embedding_model(device: Optional[str] = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", local_path: Optional[str] = None, hf_offline: bool = False):
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
        raise RuntimeError("Failed to load SentenceTransformer; ensure torch and sentence-transformers are installed and model is available.") from e


def encode_texts(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True)


def load_sim2_comments(posts_csv: Path, tox_csv: Path, tox_threshold: float, n_sample: int, seed: int) -> pd.DataFrame:
    posts = pd.read_csv(posts_csv)
    tox = pd.read_csv(tox_csv)
    exp_posts_cols = {"id", "tweet"}
    exp_tox_cols = {"id", "toxicity", "is_comment"}
    if not exp_posts_cols.issubset(posts.columns):
        raise ValueError(f"simulation posts.csv must have columns {exp_posts_cols}")
    if not exp_tox_cols.issubset(tox.columns):
        raise ValueError(f"simulation toxigen.csv must have columns {exp_tox_cols}")
    df = posts.merge(tox[list(exp_tox_cols)], on="id", how="left")
    df = df[(df["is_comment"] == True) & (df["toxicity"] >= tox_threshold)].copy()
    df["tweet"] = df["tweet"].astype(str).map(clean_text)
    df = df[df["tweet"].str.len() > 0]
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=seed)
    return df.reset_index(drop=True)


def load_voat_comments(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"MADOC parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    need = {"content", "interaction_type"}
    if not need.issubset(df.columns):
        raise ValueError(f"Expected columns {need} in {parquet_path}")
    # Use singular 'comment' based on dataset; normalize case just in case
    df = df[df["interaction_type"].str.lower() == "comment"].copy()
    df = df.dropna(subset=["content"]) 
    df["content"] = df["content"].astype(str).map(clean_text)
    df = df[df["content"].str.len() > 0]
    return df.reset_index(drop=True)


@dataclass
class ApproachResult:
    name: str
    index: int
    score: float


def nearest_by_embeddings(voat_embs: np.ndarray, query_embs: np.ndarray) -> Tuple[List[int], List[float]]:
    sims = cosine_similarity(query_embs, voat_embs)
    idx = np.asarray(sims.argmax(axis=1)).ravel()
    sc = sims.max(axis=1)
    return idx.tolist(), sc.tolist()


def maybe_apply_instruction_prefix(model_name: str, texts: List[str], is_query: bool) -> List[str]:
    name = model_name.lower()
    if "e5" in name or "gte" in name or "bge" in name:
        if is_query:
            return ["query: " + t for t in texts]
        else:
            return ["passage: " + t for t in texts]
    return texts


def l2_normalize(a: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(a, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return a / denom


def write_report(out_path: Path, sim_df: pd.DataFrame, voat_df: pd.DataFrame, approaches: Dict[str, Tuple[List[int], List[float]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("Simulation toxic comments → closest Voat comments (per approach)\n")
    lines.append(f"Sample size: {len(sim_df)}\n")
    for qi, srow in sim_df.iterrows():
        lines.append(f"=== Query {qi+1} | sim2 id {srow['id']} | toxicity {float(srow['toxicity']):.3f} ===")
        qtext = str(srow["tweet"]) if isinstance(srow["tweet"], str) else ""
        lines.append(f"sim2 text: {qtext[:600]}{'...' if len(qtext) > 600 else ''}")
        for name, (idxs, scores) in approaches.items():
            cand_idx = int(idxs[qi])
            score = float(scores[qi])
            crow = voat_df.iloc[cand_idx]
            ctext = str(crow["content"]) if isinstance(crow["content"], str) else ""
            pid = crow.get("post_id", "NA")
            lines.append(f"-- {name}: match post_id {pid} | score {score:.3f}")
            lines.append(f"voat text: {ctext[:600]}{'...' if len(ctext) > 600 else ''}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Find similar Voat comments for sampled toxic simulation comments using embedding similarity")
    ap.add_argument("--sim2-posts", type=Path, default=Path("simulation/posts.csv"))
    ap.add_argument("--sim2-tox", type=Path, default=Path("simulation/toxigen.csv"))
    ap.add_argument("--madoc-parquet", type=Path, default=Path("MADOC/voat-technology/voat_technology_madoc.parquet"))
    ap.add_argument("--tox-threshold", type=float, default=0.4)
    ap.add_argument("--n-sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--embedding-models", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Comma-separated list of embedding models to use")
    ap.add_argument("--embedding-local-path", type=str, default=None)
    ap.add_argument("--hf-offline", action="store_true")
    ap.add_argument("--cross-encoder", type=str, default=None, help="Optional CrossEncoder model to rerank top-K from the first embed model")
    ap.add_argument("--rerank-topk", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    # Load datasets
    sim_df = load_sim2_comments(args.sim2_posts, args.sim2_tox, args.tox_threshold, args.n_sample, args.seed)
    if sim_df.empty:
        print("No simulation comments found above threshold.", file=sys.stderr)
        return 1
    voat_df = load_voat_comments(args.madoc_parquet)
    if voat_df.empty:
        print("No Voat comments found in MADOC parquet.", file=sys.stderr)
        return 1

    # Prepare text corpora
    queries = sim_df["tweet"].astype(str).tolist()
    voat_texts = voat_df["content"].astype(str).tolist()

    # Embedding-only approaches: support multiple models
    approaches: Dict[str, Tuple[List[int], List[float]]] = {}
    model_names = [m.strip() for m in args.embedding_models.split(",") if m.strip()]
    cache_first = {}
    for mname in model_names:
        model = make_embedding_model(device=args.device, model_name=mname, local_path=args.embedding_local_path, hf_offline=args.hf_offline)
        q_texts = maybe_apply_instruction_prefix(mname, queries, is_query=True)
        d_texts = maybe_apply_instruction_prefix(mname, voat_texts, is_query=False)
        voat_embs = l2_normalize(encode_texts(model, d_texts))
        query_embs = l2_normalize(encode_texts(model, q_texts))
        idx, sc = nearest_by_embeddings(voat_embs, query_embs)
        key = "embed:" + mname.split("/")[-1]
        approaches[key] = (idx, sc)
        if not cache_first:
            cache_first = {"voat_embs": voat_embs, "query_embs": query_embs}

    # Optional cross-encoder reranking from the first embed model's top-K
    if args.cross_encoder and cache_first:
        try:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(args.cross_encoder, device=None)
            sims_full = cache_first["query_embs"] @ cache_first["voat_embs"].T
            rerank_idx: List[int] = []
            rerank_sc: List[float] = []
            for i in range(len(queries)):
                row = sims_full[i]
                topk = np.argsort(-row)[: max(1, args.rerank_topk)]
                pairs = [(queries[i], voat_texts[j]) for j in topk]
                scores = ce.predict(pairs)
                best_local = int(topk[int(np.argmax(scores))])
                rerank_idx.append(best_local)
                rerank_sc.append(float(np.max(scores)))
            approaches["cross-encoder:" + args.cross_encoder.split("/")[-1]] = (rerank_idx, rerank_sc)
        except Exception as e:
            print("Warning: cross-encoder rerank skipped:", e, file=sys.stderr)

    out_path = args.out or (args.sim2_posts.parent / "sim_to_voat_comment_matches.txt")
    write_report(out_path, sim_df, voat_df, approaches)
    print(f"Wrote report to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
