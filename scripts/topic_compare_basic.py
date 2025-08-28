#!/usr/bin/env python3
"""
Basic topic comparison between a cleaned Simulation corpus and a Voat parquet corpus.

Goals:
- Be robust to vocabulary collapse and UMAP/numba issues by using safe defaults.
- Use BERTopic with MiniLM embeddings + HDBSCAN (no UMAP by default).
- Represent topics via document-embedding centroids for stable cross-domain similarity.
- Compute cosine similarity, take top-K matches per Simulation topic, and export results.

Usage (recommended):
  python scripts/topic_compare_basic.py \
    --sim-csv simulation3/posts_clean.csv --sim-text-col full_text \
    --voat-parquet MADOC/voat-technology/voat_technology_madoc.parquet --voat-text-col content \
    --outdir simulation3/topic_compare_basic --save-heatmap

Outputs (in --outdir):
  - topic_info_sim.csv, topic_info_voat.csv
  - topic_matches.csv, topic_similarity.csv, summary.json
  - similarity_heatmap.png (if --save-heatmap)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# Minimal text cleaning
# -------------------------
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = s.replace("\n", " ")
    # Fix missing space after periods before capital letters
    s = re.sub(r"\.([A-Z])", r". \1", s)
    # Strip username/mention prefix at start
    s = re.sub(r"^[.@]\S+\s+", "", s)
    s = WS_RE.sub(" ", s)
    return s.strip()


def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Data loading
# -------------------------
def load_sim_corpus(path: Path, text_col: str, min_chars: int, max_docs: Optional[int], seed: int) -> Tuple[List[str], pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Simulation CSV not found: {path}")
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {path}")
    texts = df[text_col].astype(str).map(clean_text)
    texts = texts[texts.str.len() >= int(min_chars)]
    if max_docs is not None and len(texts) > max_docs:
        texts = texts.sample(n=max_docs, random_state=seed)
    return texts.tolist(), df.loc[texts.index].reset_index(drop=True)


def load_voat_corpus(path: Path, text_col: str, min_chars: int, max_docs: Optional[int], seed: int) -> Tuple[List[str], pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Voat parquet not found: {path}")
    try:
        df = pd.read_parquet(path)
    except ImportError as e:
        raise ImportError(
            "Reading parquet requires 'pyarrow' or 'fastparquet'. Install one, e.g. pip install pyarrow"
        ) from e
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {path}")
    texts = df[text_col].astype(str).map(clean_text)
    texts = texts[texts.str.len() >= int(min_chars)]
    if max_docs is not None and len(texts) > max_docs:
        texts = texts.sample(n=max_docs, random_state=seed)
    return texts.tolist(), df.loc[texts.index].reset_index(drop=True)


# -------------------------
# BERTopic (safe defaults + fallbacks)
# -------------------------
def make_embedding_model(device_preference: Optional[str] = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = device_preference or ("cuda" if torch.cuda.is_available() else "cpu")
        return SentenceTransformer(model_name, device=device)
    except Exception as e:
        raise RuntimeError(
            "Failed to load sentence-transformers embeddings. Install 'sentence-transformers' and 'torch'."
        ) from e


def build_bertopic(embedding_model, min_topic_size: int, vectorizer_min_df: int = 1, extra_stopwords: Optional[List[str]] = None):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN

    # Try UMAP; if unavailable, stay None. This improves separability and topic count.
    umap_model = None
    try:
        from umap import UMAP  # type: ignore

        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
    except Exception:
        # Fallback to TruncatedSVD if available; else stay None
        try:
            from sklearn.decomposition import TruncatedSVD  # type: ignore

            class SVDReducer:
                def __init__(self, n_components=5, random_state=42):
                    self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)

                def fit(self, X, y=None):
                    self._svd.fit(X)
                    return self

                def fit_transform(self, X, y=None):
                    return self._svd.fit_transform(X)

                def transform(self, X):
                    return self._svd.transform(X)

            umap_model = SVDReducer(n_components=5, random_state=42)
        except Exception:
            umap_model = None

    # Vectorizer: keep words starting with a letter, length >=3; min_df=1 to avoid vocabulary collapse
    extra = set(w.strip().lower() for w in (extra_stopwords or []) if w and w.strip())
    stop_words = list(ENGLISH_STOP_WORDS.union(extra)) if extra else "english"
    vectorizer_model = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=vectorizer_min_df,
        token_pattern=r"(?u)\b[^\W\d][\w\-]{2,}\b",
    )

    # HDBSCAN: favor more fine-grained clusters on high-dim embeddings
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=True,
    )
    return topic_model


@dataclass
class TopicInfo:
    topic_id: int
    size: int
    label: str
    top_words: List[str]


def _clean_tokens(words: List[str], top_n: int) -> List[str]:
    out: List[str] = []
    for w in words:
        if not w:
            continue
        if len(w) >= 4 and w.isupper():
            continue
        if re.fullmatch(r"\d+", w):
            continue
        if re.fullmatch(r"[0-9a-fA-F]{4,}", w):
            continue
        if len(w) > 30:
            continue
        out.append(w)
        if len(out) >= top_n:
            break
    return out


def extract_topic_info(model, texts: List[str], top_n_words: int) -> List[TopicInfo]:
    # Model is assumed fitted; just read topic info and top words.
    info = model.get_topic_info()
    try:
        label_map = model.get_topic_labels()
    except Exception:
        label_map = {int(row["Topic"]): row.get("Name", f"Topic {int(row['Topic'])}") for _, row in info.iterrows()}
    out: List[TopicInfo] = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        words_scores = model.get_topic(tid) or []
        top_words = _clean_tokens([w for w, _ in words_scores], top_n_words)
        label = label_map.get(tid, f"Topic {tid}")
        out.append(TopicInfo(topic_id=tid, size=int(row["Count"]), label=str(label), top_words=top_words))
    return out


def compute_topic_centroids(texts: List[str], topics: List[int], infos: List[TopicInfo], embedding_model, max_docs_per_topic: int = 200) -> np.ndarray:
    topic_to_indices = {ti.topic_id: [] for ti in infos}
    for idx, t in enumerate(topics):
        if t in topic_to_indices and len(topic_to_indices[t]) < max_docs_per_topic:
            topic_to_indices[t].append(idx)
    embs = []
    for ti in infos:
        idxs = topic_to_indices.get(ti.topic_id, [])
        if not idxs:
            embs.append(np.zeros((384,), dtype=np.float32))
            continue
        batch = [texts[i] for i in idxs]
        vecs = embedding_model.encode(batch, convert_to_numpy=True)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        embs.append(vecs.mean(axis=0))
    return np.vstack(embs) if embs else np.zeros((0, 384))


def topic_vectors_centroid(model, texts: List[str], infos: List[TopicInfo], embedding_model, max_docs_per_topic: int = 200) -> Tuple[np.ndarray, List[int]]:
    topics, _ = model.transform(texts)
    vecs = compute_topic_centroids(texts, topics, infos, embedding_model, max_docs_per_topic=max_docs_per_topic)
    order = [ti.topic_id for ti in infos]
    return vecs, order


def fit_with_fallback(model, texts: List[str], extra_stopwords: Optional[List[str]] = None):
    """Fit-transform with fallback if vectorizer collapses to empty vocabulary."""
    try:
        return model.fit_transform(texts)
    except ValueError as e:
        if "empty vocabulary" in str(e):
            print("Warning: empty vocabulary; retrying with relaxed vectorizer", file=sys.stderr)
            extra = set(w.strip().lower() for w in (extra_stopwords or []) if w and w.strip())
            stop_words = list(ENGLISH_STOP_WORDS.union(extra)) if extra else list(ENGLISH_STOP_WORDS)
            model.vectorizer_model = CountVectorizer(
                stop_words=stop_words,
                ngram_range=(1, 2),
                min_df=1,
                token_pattern=r"(?u)\b\w\w+\b",
            )
            return model.fit_transform(texts)
        raise


def run(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Basic topic comparison between Simulation (clean) and Voat parquet")
    p.add_argument("--sim-csv", type=Path, default=Path("simulation3/posts_clean.csv"))
    p.add_argument("--sim-text-col", type=str, default="full_text")
    p.add_argument("--voat-parquet", type=Path, default=Path("MADOC/voat-technology/voat_technology_madoc.parquet"))
    p.add_argument("--voat-text-col", type=str, default="content")
    p.add_argument("--outdir", type=Path, default=Path("simulation3/topic_compare_basic"))
    p.add_argument("--min-doc-chars", type=int, default=25)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--min-topic-size", type=int, default=15)
    p.add_argument("--top-n-words", type=int, default=10)
    p.add_argument("--similarity-threshold", type=float, default=0.6)
    p.add_argument("--topk-per-sim", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="Force device for embeddings: 'cuda' or 'cpu'")
    p.add_argument("--save-heatmap", action="store_true")
    p.add_argument(
        "--extra-stopwords",
        type=str,
        default=None,
        help="Comma-separated additional stopwords to remove (e.g., 'just,like,justsaying')",
    )
    args = p.parse_args(argv)

    ensure_outdir(args.outdir)

    # Load corpora
    sim_texts, sim_df = load_sim_corpus(args.sim_csv, args.sim_text_col, args.min_doc_chars, args.max_docs, args.seed)
    voat_texts, voat_df = load_voat_corpus(args.voat_parquet, args.voat_text_col, args.min_doc_chars, args.max_docs, args.seed)
    if len(sim_texts) == 0 or len(voat_texts) == 0:
        print("Empty corpus after filtering. Adjust --min-doc-chars or text columns.", file=sys.stderr)
        return 1

    # Embedding model
    embedder = make_embedding_model(args.device)

    # BERTopic models (safe defaults)
    extras_list = [s.strip() for s in args.extra_stopwords.split(",")] if args.extra_stopwords else None
    model_sim = build_bertopic(
        embedder,
        min_topic_size=args.min_topic_size,
        vectorizer_min_df=1,
        extra_stopwords=extras_list,
    )
    model_voat = build_bertopic(
        embedder,
        min_topic_size=args.min_topic_size,
        vectorizer_min_df=1,
        extra_stopwords=extras_list,
    )

    # Fit + extract info with robust fallback
    try:
        _ts, _ = fit_with_fallback(model_sim, sim_texts, extras_list)
        _tv, _ = fit_with_fallback(model_voat, voat_texts, extras_list)
    except Exception as e:
        print(f"BERTopic fitting failed: {e}", file=sys.stderr)
        return 1

    sim_infos = extract_topic_info(model_sim, sim_texts, top_n_words=args.top_n_words)
    voat_infos = extract_topic_info(model_voat, voat_texts, top_n_words=args.top_n_words)

    # Topic vectors via centroids
    sim_vecs, _ = topic_vectors_centroid(model_sim, sim_texts, sim_infos, embedder)
    voat_vecs, _ = topic_vectors_centroid(model_voat, voat_texts, voat_infos, embedder)

    # Similarity matrix and top-K matches per sim topic
    if sim_vecs.size == 0 or voat_vecs.size == 0:
        print("No topic vectors; aborting.", file=sys.stderr)
        return 1
    sim_matrix = cosine_similarity(sim_vecs, voat_vecs)

    pairs = []  # (sim_idx, voat_idx, sim)
    topk = max(1, int(args.topk_per_sim))
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix[i]
        order = np.argsort(-row)
        k = 0
        for j in order:
            if row[j] < float(args.similarity_threshold):
                break
            pairs.append((i, int(j), float(row[j])))
            k += 1
            if k >= topk:
                break

    # Exports
    sim_info_df = pd.DataFrame({
        "topic_id": [t.topic_id for t in sim_infos],
        "size": [t.size for t in sim_infos],
        "label": [t.label for t in sim_infos],
        "top_words": [" ".join(t.top_words) for t in sim_infos],
    })
    voat_info_df = pd.DataFrame({
        "topic_id": [t.topic_id for t in voat_infos],
        "size": [t.size for t in voat_infos],
        "label": [t.label for t in voat_infos],
        "top_words": [" ".join(t.top_words) for t in voat_infos],
    })
    sim_info_df.to_csv(args.outdir / "topic_info_sim.csv", index=False)
    voat_info_df.to_csv(args.outdir / "topic_info_voat.csv", index=False)

    rows = []
    for si, vi, s in pairs:
        s_info = sim_infos[si]
        v_info = voat_infos[vi]
        rows.append({
            "sim_topic_id": s_info.topic_id,
            "sim_label": s_info.label,
            "sim_size": s_info.size,
            "voat_topic_id": v_info.topic_id,
            "voat_label": v_info.label,
            "voat_size": v_info.size,
            "cosine_sim": s,
        })
    matches_df = pd.DataFrame(rows)
    matches_df.to_csv(args.outdir / "topic_matches.csv", index=False)

    sim_df = pd.DataFrame(sim_matrix, columns=[f"voat_{i}" for i in range(sim_matrix.shape[1])])
    sim_df.insert(0, "sim_idx", range(sim_matrix.shape[0]))
    sim_df.to_csv(args.outdir / "topic_similarity.csv", index=False)

    # Summary
    matched_sim_set = set([p[0] for p in pairs])
    coverage = float(len(matched_sim_set) / max(1, len(sim_infos)))
    best_sim = [p for p in pairs if p[0] in matched_sim_set]
    best_top = {}
    for r, c, s in pairs:
        if r not in best_top:
            best_top[r] = s
    mean_cos = float(np.mean(list(best_top.values()))) if best_top else 0.0
    med_cos = float(np.median(list(best_top.values()))) if best_top else 0.0
    summary = {
        "sim_docs": len(sim_texts),
        "voat_docs": len(voat_texts),
        "sim_topics": len(sim_infos),
        "voat_topics": len(voat_infos),
        "matches": len(pairs),
        "coverage_sim": coverage,
        "similarity_threshold": float(args.similarity_threshold),
        "mean_cosine_top1": mean_cos,
        "median_cosine_top1": med_cos,
        "seed": int(args.seed),
    }
    with open(args.outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Heatmap (optional)
    if args.save_heatmap and sim_matrix.size > 0:
        try:
            import matplotlib.pyplot as plt
            order = np.argsort(-sim_matrix.max(axis=1))
            scores = sim_matrix[order][:, : min(5, sim_matrix.shape[1])]
            labels = [sim_infos[i].label for i in order]
            labels = [l if len(l) <= 40 else l[:37] + "..." for l in labels]
            plt.figure(figsize=(6, max(4, len(labels) * 0.25)))
            im = plt.imshow(scores, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
            plt.colorbar(im, fraction=0.046, pad=0.04, label="Cosine similarity")
            plt.yticks(range(len(labels)), labels, fontsize=7)
            plt.xticks(range(scores.shape[1]), [f"top{j+1}" for j in range(scores.shape[1])], fontsize=8)
            plt.xlabel("Voat topics (top columns)")
            plt.ylabel("Simulation topics (sorted by max sim)")
            plt.title("Topic similarity (cosine)")
            plt.tight_layout()
            plt.savefig(args.outdir / "similarity_heatmap.png", dpi=200)
            plt.close()
        except Exception as e:
            print(f"Heatmap failed: {e}", file=sys.stderr)

    run_info = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "timestamp": int(time.time()),
    }
    with open(args.outdir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"Wrote outputs to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
