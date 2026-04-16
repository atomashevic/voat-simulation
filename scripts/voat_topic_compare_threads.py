#!/usr/bin/env python3
"""
Thread-level topic comparison between simulation and MADOC Voat.

Unlike the comment-level script, this aggregates all comments within each thread
into a single document, capturing the overall conversation theme.

Usage:
  python scripts/voat_topic_compare_threads.py \
    --sim2-posts-csv results/basic/simulation/posts.csv \
    --outdir results/topic_threads/simulation \
    --sim2-min-topic-size 10 \
    --madoc-min-topic-size 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Import from the existing script (reuse all the machinery)
import voat_topic_compare as vtc


def aggregate_threads(
    posts_df: pd.DataFrame,
    text_cols: List[str],
    min_thread_chars: int = 50,
) -> pd.DataFrame:
    """
    Aggregate all comments in each thread into a single document.

    Returns DataFrame with columns: thread_id, thread_text, comment_count
    """
    # Build text for each comment
    texts = vtc.build_text_series(posts_df, text_cols)
    posts_df = posts_df.copy()
    posts_df['_text'] = texts

    # Detect ID column (could be 'id' or 'post_id')
    id_col = 'id' if 'id' in posts_df.columns else 'post_id'

    # Group by thread and concatenate all comments
    thread_groups = posts_df.groupby('thread_id').agg({
        '_text': lambda x: ' '.join(x.dropna().astype(str)),
        id_col: 'count'  # count comments per thread
    }).reset_index()

    thread_groups.columns = ['thread_id', 'thread_text', 'comment_count']

    # Clean the aggregated thread text
    thread_groups['thread_text'] = thread_groups['thread_text'].apply(vtc.clean_text)

    # Filter by minimum length
    thread_groups = thread_groups[
        thread_groups['thread_text'].str.len() >= min_thread_chars
    ].reset_index(drop=True)

    return thread_groups


def load_thread_corpus(
    path: Path,
    text_cols: Optional[List[str]] = None,
    min_thread_chars: int = 50,
    max_threads: Optional[int] = None,
    seed: int = 42,
) -> tuple[List[str], pd.DataFrame, List[str]]:
    """Load posts/comments and aggregate by thread."""
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Detect text columns
    used_cols = vtc.detect_text_columns(df, preferred=text_cols)
    if not used_cols:
        raise ValueError("No suitable text columns found.")

    # Check if thread_id column exists - if not, try to reconstruct from parent_id
    if 'thread_id' not in df.columns:
        if 'parent_id' in df.columns and 'post_id' in df.columns:
            print(f"Reconstructing threads from parent_id structure in {path.name}", file=sys.stderr)
            # Find root posts (no parent or parent is null)
            df = df.copy()
            df['thread_id'] = None

            # Posts with no parent are thread roots
            root_mask = df['parent_id'].isna()
            df.loc[root_mask, 'thread_id'] = df.loc[root_mask, 'post_id']

            # Propagate thread_id down the tree (iteratively)
            # Build parent -> post_id mapping
            parent_to_posts = df[~root_mask].groupby('parent_id')['post_id'].apply(list).to_dict()

            # Map post_id to thread_id
            post_to_thread = df[root_mask].set_index('post_id')['thread_id'].to_dict()

            # Breadth-first propagation
            max_iterations = 10
            for iteration in range(max_iterations):
                new_mappings = {}
                for parent_id, children in parent_to_posts.items():
                    if parent_id in post_to_thread:
                        thread_id = post_to_thread[parent_id]
                        for child_id in children:
                            if child_id not in post_to_thread:
                                new_mappings[child_id] = thread_id

                if not new_mappings:
                    break

                post_to_thread.update(new_mappings)

            # Apply thread_id mapping
            df['thread_id'] = df['post_id'].map(post_to_thread)

            # Drop posts that couldn't be assigned (orphaned)
            original_count = len(df)
            df = df[df['thread_id'].notna()]
            if len(df) < original_count:
                print(f"  Dropped {original_count - len(df)} orphaned posts", file=sys.stderr)

            print(f"  Reconstructed {df['thread_id'].nunique()} threads from parent_id structure",
                  file=sys.stderr)
        else:
            print(f"Warning: No thread_id or parent_id in {path.name}, treating each post as separate document",
                  file=sys.stderr)
            # Just process normally without aggregation
            texts_series = vtc.build_text_series(df, used_cols)
            texts_series = texts_series.apply(vtc.clean_text)
            texts_series = texts_series[texts_series.str.len() >= min_thread_chars]

            if max_threads is not None and len(texts_series) > max_threads:
                texts_series = texts_series.sample(n=max_threads, random_state=seed)

            texts = texts_series.tolist()
            thread_df = pd.DataFrame({'thread_id': range(len(texts)), 'thread_text': texts, 'comment_count': 1})

            print(f"Loaded {len(texts)} individual posts/comments (no thread aggregation)", file=sys.stderr)
            return texts, thread_df, used_cols

    # Aggregate by thread
    thread_df = aggregate_threads(df, used_cols, min_thread_chars=min_thread_chars)

    # Sample if requested
    if max_threads is not None and len(thread_df) > max_threads:
        thread_df = thread_df.sample(n=max_threads, random_state=seed)

    texts = thread_df['thread_text'].tolist()

    print(f"Loaded {len(texts)} threads from {len(df)} comments "
          f"(avg {len(df)/len(texts):.1f} comments/thread)", file=sys.stderr)

    return texts, thread_df, used_cols


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Thread-level topic comparison between simulation and MADOC."
    )

    # Input/output
    parser.add_argument("--sim2-posts-csv", type=Path, required=True)
    parser.add_argument("--madoc-input", type=Path,
                       default=Path("MADOC/voat-technology/voat_technology_madoc.parquet"))
    parser.add_argument("--outdir", type=Path, required=True)

    # Text columns
    parser.add_argument("--text-cols", type=str, default=None)
    parser.add_argument("--sim2-text-cols", type=str, default=None)
    parser.add_argument("--madoc-text-cols", type=str, default=None)

    # Thread filtering
    parser.add_argument("--min-thread-chars", type=int, default=50,
                       help="Minimum characters for aggregated thread text")
    parser.add_argument("--max-threads", type=int, default=None)
    parser.add_argument("--sim2-max-threads", type=int, default=None)
    parser.add_argument("--madoc-max-threads", type=int, default=10000)

    # Topic modeling parameters
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--sim2-min-topic-size", type=int, default=None)
    parser.add_argument("--madoc-min-topic-size", type=int, default=None)
    parser.add_argument("--top-n-words", type=int, default=10)
    parser.add_argument("--vectorizer-min-df", type=int, default=1)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--topk-per-sim2", type=int, default=3)

    # Advanced parameters
    parser.add_argument("--topic-repr", type=str, default="centroid",
                       choices=["centroid", "bertopic", "hybrid"])
    parser.add_argument("--repr-alpha", type=float, default=0.5)
    parser.add_argument("--composite-alpha", type=float, default=0.8)
    parser.add_argument("--word-sim-threshold", type=float, default=0.6)
    parser.add_argument("--centroid-max-docs", type=int, default=200)

    # Text preprocessing
    parser.add_argument("--extra-stopwords", type=str, default=None)
    parser.add_argument("--remove-contraction-fragments", action="store_true")
    parser.add_argument("--df-threshold", type=float, default=0.4)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--embedding-model", type=str,
                       default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--save-heatmap", action="store_true")

    args = parser.parse_args(argv)

    vtc.set_seeds(args.seed)
    vtc.ensure_outdir(args.outdir)

    # Parse text columns
    text_cols_global = [c.strip() for c in args.text_cols.split(",")] if args.text_cols else None
    sim2_text_cols = [c.strip() for c in args.sim2_text_cols.split(",")] if args.sim2_text_cols else text_cols_global
    if args.madoc_text_cols:
        madoc_text_cols = [c.strip() for c in args.madoc_text_cols.split(",")]
    elif text_cols_global:
        madoc_text_cols = text_cols_global
    else:
        madoc_text_cols = ["content"]

    # Load thread-level corpora
    sim2_max_threads = args.sim2_max_threads if args.sim2_max_threads is not None else args.max_threads
    madoc_max_threads = args.madoc_max_threads if args.madoc_max_threads is not None else args.max_threads

    print("Loading simulation threads...", file=sys.stderr)
    sim2_texts, sim2_df, sim2_used_cols = load_thread_corpus(
        args.sim2_posts_csv,
        text_cols=sim2_text_cols,
        min_thread_chars=args.min_thread_chars,
        max_threads=sim2_max_threads,
        seed=args.seed,
    )

    print("Loading MADOC threads...", file=sys.stderr)
    madoc_texts, madoc_df, madoc_used_cols = load_thread_corpus(
        args.madoc_input,
        text_cols=madoc_text_cols,
        min_thread_chars=args.min_thread_chars,
        max_threads=madoc_max_threads,
        seed=args.seed,
    )

    if len(sim2_texts) < 10:
        print(f"Warning: Very few simulation threads ({len(sim2_texts)})", file=sys.stderr)
    if len(madoc_texts) < 50:
        print(f"Warning: Very few MADOC threads ({len(madoc_texts)})", file=sys.stderr)

    # Build embedding model
    embedding_model = vtc.make_embedding_model(
        device_preference=args.device,
        model_name=args.embedding_model,
    )

    # Topic modeling parameters
    sim2_min_topic_size = args.sim2_min_topic_size if args.sim2_min_topic_size is not None else args.min_topic_size
    madoc_min_topic_size = args.madoc_min_topic_size if args.madoc_min_topic_size is not None else args.min_topic_size

    # Adaptive sizing
    sim2_min_topic_size = vtc.adaptive_min_topic_size(len(sim2_texts), sim2_min_topic_size)
    madoc_min_topic_size = max(2, madoc_min_topic_size)

    print(f"Using min_topic_size: sim2={sim2_min_topic_size}, madoc={madoc_min_topic_size}",
          file=sys.stderr)

    # Stopwords
    common_extra_stop = set()
    if args.extra_stopwords:
        common_extra_stop.update([w.strip().lower() for w in args.extra_stopwords.split(",") if w.strip()])
    if args.remove_contraction_fragments:
        common_extra_stop.update({"don", "ve", "ll", "re", "isn", "ain", "cant", "won",
                                  "shouldn", "wouldn", "couldn"})

    sim2_stopwords = vtc.build_stopword_set(
        sim2_texts, sim2_used_cols, common_extra_stop,
        min_token_freq=0.0, df_threshold=args.df_threshold, min_docs_required=20
    )
    madoc_stopwords = vtc.build_stopword_set(
        madoc_texts, madoc_used_cols, common_extra_stop,
        min_token_freq=0.0, df_threshold=args.df_threshold, min_docs_required=100
    )

    # Build and fit topic models
    print("Fitting simulation topic model...", file=sys.stderr)
    sim2_hdbscan_kwargs = {}
    if len(sim2_texts) <= 1500:
        sim2_hdbscan_kwargs.update({
            "cluster_selection_method": "leaf",
            "allow_single_cluster": True,
            "min_samples": 1,
        })

    topic_model_sim2 = vtc.build_bertopic(
        embedding_model,
        min_topic_size=sim2_min_topic_size,
        seed=args.seed,
        vectorizer_min_df=args.vectorizer_min_df,
        hdbscan_kwargs=sim2_hdbscan_kwargs,
    )
    vtc.apply_stopwords(topic_model_sim2, sim2_stopwords)

    try:
        _topics_sim2, _ = topic_model_sim2.fit_transform(sim2_texts)
    except ValueError as e:
        if "empty vocabulary" in str(e):
            print("Warning: Falling back to SimpleTopicModel for simulation", file=sys.stderr)
            topic_model_sim2 = vtc.SimpleTopicModel(
                seed=args.seed,
                min_topic_size=sim2_min_topic_size,
                top_n_words=args.top_n_words
            )
            _topics_sim2, _ = topic_model_sim2.fit_transform(sim2_texts)
        else:
            raise

    print("Fitting MADOC topic model...", file=sys.stderr)
    topic_model_madoc = vtc.build_bertopic(
        embedding_model,
        min_topic_size=madoc_min_topic_size,
        seed=args.seed,
        vectorizer_min_df=args.vectorizer_min_df,
    )
    vtc.apply_stopwords(topic_model_madoc, madoc_stopwords)

    try:
        _topics_madoc, _ = topic_model_madoc.fit_transform(madoc_texts)
    except ValueError as e:
        if "empty vocabulary" in str(e):
            print("Warning: Falling back to SimpleTopicModel for MADOC", file=sys.stderr)
            topic_model_madoc = vtc.SimpleTopicModel(
                seed=args.seed,
                min_topic_size=madoc_min_topic_size,
                top_n_words=args.top_n_words
            )
            _topics_madoc, _ = topic_model_madoc.fit_transform(madoc_texts)
        else:
            raise

    # Extract topic info
    sim2_infos = vtc.extract_topic_info(topic_model_sim2, sim2_texts, top_n_words=args.top_n_words)
    madoc_infos = vtc.extract_topic_info(topic_model_madoc, madoc_texts, top_n_words=args.top_n_words)

    print(f"Discovered {len(sim2_infos)} simulation topics, {len(madoc_infos)} MADOC topics",
          file=sys.stderr)

    # Compute topic vectors
    topics_sim2, _ = topic_model_sim2.transform(sim2_texts)
    topics_madoc, _ = topic_model_madoc.transform(madoc_texts)

    if args.topic_repr == "centroid":
        sim2_embs = vtc.compute_topic_centroids(
            sim2_texts, topics_sim2, sim2_infos, embedding_model,
            max_docs_per_topic=args.centroid_max_docs
        )
        madoc_embs = vtc.compute_topic_centroids(
            madoc_texts, topics_madoc, madoc_infos, embedding_model,
            max_docs_per_topic=args.centroid_max_docs
        )
    elif args.topic_repr == "bertopic":
        sim2_embs = vtc.compute_topic_embeddings(topic_model_sim2, sim2_infos, embedding_model)
        madoc_embs = vtc.compute_topic_embeddings(topic_model_madoc, madoc_infos, embedding_model)
    else:  # hybrid
        sim2_c = vtc.compute_topic_centroids(
            sim2_texts, topics_sim2, sim2_infos, embedding_model,
            max_docs_per_topic=args.centroid_max_docs
        )
        madoc_c = vtc.compute_topic_centroids(
            madoc_texts, topics_madoc, madoc_infos, embedding_model,
            max_docs_per_topic=args.centroid_max_docs
        )
        sim2_b = vtc.compute_topic_embeddings(topic_model_sim2, sim2_infos, embedding_model)
        madoc_b = vtc.compute_topic_embeddings(topic_model_madoc, madoc_infos, embedding_model)
        a = float(args.repr_alpha)
        sim2_embs = a * sim2_c + (1 - a) * sim2_b
        madoc_embs = a * madoc_c + (1 - a) * madoc_b

    # Similarity and matching (reuse existing logic)
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(sim2_embs, madoc_embs) if (sim2_embs.size and madoc_embs.size) else np.zeros((len(sim2_embs), len(madoc_embs)))
    pairs, unmatched_sim2_idx, unmatched_madoc_idx, topk_indices, topk_scores = vtc.match_topics_topk(
        sim_matrix, threshold=args.similarity_threshold, topk=args.topk_per_sim2
    )

    # Rerank with composite score
    composite_alpha = float(args.composite_alpha)
    reranked_pairs = []
    heatmap_scores = np.zeros_like(topk_scores)
    for i in range(len(sim2_infos)):
        row_pairs = [(r, c, s, rk) for (r, c, s, rk) in pairs if r == i]
        if not row_pairs:
            continue
        scored = []
        for (r, c, s, rk) in row_pairs:
            sj = vtc.soft_jaccard(
                sim2_infos[r].top_words, madoc_infos[c].top_words,
                embedding_model, sim_thresh=args.word_sim_threshold
            )
            comp = composite_alpha * s + (1 - composite_alpha) * sj
            scored.append((r, c, s, rk, sj, comp))
        scored.sort(key=lambda x: -x[5])
        for new_rank, (r, c, s, rk, sj, comp) in enumerate(scored[:args.topk_per_sim2]):
            reranked_pairs.append((r, c, s, new_rank))
            heatmap_scores[r, new_rank] = s
    pairs = reranked_pairs
    topk_scores = heatmap_scores

    # Save outputs
    sim2_labels = [ti.label for ti in sim2_infos]
    madoc_labels = [ti.label for ti in madoc_infos]

    sim2_info_df = pd.DataFrame({
        "topic_id": [t.topic_id for t in sim2_infos],
        "size": [t.size for t in sim2_infos],
        "label": sim2_labels,
        "top_words": [" ".join(t.top_words) for t in sim2_infos],
    })
    madoc_info_df = pd.DataFrame({
        "topic_id": [t.topic_id for t in madoc_infos],
        "size": [t.size for t in madoc_infos],
        "label": madoc_labels,
        "top_words": [" ".join(t.top_words) for t in madoc_infos],
    })
    vtc.write_csv(sim2_info_df, args.outdir / "topic_info_sim2.csv")
    vtc.write_csv(madoc_info_df, args.outdir / "topic_info_madoc.csv")

    # Matches table
    matches_rows = []
    for r_idx, c_idx, sim, rank in pairs:
        s_info = sim2_infos[r_idx]
        m_info = madoc_infos[c_idx]
        jac = vtc.jaccard(s_info.top_words, m_info.top_words)
        sjac = vtc.soft_jaccard(
            s_info.top_words, m_info.top_words, embedding_model,
            sim_thresh=args.word_sim_threshold
        )
        matches_rows.append({
            "sim2_topic_id": s_info.topic_id,
            "sim2_label": s_info.label,
            "sim2_size": s_info.size,
            "madoc_topic_id": m_info.topic_id,
            "madoc_label": m_info.label,
            "madoc_size": m_info.size,
            "cosine_sim": sim,
            "jaccard_topn": jac,
            "soft_jaccard_topn": sjac,
            "rank": rank + 1,
        })
    matches_df = pd.DataFrame(matches_rows)
    vtc.write_csv(matches_df, args.outdir / "topic_matches.csv")

    # Full similarity matrix
    sim_df = pd.DataFrame(sim_matrix, columns=[f"madoc_{i}" for i in range(sim_matrix.shape[1])])
    sim_df.insert(0, "sim2_idx", range(sim_matrix.shape[0]))
    vtc.write_csv(sim_df, args.outdir / "topic_similarity.csv")

    # Summary
    matched_sim2_set = {p[0] for p in pairs}
    coverage = float(len(matched_sim2_set) / max(1, len(sim2_infos)))
    best_matches = [row for row in matches_rows if row.get("rank", 1) == 1]
    mean_cos = float(np.mean([m["cosine_sim"] for m in best_matches])) if best_matches else 0.0
    med_cos = float(np.median([m["cosine_sim"] for m in best_matches])) if best_matches else 0.0
    mean_jac = float(np.mean([m["jaccard_topn"] for m in best_matches])) if best_matches else 0.0
    med_jac = float(np.median([m["jaccard_topn"] for m in best_matches])) if best_matches else 0.0
    mean_sjac = float(np.mean([m["soft_jaccard_topn"] for m in best_matches])) if best_matches else 0.0
    med_sjac = float(np.median([m["soft_jaccard_topn"] for m in best_matches])) if best_matches else 0.0

    summary = {
        "sim2_threads": len(sim2_texts),
        "madoc_threads": len(madoc_texts),
        "sim2_topics": len(sim2_infos),
        "madoc_topics": len(madoc_infos),
        "matches": len(matches_rows),
        "coverage_sim2": coverage,
        "similarity_threshold": args.similarity_threshold,
        "mean_cosine": mean_cos,
        "median_cosine": med_cos,
        "mean_jaccard": mean_jac,
        "median_jaccard": med_jac,
        "mean_soft_jaccard": mean_sjac,
        "median_soft_jaccard": med_sjac,
        "unmatched_sim2_count": len(unmatched_sim2_idx),
        "unmatched_madoc_count": len(unmatched_madoc_idx),
        "seed": args.seed,
    }
    with open(args.outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Heatmap
    if args.save_heatmap and sim_matrix.size > 0:
        heatmap_path = args.outdir / "similarity_heatmap.png"
        vtc.plot_heatmap_topk(sim2_labels, topk_scores, heatmap_path)

    # Run info
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_info = {
        "args": args_dict,
        "sim2_used_text_cols": sim2_used_cols,
        "madoc_used_text_cols": madoc_used_cols,
        "timestamp": int(time.time()),
        "granularity": "thread",  # Mark this as thread-level
    }
    with open(args.outdir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    print(f"Thread-level topic analysis complete. Outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
