#!/usr/bin/env python3
"""
extract_run_metrics.py - Extract unified metrics from a single run's outputs.

Parses all analysis outputs in a run directory and produces a single metrics.json
with scalar metrics suitable for aggregation across 30 runs.

Usage:
    python scripts/extract_run_metrics.py --run-dir results/run01
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def safe_float(val: Any) -> Optional[float]:
    """Convert to float, return None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def extract_activity_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract activity metrics from posts.csv and users.csv."""
    metrics: Dict[str, Any] = {}

    posts_path = run_dir / "posts.csv"
    if posts_path.exists():
        try:
            df = pd.read_csv(posts_path)
            metrics["total_posts"] = len(df)

            # Count comments (comment_to != -1)
            if "comment_to" in df.columns:
                comments = (df["comment_to"] != -1).sum()
                metrics["comments"] = int(comments)
                metrics["root_posts"] = int(len(df) - comments)

            if "user_id" in df.columns:
                metrics["unique_users"] = int(df["user_id"].nunique())
                posts_per_user = df.groupby("user_id").size()
                metrics["mean_posts_per_user"] = float(posts_per_user.mean())
                metrics["median_posts_per_user"] = float(posts_per_user.median())
                metrics["max_posts_per_user"] = int(posts_per_user.max())

            if "thread_id" in df.columns:
                thread_sizes = df.groupby("thread_id").size()
                metrics["avg_thread_length"] = float(thread_sizes.mean())
                metrics["num_threads"] = int(df["thread_id"].nunique())

            # Daily active users (round / 24 = day)
            if "round" in df.columns and "user_id" in df.columns:
                df_copy = df.copy()
                df_copy["day"] = df_copy["round"] // 24
                daily_active = df_copy.groupby("day")["user_id"].nunique()
                metrics["mean_daily_active_users"] = float(daily_active.mean())
                metrics["num_days"] = int(df_copy["day"].nunique())

        except Exception as e:
            logger.warning(f"Error reading posts.csv: {e}")

    users_path = run_dir / "users.csv"
    if users_path.exists():
        try:
            users_df = pd.read_csv(users_path)
            metrics["total_users_registered"] = len(users_df)
        except Exception as e:
            logger.warning(f"Error reading users.csv: {e}")

    return metrics


def extract_toxicity_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract toxicity metrics from toxigen.csv and toxigen_summary.txt."""
    metrics: Dict[str, Any] = {}

    tox_path = run_dir / "toxigen.csv"
    if tox_path.exists():
        try:
            df = pd.read_csv(tox_path)
            if "toxicity" in df.columns:
                metrics["mean"] = float(df["toxicity"].mean())
                metrics["median"] = float(df["toxicity"].median())
                metrics["std"] = float(df["toxicity"].std())
                metrics["max"] = float(df["toxicity"].max())
                metrics["p90"] = float(df["toxicity"].quantile(0.9))
                metrics["p95"] = float(df["toxicity"].quantile(0.95))

                # By post type if available
                if "post_type" in df.columns:
                    type_means = df.groupby("post_type")["toxicity"].mean().to_dict()
                    for k, v in type_means.items():
                        key = k.lower().replace(" ", "_")
                        metrics[f"{key}_mean"] = float(v)

                # Fraction above thresholds
                metrics["frac_above_0.5"] = float((df["toxicity"] > 0.5).mean())
                metrics["frac_above_0.8"] = float((df["toxicity"] > 0.8).mean())
        except Exception as e:
            logger.warning(f"Error reading toxigen.csv: {e}")

    return metrics


def extract_network_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract network metrics from network_analysis.txt."""
    metrics: Dict[str, Any] = {}

    # Try both old and new names
    for fname in ["network_analysis.txt", "enhanced_network_analysis.txt"]:
        txt_path = run_dir / fname
        if txt_path.exists():
            try:
                content = txt_path.read_text(encoding="utf-8", errors="ignore")
                # Prefer anchored patterns to avoid accidental early matches
                # such as "Average core size: 78.1 nodes".
                patterns = {
                    # Network statistics (whole graph)
                    "nodes": r"(?m)^\s*-\s*Num\s+Nodes:\s*([\d,]+)\s*$",
                    "edges": r"(?m)^\s*-\s*Num\s+Edges:\s*([\d,]+)\s*$",
                    "avg_degree": r"(?m)^\s*-\s*Avg\s+Degree:\s*([\d.]+)\s*$",
                    "avg_weighted_degree": r"(?m)^\s*-\s*Avg\s+Weighted\s+Degree:\s*([\d.]+)\s*$",
                    "avg_clustering": r"(?m)^\s*-\s*Avg\s+Clustering:\s*([\d.]+)\s*$",
                    "density_total": r"(?m)^\s*-\s*Density:\s*([\d.]+)\s*$",
                    "lcc_nodes": r"(?m)^\s*-\s*Largest\s+Component\s+Size:\s*([\d,]+)\s*$",
                    "lcc_ratio": r"(?m)^\s*-\s*Largest\s+Component\s+Ratio:\s*([\d.]+)\s*$",
                    # Largest connected component stats (from scope section)
                    "density_lcc": r"(?m)^\s*-\s*Component\s+density:\s*([\d.]+)\s*$",
                    # Best partition (core/periphery)
                    "core_nodes": r"(?m)^\s*-\s*Core\s+Size:\s*([\d,]+)\s+nodes",
                    "periphery_nodes": r"(?m)^\s*-\s*Periphery\s+Size:\s*([\d,]+)\s+nodes",
                    "core_pct": r"(?m)^\s*-\s*Core\s+Size:\s*[\d,]+\s+nodes\s*\(([\d.]+)%\s+of\s+analyzed\s+component\)",
                    "modularity": r"(?m)^\s*-\s*Modularity:\s*([-\d.]+)\s*$",
                    "core_density": r"(?m)^\s*-\s*Core\s+Density:\s*([\d.]+)\s*$",
                    "core_periphery_density": r"(?m)^\s*-\s*Core-Periphery\s+Density:\s*([\d.]+)\s*$",
                    "core_avg_degree": r"(?m)^\s*-\s*Core\s+Avg\s+Degree:\s*([\d.]+)\s*$",
                }
                for key, pattern in patterns.items():
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        val_str = match.group(1).replace(",", "")
                        metrics[key] = safe_float(val_str)

                # Backwards-compatible aliases used by aggregators.
                if "density_lcc" in metrics:
                    metrics["density"] = metrics["density_lcc"]
                elif "density_total" in metrics:
                    metrics["density"] = metrics["density_total"]

                # Derived: core % of analyzed component if missing.
                if (
                    "core_pct" not in metrics
                    and metrics.get("core_nodes") is not None
                    and metrics.get("periphery_nodes") is not None
                ):
                    denom = metrics["core_nodes"] + metrics["periphery_nodes"]
                    if denom:
                        metrics["core_pct"] = 100.0 * metrics["core_nodes"] / denom
                break
            except Exception as e:
                logger.warning(f"Error reading {fname}: {e}")

    return metrics


def extract_entropy_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract convergence entropy metrics from entropy_agg.json and pairs_all.csv."""
    metrics: Dict[str, Any] = {}

    # Try flat outputs first, then legacy names.
    for fname in ["entropy_agg.json", "agg_stats.json"]:
        json_path = run_dir / fname
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, dict) and isinstance(data.get("overall"), dict):
                    overall = data["overall"]
                    h_stats = overall.get("H") if isinstance(overall.get("H"), dict) else {}
                    hpt_stats = (
                        overall.get("H_per_token")
                        if isinstance(overall.get("H_per_token"), dict)
                        else {}
                    )
                    # Overall H
                    metrics["H_mean"] = safe_float(h_stats.get("mean"))
                    metrics["H_median"] = safe_float(h_stats.get("median"))
                    metrics["H_std"] = safe_float(h_stats.get("std"))
                    metrics["H_p10"] = safe_float(h_stats.get("p10"))
                    metrics["H_p90"] = safe_float(h_stats.get("p90"))
                    # Overall H per token
                    metrics["H_per_token_mean"] = safe_float(hpt_stats.get("mean"))
                    metrics["H_per_token_median"] = safe_float(hpt_stats.get("median"))
                    metrics["H_per_token_std"] = safe_float(hpt_stats.get("std"))
                    metrics["H_per_token_p10"] = safe_float(hpt_stats.get("p10"))
                    metrics["H_per_token_p90"] = safe_float(hpt_stats.get("p90"))
                    metrics["n_pairs"] = safe_float(
                        h_stats.get("count") or hpt_stats.get("count")
                    )
                else:
                    # Fallback for older flat payloads.
                    for key in [
                        "H_mean",
                        "H_median",
                        "H_per_token_mean",
                        "H_per_token_median",
                        "n_pairs",
                    ]:
                        if isinstance(data, dict) and key in data:
                            metrics[key] = safe_float(data[key])
                break
            except Exception as e:
                logger.warning(f"Error reading {fname}: {e}")

    # Parse pairs_all.csv for inter/intra breakdown
    pairs_path = run_dir / "pairs_all.csv"
    if pairs_path.exists():
        try:
            df = pd.read_csv(pairs_path)
            if "pair_type" in df.columns and "H_per_token" in df.columns:
                for ptype in ["interpersonal", "intrapersonal"]:
                    subset = df[df["pair_type"] == ptype]["H_per_token"]
                    if len(subset) > 0:
                        prefix = "inter" if ptype == "interpersonal" else "intra"
                        metrics[f"{prefix}_hpt_mean"] = float(subset.mean())
                        metrics[f"{prefix}_hpt_n"] = len(subset)
        except Exception as e:
            logger.warning(f"Error reading pairs_all.csv: {e}")

    return metrics


def extract_topic_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract topic metrics from topic_summary.json or summary.json."""
    metrics: Dict[str, Any] = {}

    for fname in ["topic_summary.json", "summary.json"]:
        json_path = run_dir / fname
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, dict) and ("sim2_topics" in data or "madoc_topics" in data):
                    wanted = [
                        "sim2_docs",
                        "madoc_docs",
                        "sim2_topics",
                        "madoc_topics",
                        "matches",
                        "coverage_sim2",
                        "similarity_threshold",
                        "mean_cosine",
                        "median_cosine",
                        "mean_jaccard",
                        "median_jaccard",
                        "mean_soft_jaccard",
                        "median_soft_jaccard",
                        "unmatched_sim2_count",
                        "unmatched_madoc_count",
                    ]
                    for key in wanted:
                        if key in data:
                            metrics[key] = safe_float(data[key])
                else:
                    # Legacy BERTopic summary shapes.
                    for key in [
                        "n_topics",
                        "n_topics_sim",
                        "n_topics_madoc",
                        "coherence",
                        "diversity",
                    ]:
                        if isinstance(data, dict) and key in data:
                            metrics[key] = safe_float(data[key])
                    if (
                        isinstance(data, dict)
                        and "similarity" in data
                        and isinstance(data["similarity"], dict)
                    ):
                        for k, v in data["similarity"].items():
                            metrics[f"sim_{k}"] = safe_float(v)
                break
            except Exception as e:
                logger.warning(f"Error reading {fname}: {e}")

    return metrics


def extract_embedding_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract embedding similarity metrics from sim_voat_embedding_similarity.json."""
    metrics: Dict[str, Any] = {}

    for fname in ["sim_voat_embedding_similarity.json", "embedding_similarity.json"]:
        json_path = run_dir / fname
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                metrics["embedding_model"] = data.get("embedding_model")
                for item in data["results"]:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label", "")).lower()
                    if label not in {"posts", "comments"}:
                        continue
                    for key in [
                        "mean_cosine",
                        "median_cosine",
                        "count_ge_high",
                        "count_ge_mid_lt_high",
                        "threshold_mid",
                        "threshold_high",
                        "n_sim2",
                        "n_voat",
                    ]:
                        if key in item:
                            metrics[f"{label}_{key}"] = safe_float(item[key])
            elif isinstance(data, dict):
                # Legacy nested structure by content type.
                for content_type in ["posts", "comments"]:
                    if content_type in data and isinstance(data[content_type], dict):
                        sub = data[content_type]
                        for key in [
                            "mean_cosine",
                            "median_cosine",
                            "std_cosine",
                            "n_pairs",
                        ]:
                            if key in sub:
                                metrics[f"{content_type}_{key}"] = safe_float(sub[key])
            break
        except Exception as e:
            logger.warning(f"Error reading {fname}: {e}")
            break

    return metrics


def extract_thread_topic_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract thread-level topic metrics from topic_threads/summary.json."""
    metrics: Dict[str, Any] = {}
    json_path = run_dir / "topic_threads" / "summary.json"
    if not json_path.exists():
        return metrics
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(data, dict):
            wanted = [
                "sim2_threads",
                "madoc_threads",
                "sim2_topics",
                "madoc_topics",
                "matches",
                "coverage_sim2",
                "similarity_threshold",
                "mean_cosine",
                "median_cosine",
                "mean_jaccard",
                "median_jaccard",
                "mean_soft_jaccard",
                "median_soft_jaccard",
                "unmatched_sim2_count",
                "unmatched_madoc_count",
            ]
            for key in wanted:
                if key in data:
                    metrics[key] = safe_float(data[key])
    except Exception as e:
        logger.warning(f"Error reading thread topic summary: {e}")
    return metrics


def extract_semantic_diversity_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract a compact numeric summary from diversity/semantic_diversity_summary.csv."""
    metrics: Dict[str, Any] = {}
    csv_path = run_dir / "diversity" / "semantic_diversity_summary.csv"
    if not csv_path.exists():
        return metrics
    try:
        df = pd.read_csv(csv_path)
        if "mean_pairwise_cosine_distance" not in df.columns:
            return metrics
        if "status" in df.columns:
            ok_df = df[df["status"] == "ok"]
        else:
            ok_df = df
        vals = ok_df["mean_pairwise_cosine_distance"].dropna()
        if len(vals) == 0:
            return metrics
        metrics["mean_pairwise_cosine_distance_mean"] = float(vals.mean())
        metrics["mean_pairwise_cosine_distance_median"] = float(vals.median())
        metrics["mean_pairwise_cosine_distance_n"] = int(vals.shape[0])
    except Exception as e:
        logger.warning(f"Error reading semantic diversity summary: {e}")
    return metrics


def extract_all_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract all metrics from a run directory."""
    run_name = run_dir.name

    result = {
        "run": run_name,
        "activity": extract_activity_metrics(run_dir),
        "toxicity": extract_toxicity_metrics(run_dir),
        "network": extract_network_metrics(run_dir),
        "entropy": extract_entropy_metrics(run_dir),
        "topic": extract_topic_metrics(run_dir),
        "topic_threads": extract_thread_topic_metrics(run_dir),
        "embedding": extract_embedding_metrics(run_dir),
        "semantic_diversity": extract_semantic_diversity_metrics(run_dir),
    }

    # Count how many categories have data
    categories_with_data = sum(
        1
        for cat in [
            "activity",
            "toxicity",
            "network",
            "entropy",
            "topic",
            "topic_threads",
            "embedding",
            "semantic_diversity",
        ]
        if result.get(cat)
    )
    result["_meta"] = {
        "categories_with_data": categories_with_data,
        "run_dir": str(run_dir),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract unified metrics from a run directory")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to run output directory")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: <run-dir>/metrics.json)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    output_path = args.output or (run_dir / "metrics.json")

    logger.info(f"Extracting metrics from {run_dir}")
    metrics = extract_all_metrics(run_dir)

    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Wrote metrics to {output_path}")

    # Summary
    cats = metrics.get("_meta", {}).get("categories_with_data", 0)
    logger.info(f"Categories with data: {cats}/8")


if __name__ == "__main__":
    main()
