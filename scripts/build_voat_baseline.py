#!/usr/bin/env python3
"""
build_voat_baseline.py
======================
Aggregate per-sample analysis outputs from a Voat samples directory into a
single baseline JSON file compatible with aggregate_30runs.py (--voat-ci).

Reads from each sample_N subdirectory:
  - full_results.json       → user dynamics, toxicity, thread metrics
  - avg_interactions.json   → interaction rate / active user metrics
  - post_distribution.json  → post/comment ratio
  - enhanced_network_analysis.txt (optional) → network topology metrics

Output format (flat dict):
  {
    "metric_name": {
      "mean": float, "ci_lower": float, "ci_upper": float,
      "std": float, "n_samples": int
    },
    ...
  }

Usage:
    python scripts/build_voat_baseline.py \\
        --samples-dir MADOC/voat-technology-nonoverlap \\
        --output MADOC/voat-technology-nonoverlap/baseline_ci.json

    # or against the original overlapping samples:
    python scripts/build_voat_baseline.py \\
        --samples-dir MADOC/voat-technology \\
        --output MADOC/voat-technology/baseline_ci.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_metrics import (
    compute_repeated_interaction_pct_from_parquet,
    summarize_values,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def metric_entry(
    values: List[float],
    confidence: float,
    ci_method: str,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    return summarize_values(
        values,
        confidence=confidence,
        ci_method=ci_method,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Per-sample extractors
# ---------------------------------------------------------------------------

def extract_summary_metrics(sample_dir: Path) -> Optional[Dict[str, float]]:
    """Extract scalar summary metrics from full_results.json,
    avg_interactions.json, and post_distribution.json."""

    full_results_path = sample_dir / "full_results.json"
    avg_int_path      = sample_dir / "avg_interactions.json"
    post_dist_path    = sample_dir / "post_distribution.json"

    if not full_results_path.exists():
        logger.warning(f"  Missing full_results.json in {sample_dir.name}")
        return None

    with open(full_results_path) as f:
        fr = json.load(f)

    metrics: Dict[str, float] = {}

    # -- user dynamics -------------------------------------------------------
    ud = fr.get("user_dynamics", {})
    _safe(metrics, "unique_users",         ud, "total_unique_users")
    _safe(metrics, "avg_new_users_pct",    ud, "avg_new_users_percentage_after_day10")
    _safe(metrics, "avg_churn_pct",        ud, "avg_churn_percentage_after_day10")

    ai = fr.get("avg_interactions", {})
    if ai:
        _safe(metrics, "avg_unique_active_users_per_day", ai, "avg_unique_active_users_per_day")
        _safe(metrics, "avg_interactions_per_user_per_day", ai, "overall_avg_interactions_per_user_per_day")
    elif avg_int_path.exists():
        with open(avg_int_path) as f:
            ai = json.load(f)
        _safe(metrics, "avg_unique_active_users_per_day", ai, "avg_unique_active_users_per_day")
        _safe(metrics, "avg_interactions_per_user_per_day", ai, "overall_avg_interactions_per_user_per_day")

    # -- post / comment ratio ------------------------------------------------
    pd = fr.get("post_distribution", {})
    if pd:
        _safe(metrics, "post_comment_ratio", pd, "post_comment_ratio")
        _safe(metrics, "avg_posts_per_day", pd, "avg_posts_per_day")
        _safe(metrics, "avg_comments_per_day", pd, "avg_comments_per_day")
        _safe(metrics, "root_posts", pd, "total_posts")
        _safe(metrics, "comments", pd, "total_comments")
        if metrics.get("root_posts") is not None and metrics.get("comments") is not None:
            metrics["total_interactions"] = metrics["root_posts"] + metrics["comments"]
    elif post_dist_path.exists():
        with open(post_dist_path) as f:
            pd = json.load(f)
        _safe(metrics, "post_comment_ratio", pd, "post_comment_ratio")
        _safe(metrics, "avg_posts_per_day", pd, "avg_posts_per_day")
        _safe(metrics, "avg_comments_per_day", pd, "avg_comments_per_day")

    # -- toxicity ------------------------------------------------------------
    tox = fr.get("toxicity", {})
    ts  = tox.get("toxicity_stats", {})
    if ts and tox.get("has_toxicity_data"):
        _safe(metrics, "avg_toxicity",    ts, "mean")
        _safe(metrics, "median_toxicity", ts, "median")
    interaction_toxicity = tox.get("interaction_toxicity", {})
    if interaction_toxicity:
        posts_tox = interaction_toxicity.get("posts", {})
        comments_tox = interaction_toxicity.get("comments", {})
        _safe(metrics, "posts_mean", posts_tox, "mean")
        _safe(metrics, "comments_mean", comments_tox, "mean")

    # -- thread length -------------------------------------------------------
    tl = fr.get("thread_length", {})
    if tl.get("success"):
        _safe(metrics, "avg_thread_length",    tl, "avg_thread_length")
        _safe(metrics, "median_thread_length", tl, "median_thread_length")

    # -- thread activity -----------------------------------------------------
    ta = fr.get("thread_activity", {})
    if ta.get("success"):
        _safe(metrics, "avg_thread_active_days",    ta, "avg_active_days")
        _safe(metrics, "median_thread_active_days", ta, "median_active_days")

    return metrics


def _safe(out: dict, key: str, src: dict, src_key: str) -> None:
    val = src.get(src_key)
    if val is not None:
        try:
            out[key] = float(val)
        except (TypeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Network metrics from enhanced_network_analysis.txt  (optional)
# ---------------------------------------------------------------------------

_NET_PATTERNS = {
    "nodes": r"(?m)^\s*-\s*Num Nodes:\s*([\d,]+)\s*$",
    "edges": r"(?m)^\s*-\s*Num Edges:\s*([\d,]+)\s*$",
    "avg_degree": r"(?m)^\s*-\s*Avg Degree:\s*([0-9.eE+\-]+)\s*$",
    "avg_weighted_degree": r"(?m)^\s*-\s*Avg Weighted Degree:\s*([0-9.eE+\-]+)\s*$",
    "avg_clustering": r"(?m)^\s*-\s*Avg Clustering:\s*([0-9.eE+\-]+)\s*$",
    "density_total": r"(?m)^\s*-\s*Density:\s*([0-9.eE+\-]+)\s*$",
    "lcc_nodes": r"(?m)^\s*-\s*Largest Component Size:\s*([\d,]+)\s*$",
    "lcc_ratio": r"(?m)^\s*-\s*Largest Component Ratio:\s*([0-9.eE+\-]+)\s*$",
    "density": r"(?m)^\s*-\s*Component density:\s*([0-9.eE+\-]+)\s*$",
    "core_nodes": r"(?m)^\s*-\s*Core Size:\s*([\d,]+)\s+nodes",
    "periphery_nodes": r"(?m)^\s*-\s*Periphery Size:\s*([\d,]+)\s+nodes",
    "core_pct": r"(?m)^\s*-\s*Core Size:\s*[\d,]+\s+nodes\s*\(([0-9.eE+\-]+)%\s+of analyzed component\)",
    "core_density": r"(?m)^\s*-\s*Core Density:\s*([0-9.eE+\-]+)\s*$",
    "core_periphery_density": r"(?m)^\s*-\s*Core-Periphery Density:\s*([0-9.eE+\-]+)\s*$",
    "core_avg_degree": r"(?m)^\s*-\s*Core Avg Degree:\s*([0-9.eE+\-]+)\s*$",
}


def extract_network_metrics(sample_dir: Path) -> Optional[Dict[str, float]]:
    txt_path = sample_dir / "enhanced_network_analysis.txt"
    if not txt_path.exists():
        return None

    text = txt_path.read_text(errors="ignore")
    metrics: Dict[str, float] = {}

    for metric, pattern in _NET_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                metrics[metric] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

    if metrics.get("lcc_ratio") is not None:
        metrics["lcc_share"] = 100.0 * metrics["lcc_ratio"]

    parquet_files = sorted(sample_dir.glob("*.parquet"))
    if parquet_files:
        try:
            metrics["repeated_interaction_pct"] = compute_repeated_interaction_pct_from_parquet(parquet_files[0])
        except Exception as exc:
            logger.warning("  Failed repeated interaction calculation in %s: %s", sample_dir.name, exc)

    return metrics if metrics else None


# ---------------------------------------------------------------------------
def build_baseline(
    samples_dir: Path,
    confidence: float = 0.99,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    sample_dirs = sorted(
        [d for d in samples_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    logger.info(f"Found {len(sample_dirs)} sample directories in {samples_dir}")

    # Extract per-sample metrics
    all_summary: Dict[str, List[float]] = {}
    all_network: Dict[str, List[float]] = {}
    n_loaded = 0
    n_network = 0

    for sd in sample_dirs:
        sm = extract_summary_metrics(sd)
        if sm is None:
            continue
        n_loaded += 1
        for k, v in sm.items():
            all_summary.setdefault(k, []).append(v)

        nm = extract_network_metrics(sd)
        if nm:
            n_network += 1
            for k, v in nm.items():
                all_network.setdefault(k, []).append(v)

    logger.info(f"Loaded summary metrics from {n_loaded}/{len(sample_dirs)} samples")
    if n_network:
        logger.info(f"Loaded network metrics from {n_network}/{len(sample_dirs)} samples")
    else:
        logger.info("No enhanced_network_analysis.txt found — network metrics omitted")

    baseline: Dict[str, Any] = {}
    for metric, values in sorted(all_summary.items()):
        baseline[metric] = metric_entry(values, confidence, ci_method, n_bootstrap, seed)
    for metric, values in sorted(all_network.items()):
        baseline[metric] = metric_entry(values, confidence, ci_method, n_bootstrap, seed)
    return baseline


def print_summary(baseline: Dict[str, Any]) -> None:
    print(f"\n{'Metric':<38} {'Mean':>10} {'CI Lower':>10} {'CI Upper':>10}  N")
    print("-" * 76)
    for metric, entry in sorted(baseline.items()):
        mean = entry['mean']
        lo   = entry['ci_lower']
        hi   = entry['ci_upper']
        n    = entry['n_samples']
        print(f"  {metric:<36} {mean:>10.4f} {lo:>10.4f} {hi:>10.4f}  {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Voat baseline CI JSON from sample directories")
    parser.add_argument(
        "--samples-dir", type=Path,
        default=Path("MADOC/voat-technology-nonoverlap"),
        help="Directory containing sample_1/, sample_2/, … subdirectories",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output JSON path (default: <samples-dir>/baseline_ci.json)",
    )
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument(
        "--ci-method",
        choices=["bootstrap", "t"],
        default="bootstrap",
        help="Confidence interval method for the baseline summary",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="Bootstrap resamples when --ci-method=bootstrap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples_dir = args.samples_dir
    output_path = args.output or (samples_dir / "baseline_ci.json")

    try:
        baseline = build_baseline(
            samples_dir=samples_dir,
            confidence=args.confidence,
            ci_method=args.ci_method,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2))
    logger.info(f"Wrote {len(baseline)} metrics → {output_path}")
    print_summary(baseline)


if __name__ == "__main__":
    main()
