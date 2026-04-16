#!/usr/bin/env python3
"""
aggregate_30runs.py - Aggregate metrics from 30 simulation runs and compute CIs.

Reads metrics.json from each run directory, computes mean/SD/95% CI for each
metric, compares to Voat baseline, and outputs summary files.

Usage:
    python scripts/aggregate_30runs.py --results-dir results --output-dir results/aggregate
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_metrics import (
    compute_repeated_interaction_pct_from_posts,
    summarize_values,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten nested metrics dict into flat key-value pairs."""
    result = {}
    for key, val in metrics.items():
        if key.startswith("_"):  # Skip metadata
            continue
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(val, dict):
            result.update(flatten_metrics(val, f"{full_key}."))
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            result[full_key] = float(val)
    return result


def load_run_metrics(results_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load metrics.json from all run directories."""
    metrics_list = []
    run_names = []

    for run_dir in sorted(results_dir.glob("run*")):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            logger.warning(f"Missing metrics.json in {run_dir}")
            continue

        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8", errors="ignore"))
            data = derive_run_metrics(run_dir, data)
            metrics_list.append(data)
            run_names.append(run_dir.name)
        except Exception as e:
            logger.warning(f"Error loading {metrics_path}: {e}")

    return metrics_list, run_names


def derive_run_metrics(run_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    derived = deepcopy(metrics)

    network = derived.setdefault("network", {})
    posts_path = run_dir / "posts.csv"
    if posts_path.exists() and "repeated_interaction_pct" not in network:
        try:
            network["repeated_interaction_pct"] = compute_repeated_interaction_pct_from_posts(posts_path)
        except Exception as exc:
            logger.warning("Failed repeated interaction derivation for %s: %s", run_dir.name, exc)

    toxicity = derived.setdefault("toxicity", {})
    toxigen_path = run_dir / "toxigen.csv"
    if toxigen_path.exists() and "post_mean" not in toxicity:
        try:
            tox_df = pd.read_csv(toxigen_path)
            if {"toxicity", "post_type"}.issubset(tox_df.columns):
                normalized = (
                    tox_df["post_type"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace("comments", "comment", regex=False)
                    .str.replace("posts", "post", regex=False)
                )
                post_mask = normalized != "comment"
                comment_mask = normalized == "comment"
                if post_mask.any():
                    toxicity["post_mean"] = float(tox_df.loc[post_mask, "toxicity"].mean())
                if comment_mask.any() and "comment_mean" not in toxicity:
                    toxicity["comment_mean"] = float(tox_df.loc[comment_mask, "toxicity"].mean())
        except Exception as exc:
            logger.warning("Failed toxicity post_mean derivation for %s: %s", run_dir.name, exc)

    entropy = derived.setdefault("entropy", {})
    pairs_path = run_dir / "pairs_all.csv"
    if pairs_path.exists():
        try:
            pairs_df = pd.read_csv(pairs_path)
            if {"lag", "H_per_token"}.issubset(pairs_df.columns):
                lag_df = pairs_df[["lag", "H_per_token"]].dropna()
                for lag in [1, 2, 3]:
                    subset = lag_df[lag_df["lag"] == lag]["H_per_token"]
                    if not subset.empty:
                        entropy[f"hpt_lag_{lag}"] = float(subset.mean())
                subset_4 = lag_df[lag_df["lag"] >= 4]["H_per_token"]
                if not subset_4.empty:
                    entropy["hpt_lag_4_plus"] = float(subset_4.mean())
        except Exception as exc:
            logger.warning("Failed entropy lag derivation for %s: %s", run_dir.name, exc)

    return derived


# Mapping from sim metric names (flattened, with namespace prefix) to Voat baseline keys
VOAT_METRIC_ALIASES: Dict[str, str] = {
    "activity.root_posts":                "root_posts",
    "activity.comments":                  "comments",
    "activity.total_posts":               "total_interactions",
    "activity.unique_users":               "unique_users",
    "activity.mean_daily_active_users":    "avg_unique_active_users_per_day",
    "activity.mean_posts_per_user":        "avg_interactions_per_user_per_day",
    "activity.avg_thread_length":          "avg_thread_length",
    "toxicity.mean":                       "avg_toxicity",
    "toxicity.post_mean":                  "posts_mean",
    "toxicity.comment_mean":               "comments_mean",
    "network.nodes":                       "nodes",
    "network.edges":                       "edges",
    "network.density":                     "density",
    "network.avg_degree":                  "avg_degree",
    "network.avg_weighted_degree":         "avg_weighted_degree",
    "network.avg_clustering":              "avg_clustering",
    "network.lcc_nodes":                   "lcc_nodes",
    "network.lcc_ratio":                   "lcc_ratio",
    "network.core_nodes":                  "core_nodes",
    "network.core_pct":                    "core_pct",
    "network.core_density":                "core_density",
    "network.core_periphery_density":      "core_periphery_density",
    "network.repeated_interaction_pct":    "repeated_interaction_pct",
}


def load_voat_baseline(voat_ci_path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    """Load Voat baseline CIs if available."""
    if voat_ci_path is None or not voat_ci_path.exists():
        return {}

    try:
        if voat_ci_path.suffix == ".json":
            return json.loads(voat_ci_path.read_text(encoding="utf-8", errors="ignore"))
        elif voat_ci_path.suffix == ".csv":
            df = pd.read_csv(voat_ci_path)
            result = {}
            for _, row in df.iterrows():
                metric = row.get("metric", row.get("Metric", ""))
                if metric:
                    result[metric] = {
                        "mean": row.get("mean", row.get("Mean")),
                        "ci_lower": row.get("ci_lower", row.get("CI_Lower")),
                        "ci_upper": row.get("ci_upper", row.get("CI_Upper")),
                    }
            return result
    except Exception as e:
        logger.warning(f"Error loading Voat baseline: {e}")

    return {}


def aggregate_metrics(
    metrics_list: List[Dict[str, Any]],
    run_names: List[str],
    voat_baseline: Dict[str, Dict[str, float]],
    confidence: float = 0.99,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate metrics across runs and compute CIs.

    Returns:
        all_runs_df: DataFrame with one row per run
        ci_df: DataFrame with aggregate stats per metric
    """
    # Flatten all metrics
    flat_list = []
    for metrics in metrics_list:
        flat = flatten_metrics(metrics)
        flat_list.append(flat)

    # Get all unique metric names
    all_keys = set()
    for flat in flat_list:
        all_keys.update(flat.keys())
    all_keys = sorted(all_keys)

    # Build per-run DataFrame
    rows = []
    for run_name, flat in zip(run_names, flat_list):
        row = {"run": run_name}
        row.update({k: flat.get(k) for k in all_keys})
        rows.append(row)

    all_runs_df = pd.DataFrame(rows)

    # Compute aggregate stats per metric
    ci_rows = []
    for metric in all_keys:
        values = [flat.get(metric) for flat in flat_list if flat.get(metric) is not None]
        values = [v for v in values if not np.isnan(v)]

        if len(values) == 0:
            continue

        summary = summarize_values(
            values,
            confidence=confidence,
            ci_method=ci_method,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        mean_val = summary["mean"]
        ci_lower = summary["ci_lower"]
        ci_upper = summary["ci_upper"]
        sd_val = summary["std"]

        row = {
            "metric": metric,
            "n_runs": len(values),
            "ci_method": summary.get("ci_method", ci_method),
            "mean": mean_val,
            "sd": sd_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "min": np.min(values),
            "max": np.max(values),
        }

        # Compare to Voat baseline if available (try alias first, then direct key)
        voat_key = VOAT_METRIC_ALIASES.get(metric, metric)
        voat_info = voat_baseline.get(voat_key, {})
        if voat_info:
            voat_mean = voat_info.get("mean")
            if voat_mean is not None and not np.isnan(voat_mean):
                row["voat_mean"] = voat_mean
                row["voat_ci_lower"] = voat_info.get("ci_lower")
                row["voat_ci_upper"] = voat_info.get("ci_upper")
                row["delta"] = mean_val - voat_mean
                row["delta_pct"] = 100 * (mean_val - voat_mean) / abs(voat_mean) if voat_mean != 0 else np.nan

                # Check CI overlap
                voat_ci_lower = voat_info.get("ci_lower", voat_mean)
                voat_ci_upper = voat_info.get("ci_upper", voat_mean)
                overlap = not (ci_upper < voat_ci_lower or ci_lower > voat_ci_upper)
                row["ci_overlap"] = overlap

        ci_rows.append(row)

    ci_df = pd.DataFrame(ci_rows)

    return all_runs_df, ci_df


def generate_report(
    all_runs_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    output_dir: Path,
    confidence: float,
    ci_method: str,
) -> Dict[str, Any]:
    """Generate summary report."""
    n_runs = len(all_runs_df)
    n_metrics = len(ci_df)

    # Key metrics summary
    key_metrics = [
        "activity.total_posts",
        "activity.unique_users",
        "activity.mean_posts_per_user",
        "toxicity.mean",
        "network.density",
        "network.core_pct",
        "entropy.inter_hpt_mean",
        "entropy.intra_hpt_mean",
    ]

    key_summary = {}
    for metric in key_metrics:
        row = ci_df[ci_df["metric"] == metric]
        if len(row) > 0:
            r = row.iloc[0]
            key_summary[metric] = {
                "mean": r["mean"],
                "ci": f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]",
                "sd": r["sd"],
            }
            if "voat_mean" in r and pd.notna(r["voat_mean"]):
                key_summary[metric]["voat_mean"] = r["voat_mean"]
                key_summary[metric]["delta_pct"] = r.get("delta_pct")

    report = {
        "n_runs": n_runs,
        "n_metrics": n_metrics,
        "confidence": confidence,
        "ci_method": ci_method,
        "key_metrics": key_summary,
        "metrics_with_voat_comparison": int(ci_df["voat_mean"].notna().sum()) if "voat_mean" in ci_df.columns else 0,
        "output_files": {
            "all_runs": str(output_dir / "sim_metrics_all.csv"),
            "ci_summary": str(output_dir / "sim_metrics_ci.csv"),
            "report": str(output_dir / "benchmark_report.json"),
        },
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics from 30 simulation runs")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing runXX subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/aggregate)",
    )
    parser.add_argument(
        "--voat-ci",
        type=Path,
        default=None,
        help="Path to Voat baseline CI file (JSON or CSV)",
    )
    parser.add_argument("--confidence", type=float, default=0.99, help="Confidence level for CI calculation")
    parser.add_argument(
        "--ci-method",
        choices=["bootstrap", "t"],
        default="bootstrap",
        help="Confidence interval method used to summarize run-level metrics",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="Bootstrap resamples when --ci-method=bootstrap")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir or (results_dir / "aggregate")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all run metrics
    logger.info(f"Loading metrics from {results_dir}")
    metrics_list, run_names = load_run_metrics(results_dir)

    if len(metrics_list) == 0:
        logger.error("No metrics.json files found")
        sys.exit(1)

    logger.info(f"Loaded {len(metrics_list)} runs: {run_names}")

    # Load Voat baseline if provided
    voat_baseline = {}
    if args.voat_ci:
        voat_baseline = load_voat_baseline(args.voat_ci)
        logger.info(f"Loaded Voat baseline with {len(voat_baseline)} metrics")

    # Aggregate
    all_runs_df, ci_df = aggregate_metrics(
        metrics_list,
        run_names,
        voat_baseline,
        confidence=args.confidence,
        ci_method=args.ci_method,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    # Save outputs
    all_runs_path = output_dir / "sim_metrics_all.csv"
    ci_path = output_dir / "sim_metrics_ci.csv"
    report_path = output_dir / "benchmark_report.json"

    all_runs_df.to_csv(all_runs_path, index=False)
    ci_df.to_csv(ci_path, index=False)
    logger.info(f"Wrote {len(all_runs_df)} runs to {all_runs_path}")
    logger.info(f"Wrote {len(ci_df)} metrics to {ci_path}")

    # Generate report
    report = generate_report(all_runs_df, ci_df, output_dir, confidence=args.confidence, ci_method=args.ci_method)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote report to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"AGGREGATION SUMMARY: {len(metrics_list)} runs, {len(ci_df)} metrics")
    print("=" * 60)
    for metric, info in report.get("key_metrics", {}).items():
        print(f"  {metric}: {info['mean']:.4f} {info['ci']}")
        if "voat_mean" in info:
            print(f"    vs Voat: {info['voat_mean']:.4f} (delta: {info.get('delta_pct', 0):.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
