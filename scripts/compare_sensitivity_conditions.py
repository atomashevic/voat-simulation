#!/usr/bin/env python3
"""
compare_sensitivity_conditions.py - Compare sensitivity analysis conditions.

Loads per-condition aggregated metrics, computes cross-condition contrasts
with bootstrap CIs, compares to Voat baseline, and generates tables + plots
for each comparison group (persona: c0/c1/c2, temperature: c0/c3/c4).

Usage:
    python scripts/compare_sensitivity_conditions.py \\
        --sensitivity-dir results/sensitivity \\
        --groups "persona:c0,c1,c2" "temperature:c0,c3,c4" \\
        --output-dir results/sensitivity/comparisons \\
        --voat-madoc-root MADOC/voat-technology \\
        --voat-parquet MADOC/voat-technology/voat_technology_madoc.parquet

Temporal note:
    Sensitivity runs use 10 simulated days; Voat samples span 30 days.
    Activity totals (posts, comments) are normalized to per-day rates for
    Voat comparison. Condition-vs-condition comparisons use raw totals
    since all conditions run 10 days.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Add repo root to path so aggregate_30runs is importable ───────────────────
_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from aggregate_30runs import (  # noqa: E402
    load_run_metrics,
    aggregate_metrics,
    bootstrap_ci,
    flatten_metrics,
)

# ── Constants ─────────────────────────────────────────────────────────────────
VOAT_NUM_DAYS = 7   # use 7-day sliced samples to match sensitivity run duration
SIM_NUM_DAYS_DEFAULT = 10  # sensitivity runs are 10-day simulations

# Metrics whose raw totals should NOT be compared directly to Voat
DURATION_SENSITIVE = {
    "activity.total_posts",
    "activity.comments",
    "activity.root_posts",
    "activity.unique_users",
    "activity.num_threads",
    "activity.total_users_registered",
}

# Metric categories for output tables and forest plots
METRIC_CATEGORIES: Dict[str, List[str]] = {
    "activity": [
        "activity.posts_per_day",
        "activity.comments_per_day",
        "activity.mean_posts_per_user",
        "activity.avg_thread_length",
        "activity.mean_daily_active_users",
    ],
    "network": [
        "network.density",
        "network.avg_clustering",
        "network.core_pct",
        "network.core_density",
        "network.modularity",
    ],
    "toxicity": [
        "toxicity.mean",
        "toxicity.p90",
        "toxicity.frac_above_0.5",
        "toxicity.comment_mean",
    ],
    "topic": [
        "topic.coverage_sim2",
        "topic.mean_cosine",
        "topic.sim2_topics",
    ],
    "entropy": [
        "entropy.H_per_token_mean",
        "entropy.inter_hpt_mean",
        "entropy.intra_hpt_mean",
    ],
}

# Colors per condition (tab10 palette, c0=blue always)
CONDITION_COLORS: Dict[str, str] = {
    "c0":  "#1f77b4",  # blue   – baseline
    "c1":  "#ff7f0e",  # orange – neutral persona
    "c2":  "#2ca02c",  # green  – no-politics persona
    "c3":  "#9467bd",  # purple – low temperature
    "c4":  "#d62728",  # red    – high temperature
    "c5":  "#8c564b",  # brown  – budget flat
    "c6":  "#e377c2",  # pink   – budget steep
    "c7":  "#7f7f7f",  # grey   – CPR low
    "c8":  "#bcbd22",  # yellow-green – CPR high
    "c9":  "#17becf",  # cyan   – churn low
    "c10": "#aec7e8",  # light blue – churn high
}

DEFAULT_LABELS: Dict[str, str] = {
    "c0":  "Baseline",
    "c1":  "NeutralPersona",
    "c2":  "NoPolPersona",
    "c3":  "TempLow",
    "c4":  "TempHigh",
    "c5":  "BudgetFlat",
    "c6":  "BudgetSteep",
    "c7":  "CPRLow",
    "c8":  "CPRHigh",
    "c9":  "ChurnLow",
    "c10": "ChurnHigh",
}

# ── Metric derivation ─────────────────────────────────────────────────────────

def derive_daily_rates(flat: Dict[str, float]) -> Dict[str, float]:
    """Add posts_per_day / comments_per_day / threads_per_day to flat dict."""
    result = dict(flat)
    num_days = flat.get("activity.num_days") or SIM_NUM_DAYS_DEFAULT
    if num_days and num_days > 0:
        for src, dst in [
            ("activity.total_posts",  "activity.posts_per_day"),
            ("activity.comments",     "activity.comments_per_day"),
            ("activity.num_threads",  "activity.threads_per_day"),
        ]:
            val = flat.get(src)
            if val is not None:
                result[dst] = val / num_days
    return result


# ── Data loading ──────────────────────────────────────────────────────────────

def load_condition_data(
    sensitivity_dir: Path,
    conditions: List[str],
    n_bootstrap: int = 2000,
) -> Dict[str, Dict[str, Any]]:
    """Load and aggregate metrics for each condition.

    Returns:
        dict: cond -> {"ci_df": DataFrame, "all_runs_df": DataFrame,
                       "flat_list": list of flat metric dicts, "n_runs": int}
    """
    result: Dict[str, Dict] = {}

    for cond in conditions:
        cond_dir = sensitivity_dir / cond
        if not cond_dir.exists():
            logger.warning(f"Condition directory not found: {cond_dir}")
            continue

        metrics_list, run_names = load_run_metrics(cond_dir)
        if not metrics_list:
            logger.warning(f"No metrics.json found for condition {cond}")
            continue

        # Build flat list with derived daily rate metrics
        flat_list = []
        for m in metrics_list:
            flat = flatten_metrics(m)
            flat = derive_daily_rates(flat)
            flat_list.append(flat)

        # Base aggregation (uses aggregate_metrics from aggregate_30runs)
        all_runs_df, ci_df = aggregate_metrics(metrics_list, run_names, {})

        # Compute CI for derived daily rate metrics not in metrics.json
        derived_rows = []
        for metric in ["activity.posts_per_day", "activity.comments_per_day", "activity.threads_per_day"]:
            values = [f[metric] for f in flat_list if metric in f and f[metric] is not None]
            values = [v for v in values if not np.isnan(float(v))]
            if len(values) >= 2:
                mean, ci_lo, ci_hi = bootstrap_ci(values, n_bootstrap=n_bootstrap)
                derived_rows.append({
                    "metric": metric,
                    "n_runs": len(values),
                    "mean": mean,
                    "sd": float(np.std(values, ddof=1)),
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                })

        if derived_rows:
            ci_df = pd.concat([ci_df, pd.DataFrame(derived_rows)], ignore_index=True)

        result[cond] = {
            "ci_df": ci_df,
            "all_runs_df": all_runs_df,
            "flat_list": flat_list,
            "n_runs": len(metrics_list),
        }
        logger.info(f"Loaded {len(metrics_list)} runs for condition {cond}")

    return result


# ── Voat baseline ─────────────────────────────────────────────────────────────

def _parse_float_from_text(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _metrics_from_7d_parquet(parquet_path: Path) -> Dict[str, float]:
    """Compute activity and toxicity metrics from a 7-day sliced MADOC parquet.

    The parquet has columns: post_id, publish_date, user_id, parent_id,
    interaction_type (posts/comments), toxicity_toxigen, ...
    """
    try:
        import pandas as pd
    except ImportError:
        return {}

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.warning(f"Could not read parquet {parquet_path}: {e}")
        return {}

    if df.empty:
        return {}

    m: Dict[str, float] = {}

    # Determine actual span in days
    if "publish_date" in df.columns:
        ts = df["publish_date"]
        n_days = max((ts.max() - ts.min()) / 86400, 1.0)
    else:
        n_days = float(VOAT_NUM_DAYS)

    posts_mask = df.get("interaction_type", pd.Series(dtype=str)) == "posts"
    comments_mask = ~posts_mask

    n_posts = int(posts_mask.sum())
    n_comments = int(comments_mask.sum())
    n_total = len(df)

    # Activity
    m["activity.posts_per_day"] = n_posts / n_days
    m["activity.comments_per_day"] = n_comments / n_days

    if "user_id" in df.columns:
        unique_users = df["user_id"].nunique()
        m["activity.unique_users"] = float(unique_users)
        if unique_users > 0 and n_posts > 0:
            m["activity.mean_posts_per_user"] = n_posts / unique_users

    # Thread length: root posts (no parent) define threads
    if "parent_id" in df.columns:
        root_posts = int(df["parent_id"].isna().sum())
        if root_posts > 0:
            m["activity.avg_thread_length"] = n_total / root_posts

    # Toxicity from toxicity_toxigen column
    if "toxicity_toxigen" in df.columns:
        tox = df["toxicity_toxigen"].dropna()
        if len(tox) > 0:
            m["toxicity.mean"] = float(tox.mean())
            m["toxicity.p90"] = float(tox.quantile(0.9))
            m["toxicity.frac_above_0.5"] = float((tox >= 0.5).mean())

        tox_comments = df.loc[comments_mask, "toxicity_toxigen"].dropna()
        if len(tox_comments) > 0:
            m["toxicity.comment_mean"] = float(tox_comments.mean())

    return m


def compute_voat_baseline(
    madoc_root: Path,
    n_bootstrap: int = 2000,
) -> Dict[str, Dict[str, float]]:
    """Compute Voat baseline metrics from 7-day sliced MADOC samples with bootstrap CI.

    Activity and toxicity are computed directly from 7-day parquets.
    Network metrics are read from enhanced_network_analysis.txt (per-sample).
    Returns: metric -> {"mean", "ci_lower", "ci_upper"}
    """
    sample_metrics_list: List[Dict[str, float]] = []

    for sample_dir in sorted(madoc_root.glob("sample_*")):
        if not sample_dir.is_dir():
            continue
        m: Dict[str, float] = {}

        # Activity + toxicity: load directly from 7d parquet
        pq_7d_files = sorted(sample_dir.glob("*_7d.parquet"))
        if pq_7d_files:
            pq_metrics = _metrics_from_7d_parquet(pq_7d_files[0])
            m.update(pq_metrics)
        else:
            logger.warning(f"No 7d parquet found in {sample_dir}; activity/toxicity will be missing")

        # Network from enhanced_network_analysis.txt (computed on full data; structure
        # is comparable since we use the same underlying user-interaction graph)
        net_path = sample_dir / "enhanced_network_analysis.txt"
        if net_path.exists():
            try:
                txt = net_path.read_text(encoding="utf-8", errors="ignore")
                m["network.density"] = _parse_float_from_text(
                    txt, r"(?m)^\s*-\s*(?:Component\s+)?[Dd]ensity:\s*([\d.eE+\-]+)\s*$"
                )
                m["network.avg_clustering"] = _parse_float_from_text(
                    txt, r"(?m)^\s*-\s*Avg\s+Clustering:\s*([\d.eE+\-]+)\s*$"
                )
                m["network.modularity"] = _parse_float_from_text(
                    txt, r"(?m)^\s*-\s*Modularity:\s*([-\d.eE+\-]+)\s*$"
                )
                m["network.core_density"] = _parse_float_from_text(
                    txt, r"(?m)^\s*-\s*Core\s+Density:\s*([\d.eE+\-]+)\s*$"
                )
                m["network.core_pct"] = _parse_float_from_text(
                    txt, r"(?m)^\s*-\s*Core\s+Size:\s*[\d,]+\s+nodes\s*\(([\d.]+)%\s+of\s+analyzed\s+component\)"
                )
            except Exception as e:
                logger.warning(f"Error reading {net_path}: {e}")

        # Remove None values before appending
        m_clean = {k: v for k, v in m.items() if v is not None}
        if m_clean:
            sample_metrics_list.append(m_clean)

    logger.info(f"Loaded Voat baseline from {len(sample_metrics_list)} MADOC samples (7d slices)")

    # Bootstrap CI per metric
    all_metric_keys: set = set()
    for sm in sample_metrics_list:
        all_metric_keys.update(sm.keys())

    result: Dict[str, Dict[str, float]] = {}
    for metric in all_metric_keys:
        values = [sm[metric] for sm in sample_metrics_list if metric in sm]
        if len(values) >= 2:
            mean, ci_lo, ci_hi = bootstrap_ci(values, n_bootstrap=n_bootstrap)
            result[metric] = {"mean": mean, "ci_lower": ci_lo, "ci_upper": ci_hi}

    logger.info(f"Voat baseline: {len(result)} metrics with CI")
    return result


# ── Comparison table builder ───────────────────────────────────────────────────

def _get_ci_row(ci_df: pd.DataFrame, metric: str) -> Optional[pd.Series]:
    rows = ci_df[ci_df["metric"] == metric]
    return rows.iloc[0] if len(rows) > 0 else None


def build_comparison_table(
    condition_data: Dict[str, Dict],
    conditions: List[str],
    baseline_cond: str,
    metrics: List[str],
    voat_baseline: Dict[str, Dict],
) -> pd.DataFrame:
    """Build a comparison table for a list of metrics across conditions."""
    rows = []

    for metric in metrics:
        row: Dict[str, Any] = {"metric": metric}

        # Gather per-condition stats
        baseline_mean = baseline_ci_lo = baseline_ci_hi = None
        for cond in conditions:
            if cond not in condition_data:
                for suffix in ("mean", "ci_lo", "ci_hi"):
                    row[f"{cond}_{suffix}"] = None
                continue
            r = _get_ci_row(condition_data[cond]["ci_df"], metric)
            if r is None:
                for suffix in ("mean", "ci_lo", "ci_hi"):
                    row[f"{cond}_{suffix}"] = None
                continue
            row[f"{cond}_mean"] = r["mean"]
            row[f"{cond}_ci_lo"] = r["ci_lower"]
            row[f"{cond}_ci_hi"] = r["ci_upper"]
            if cond == baseline_cond:
                baseline_mean = r["mean"]
                baseline_ci_lo = r["ci_lower"]
                baseline_ci_hi = r["ci_upper"]

        # Deltas vs baseline (condition-vs-condition)
        for cond in conditions:
            if cond == baseline_cond:
                continue
            cm = row.get(f"{cond}_mean")
            clo = row.get(f"{cond}_ci_lo")
            chi = row.get(f"{cond}_ci_hi")
            if cm is not None and baseline_mean is not None:
                delta = cm - baseline_mean
                row[f"{cond}_delta"] = delta
                row[f"{cond}_delta_pct"] = (
                    100 * delta / abs(baseline_mean) if baseline_mean != 0 else None
                )
                if all(v is not None for v in [clo, chi, baseline_ci_lo, baseline_ci_hi]):
                    row[f"{cond}_ci_overlap_c0"] = not (
                        chi < baseline_ci_lo or clo > baseline_ci_hi
                    )

        # Voat comparison
        voat = voat_baseline.get(metric, {})
        if voat:
            row["voat_mean"] = voat.get("mean")
            row["voat_ci_lo"] = voat.get("ci_lower")
            row["voat_ci_hi"] = voat.get("ci_upper")
            row["voat_normalized"] = metric not in DURATION_SENSITIVE

            for cond in conditions:
                cm = row.get(f"{cond}_mean")
                clo = row.get(f"{cond}_ci_lo")
                chi = row.get(f"{cond}_ci_hi")
                vm = voat.get("mean")
                vlo = voat.get("ci_lower")
                vhi = voat.get("ci_upper")
                if cm is not None and vm is not None:
                    row[f"{cond}_voat_delta"] = cm - vm
                    if all(v is not None for v in [clo, chi, vlo, vhi]):
                        row[f"{cond}_voat_ci_overlap"] = not (chi < vlo or clo > vhi)

        rows.append(row)

    return pd.DataFrame(rows)


# ── JSON summary builder ───────────────────────────────────────────────────────

def build_group_json(
    group_name: str,
    conditions: List[str],
    condition_labels: Dict[str, str],
    condition_data: Dict[str, Dict],
    voat_baseline: Dict[str, Dict],
) -> Dict[str, Any]:
    """Build JSON summary for a comparison group."""
    baseline = conditions[0]
    all_metrics = [m for ms in METRIC_CATEGORIES.values() for m in ms]

    summary: Dict[str, Any] = {
        "group": group_name,
        "conditions": conditions,
        "condition_labels": {c: condition_labels.get(c, c) for c in conditions},
        "baseline": baseline,
        "n_runs_per_condition": {
            c: condition_data[c]["n_runs"]
            for c in conditions
            if c in condition_data
        },
        "temporal_note": (
            f"Sensitivity runs: {SIM_NUM_DAYS_DEFAULT} days, Voat samples: {VOAT_NUM_DAYS} days. "
            "Activity totals normalized to daily rates for Voat comparison."
        ),
        "metrics": {},
        "notable_differences": [],
    }

    for metric in all_metrics:
        entry: Dict[str, Any] = {}

        for cond in conditions:
            if cond not in condition_data:
                continue
            r = _get_ci_row(condition_data[cond]["ci_df"], metric)
            if r is None:
                continue

            cond_entry: Dict[str, Any] = {
                "mean": r["mean"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "n_runs": int(r["n_runs"]),
            }

            if cond != baseline:
                br = _get_ci_row(condition_data[baseline]["ci_df"], metric)
                if br is not None:
                    delta = r["mean"] - br["mean"]
                    delta_pct = (
                        100 * delta / abs(br["mean"]) if br["mean"] != 0 else None
                    )
                    ci_overlap = not (r["ci_upper"] < br["ci_lower"] or r["ci_lower"] > br["ci_upper"])
                    cond_entry.update({
                        "delta_vs_baseline": delta,
                        "delta_pct_vs_baseline": delta_pct,
                        "ci_overlap_baseline": ci_overlap,
                    })
                    if delta_pct is not None and abs(delta_pct) > 10 and not ci_overlap:
                        summary["notable_differences"].append({
                            "metric": metric,
                            "condition": cond,
                            "label": condition_labels.get(cond, cond),
                            "delta_pct": round(delta_pct, 2),
                            "ci_overlap": ci_overlap,
                        })

            voat = voat_baseline.get(metric, {})
            if voat.get("mean") is not None:
                cond_entry["voat_mean"] = voat["mean"]
                cond_entry["voat_delta"] = r["mean"] - voat["mean"]
                if voat.get("ci_lower") is not None and voat.get("ci_upper") is not None:
                    cond_entry["voat_ci_overlap"] = not (
                        r["ci_upper"] < voat["ci_lower"] or r["ci_lower"] > voat["ci_upper"]
                    )

            entry[cond] = cond_entry

        if entry:
            summary["metrics"][metric] = entry

    return summary


# ── Plots ─────────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#fafafa")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_forest(
    condition_data: Dict[str, Dict],
    conditions: List[str],
    condition_labels: Dict[str, str],
    voat_baseline: Dict[str, Dict],
    group_name: str,
    output_dir: Path,
) -> None:
    """Multi-panel forest plot: one subplot per metric category."""
    n_cats = len(METRIC_CATEGORIES)
    fig, axes = plt.subplots(1, n_cats, figsize=(4.5 * n_cats, 7))
    if n_cats == 1:
        axes = [axes]

    avail_conds = [c for c in conditions if c in condition_data]
    offsets = np.linspace(-0.3, 0.3, len(avail_conds))

    for ax, (cat_name, metrics) in zip(axes, METRIC_CATEGORIES.items()):
        y_pos = np.arange(len(metrics))
        for ci, cond in enumerate(avail_conds):
            ci_df = condition_data[cond]["ci_df"]
            color = CONDITION_COLORS.get(cond, f"C{ci}")
            label = condition_labels.get(cond, cond)
            for mi, metric in enumerate(metrics):
                r = _get_ci_row(ci_df, metric)
                if r is None:
                    continue
                y = y_pos[mi] + offsets[ci]
                xerr_lo = r["mean"] - r["ci_lower"]
                xerr_hi = r["ci_upper"] - r["mean"]
                ax.errorbar(
                    r["mean"], y,
                    xerr=[[max(0.0, xerr_lo)], [max(0.0, xerr_hi)]],
                    fmt="o", color=color, capsize=4, linewidth=1.5, markersize=6,
                    label=label if mi == 0 else "_nolegend_",
                )

        # Voat reference lines
        for mi, metric in enumerate(metrics):
            vm = voat_baseline.get(metric, {}).get("mean")
            if vm is not None:
                ax.axvline(vm, color="gray", linestyle="--", alpha=0.5, linewidth=1.0)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([m.split(".")[-1] for m in metrics], fontsize=8)
        ax.set_title(cat_name.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("Value", fontsize=8)
        _style_ax(ax)

    handles = [
        mpatches.Patch(
            color=CONDITION_COLORS.get(c, f"C{i}"),
            label=condition_labels.get(c, c),
        )
        for i, c in enumerate(avail_conds)
    ]
    handles.append(
        mpatches.Patch(color="gray", alpha=0.5, label="Voat ref (dashed)")
    )
    fig.legend(
        handles=handles, loc="lower center", ncol=len(avail_conds) + 1,
        bbox_to_anchor=(0.5, -0.04), fontsize=9,
    )
    fig.suptitle(f"Condition Comparison — {group_name}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    path = output_dir / "plots" / "forest_plot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved forest plot → {path}")


def _gaussian_kde(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Silverman's rule KDE (mirrors workshop_toxicity_pooled.gaussian_kde)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid, dtype=float)
    n = x.size
    std = np.std(x, ddof=1) if n > 1 else 0.0
    bw = max(1.06 * std * (n ** (-0.2)), 1e-3) if std > 0 else 0.05
    u = (grid[:, None] - x[None, :]) / bw
    kern = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    return kern.sum(axis=1) / (n * bw)


def _compute_kde_ensemble(
    runs: List[np.ndarray], grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kdes = np.array([_gaussian_kde(r, grid) for r in runs])
    return np.mean(kdes, axis=0), np.percentile(kdes, 5, axis=0), np.percentile(kdes, 95, axis=0)


def _load_tox_arrays(cond_dir: Path) -> List[np.ndarray]:
    arrays = []
    for run_dir in sorted(cond_dir.glob("run*")):
        tox_file = run_dir / "toxigen.csv"
        if tox_file.exists():
            try:
                df = pd.read_csv(tox_file)
                col = next((c for c in ["toxicity", "toxicity_score"] if c in df.columns), None)
                if col:
                    scores = pd.to_numeric(df[col], errors="coerce").dropna().values
                    if len(scores) > 0:
                        arrays.append(scores)
            except Exception:
                pass
    return arrays


def _load_voat_tox_arrays(madoc_root: Path) -> List[np.ndarray]:
    arrays = []
    for sample_dir in sorted(madoc_root.glob("sample_*")):
        parquet_files = list(sample_dir.glob("*.parquet"))
        if parquet_files:
            try:
                df = pd.read_parquet(parquet_files[0])
                col = next((c for c in ["toxicity", "toxicity_toxigen", "toxigen"] if c in df.columns), None)
                if col:
                    scores = pd.to_numeric(df[col], errors="coerce").dropna().values
                    if len(scores) > 0:
                        arrays.append(scores)
            except Exception:
                pass
    return arrays


def plot_toxicity_kde(
    conditions: List[str],
    condition_labels: Dict[str, str],
    sensitivity_dir: Path,
    madoc_root: Path,
    group_name: str,
    output_dir: Path,
) -> None:
    """Toxicity KDE ensemble per condition + Voat reference."""
    grid = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)

    # Voat reference
    voat_arrays = _load_voat_tox_arrays(madoc_root)
    if voat_arrays:
        vmean, vlo, vhi = _compute_kde_ensemble(voat_arrays, grid)
        ax.fill_between(grid, vlo, vhi, color="gray", alpha=0.12)
        ax.plot(grid, vmean, color="gray", linewidth=2.5, linestyle="--",
                label=f"Voat (n={len(voat_arrays)} samples)")

    for cond in conditions:
        arrays = _load_tox_arrays(sensitivity_dir / cond)
        if not arrays:
            continue
        color = CONDITION_COLORS.get(cond, "C0")
        label = condition_labels.get(cond, cond)
        mean_kde, lo, hi = _compute_kde_ensemble(arrays, grid)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.15)
        ax.plot(grid, mean_kde, color=color, linewidth=2.5,
                label=f"{label} (n={len(arrays)} runs)")

    ax.set_xlabel("Toxicity Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Toxicity Distribution by Condition — {group_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    path = output_dir / "plots" / "toxicity_kde.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved toxicity KDE → {path}")


def plot_entropy_scatter(
    condition_data: Dict[str, Dict],
    conditions: List[str],
    condition_labels: Dict[str, str],
    group_name: str,
    output_dir: Path,
) -> None:
    """Scatter: inter_hpt_mean vs intra_hpt_mean per run, with 95% confidence ellipses."""
    from matplotlib.patches import Ellipse as MEllipse

    fig, ax = plt.subplots(figsize=(8, 7))
    _style_ax(ax)

    for cond in conditions:
        if cond not in condition_data:
            continue
        flat_list = condition_data[cond]["flat_list"]
        color = CONDITION_COLORS.get(cond, "C0")
        label = condition_labels.get(cond, cond)

        x_vals = [f.get("entropy.inter_hpt_mean") for f in flat_list]
        y_vals = [f.get("entropy.intra_hpt_mean") for f in flat_list]
        x_clean = [v for v in x_vals if v is not None]
        y_clean = [v for v in y_vals if v is not None and x_vals[y_vals.index(v)] is not None]

        # Align pairs
        pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x is not None and y is not None]
        if not pairs:
            continue
        xs, ys = zip(*pairs)

        ax.scatter(xs, ys, color=color, alpha=0.7, s=60, zorder=3, label=label)

        if len(xs) >= 3:
            try:
                cov = np.cov(xs, ys)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                chi2_95 = 5.991  # chi-squared df=2 at 95%
                width = 2 * np.sqrt(max(vals[0] * chi2_95, 0))
                height = 2 * np.sqrt(max(vals[1] * chi2_95, 0))
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                ell = MEllipse(
                    xy=(np.mean(xs), np.mean(ys)),
                    width=width, height=height, angle=angle,
                    color=color, fill=False, linewidth=2, linestyle="--", alpha=0.6,
                )
                ax.add_patch(ell)
            except np.linalg.LinAlgError:
                pass

    # Diagonal reference
    lim_lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    lim_hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.25, linewidth=1, label="inter=intra")

    ax.set_xlabel("Inter-pair H per token (mean)", fontsize=12)
    ax.set_ylabel("Intra-pair H per token (mean)", fontsize=12)
    ax.set_title(f"Convergence Entropy: Inter vs Intra — {group_name}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    path = output_dir / "plots" / "entropy_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved entropy scatter → {path}")


def plot_radar(
    condition_data: Dict[str, Dict],
    conditions: List[str],
    condition_labels: Dict[str, str],
    voat_baseline: Dict[str, Dict],
    group_name: str,
    output_dir: Path,
) -> None:
    """Radar chart: 6 key metrics normalized across conditions."""
    radar_metrics = [
        ("activity.posts_per_day",      "Posts/day"),
        ("network.core_pct",            "Core %"),
        ("toxicity.mean",               "Toxicity"),
        ("topic.mean_cosine",           "Topic sim."),
        ("entropy.H_per_token_mean",    "Entropy H"),
        ("network.density",             "Density"),
    ]
    keys = [m[0] for m in radar_metrics]
    labels = [m[1] for m in radar_metrics]

    # Collect all values (conditions + voat) for normalization bounds
    all_vals: Dict[str, List[float]] = {k: [] for k in keys}
    avail = [c for c in conditions if c in condition_data]
    for cond in avail:
        ci_df = condition_data[cond]["ci_df"]
        for k in keys:
            r = _get_ci_row(ci_df, k)
            if r is not None:
                all_vals[k].append(r["mean"])
    for k in keys:
        vm = voat_baseline.get(k, {}).get("mean")
        if vm is not None:
            all_vals[k].append(vm)

    def normalize(val: float, k: str) -> float:
        vs = all_vals[k]
        lo, hi = (min(vs), max(vs)) if vs else (0, 1)
        return (val - lo) / (hi - lo) if hi > lo else 0.5

    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_facecolor("#fafafa")

    for cond in avail:
        ci_df = condition_data[cond]["ci_df"]
        values = []
        for k in keys:
            r = _get_ci_row(ci_df, k)
            values.append(normalize(r["mean"], k) if r is not None else 0.5)
        values += values[:1]
        color = CONDITION_COLORS.get(cond, "C0")
        label = condition_labels.get(cond, cond)
        ax.plot(angles, values, color=color, linewidth=2.5, label=label)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels([])
    ax.set_title(f"Condition Radar — {group_name}", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9)
    ax.grid(color="gray", linestyle="--", alpha=0.3)

    path = output_dir / "plots" / "radar_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved radar chart → {path}")


def plot_degree_dist(
    conditions: List[str],
    condition_labels: Dict[str, str],
    sensitivity_dir: Path,
    group_name: str,
    output_dir: Path,
) -> None:
    """Log-log CCDF degree distribution overlay per condition."""
    try:
        from workshop_degree_analysis import build_network_from_posts
    except ImportError:
        logger.warning("workshop_degree_analysis not importable; skipping degree dist plot")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    _style_ax(ax)

    for cond in conditions:
        cond_dir = sensitivity_dir / cond
        if not cond_dir.exists():
            continue
        all_degrees: List[List[int]] = []
        for run_dir in sorted(cond_dir.glob("run*")):
            posts_csv = run_dir / "posts.csv"
            if posts_csv.exists():
                try:
                    G = build_network_from_posts(posts_csv)
                    degs = [d for _, d in G.degree() if d > 0]
                    if degs:
                        all_degrees.append(degs)
                except Exception:
                    pass

        if not all_degrees:
            continue

        max_deg = max(max(d) for d in all_degrees)
        x_grid = np.logspace(0, np.log10(max_deg + 1), 100)
        ccdfs = []
        for degs in all_degrees:
            arr = np.array(degs)
            ccdf = np.array([np.mean(arr >= xi) for xi in x_grid])
            ccdfs.append(ccdf)

        ccdfs_arr = np.array(ccdfs)
        mean_ccdf = np.mean(ccdfs_arr, axis=0)
        lo = np.percentile(ccdfs_arr, 5, axis=0)
        hi = np.percentile(ccdfs_arr, 95, axis=0)

        color = CONDITION_COLORS.get(cond, "C0")
        label = condition_labels.get(cond, cond)
        ax.fill_between(x_grid, lo, hi, color=color, alpha=0.15)
        ax.plot(x_grid, mean_ccdf, color=color, linewidth=2.5, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree", fontsize=12)
    ax.set_ylabel("P(X ≥ x)", fontsize=12)
    ax.set_title(f"Degree Distribution (CCDF) — {group_name}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    path = output_dir / "plots" / "degree_dist.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved degree dist → {path}")


def plot_activity_timeseries(
    conditions: List[str],
    condition_labels: Dict[str, str],
    sensitivity_dir: Path,
    group_name: str,
    output_dir: Path,
) -> None:
    """Cumulative activity growth per condition with CI bands."""
    try:
        from workshop_activity_growth_ci import load_run_counts
    except ImportError:
        logger.warning("workshop_activity_growth_ci not importable; skipping timeseries plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ts_metrics = [
        "cumulative_posts",
        "cumulative_comments",
        "cumulative_users",
        "cumulative_active_users",
    ]
    ts_titles = ["Cumulative Posts", "Cumulative Comments", "Registered Users", "Active Users"]

    for cond in conditions:
        cond_dir = sensitivity_dir / cond
        if not cond_dir.exists():
            continue
        color = CONDITION_COLORS.get(cond, "C0")
        label = condition_labels.get(cond, cond)

        per_run: Dict[str, List] = {m: [] for m in ts_metrics}
        for run_dir in sorted(cond_dir.glob("run*")):
            try:
                ts = load_run_counts(run_dir)
                ts = ts.copy()
                ts["day"] = ts.index // 24
                day_ts = ts.groupby("day").last()
                for m in ts_metrics:
                    if m in day_ts.columns:
                        per_run[m].append(day_ts[m].values)
            except Exception:
                pass

        for ax, metric, title in zip(axes.flat, ts_metrics, ts_titles):
            series = per_run[metric]
            if not series:
                continue
            max_len = max(len(s) for s in series)
            padded = np.array([
                np.pad(s, (0, max_len - len(s)), mode="edge") for s in series
            ])
            days = np.arange(max_len)
            mean = np.mean(padded, axis=0)
            lo = np.percentile(padded, 5, axis=0)
            hi = np.percentile(padded, 95, axis=0)
            ax.fill_between(days, lo, hi, color=color, alpha=0.15)
            ax.plot(days, mean, color=color, linewidth=2.5, label=label)

    for ax, title in zip(axes.flat, ts_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Day", fontsize=9)
        _style_ax(ax)

    handles, lbls = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, lbls, loc="lower center", ncol=len(conditions),
            bbox_to_anchor=(0.5, -0.03), fontsize=9,
        )
    fig.suptitle(f"Activity Growth — {group_name}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    path = output_dir / "plots" / "activity_timeseries.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved activity timeseries → {path}")


# ── Group processing ──────────────────────────────────────────────────────────

def process_group(
    group_name: str,
    conditions: List[str],
    condition_labels: Dict[str, str],
    condition_data: Dict[str, Dict],
    voat_baseline: Dict[str, Dict],
    sensitivity_dir: Path,
    madoc_root: Path,
    output_dir: Path,
    no_plots: bool,
    n_bootstrap: int,
) -> None:
    """Process one comparison group: tables, JSON, plots."""
    avail = [c for c in conditions if c in condition_data]
    if len(avail) < 2:
        logger.warning(f"Skipping group '{group_name}': fewer than 2 conditions have data")
        return

    group_dir = output_dir / group_name
    tables_dir = group_dir / "tables"
    plots_dir = group_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline = avail[0]
    logger.info(f"── Group '{group_name}': {avail} (baseline={baseline}) ──")

    # Write per-category CSV tables
    for cat_name, metrics in METRIC_CATEGORIES.items():
        table = build_comparison_table(
            condition_data, avail, baseline, metrics, voat_baseline
        )
        table.to_csv(tables_dir / f"{cat_name}_table.csv", index=False)
        logger.info(f"  Table: {cat_name} ({len(table)} rows)")

    # Write JSON summary
    summary = build_group_json(
        group_name, avail, condition_labels, condition_data, voat_baseline
    )
    json_path = group_dir / f"comparison_{'_'.join(avail)}.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"  JSON: {json_path}")

    if summary["notable_differences"]:
        logger.info(f"  Notable differences (|Δ%|>10, no CI overlap):")
        for nd in summary["notable_differences"]:
            logger.info(f"    {nd['metric']} @ {nd['label']}: {nd['delta_pct']:+.1f}%")

    if not no_plots:
        plot_forest(condition_data, avail, condition_labels, voat_baseline, group_name, group_dir)
        plot_toxicity_kde(avail, condition_labels, sensitivity_dir, madoc_root, group_name, group_dir)
        plot_entropy_scatter(condition_data, avail, condition_labels, group_name, group_dir)
        plot_radar(condition_data, avail, condition_labels, voat_baseline, group_name, group_dir)
        plot_degree_dist(avail, condition_labels, sensitivity_dir, group_name, group_dir)
        plot_activity_timeseries(avail, condition_labels, sensitivity_dir, group_name, group_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sensitivity analysis conditions across groups"
    )
    parser.add_argument(
        "--sensitivity-dir",
        type=Path,
        default=Path("results/sensitivity"),
        help="Root directory with c0/run0/ … c4/run9/ subdirectories",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["persona:c0,c1,c2", "temperature:c0,c3,c4"],
        help='Group specs like "persona:c0,c1,c2" "temperature:c0,c3,c4"',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sensitivity/comparisons"),
    )
    parser.add_argument(
        "--voat-parquet",
        type=Path,
        default=Path("MADOC/voat-technology/voat_technology_madoc.parquet"),
        help="Path to combined Voat parquet (used for toxicity KDE plot)",
    )
    parser.add_argument(
        "--voat-madoc-root",
        type=Path,
        default=Path("MADOC/voat-technology"),
        help="Root of MADOC sample directories (used for Voat baseline metrics)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--labels",
        type=str,
        default="c0=Baseline,c1=NeutralPersona,c2=NoPolPersona,c3=TempLow,c4=TempHigh",
        help="Comma-separated condition=label pairs",
    )
    args = parser.parse_args()

    # Parse condition labels
    condition_labels = dict(DEFAULT_LABELS)
    for pair in args.labels.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            condition_labels[k.strip()] = v.strip()

    # Parse groups and collect all conditions needed
    groups: Dict[str, List[str]] = {}
    all_conditions: set = set()
    for g in args.groups:
        if ":" not in g:
            logger.warning(f"Skipping malformed group spec: {g!r} (expected 'name:c0,c1,...')")
            continue
        name, conds_str = g.split(":", 1)
        cond_list = [c.strip() for c in conds_str.split(",")]
        groups[name] = cond_list
        all_conditions.update(cond_list)

    if not groups:
        logger.error("No valid groups specified.")
        sys.exit(1)

    # Load condition data
    logger.info(f"Loading condition data from {args.sensitivity_dir}")
    condition_data = load_condition_data(
        args.sensitivity_dir, sorted(all_conditions), args.n_bootstrap
    )

    if not condition_data:
        logger.error(
            "No condition data loaded. "
            "Run run_sensitivity_pipeline.sh first to generate metrics.json files."
        )
        sys.exit(1)

    # Compute Voat baseline
    logger.info(f"Computing Voat baseline from {args.voat_madoc_root}")
    voat_baseline: Dict[str, Dict] = {}
    if args.voat_madoc_root.exists():
        voat_baseline = compute_voat_baseline(args.voat_madoc_root, args.n_bootstrap)
    else:
        logger.warning(f"MADOC root not found: {args.voat_madoc_root}")

    # Process each group
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group_name, conditions in groups.items():
        process_group(
            group_name=group_name,
            conditions=conditions,
            condition_labels=condition_labels,
            condition_data=condition_data,
            voat_baseline=voat_baseline,
            sensitivity_dir=args.sensitivity_dir,
            madoc_root=args.voat_madoc_root,
            output_dir=args.output_dir,
            no_plots=args.no_plots,
            n_bootstrap=args.n_bootstrap,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
