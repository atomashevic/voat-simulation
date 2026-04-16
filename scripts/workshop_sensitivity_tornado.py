#!/usr/bin/env python3
"""
workshop_sensitivity_tornado.py — Tornado plot + SI table for OAT sensitivity analysis.

Reads all 5 comparison JSONs (persona, temperature, budget, cpr, churn),
assembles a unified delta-% matrix, and produces:
  1. tornado_plot.png  — multi-panel forest/tornado figure (one panel per
                         metric category), conditions on y-axis, delta% on x-axis,
                         filled bars = significant (no CI overlap), hatched = overlapping
  2. sensitivity_table.csv — full conditions × metrics table with delta%, CI bounds,
                              and significance flag; suitable for SI

Usage:
    python scripts/workshop_sensitivity_tornado.py \
        --comparisons-dir results/sensitivity/comparisons \
        --output-dir results/sensitivity/comparisons/SI
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Condition metadata ─────────────────────────────────────────────────────────
CONDITION_META: Dict[str, Dict] = {
    "c1":  {"label": "Neutral\nPersona",    "group": "Persona",      "color": "#ff7f0e"},
    "c2":  {"label": "No-Politics\nPersona","group": "Persona",      "color": "#2ca02c"},
    "c3":  {"label": "Temp\nLow (0.3)",     "group": "Temperature",  "color": "#9467bd"},
    "c4":  {"label": "Temp\nHigh (0.9)",    "group": "Temperature",  "color": "#d62728"},
    "c5":  {"label": "Budget\nFlat (s=1.5)","group": "Budget",       "color": "#8c564b"},
    "c6":  {"label": "Budget\nSteep(s=3.5)","group": "Budget",       "color": "#e377c2"},
    "c7":  {"label": "CPR\nLow (2:1)",      "group": "CPR",          "color": "#7f7f7f"},
    "c8":  {"label": "CPR\nHigh (50:1)",    "group": "CPR",          "color": "#bcbd22"},
    "c9":  {"label": "Churn\nLow",          "group": "Churn",        "color": "#17becf"},
    "c10": {"label": "Churn\nHigh",         "group": "Churn",        "color": "#1f77b4"},
}

GROUP_COLORS: Dict[str, str] = {
    "Persona":     "#ff7f0e",
    "Temperature": "#9467bd",
    "Budget":      "#8c564b",
    "CPR":         "#7f7f7f",
    "Churn":       "#17becf",
}

# Distinct tints for category header bars and column backgrounds
CAT_TINTS: Dict[str, str] = {
    "Activity":  "#ddeeff",
    "Network":   "#ddeedc",
    "Toxicity":  "#fddcdc",
    "Topic":     "#ecddf5",
    "Entropy":   "#fef0d5",
}

# Source JSON for each group
GROUP_FILES: Dict[str, str] = {
    "persona":     "persona/comparison_c0_c1_c2.json",
    "temperature": "temperature/comparison_c0_c3_c4.json",
    "budget":      "budget/comparison_c0_c5_c6.json",
    "cpr":         "cpr/comparison_c0_c7_c8.json",
    "churn":       "churn/comparison_c0_c9_c10.json",
}

# Metric display names and categories (full set — used for CSV table)
METRIC_CATEGORIES: Dict[str, List[Tuple[str, str]]] = {
    "Activity": [
        ("activity.posts_per_day",          "Posts / day"),
        ("activity.comments_per_day",        "Comments / day"),
        ("activity.mean_posts_per_user",     "Posts / user"),
        ("activity.avg_thread_length",       "Avg thread length"),
        ("activity.mean_daily_active_users", "Daily active users"),
    ],
    "Network": [
        ("network.density",        "Density"),
        ("network.avg_clustering", "Avg clustering"),
        ("network.core_pct",       "Core %"),
        ("network.core_density",   "Core density"),
        ("network.modularity",     "Modularity"),
    ],
    "Toxicity": [
        ("toxicity.mean",           "Mean toxicity"),
        ("toxicity.p90",            "Toxicity p90"),
        ("toxicity.frac_above_0.5", "Frac > 0.5"),
        ("toxicity.comment_mean",   "Comment tox. mean"),
    ],
    "Topic": [
        ("topic.coverage_sim2", "Topic coverage"),
        ("topic.mean_cosine",   "Mean cosine sim."),
        ("topic.sim2_topics",   "N topics"),
    ],
    "Entropy": [
        ("entropy.H_per_token_mean", "H / token"),
        ("entropy.inter_hpt_mean",   "Inter-thread H/T"),
        ("entropy.intra_hpt_mean",   "Intra-thread H/T"),
    ],
}

# Reduced metric set used in the heatmap plots (SI figure)
PLOT_METRIC_CATEGORIES: Dict[str, List[Tuple[str, str]]] = {
    "Activity": [
        ("activity.comments_per_day",        "Comments / day"),
        ("activity.avg_thread_length",       "Thread length"),
        ("activity.mean_daily_active_users", "DAU"),
    ],
    "Network": [
        ("network.density",        "Density"),
        ("network.avg_clustering", "Clustering"),
    ],
    "Toxicity": [
        ("toxicity.mean", "Mean toxicity"),
    ],
    "Topic": [
        ("topic.coverage_sim2", "Topic coverage"),
        ("topic.sim2_topics",   "N topics"),
        ("topic.mean_cosine",   "Mean cosine"),
    ],
    "Entropy": [
        ("entropy.H_per_token_mean", "H / token"),
        ("entropy.inter_hpt_mean",   "Inter-thread H/T"),
        ("entropy.intra_hpt_mean",   "Intra-thread H/T"),
    ],
}

CONDITIONS_ORDERED = ["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_deltas(comparisons_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load delta%, CI overlap, and raw means for every condition × metric."""
    data: Dict[str, Dict[str, Any]] = {c: {} for c in CONDITIONS_ORDERED}

    for group, fname in GROUP_FILES.items():
        fpath = comparisons_dir / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        jdata = json.loads(fpath.read_text())
        for metric_key, conds in jdata["metrics"].items():
            for cond, entry in conds.items():
                if cond == "c0" or cond not in data:
                    continue
                data[cond][metric_key] = {
                    "mean":       entry.get("mean"),
                    "ci_lower":   entry.get("ci_lower"),
                    "ci_upper":   entry.get("ci_upper"),
                    "delta_pct":  entry.get("delta_pct_vs_baseline"),
                    "significant": not entry.get("ci_overlap_baseline", True),
                    "c0_mean":    conds.get("c0", {}).get("mean"),
                }
    return data


# ── Table builder ──────────────────────────────────────────────────────────────

def build_table(data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Build conditions × metrics summary DataFrame."""
    rows = []
    for cat, metrics in METRIC_CATEGORIES.items():
        for metric_key, metric_label in metrics:
            row: Dict[str, Any] = {
                "category":     cat,
                "metric_key":   metric_key,
                "metric_label": metric_label,
            }
            # baseline mean from any condition that has it
            c0_mean = None
            for cond in CONDITIONS_ORDERED:
                entry = data[cond].get(metric_key, {})
                if entry.get("c0_mean") is not None:
                    c0_mean = entry["c0_mean"]
                    break
            row["c0_mean"] = c0_mean

            for cond in CONDITIONS_ORDERED:
                entry = data[cond].get(metric_key, {})
                row[f"{cond}_mean"]      = entry.get("mean")
                row[f"{cond}_delta_pct"] = entry.get("delta_pct")
                row[f"{cond}_sig"]       = entry.get("significant", False)
            rows.append(row)

    return pd.DataFrame(rows)


# ── Tornado / heatmap plot ─────────────────────────────────────────────────────

PANEL_SPLITS: List[Tuple[str, List[str]]] = [
    ("Activity and network", ["Activity", "Network"]),
    ("Toxicity, topic, and entropy", ["Toxicity", "Topic", "Entropy"]),
]

GROUPS: List[Tuple[str, List[str]]] = [
    ("Persona",     ["c1", "c2"]),
    ("Temperature", ["c3", "c4"]),
    ("Budget",      ["c5", "c6"]),
    ("CPR",         ["c7", "c8"]),
    ("Churn",       ["c9", "c10"]),
]


def _build_row_items() -> List[Tuple[str, str]]:
    """Return ordered list of ("sep"|"cond", name) for the 10-condition layout."""
    items: List[Tuple[str, str]] = []
    for gname, conds in GROUPS:
        items.append(("sep", gname))
        for c in conds:
            items.append(("cond", c))
    return items


def _draw_heatmap_panel(
    data: Dict[str, Dict[str, Any]],
    panel_cats: List[str],
    panel_title: str,
    output_path: Path,
    cap: float = 60.0,
    metric_source: Optional[Dict[str, List[Tuple[str, str]]]] = None,
) -> None:
    """
    Draw one heatmap panel for the given metric categories.

    Fixes vs. the old single-panel version:
    - xlim extends left to LABEL_OFFSET so row labels are never clipped
    - category divider lines drawn with ax.plot() in data coordinates
    - larger cells (1.0 × 1.0) and fonts (8-9 pt) for readability
    - outer border drawn explicitly
    - clip_on=False on row labels as a belt-and-suspenders guard
    """
    if metric_source is None:
        metric_source = PLOT_METRIC_CATEGORIES

    # ── Collect metrics ────────────────────────────────────────────────────
    panel_metrics: List[Tuple[str, str, str]] = []  # (key, label, cat)
    for cat in panel_cats:
        for key, label in metric_source[cat]:
            panel_metrics.append((key, label, cat))
    n_cols = len(panel_metrics)

    row_items = _build_row_items()
    cond_list = [v for t, v in row_items if t == "cond"]
    n_conds = len(cond_list)
    n_seps  = len([1 for t, _ in row_items if t == "sep"])

    # ── Build value/significance matrices ─────────────────────────────────
    val_mat = np.zeros((n_conds, n_cols))
    sig_mat = np.zeros((n_conds, n_cols), dtype=bool)
    txt_mat = [["" for _ in range(n_cols)] for _ in range(n_conds)]

    for ri, cond in enumerate(cond_list):
        for ci, (mkey, _, _) in enumerate(panel_metrics):
            entry = data[cond].get(mkey, {})
            delta = entry.get("delta_pct")
            sig   = entry.get("significant", False)
            if delta is None:
                txt_mat[ri][ci] = "—"
            else:
                val_mat[ri, ci] = float(np.clip(delta, -cap, cap))
                sig_mat[ri, ci] = sig
                txt_mat[ri][ci] = f"{delta:+.0f}"

    # ── Layout constants (all in data units; 1 cell = 1.0 × 1.0) ──────────
    SEP_H       = 0.50  # height of group-separator strip
    LABEL_OFF   = 3.2   # data units reserved LEFT of x=0 for row labels
    CAT_STRIP_H = 0.48  # colored category strip sits directly above heatmap
    COL_LABEL_H = 2.4   # space for rotated column labels above the strip
    HEAD_BANDS  = CAT_STRIP_H + COL_LABEL_H  # total header space
    CAT_BAR_H   = CAT_STRIP_H  # alias kept for fig_h calc
    CBAR_BELOW  = 1.2   # data units below y=0 for colorbar region

    # total heatmap height (bottom=0, top=total_h)
    total_h = n_conds * 1.0 + n_seps * SEP_H

    # Precompute y_bottom for every row item (top-down)
    y_bottom: Dict[Tuple[str, str], float] = {}
    y = total_h
    for item_type, item_val in row_items:
        h = SEP_H if item_type == "sep" else 1.0
        y -= h
        y_bottom[(item_type, item_val)] = y

    # ── Figure sizing ──────────────────────────────────────────────────────
    SCALE = 0.72          # inches per data unit — wider cells with fewer columns
    fig_w = (LABEL_OFF + n_cols) * SCALE
    fig_h = (CBAR_BELOW + total_h + HEAD_BANDS + 0.5) * SCALE

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-LABEL_OFF, n_cols)
    ax.set_ylim(-CBAR_BELOW, total_h + HEAD_BANDS)
    ax.axis("off")

    cmap     = plt.cm.RdBu_r
    norm_obj = plt.Normalize(vmin=-cap, vmax=cap)

    # ── Draw cells & separators ────────────────────────────────────────────
    cond_idx = 0
    for item_type, item_val in row_items:
        yb = y_bottom[(item_type, item_val)]
        h  = SEP_H if item_type == "sep" else 1.0

        if item_type == "sep":
            gc = GROUP_COLORS[item_val]
            ax.add_patch(plt.Rectangle(
                (0, yb), n_cols, h,
                facecolor=gc, alpha=0.15, zorder=1, linewidth=0))
            ax.text(n_cols / 2, yb + h / 2, item_val,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=gc, zorder=2)
        else:
            cond = item_val
            meta = CONDITION_META[cond]
            # Row label — clip_on=False ensures it paints outside xlim clip region
            ax.text(-0.22, yb + 0.5,
                    meta["label"].replace("\n", " "),
                    ha="right", va="center",
                    fontsize=9, color=GROUP_COLORS[meta["group"]],
                    clip_on=False)

            for ci in range(n_cols):
                val = val_mat[cond_idx, ci]
                sig = sig_mat[cond_idx, ci]
                txt = txt_mat[cond_idx][ci]

                rgba = cmap(norm_obj(val))
                ax.add_patch(plt.Rectangle(
                    (ci, yb), 1.0, 1.0,
                    facecolor=rgba, zorder=1, linewidth=0))

                # Significance: inset border avoids overlap artefacts
                if sig:
                    ax.add_patch(plt.Rectangle(
                        (ci + 0.05, yb + 0.05), 0.90, 0.90,
                        facecolor="none", edgecolor="black",
                        linewidth=2.2, zorder=3))

                brightness = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                tc  = "white" if brightness < 0.55 else "#111111"
                fw  = "bold" if sig else "normal"
                ax.text(ci + 0.5, yb + 0.5, txt,
                        ha="center", va="center",
                        fontsize=8, color=tc, fontweight=fw, zorder=4)

            cond_idx += 1

    # ── Outer border ──────────────────────────────────────────────────────
    ax.plot([0, n_cols, n_cols, 0, 0],
            [0, 0, total_h, total_h, 0],
            color="#333", linewidth=1.2, zorder=6)

    # ── Column headers (rotated — start above the category strip) ────────────
    for ci, (_, mlabel, _) in enumerate(panel_metrics):
        ax.text(ci + 0.5, total_h + CAT_STRIP_H + 0.08, mlabel,
                ha="left", va="bottom",
                fontsize=8.5, rotation=48, rotation_mode="anchor",
                clip_on=False)

    # ── Column background tints (per category, full heatmap height) ──────────
    cumcol = 0
    for cat in panel_cats:
        w = len(metric_source[cat])
        tint = CAT_TINTS.get(cat, "#f0f0f0")
        ax.add_patch(plt.Rectangle(
            (cumcol, 0), w, total_h,
            facecolor=tint, alpha=0.25, zorder=0, linewidth=0))
        cumcol += w

    # ── Category strip — sits directly on top of heatmap, below col labels ──────
    # Each category gets a solid tinted strip with its name.
    # A 2-pt white border on every side creates clear visual separation between
    # adjacent strips regardless of how narrow each strip is.
    strip_y = total_h  # bottom of strip = top of heatmap
    cumcol = 0
    for cat in panel_cats:
        w    = len(metric_source[cat])
        tint = CAT_TINTS.get(cat, "#f0f0f0")
        PAD  = 0.04  # inset so white borders between adjacent strips are visible
        ax.add_patch(plt.Rectangle(
            (cumcol + PAD, strip_y + PAD),
            w - 2 * PAD, CAT_STRIP_H - 2 * PAD,
            facecolor=tint, alpha=1.0, zorder=3,
            linewidth=2.5, edgecolor="white"))
        ax.text(cumcol + w / 2, strip_y + CAT_STRIP_H / 2, cat,
                ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#222222", zorder=4,
                clip_on=False)
        cumcol += w

    # (no vertical divider lines — category strips provide sufficient grouping)

    # ── Colorbar (placed in the bottom margin via inset_axes) ─────────────
    # Convert data coordinates → axes fraction for placement
    data_to_ax_x = lambda xd: (xd - (-LABEL_OFF)) / (n_cols + LABEL_OFF)
    data_to_ax_y = lambda yd: (yd - (-CBAR_BELOW)) / (total_h + HEAD_BANDS + CBAR_BELOW)

    cbar_left   = data_to_ax_x(n_cols * 0.15)
    cbar_width  = data_to_ax_x(n_cols * 0.70) - cbar_left
    cbar_bottom = data_to_ax_y(-CBAR_BELOW + 0.15)
    cbar_height = data_to_ax_y(-CBAR_BELOW + 0.45) - cbar_bottom

    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "Δ% vs baseline (C0)   [colour scale capped at ±60%]",
        fontsize=8)
    cbar.set_ticks([-60, -40, -20, 0, 20, 40, 60])
    cbar.ax.tick_params(labelsize=7.5)

    # ── Title ─────────────────────────────────────────────────────────────
    fig.text(0.5, 0.995,
             f"OAT sensitivity — {panel_title}",
             ha="center", va="top",
             fontsize=11, fontweight="bold")
    fig.text(0.5, 0.975,
             "Bold border = non-overlapping 99% bootstrap CI  |  colour = Δ% from baseline",
             ha="center", va="top", fontsize=8, color="#444")

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_tornado(data: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """
    Produce two heatmap panels saved to output_dir:
      tornado_activity_network.png
      tornado_toxicity_topic_entropy.png
    """
    for panel_title, panel_cats in PANEL_SPLITS:
        slug = panel_title.lower().replace(" ", "_").replace(",", "").replace("&", "and").replace("/", "_")
        out_path = output_dir / f"tornado_{slug}.png"
        print(f"  Panel: {panel_title} ({', '.join(panel_cats)})")
        _draw_heatmap_panel(data, panel_cats, panel_title, out_path)


# ── SI table formatter ─────────────────────────────────────────────────────────

def save_si_table(df: pd.DataFrame, output_path: Path) -> None:
    """Save a publication-ready CSV with delta% and significance markers."""
    rows = []
    for _, r in df.iterrows():
        row: Dict[str, Any] = {
            "Category": r["category"],
            "Metric": r["metric_label"],
            "Baseline (C0) mean": _fmt(r["c0_mean"]),
        }
        for cond in CONDITIONS_ORDERED:
            meta = CONDITION_META[cond]
            delta = r.get(f"{cond}_delta_pct")
            sig   = r.get(f"{cond}_sig", False)
            if delta is None:
                row[f"{meta['label'].replace(chr(10),' ')} Δ%"] = "—"
            else:
                marker = "*" if sig else ""
                row[f"{meta['label'].replace(chr(10),' ')} Δ%"] = f"{delta:+.1f}{marker}"
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"Saved SI table → {output_path}")


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if abs(v) < 0.01:
        return f"{v:.4f}"
    if abs(v) < 1:
        return f"{v:.3f}"
    return f"{v:.2f}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons-dir", type=Path,
                        default=Path("results/sensitivity/comparisons"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/sensitivity/comparisons/SI"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sensitivity deltas...")
    data = load_all_deltas(args.comparisons_dir)

    print("Building summary table...")
    df = build_table(data)
    save_si_table(df, args.output_dir / "sensitivity_table.csv")

    print("Generating tornado plots (2 panels)...")
    plot_tornado(data, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
