#!/usr/bin/env python3
"""
Find the best simulation run to pair with a Voat sample and generate visualizations.

This script:
1. Loads network metrics from all 30 simulation runs
2. Either uses an explicit --sim-run or finds the best match using z-scored distance
3. Generates side-by-side core-periphery visualization
4. Generates degree distribution overlay figure
5. Saves metrics comparison table

Usage:
    # Auto-find best match for sample_2
    python voat_sim_best_pair_viz.py --voat-sample sample_2

    # Use explicit pair (sample_8 <-> run29) for visual similarity
    python voat_sim_best_pair_viz.py --voat-sample sample_8 --sim-run run29
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Metrics to compare for similarity
COMPARE_METRICS = ["density", "avg_degree", "core_pct", "lcc_ratio", "avg_clustering"]

# Style constants
CORE_COLOR = "#e74c3c"  # Red
PERIPHERY_COLOR = "#3498db"  # Blue


def parse_network_stats(stats_file: Path) -> Dict[str, float]:
    """Parse network statistics from enhanced_network_analysis.txt."""
    stats = {}
    try:
        text = stats_file.read_text()

        patterns = {
            "density": r"Density:\s*([\d.]+)",
            "avg_degree": r"Avg Degree:\s*([\d.]+)",
            "avg_clustering": r"Avg Clustering:\s*([\d.]+)",
            "lcc_ratio": r"Largest Component Ratio:\s*([\d.]+)",
            "core_pct": r"Core Size:\s*\d+\s*nodes?\s*\(([\d.]+)%",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                stats[key] = float(match.group(1))

    except Exception as e:
        logger.warning("Failed to parse %s: %s", stats_file, e)

    return stats


def load_sim_metrics(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load network metrics from all simulation runs."""
    metrics = {}

    for run_dir in sorted(results_dir.glob("run*")):
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name
        metrics_file = run_dir / "metrics.json"

        if metrics_file.exists():
            try:
                data = json.loads(metrics_file.read_text())
                net = data.get("network", {})
                metrics[run_name] = {
                    "density": net.get("density", net.get("density_lcc", 0)),
                    "avg_degree": net.get("avg_degree", 0),
                    "avg_clustering": net.get("avg_clustering", 0),
                    "lcc_ratio": net.get("lcc_ratio", 0),
                    "core_pct": net.get("core_pct", 0),
                    "nodes": net.get("nodes", 0),
                    "edges": net.get("edges", 0),
                    "lcc_nodes": net.get("lcc_nodes", 0),
                    "core_nodes": net.get("core_nodes", 0),
                }
            except Exception as e:
                logger.warning("Failed to load %s: %s", metrics_file, e)

    logger.info("Loaded metrics for %d simulation runs", len(metrics))
    return metrics


def load_voat_sample_metrics(voat_dir: Path, sample_name: str) -> Dict[str, float]:
    """Load network metrics for a specific Voat sample."""
    sample_dir = voat_dir / sample_name
    stats_file = sample_dir / "enhanced_network_analysis.txt"

    if not stats_file.exists():
        logger.error("Could not find %s", stats_file)
        return {}

    stats = parse_network_stats(stats_file)

    # Also parse node/edge counts
    try:
        text = stats_file.read_text()

        match = re.search(r"Num Nodes:\s*(\d+)", text)
        if match:
            stats["nodes"] = int(match.group(1))

        match = re.search(r"Num Edges:\s*(\d+)", text)
        if match:
            stats["edges"] = int(match.group(1))

        match = re.search(r"Largest Component Size:\s*(\d+)", text)
        if match:
            stats["lcc_nodes"] = int(match.group(1))

        match = re.search(r"Core Size:\s*(\d+)\s*nodes?", text)
        if match:
            stats["core_nodes"] = int(match.group(1))

    except Exception as e:
        logger.warning("Failed to parse additional stats: %s", e)

    logger.info("Loaded metrics for %s: %s", sample_name, stats)
    return stats


def find_best_match(
    sim_metrics: Dict[str, Dict[str, float]],
    target_metrics: Dict[str, float],
) -> Tuple[str, float, pd.DataFrame]:
    """Find the simulation run most similar to target metrics using z-scored distance.

    Returns:
        Tuple of (best_run_name, distance, rankings_dataframe)
    """
    # Build matrix for z-scoring: all sims + target
    all_names = list(sim_metrics.keys()) + ["target"]
    all_values = []

    for name in sim_metrics:
        row = [sim_metrics[name].get(k, 0) for k in COMPARE_METRICS]
        all_values.append(row)

    target_row = [target_metrics.get(k, 0) for k in COMPARE_METRICS]
    all_values.append(target_row)

    arr = np.array(all_values)

    # Z-score normalization
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)
    stds[stds == 0] = 1
    z_arr = (arr - means) / stds

    # Target is the last row
    target_z = z_arr[-1]
    sim_z = z_arr[:-1]
    sim_names = list(sim_metrics.keys())

    # Compute distances
    distances = []
    for i, sim_name in enumerate(sim_names):
        dist = np.linalg.norm(sim_z[i] - target_z)
        distances.append((sim_name, dist))

    # Sort by distance
    distances.sort(key=lambda x: x[1])

    # Create rankings dataframe
    rankings = pd.DataFrame([
        {
            "rank": i + 1,
            "run": name,
            "distance": dist,
            **sim_metrics[name]
        }
        for i, (name, dist) in enumerate(distances)
    ])

    best_run, best_dist = distances[0]
    logger.info("Best match: %s (distance: %.4f)", best_run, best_dist)

    return best_run, best_dist, rankings


def build_network_from_posts(posts_csv: Path) -> nx.Graph:
    """Build undirected network from posts.csv."""
    df = pd.read_csv(posts_csv)
    G = nx.Graph()

    G.add_nodes_from(df["user_id"].unique())

    comment_df = df[df["comment_to"] != -1]
    post_users = df.set_index("id")["user_id"].to_dict()

    for _, row in comment_df.iterrows():
        commenter = row["user_id"]
        parent_id = row["comment_to"]
        parent_user = post_users.get(parent_id)

        if parent_user and parent_user != commenter:
            if G.has_edge(commenter, parent_user):
                G[commenter][parent_user]["weight"] += 1
            else:
                G.add_edge(commenter, parent_user, weight=1)

    return G


def build_network_from_parquet(parquet_path: Path) -> nx.Graph:
    """Build undirected network from Voat parquet."""
    df = pd.read_parquet(parquet_path)
    G = nx.Graph()

    if "user_id" in df.columns:
        G.add_nodes_from(df["user_id"].dropna().unique())

    if "parent_id" in df.columns and "user_id" in df.columns:
        post_users = df.set_index("post_id")["user_id"].to_dict()

        for _, row in df.iterrows():
            if pd.notna(row.get("parent_id")) and row["parent_id"] != row["post_id"]:
                child_user = row["user_id"]
                parent_user = post_users.get(row["parent_id"])

                if parent_user and parent_user != child_user:
                    if G.has_edge(child_user, parent_user):
                        G[child_user][parent_user]["weight"] += 1
                    else:
                        G.add_edge(child_user, parent_user, weight=1)

    return G


def load_core_periphery(cp_file: Path) -> Dict:
    """Load core-periphery membership from CSV."""
    try:
        df = pd.read_csv(cp_file)
        user_col = "user_id" if "user_id" in df.columns else df.columns[0]

        if "is_core" in df.columns:
            return df.set_index(user_col)["is_core"].to_dict()
        elif "cp_label" in df.columns:
            return {row[user_col]: row["cp_label"] == 0 for _, row in df.iterrows()}

    except Exception as e:
        logger.warning("Failed to load core-periphery from %s: %s", cp_file, e)

    return {}


def get_lcc(G: nx.Graph) -> nx.Graph:
    """Extract largest connected component."""
    if len(G.nodes()) == 0:
        return G
    if nx.is_connected(G):
        return G
    lcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc_nodes).copy()


def plot_network(
    G: nx.Graph,
    is_core: Dict,
    ax: plt.Axes,
    title: str,
    metrics: Dict[str, float],
    show_legend: bool = True,
) -> None:
    """Plot network with core-periphery coloring."""

    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    # Get LCC for visualization
    G = get_lcc(G)

    # Node colors
    node_colors = [CORE_COLOR if is_core.get(node, False) else PERIPHERY_COLOR
                   for node in G.nodes()]

    # Node sizes based on degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [20 + 100 * (degrees[n] / max_deg) for n in G.nodes()]

    # Spring layout
    k = 2 / np.sqrt(len(G.nodes())) if len(G.nodes()) > 0 else 1
    pos = nx.spring_layout(G, k=k, iterations=50, seed=42)

    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color="gray", width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                          node_size=node_sizes, alpha=0.8)

    ax.set_title(title, fontsize=14, weight="bold")
    ax.axis("off")

    # Add metrics inset
    core_pct = metrics.get("core_pct", 0)
    lcc_nodes = int(metrics.get("lcc_nodes", len(G.nodes())))
    avg_deg = metrics.get("avg_degree", 0)

    inset_text = f"LCC: {lcc_nodes} nodes\nAvg degree: {avg_deg:.2f}\nCore: {core_pct:.1f}%"
    ax.text(0.02, 0.98, inset_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    if show_legend:
        legend_elements = [
            Patch(facecolor=CORE_COLOR, label="Core"),
            Patch(facecolor=PERIPHERY_COLOR, label="Periphery"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)


def create_core_periphery_figure(
    voat_sample: str,
    sim_run: str,
    voat_dir: Path,
    results_dir: Path,
    voat_metrics: Dict[str, float],
    sim_metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """Create side-by-side core-periphery visualization."""

    # Load Voat network
    voat_sample_dir = voat_dir / voat_sample
    voat_parquet = list(voat_sample_dir.glob("*.parquet"))
    voat_cp = voat_sample_dir / "enhanced_core_periphery_membership.csv"

    G_voat = build_network_from_parquet(voat_parquet[0]) if voat_parquet else nx.Graph()
    is_core_voat = load_core_periphery(voat_cp) if voat_cp.exists() else {}

    # Load simulation network
    sim_dir = results_dir / sim_run
    sim_posts = sim_dir / "posts.csv"
    sim_cp = sim_dir / "core_periphery.csv"

    G_sim = build_network_from_posts(sim_posts) if sim_posts.exists() else nx.Graph()
    is_core_sim = load_core_periphery(sim_cp) if sim_cp.exists() else {}

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Extract sample/run numbers for titles
    voat_num = voat_sample.split("_")[-1]
    sim_num = sim_run.replace("run", "")

    plot_network(G_voat, is_core_voat, ax1, f"a) Voat (sample {voat_num})",
                 voat_metrics, show_legend=True)
    plot_network(G_sim, is_core_sim, ax2, f"b) Simulation (run {sim_num})",
                 sim_metrics, show_legend=False)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved core-periphery figure to %s", output_path)


def create_degree_distribution_figure(
    voat_sample: str,
    sim_run: str,
    voat_dir: Path,
    results_dir: Path,
    output_path: Path,
) -> None:
    """Create degree distribution overlay figure."""

    # Load Voat network
    voat_sample_dir = voat_dir / voat_sample
    voat_parquet = list(voat_sample_dir.glob("*.parquet"))
    G_voat = build_network_from_parquet(voat_parquet[0]) if voat_parquet else nx.Graph()

    # Load simulation network
    sim_dir = results_dir / sim_run
    sim_posts = sim_dir / "posts.csv"
    G_sim = build_network_from_posts(sim_posts) if sim_posts.exists() else nx.Graph()

    # Get LCCs
    G_voat = get_lcc(G_voat)
    G_sim = get_lcc(G_sim)

    # Get degree sequences
    voat_degrees = [d for _, d in G_voat.degree()]
    sim_degrees = [d for _, d in G_sim.degree()]

    # Extract sample/run numbers
    voat_num = voat_sample.split("_")[-1]
    sim_num = sim_run.replace("run", "")

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Histogram overlay
    max_deg = max(max(voat_degrees), max(sim_degrees))
    bins = np.arange(0, min(max_deg + 2, 50), 1)

    ax1.hist(voat_degrees, bins=bins, alpha=0.6, label=f"Voat sample {voat_num}",
             color=PERIPHERY_COLOR, density=True)
    ax1.hist(sim_degrees, bins=bins, alpha=0.6, label=f"Simulation run {sim_num}",
             color=CORE_COLOR, density=True)
    ax1.set_xlabel("Degree", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Degree Distribution (Histogram)", fontsize=13, weight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 40)

    # Panel 2: Log-log CCDF
    for degrees, label, color in [
        (voat_degrees, f"Voat sample {voat_num}", PERIPHERY_COLOR),
        (sim_degrees, f"Simulation run {sim_num}", CORE_COLOR)
    ]:
        sorted_deg = np.sort(degrees)[::-1]
        ccdf = np.arange(1, len(sorted_deg) + 1) / len(sorted_deg)
        ax2.loglog(sorted_deg, ccdf, "o-", label=label, color=color,
                  markersize=4, alpha=0.7, linewidth=1.5)

    ax2.set_xlabel("Degree", fontsize=12)
    ax2.set_ylabel("CCDF", fontsize=12)
    ax2.set_title("Degree Distribution (Log-Log CCDF)", fontsize=13, weight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved degree distribution figure to %s", output_path)


def save_metrics_table(
    voat_sample: str,
    sim_run: str,
    voat_metrics: Dict[str, float],
    sim_metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """Save metrics comparison table as CSV."""

    rows = []
    for metric in ["nodes", "edges", "lcc_nodes", "lcc_ratio", "avg_degree",
                   "avg_clustering", "density", "core_pct", "core_nodes"]:
        rows.append({
            "metric": metric,
            "voat": voat_metrics.get(metric, "N/A"),
            "simulation": sim_metrics.get(metric, "N/A"),
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved metrics table to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Find best sim-voat pair and visualize")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--voat-dir", type=Path, default=Path("MADOC/voat-technology"))
    parser.add_argument("--voat-sample", type=str, default="sample_2",
                        help="Voat sample to compare (e.g., sample_8)")
    parser.add_argument("--sim-run", type=str, default=None,
                        help="Explicit simulation run to pair with (e.g., run29). "
                             "If not specified, auto-finds best match.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    voat_dir = args.voat_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    voat_sample = args.voat_sample

    # Load all simulation metrics
    sim_metrics_all = load_sim_metrics(results_dir)

    # Load target Voat sample metrics
    voat_metrics = load_voat_sample_metrics(voat_dir, voat_sample)

    if not sim_metrics_all or not voat_metrics:
        logger.error("Could not load required metrics")
        return

    # Either use explicit sim-run or find best match
    if args.sim_run:
        # Use explicit pairing
        best_run = args.sim_run
        if best_run not in sim_metrics_all:
            logger.error("Specified --sim-run '%s' not found in results", best_run)
            logger.info("Available runs: %s", ", ".join(sorted(sim_metrics_all.keys())))
            return
        distance = None
        rankings = None
        print("\n" + "=" * 60)
        print(f"EXPLICIT PAIRING: {voat_sample} <-> {best_run}")
        print("=" * 60)
    else:
        # Find best match automatically
        best_run, distance, rankings = find_best_match(sim_metrics_all, voat_metrics)
        print("\n" + "=" * 60)
        print(f"TOP 10 SIMULATION RUNS MATCHING {voat_sample}")
        print("=" * 60)
        print(rankings.head(10).to_string(index=False))

    # Get numbers for file naming
    voat_num = voat_sample.split("_")[-1]
    sim_num = best_run.replace("run", "").zfill(2)

    # Create visualizations
    print(f"\nGenerating visualizations for {voat_sample} <-> {best_run}...")

    create_core_periphery_figure(
        voat_sample, best_run, voat_dir, results_dir,
        voat_metrics, sim_metrics_all[best_run],
        output_dir / f"core_periphery_comparison_{voat_num}_{sim_num}.png"
    )

    create_degree_distribution_figure(
        voat_sample, best_run, voat_dir, results_dir,
        output_dir / f"degree_distribution_pair_{voat_num}_{sim_num}.png"
    )

    # Save metrics table
    save_metrics_table(
        voat_sample, best_run, voat_metrics, sim_metrics_all[best_run],
        output_dir.parent / "data" / f"best_pair_metrics_{voat_num}_{sim_num}.csv"
    )

    # Save rankings (if auto-matched)
    if rankings is not None:
        rankings.to_csv(output_dir.parent / "data" / f"sim_rankings_vs_{voat_sample}.csv", index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pair: {voat_sample} <-> {best_run}")
    if distance is not None:
        print(f"Z-scored Euclidean distance: {distance:.4f}")
    else:
        print("(Explicit pairing - no distance computed)")
    print(f"\nKey metrics comparison:")
    print(f"  {'Metric':<20} {'Voat':>12} {'Simulation':>12} {'Ratio':>10}")
    print(f"  {'-'*54}")
    for metric in ["avg_degree", "lcc_ratio", "core_pct", "density", "avg_clustering"]:
        v = voat_metrics.get(metric, 0)
        s = sim_metrics_all[best_run].get(metric, 0)
        ratio = s / v if v > 0 else 0
        print(f"  {metric:<20} {v:>12.4f} {s:>12.4f} {ratio:>10.2f}x")

    print(f"\nFiles generated:")
    print(f"  - {output_dir}/core_periphery_comparison_{voat_num}_{sim_num}.png")
    print(f"  - {output_dir}/degree_distribution_pair_{voat_num}_{sim_num}.png")
    print(f"  - {output_dir.parent}/data/best_pair_metrics_{voat_num}_{sim_num}.csv")
    if rankings is not None:
        print(f"  - {output_dir.parent}/data/sim_rankings_vs_{voat_sample}.csv")


if __name__ == "__main__":
    main()
