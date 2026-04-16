#!/usr/bin/env python3
"""
Find the most similar simulation-Voat network pair and create side-by-side visualization.

Compares network metrics (density, avg_degree, core_pct, lcc_ratio, avg_clustering)
between 30 simulation runs and 30 Voat samples, finds the pair with minimum
z-scored Euclidean distance, and generates a side-by-side core-periphery visualization.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Metrics to compare for similarity
COMPARE_METRICS = ["density", "avg_degree", "core_pct", "lcc_ratio", "avg_clustering"]


def parse_network_stats(stats_file: Path) -> Dict[str, float]:
    """Parse network statistics from enhanced_network_analysis.txt."""
    stats = {}
    try:
        text = stats_file.read_text()
        
        # Parse key metrics using regex
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
        
        # Try metrics.json first
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            try:
                data = json.loads(metrics_file.read_text())
                net = data.get("network", {})
                metrics[run_name] = {
                    "density": net.get("density", 0),
                    "avg_degree": net.get("avg_degree", 0),
                    "avg_clustering": net.get("avg_clustering", 0),
                    "lcc_ratio": net.get("lcc_ratio", 0),
                    "core_pct": net.get("core_pct", 0),
                }
                continue
            except Exception:
                pass
        
        # Fallback to parsing network_analysis.txt
        for stats_file in [run_dir / "network_analysis.txt", run_dir / "enhanced_network_analysis.txt"]:
            if stats_file.exists():
                stats = parse_network_stats(stats_file)
                if stats:
                    metrics[run_name] = stats
                    break
    
    logger.info("Loaded metrics for %d simulation runs", len(metrics))
    return metrics


def load_voat_metrics(voat_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load network metrics from all Voat samples."""
    metrics = {}
    
    for sample_dir in sorted(voat_dir.glob("sample_*")):
        if not sample_dir.is_dir():
            continue
            
        sample_name = sample_dir.name
        
        for stats_file in [sample_dir / "enhanced_network_analysis.txt", sample_dir / "network_analysis.txt"]:
            if stats_file.exists():
                stats = parse_network_stats(stats_file)
                if stats:
                    metrics[sample_name] = stats
                    break
    
    logger.info("Loaded metrics for %d Voat samples", len(metrics))
    return metrics


def find_most_similar_pair(
    sim_metrics: Dict[str, Dict[str, float]],
    voat_metrics: Dict[str, Dict[str, float]],
    rank: int = 1,
) -> Tuple[str, str, float]:
    """Find the sim-voat pair with minimum z-scored Euclidean distance.
    
    Args:
        rank: 1 for best, 2 for second-best, etc.
    """
    
    # Build matrices for z-scoring
    all_names = list(sim_metrics.keys()) + list(voat_metrics.keys())
    all_values = []
    
    for name in all_names:
        m = sim_metrics.get(name) or voat_metrics.get(name)
        row = [m.get(k, 0) for k in COMPARE_METRICS]
        all_values.append(row)
    
    arr = np.array(all_values)
    
    # Z-score normalization
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)
    stds[stds == 0] = 1  # Avoid division by zero
    z_arr = (arr - means) / stds
    
    # Split back to sim and voat
    n_sim = len(sim_metrics)
    sim_z = z_arr[:n_sim]
    voat_z = z_arr[n_sim:]
    
    sim_names = list(sim_metrics.keys())
    voat_names = list(voat_metrics.keys())
    
    # Compute all distances and sort
    all_pairs = []
    for i, sim_name in enumerate(sim_names):
        for j, voat_name in enumerate(voat_names):
            dist = np.linalg.norm(sim_z[i] - voat_z[j])
            all_pairs.append((dist, sim_name, voat_name))
    
    all_pairs.sort(key=lambda x: x[0])
    
    # Get the pair at specified rank (1-indexed)
    idx = min(rank - 1, len(all_pairs) - 1)
    min_dist, best_sim, best_voat = all_pairs[idx]
    
    logger.info("Rank %d pair: %s <-> %s (distance: %.3f)", rank, best_sim, best_voat, min_dist)
    
    return best_sim, best_voat, min_dist


def build_network_from_posts(posts_csv: Path) -> nx.Graph:
    """Build undirected network from posts.csv."""
    df = pd.read_csv(posts_csv)
    G = nx.Graph()
    
    # Add all users as nodes
    G.add_nodes_from(df["user_id"].unique())
    
    # Add edges from comments
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
    
    # Add all users
    if "user_id" in df.columns:
        G.add_nodes_from(df["user_id"].dropna().unique())
    
    # Create edges from parent-child relationships
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


def load_core_periphery(cp_file: Path) -> Dict[int, bool]:
    """Load core-periphery membership from CSV."""
    try:
        df = pd.read_csv(cp_file)
        user_col = "user_id" if "user_id" in df.columns else df.columns[0]
        core_col = "is_core" if "is_core" in df.columns else "cp_label"
        
        if core_col == "is_core":
            return df.set_index(user_col)["is_core"].to_dict()
        else:
            # cp_label: 0 = core, 1 = periphery
            return {row[user_col]: row[core_col] == 0 for _, row in df.iterrows()}
    except Exception as e:
        logger.warning("Failed to load core-periphery from %s: %s", cp_file, e)
        return {}


def plot_network(
    G: nx.Graph,
    is_core: Dict,
    ax: plt.Axes,
    title: str,
    show_legend: bool = True,
) -> None:
    """Plot network with core-periphery coloring."""
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Get largest connected component for better visualization
    if not nx.is_connected(G):
        lcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc_nodes).copy()
    
    # Node colors based on core membership
    node_colors = []
    for node in G.nodes():
        if is_core.get(node, False):
            node_colors.append("#e74c3c")  # Red for core
        else:
            node_colors.append("#3498db")  # Blue for periphery
    
    # Node sizes based on degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [20 + 80 * (degrees[n] / max_deg) for n in G.nodes()]
    
    # Layout
    pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=50, seed=42)
    
    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="gray", width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    ax.set_title(title, fontsize=12, weight="bold")
    ax.axis("off")
    
    # Add legend only if requested
    if show_legend:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", label="Core"),
            Patch(facecolor="#3498db", label="Periphery"),
        ]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=9)


def create_comparison_figure(
    sim_run: str,
    voat_sample: str,
    results_dir: Path,
    voat_dir: Path,
    output_path: Path,
    sim_metrics: Dict[str, float],
    voat_metrics_dict: Dict[str, float],
) -> None:
    """Create side-by-side network visualization."""
    
    # Load simulation network and core-periphery
    sim_dir = results_dir / sim_run
    sim_posts = sim_dir / "posts.csv"
    sim_cp = sim_dir / "core_periphery.csv"
    
    if sim_posts.exists():
        G_sim = build_network_from_posts(sim_posts)
    else:
        G_sim = nx.Graph()
    
    is_core_sim = load_core_periphery(sim_cp) if sim_cp.exists() else {}
    
    # Load Voat network and core-periphery
    voat_sample_dir = voat_dir / voat_sample
    voat_parquet = list(voat_sample_dir.glob("*.parquet"))
    voat_cp = voat_sample_dir / "enhanced_core_periphery_membership.csv"
    
    if voat_parquet:
        G_voat = build_network_from_parquet(voat_parquet[0])
    else:
        G_voat = nx.Graph()
    
    is_core_voat = load_core_periphery(voat_cp) if voat_cp.exists() else {}
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot both networks (no legend on left to avoid overlap with text box)
    plot_network(G_sim, is_core_sim, ax1, f"Simulation ({sim_run})", show_legend=False)
    plot_network(G_voat, is_core_voat, ax2, f"Voat ({voat_sample})", show_legend=True)
    
    # Add metrics comparison as text (positioned on the left, larger font)
    metrics_text = "Metrics Comparison:\n"
    metrics_text += f"{'Metric':<15} {'Sim':>10} {'Voat':>10}\n"
    metrics_text += "-" * 37 + "\n"
    for m in COMPARE_METRICS:
        sim_val = sim_metrics.get(m, 0)
        voat_val = voat_metrics_dict.get(m, 0)
        metrics_text += f"{m:<15} {sim_val:>10.4f} {voat_val:>10.4f}\n"
    
    fig.text(0.02, 0.02, metrics_text, ha="left", fontsize=11, family="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.95, pad=0.6))
    
    fig.suptitle("Most Similar Network Pair: Simulation vs Voat", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find most similar sim-voat network pair and visualize."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing run01, run02, ... subdirectories.",
    )
    parser.add_argument(
        "--voat-dir",
        type=Path,
        default=Path("MADOC/voat-technology"),
        help="Directory containing Voat sample_1, sample_2, ... subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workshop/figures/network_pair_most_similar.png"),
        help="Output file for the generated figure.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Which ranked pair to use (1=best, 2=second-best, etc.).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    voat_dir = args.voat_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    
    # Load metrics
    sim_metrics = load_sim_metrics(results_dir)
    voat_metrics = load_voat_metrics(voat_dir)
    
    if not sim_metrics or not voat_metrics:
        logger.error("Could not load metrics from both sources")
        return
    
    # Find similar pair at specified rank
    best_sim, best_voat, distance = find_most_similar_pair(sim_metrics, voat_metrics, rank=args.rank)
    
    # Create visualization
    create_comparison_figure(
        best_sim,
        best_voat,
        results_dir,
        voat_dir,
        output_path,
        sim_metrics[best_sim],
        voat_metrics[best_voat],
    )


if __name__ == "__main__":
    main()

