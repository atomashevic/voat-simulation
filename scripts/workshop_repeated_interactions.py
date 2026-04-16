#!/usr/bin/env python3
"""
Visualize repeated-interaction rates across the 60-network main benchmark.

The comparison uses 30 simulation runs and 30 matched Voat windows.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_network_from_posts(posts_csv: Path) -> nx.Graph:
    """Build undirected network from simulation posts.csv."""
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
    
    if "user_id" not in df.columns:
        return G
    
    G.add_nodes_from(df["user_id"].dropna().unique())
    
    if "parent_id" in df.columns:
        post_users = df.set_index("post_id")["user_id"].to_dict()
        
        for _, row in df.iterrows():
            if pd.notna(row.get("parent_id")) and row["parent_id"] != row["post_id"]:
                child_user = row["user_id"]
                parent_user = post_users.get(row["parent_id"])
                
                if parent_user and child_user and parent_user != child_user:
                    if G.has_edge(child_user, parent_user):
                        G[child_user][parent_user]["weight"] += 1
                    else:
                        G.add_edge(child_user, parent_user, weight=1)
    
    return G


def compute_repeated_interaction_pct(G: nx.Graph) -> float:
    """Compute percentage of edges with weight > 1."""
    if G.number_of_edges() == 0:
        return 0.0
    
    repeated = sum(1 for u, v, d in G.edges(data=True) if d.get("weight", 1) > 1)
    total = G.number_of_edges()
    
    return 100.0 * repeated / total


def load_all_repeated_pcts(
    results_dir: Path,
    voat_dir: Path,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Load repeated interaction percentages from all networks."""
    
    sim_pcts = []
    voat_pcts = []
    
    # Load simulation networks
    for run_dir in sorted(results_dir.glob("run*")):
        if not run_dir.is_dir():
            continue
        
        posts_file = run_dir / "posts.csv"
        if posts_file.exists():
            try:
                G = build_network_from_posts(posts_file)
                pct = compute_repeated_interaction_pct(G)
                sim_pcts.append((run_dir.name, pct))
                logger.debug("%s: %.1f%% repeated (edges=%d)", run_dir.name, pct, G.number_of_edges())
            except Exception as e:
                logger.warning("Failed to load %s: %s", run_dir.name, e)
    
    logger.info("Loaded %d simulation networks", len(sim_pcts))
    
    # Load Voat networks
    for sample_dir in sorted(voat_dir.glob("sample_*")):
        if not sample_dir.is_dir():
            continue
        
        parquet_files = list(sample_dir.glob("*.parquet"))
        if parquet_files:
            try:
                G = build_network_from_parquet(parquet_files[0])
                pct = compute_repeated_interaction_pct(G)
                voat_pcts.append((sample_dir.name, pct))
                logger.debug("%s: %.1f%% repeated (edges=%d)", sample_dir.name, pct, G.number_of_edges())
            except Exception as e:
                logger.warning("Failed to load %s: %s", sample_dir.name, e)
    
    logger.info("Loaded %d Voat networks", len(voat_pcts))
    
    return sim_pcts, voat_pcts


def plot_repeated_interactions(
    sim_pcts: List[Tuple[str, float]],
    voat_pcts: List[Tuple[str, float]],
    output_path: Path,
) -> None:
    """Create comparison plot of repeated interaction percentages."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set background
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Colors
    sim_color = "#1f77b4"
    voat_color = "#ff7f0e"
    
    # Extract values
    sim_vals = [pct for name, pct in sim_pcts]
    voat_vals = [pct for name, pct in voat_pcts]
    
    # Create positions for strip plot
    np.random.seed(42)
    sim_x = np.random.normal(0, 0.08, len(sim_vals))
    voat_x = np.random.normal(1, 0.08, len(voat_vals))
    
    # Plot individual points
    ax.scatter(sim_x, sim_vals, c=sim_color, alpha=0.6, s=80, edgecolors='white', linewidths=0.5)
    ax.scatter(voat_x, voat_vals, c=voat_color, alpha=0.6, s=80, edgecolors='white', linewidths=0.5)
    
    # Plot means with error bars (std)
    sim_mean, sim_std = np.mean(sim_vals), np.std(sim_vals)
    voat_mean, voat_std = np.mean(voat_vals), np.std(voat_vals)
    
    ax.errorbar(0, sim_mean, yerr=sim_std, fmt='D', color='black', markersize=12, 
                capsize=8, capthick=2, elinewidth=2, zorder=10)
    ax.errorbar(1, voat_mean, yerr=voat_std, fmt='D', color='black', markersize=12,
                capsize=8, capthick=2, elinewidth=2, zorder=10)
    
    # Add mean labels
    ax.text(0.15, sim_mean, f'{sim_mean:.1f}%', fontsize=12, fontweight='bold', 
            va='center', color='#333333')
    ax.text(1.15, voat_mean, f'{voat_mean:.1f}%', fontsize=12, fontweight='bold',
            va='center', color='#333333')
    
    # Styling
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Simulation\n(n={len(sim_vals)})', f'Voat\n(n={len(voat_vals)})'], 
                       fontsize=13, fontweight='bold')
    ax.set_ylabel("% Edges with Weight > 1\n(Repeated Interactions)", fontsize=13, fontweight='bold')
    ax.set_title(
        "Repeated Interactions: Simulation vs Voat\n"
        "Percentage of Edges Representing Multiple Interactions",
        fontsize=15,
        fontweight='bold',
        pad=15,
    )
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', color='#cccccc')
    
    # Tick styling
    ax.tick_params(axis='both', labelsize=11, colors='#333333')
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, max(max(sim_vals), max(voat_vals)) * 1.15)
    
    # Summary stats box
    stats_text = (
        f"Summary\n"
        f"{'─' * 20}\n"
        f"Simulation:\n"
        f"  Mean: {sim_mean:.1f}%\n"
        f"  Std:  {sim_std:.1f}%\n\n"
        f"Voat:\n"
        f"  Mean: {voat_mean:.1f}%\n"
        f"  Std:  {voat_std:.1f}%"
    )
    
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#dddddd", alpha=0.95))
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize repeated interactions across networks."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing simulation run subdirectories.",
    )
    parser.add_argument(
        "--voat-dir",
        type=Path,
        default=Path("MADOC/voat-technology-midlife30"),
        help="Directory containing matched Voat window subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workshop/figures/repeated_interactions_60networks.png"),
        help="Output file for the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    voat_dir = args.voat_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    
    sim_pcts, voat_pcts = load_all_repeated_pcts(results_dir, voat_dir)
    
    if not sim_pcts and not voat_pcts:
        logger.error("No networks loaded")
        return
    
    plot_repeated_interactions(sim_pcts, voat_pcts, output_path)


if __name__ == "__main__":
    main()
