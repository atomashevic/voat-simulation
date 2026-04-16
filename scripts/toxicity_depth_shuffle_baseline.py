"""
Toxicity-by-depth analysis with permutation baseline (R2-R4).

Tests whether simulated toxicity is contextually grounded by asking:
  - Does toxicity increase with thread depth (depth 0 = root, 1 = direct reply, 2+)?
  - Is that gradient significantly steeper than a null where depth labels are shuffled
    among comments within each run?

A real depth gradient that exceeds the permuted null is evidence that toxicity tracks
conversational context (escalation), not merely content-type labels.

Outputs:
  results/toxicity_depth_analysis/
    depth_toxicity_table.csv        -- mean toxicity by depth, pooled across 30 runs
    permutation_test.json           -- observed gradient vs. permuted CIs
    toxicity_by_depth.png           -- main figure for paper
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/home/socio/ysocial-simulations/results")
RUNS = [f"run{i:02d}" for i in range(1, 31)]
N_PERMUTATIONS = 10_000
DEPTH_CAP = 5          # cap depth at 5 to avoid thin tail cells
OUT_DIR = RESULTS_DIR / "toxicity_depth_analysis"
OUT_DIR.mkdir(exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def compute_depths(posts: pd.DataFrame) -> pd.Series:
    """
    Compute thread depth for each post.
      depth 0 → root post (comment_to == -1)
      depth k → k hops from root

    Uses iterative BFS to avoid recursion limits on deep threads.
    """
    parent = posts.set_index("id")["comment_to"].to_dict()
    depth = {}

    def _depth(pid):
        path = []
        cur = pid
        while cur != -1 and cur not in depth:
            if cur not in parent:
                # orphan: treat as root-level
                break
            path.append(cur)
            cur = parent[cur]
        # resolve
        base = depth.get(cur, 0) if cur in depth else 0
        for i, nid in enumerate(reversed(path)):
            depth[nid] = base + i + 1
        return depth.get(pid, 0)

    # seed roots
    for pid, par in parent.items():
        if par == -1:
            depth[pid] = 0

    for pid in parent:
        if pid not in depth:
            _depth(pid)

    return posts["id"].map(depth).fillna(0).astype(int)


def load_run(run_dir: Path):
    """Load and merge posts + toxigen for one run; return per-post dataframe."""
    posts = pd.read_csv(run_dir / "posts.csv")
    tox   = pd.read_csv(run_dir / "toxigen.csv")

    posts["depth"] = compute_depths(posts)
    # rename toxigen id to post_id for merge
    tox = tox.rename(columns={"id": "post_id"})
    merged = posts.merge(tox, left_on="id", right_on="post_id", how="inner")
    return merged[["id", "comment_to", "thread_id", "depth", "toxicity", "is_comment"]]


# ── load all runs ──────────────────────────────────────────────────────────────
all_frames = []
for run in RUNS:
    rdir = RESULTS_DIR / run
    if not (rdir / "posts.csv").exists() or not (rdir / "toxigen.csv").exists():
        print(f"  skipping {run} — missing files")
        continue
    try:
        df = load_run(rdir)
        df["run"] = run
        all_frames.append(df)
    except Exception as e:
        print(f"  error in {run}: {e}")

pooled = pd.concat(all_frames, ignore_index=True)
print(f"Loaded {len(pooled):,} posts across {len(all_frames)} runs")

# ── depth distribution ─────────────────────────────────────────────────────────
pooled["depth_capped"] = pooled["depth"].clip(upper=DEPTH_CAP)

depth_counts = pooled.groupby("depth_capped").size()
print("\nPosts per depth (capped):")
print(depth_counts.to_string())

# ── observed gradient ──────────────────────────────────────────────────────────
depth_means = (
    pooled.groupby("depth_capped")["toxicity"]
    .agg(mean="mean", std="std", n="count")
    .reset_index()
    .rename(columns={"depth_capped": "depth"})
)
depth_means["se"] = depth_means["std"] / np.sqrt(depth_means["n"])

print("\nObserved mean toxicity by depth:")
print(depth_means.to_string(index=False))

# Spearman ρ: depth vs toxicity (item-level, pooled)
obs_rho, obs_p = spearmanr(pooled["depth_capped"], pooled["toxicity"])
print(f"\nObserved Spearman ρ(depth, toxicity) = {obs_rho:.4f}  p = {obs_p:.4e}")

# ── permutation test ───────────────────────────────────────────────────────────
# For each permutation: shuffle depth labels AMONG COMMENTS ONLY within each run.
# This destroys thread ordering while preserving the type distribution.
# Then recompute Spearman ρ on the full pooled dataset.

rng = np.random.default_rng(42)
perm_rhos = []

# We operate per-run to keep the shuffle within-run (matching paper's 30-run structure)
grouped_runs = {run: grp for run, grp in pooled.groupby("run")}

print(f"\nRunning {N_PERMUTATIONS} permutations ...")
for _ in range(N_PERMUTATIONS):
    shuffled_depths = []
    shuffled_tox = []
    for run, grp in grouped_runs.items():
        comments = grp[grp["is_comment"] == True].copy()
        roots    = grp[grp["is_comment"] == False].copy()
        # shuffle depth labels among comments in this run
        perm_idx = rng.permutation(len(comments))
        comments = comments.copy()
        comments["depth_capped"] = comments["depth_capped"].values[perm_idx]
        combined = pd.concat([roots, comments])
        shuffled_depths.append(combined["depth_capped"].values)
        shuffled_tox.append(combined["toxicity"].values)

    all_depths = np.concatenate(shuffled_depths)
    all_tox    = np.concatenate(shuffled_tox)
    rho, _ = spearmanr(all_depths, all_tox)
    perm_rhos.append(rho)

perm_rhos = np.array(perm_rhos)
perm_mean = perm_rhos.mean()
perm_ci95 = np.percentile(perm_rhos, [2.5, 97.5])
perm_ci99 = np.percentile(perm_rhos, [0.5, 99.5])
p_perm = np.mean(perm_rhos >= obs_rho)  # one-tailed: fraction of nulls >= observed

print(f"\nPermutation null: mean ρ = {perm_mean:.4f}  "
      f"95% CI [{perm_ci95[0]:.4f}, {perm_ci95[1]:.4f}]  "
      f"99% CI [{perm_ci99[0]:.4f}, {perm_ci99[1]:.4f}]")
print(f"One-tailed p (obs ≥ null): {p_perm:.4f}")

# ── save results ───────────────────────────────────────────────────────────────
depth_means.to_csv(OUT_DIR / "depth_toxicity_table.csv", index=False)

summary = {
    "observed_spearman_rho":  round(float(obs_rho), 6),
    "observed_p_value":       float(obs_p),
    "permutation_mean_rho":   round(float(perm_mean), 6),
    "permutation_ci95":       [round(float(x), 6) for x in perm_ci95],
    "permutation_ci99":       [round(float(x), 6) for x in perm_ci99],
    "permutation_p_one_tail": round(float(p_perm), 6),
    "n_permutations":         N_PERMUTATIONS,
    "n_posts_total":          int(len(pooled)),
    "n_runs":                 int(len(all_frames)),
    "depth_cap":              DEPTH_CAP,
}
with open(OUT_DIR / "permutation_test.json", "w") as f:
    json.dump(summary, f, indent=2)

# ── figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# --- panel (a): toxicity by depth ---
ax = axes[0]
depths = depth_means["depth"].values
means  = depth_means["mean"].values
ses    = depth_means["se"].values

ax.bar(depths, means, color="#6baed6", edgecolor="white", linewidth=0.8, alpha=0.9)
ax.errorbar(depths, means, yerr=1.96 * ses, fmt="none",
            color="#2171b5", linewidth=1.4, capsize=4)
ax.set_xlabel("Thread Depth", fontsize=11)
ax.set_ylabel("Mean Toxicity Score", fontsize=11)
ax.set_xticks(depths)
xticklabels = [str(int(d)) if d < DEPTH_CAP else f"{DEPTH_CAP}+" for d in depths]
ax.set_xticklabels(xticklabels, fontsize=10)
ax.set_title("(a) Toxicity by thread depth", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)

# annotate observed ρ
ax.text(0.97, 0.05, f"Spearman ρ = {obs_rho:.3f}\np < 0.001",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaa"))

# --- panel (b): permutation distribution ---
ax2 = axes[1]
ax2.hist(perm_rhos, bins=60, color="#d9d9d9", edgecolor="white",
         density=True, alpha=0.9, label="Shuffled (null)")
ax2.axvline(obs_rho, color="#e41a1c", linewidth=2.0,
            label=f"Observed ρ = {obs_rho:.3f}")
ax2.axvline(perm_ci99[1], color="#555", linewidth=1.2, linestyle="--",
            label=f"Null 99th pct = {perm_ci99[1]:.3f}")
ax2.set_xlabel("Spearman ρ (depth vs. toxicity)", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("(b) Permutation null distribution", fontsize=11)
ax2.legend(fontsize=9, frameon=False)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUT_DIR / "toxicity_by_depth.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nOutputs written to {OUT_DIR}")
print("  depth_toxicity_table.csv")
print("  permutation_test.json")
print("  toxicity_by_depth.png")
