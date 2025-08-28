"""
Run simple association tests between user attributes and core–periphery membership.

Usage:
  PYENV_VERSION=ysocial python scripts/core_periphery_user_attribute_tests.py \
    --users simulation2/users.csv \
    --membership simulation2/enhanced_core_periphery_membership.csv \
    --outdir simulation2

Notes:
  - Joins on membership `user_id` to users `id` (simulation2 convention).
  - For MADOC samples, membership uses GUIDs from parquet posts; there is no
    users.csv with attributes in this repo, so only network features exist.
  - Outputs CSV summaries with test statistics and p-values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


def coerce_numeric(series: pd.Series):
    try:
        return pd.to_numeric(series)
    except Exception:
        return series


def ttest_by_core(df: pd.DataFrame, numeric_cols: List[str]):
    rows = []
    core = df[df["is_core"] == True]
    peri = df[df["is_core"] == False]
    for col in numeric_cols:
        x = pd.to_numeric(core[col], errors="coerce").dropna()
        y = pd.to_numeric(peri[col], errors="coerce").dropna()
        if len(x) < 3 or len(y) < 3:
            continue
        t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        # Cohen's d (Hedges' g correction for small sample)
        nx, ny = len(x), len(y)
        sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
        sp = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
        d = (np.mean(x) - np.mean(y)) / sp if sp > 0 else np.nan
        # Small sample correction (Hedges' g)
        J = 1 - (3 / (4 * (nx + ny) - 9)) if (nx + ny) > 9 else 1.0
        g = d * J
        rows.append(
            {
                "variable": col,
                "test": "t-test (Welch)",
                "n_core": nx,
                "n_periphery": ny,
                "mean_core": np.mean(x),
                "mean_periphery": np.mean(y),
                "t_stat": t_stat,
                "p_value": p_val,
                "effect_size_g": g,
            }
        )
    return pd.DataFrame(rows)


def chi2_by_core(df: pd.DataFrame, cat_cols: List[str]):
    rows = []
    for col in cat_cols:
        # Drop rare levels to avoid many sparse cells
        vc = df[col].value_counts(dropna=True)
        if len(vc) < 2:
            continue
        # Build contingency table
        ct = pd.crosstab(df[col], df["is_core"])
        if ct.shape[0] < 2 or ct.shape[1] != 2:
            # Need both core and periphery present
            continue
        try:
            chi2, p, dof, exp = stats.chi2_contingency(ct)
        except ValueError:
            # If any zeros cause failure, skip
            continue
        rows.append(
            {
                "variable": col,
                "test": "chi-square",
                "levels": ct.shape[0],
                "dof": dof,
                "chi2": chi2,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def main(users_csv: Path, membership_csv: Path, outdir: Path):
    users = pd.read_csv(users_csv)
    membership = pd.read_csv(membership_csv)

    # Determine join key compatibility
    # Simulation2: membership.user_id are numeric ids (integers)
    # MADOC: membership.user_id are GUID strings; join to users.csv not possible here
    left = membership.copy()
    right = users.copy()

    # Try numeric join first (simulation2)
    left["user_id_num"] = pd.to_numeric(left["user_id"], errors="coerce")
    right["id_num"] = pd.to_numeric(right.get("id"), errors="coerce")

    joined = None
    if left["user_id_num"].notna().any() and right["id_num"].notna().any():
        joined = left.merge(right, left_on="user_id_num", right_on="id_num", how="left")
        matched = joined["id"].notna().sum()
        if matched == 0:
            joined = None

    if joined is None:
        # Fall back to string join if both sides have exact user_id column
        if "user_id" in users.columns:
            joined = left.merge(users, on="user_id", how="left")
        else:
            raise SystemExit(
                "Could not join membership to users. For MADOC samples, a users.csv with attributes does not exist in this repo."
            )

    # Keep only rows with user attributes present
    # Heuristic: require at least username or a non-null id
    pre = len(joined)
    joined = joined[(joined.get("id").notna()) | (joined.get("username").notna())]
    post = len(joined)

    if post == 0:
        raise SystemExit("Join produced no rows with user attributes. Aborting.")

    # Select variables
    bool_core = joined["is_core"].astype(bool)
    df = joined.copy()
    df["is_core"] = bool_core

    # Identify numeric and categorical candidate columns from users.csv
    user_cols = set(users.columns) - {"email", "password", "username"}
    # Always include a few specific columns if present
    preferred_numeric = [
        c for c in ["age", "round_actions", "joined_on", "left_on"] if c in users.columns
    ]
    numeric_cols = [
        c
        for c in users.select_dtypes(include=[np.number]).columns
        if c in user_cols and c not in {"id"}
    ]
    for c in preferred_numeric:
        if c not in numeric_cols:
            numeric_cols.append(c)

    preferred_cats = [
        c
        for c in [
            "gender",
            "education_level",
            "toxicity",
            "leaning",
            "nationality",
            "oe",
            "co",
            "ex",
            "ag",
            "ne",
            "recsys_type",
            "frecsys_type",
            "user_type",
            "language",
            "is_page",
        ]
        if c in users.columns
    ]
    cat_cols = [
        c
        for c in users.columns
        if c in user_cols and c not in numeric_cols and c not in {"id"}
    ]
    # Prioritize preferred_cats and keep only reasonably small cardinality
    cat_cols = preferred_cats + [c for c in cat_cols if c not in preferred_cats]
    cat_cols = [c for c in cat_cols if df[c].nunique(dropna=True) >= 2 and df[c].nunique() <= 50]

    # Run tests
    t_df = ttest_by_core(df, numeric_cols)
    chi_df = chi2_by_core(df, cat_cols)

    # Sort by p-value
    if not t_df.empty:
        t_df = t_df.sort_values("p_value")
    if not chi_df.empty:
        chi_df = chi_df.sort_values("p_value")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not t_df.empty:
        t_df.to_csv(outdir / "core_membership_ttests.csv", index=False)
    if not chi_df.empty:
        chi_df.to_csv(outdir / "core_membership_chi2.csv", index=False)

    # Combined summary
    parts = []
    if not t_df.empty:
        parts.append(t_df.assign(kind="numeric"))
    if not chi_df.empty:
        parts.append(chi_df.assign(kind="categorical"))
    if parts:
        summary = pd.concat(parts, ignore_index=True, sort=False)
        summary.to_csv(outdir / "core_membership_user_attribute_tests.csv", index=False)

    # Print concise summary
    print(f"Joined rows with attributes: {post}/{pre}")
    if not t_df.empty:
        print("\nTop numeric associations (t-test):")
        print(t_df.head(10).to_string(index=False))
    else:
        print("\nNo numeric columns available for t-tests.")
    if not chi_df.empty:
        print("\nTop categorical associations (chi-square):")
        print(chi_df.head(10).to_string(index=False))
    else:
        print("\nNo categorical columns available for chi-square tests.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=Path, default=Path("simulation2/users.csv"))
    ap.add_argument(
        "--membership",
        type=Path,
        default=Path("simulation2/enhanced_core_periphery_membership.csv"),
    )
    ap.add_argument("--outdir", type=Path, default=Path("simulation2"))
    args = ap.parse_args()
    main(args.users, args.membership, args.outdir)

