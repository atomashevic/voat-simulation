#!/usr/bin/env python3
"""Aggregate comparisons between simulations and MADOC Voat samples."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline import (
    load_madoc_samples,
    load_simulation_run,
    summarize_madoc_sample,
    summarize_simulation_run,
)
from scripts.pipeline.reporting import (
    build_comparison_table,
    build_report_payload,
    write_json_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_embedding_similarity(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def parse_entropy_pairs_csv(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    inter_sum = intra_sum = 0.0
    inter_n = intra_n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = next(f).strip().split(",")
        idx_pair = header.index("pair_type")
        idx_hpt = header.index("H_per_token")
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(idx_pair, idx_hpt):
                continue
            pair = parts[idx_pair]
            try:
                hpt = float(parts[idx_hpt])
            except ValueError:
                continue
            if pair == "interpersonal":
                inter_sum += hpt
                inter_n += 1
            elif pair == "intrapersonal":
                intra_sum += hpt
                intra_n += 1
    if inter_n > 0:
        out["inter_hpt_mean"] = inter_sum / inter_n
    if intra_n > 0:
        out["intra_hpt_mean"] = intra_sum / intra_n
    return out


def aggregate_network_voat_from_csv(csv_path: Path) -> Dict[str, Any]:
    out = {
        "avg_degree": [],
        "density": [],
        "avg_clustering": [],
        "largest_component_ratio": [],
        "core_pct": [],
    }
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        header = next(f).rstrip("\n").split(",")
        idx_name = 0
        idx_avg_deg = header.index("Avg Degree")
        idx_density = header.index("Density")
        idx_avg_clust = header.index("Avg Clustering")
        idx_lcc = header.index("Largest Component Ratio")
        idx_core_pct = header.index("Core Percentage") if "Core Percentage" in header else None
        for line in f:
            parts = line.rstrip("\n").split(",")
            if not parts or parts[0].startswith("Simulation"):
                continue
            name = parts[idx_name]
            if not name.startswith("sample_"):
                continue
            try:
                out["avg_degree"].append(float(parts[idx_avg_deg]))
                out["density"].append(float(parts[idx_density]))
                out["avg_clustering"].append(float(parts[idx_avg_clust]))
                out["largest_component_ratio"].append(float(parts[idx_lcc]))
                if idx_core_pct is not None and parts[idx_core_pct]:
                    out["core_pct"].append(float(parts[idx_core_pct]))
            except ValueError:
                continue
    return {k: _mean_sd(v) for k, v in out.items()}


def summarize_madoc_samples(root: Path):
    samples = load_madoc_samples(root)
    return [summarize_madoc_sample(sample) for sample in samples]


def summarize_simulations(sim_dirs: Sequence[Path]):
    summaries = []
    for sim_dir in sim_dirs:
        run = load_simulation_run(sim_dir)
        summaries.append(summarize_simulation_run(run))
    return summaries


def _mean_sd(values: Iterable[float]) -> Dict[str, Optional[float]]:
    vals = list(values)
    if not vals:
        return {"mean": None, "sd": None, "n": 0}
    mean_val = sum(vals) / len(vals)
    if len(vals) == 1:
        return {"mean": mean_val, "sd": None, "n": 1}
    variance = sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)
    return {"mean": mean_val, "sd": variance ** 0.5, "n": len(vals)}


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen: set[str] = set()
    ordered: List[Path] = []
    for path in paths:
        p = Path(path)
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(p)
    return ordered


def _extract_section(items, section: str, key: str) -> List[float]:
    vals: List[float] = []
    for item in items:
        section_dict = getattr(item, section, None)
        if not isinstance(section_dict, dict):
            continue
        value = section_dict.get(key)
        if value is not None:
            vals.append(float(value))
    return vals


def build_voat_sample_summary(madoc_metrics) -> Dict[str, Any]:
    targets = {
        "users": ("activity", "unique_users"),
        "posts": ("activity", "posts_total"),
        "comments": ("activity", "comments_total"),
        "avg_thread_len": ("activity", "avg_thread_length"),
        "tox_posts_mean": ("toxicity", "posts_mean"),
        "tox_comments_mean": ("toxicity", "comments_mean"),
        "active_users_mean_per_day": ("activity", "mean_daily_active_users"),
    }
    summary = {"n_samples": len(madoc_metrics)}
    for label, (section, key) in targets.items():
        summary[label] = _mean_sd(_extract_section(madoc_metrics, section, key))
    return summary


def collect_simulation_artifacts(sim_dirs: Sequence[Path]) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    for sim_dir in sim_dirs:
        sim_path = Path(sim_dir)
        name = sim_path.name

        embedding_candidates = [
            sim_path / "sim_voat_embedding_similarity.json",
            sim_path / "embedding_similarity.json",
        ]
        embedding_path = next((p for p in embedding_candidates if p.exists()), embedding_candidates[0])

        entropy_candidates = [
            sim_path / "pairs_all.csv",
            sim_path / "convergence_entropy" / "pairs_all.csv",
        ]
        entropy_path = next((p for p in entropy_candidates if p.exists()), entropy_candidates[0])

        artifacts[name] = {
            "embedding": parse_embedding_similarity(embedding_path),
            "entropy": parse_entropy_pairs_csv(entropy_path),
        }
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MADOC vs simulation metrics")
    parser.add_argument("--madoc-root", type=Path, default=Path("MADOC/voat-technology"))
    parser.add_argument("--sim-dirs", nargs="*", type=Path, default=None, help="Simulation directories to include")
    parser.add_argument("--sim-dir", type=Path, help="(deprecated) primary simulation dir")
    parser.add_argument("--sim3-dir", type=Path, help="(deprecated) secondary simulation dir")
    parser.add_argument("--voat-comp-csv", type=Path, default=Path("simulation3/voat-comparison/network_comparison_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/latest/madoc_comparison_summary.json"))
    args = parser.parse_args()

    sim_dirs: List[Path] = []
    if args.sim_dirs:
        sim_dirs.extend(Path(p) for p in args.sim_dirs)
    if args.sim_dir:
        sim_dirs.append(args.sim_dir)
    if args.sim3_dir:
        sim_dirs.append(args.sim3_dir)
    if not sim_dirs:
        sim_dirs = [Path("simulation")]
    sim_dirs = _dedupe_paths(sim_dirs)

    madoc_metrics = summarize_madoc_samples(args.madoc_root)
    sim_metrics = summarize_simulations(sim_dirs)

    comparison_rows = build_comparison_table(madoc_metrics, sim_metrics)
    base_config = {
        "madoc_root": str(args.madoc_root),
        "sim_dirs": [str(p) for p in sim_dirs],
        "voat_comp_csv": str(args.voat_comp_csv),
    }
    payload = build_report_payload(madoc_metrics, sim_metrics, comparison_rows, base_config)
    payload["simulation_artifacts"] = collect_simulation_artifacts(sim_dirs)
    payload["voat_network"] = aggregate_network_voat_from_csv(args.voat_comp_csv)
    payload["voat_samples_summary"] = build_voat_sample_summary(madoc_metrics)

    write_json_report(args.out, payload)


if __name__ == "__main__":
    main()
