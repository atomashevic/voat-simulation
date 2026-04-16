#!/usr/bin/env python3
"""Unified analytical pipeline comparing MADOC Voat samples against simulations."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline import (
    build_comparison_table,
    build_report_payload,
    load_madoc_samples,
    load_simulation_run,
    summarize_madoc_sample,
    summarize_simulation_run,
    write_comparison_csv,
    write_json_report,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MADOC Voat samples with simulation runs")
    parser.add_argument("--madoc-root", type=Path, default=Path("MADOC/voat-technology"))
    parser.add_argument(
        "--madoc-samples",
        nargs="*",
        help="Optional list of sample directory names (e.g., sample_1 sample_5)",
    )
    parser.add_argument(
        "--sim-dirs",
        nargs="+",
        type=Path,
        default=[Path("simulation")],
        help="Simulation directories to include in the comparison",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/latest"))
    parser.add_argument("--report-prefix", default="madoc_sim_comparison")
    parser.add_argument("--table-csv", type=Path, help="Optional explicit path for the comparison CSV table")
    parser.add_argument("--json", type=Path, help="Optional explicit path for the JSON report")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the comparison table to stdout after generation",
    )
    return parser.parse_args()


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    ordered: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    sim_dirs = _dedupe_paths([Path(p) for p in args.sim_dirs])
    if not sim_dirs:
        raise SystemExit("No simulation directories provided")

    madoc_samples = load_madoc_samples(args.madoc_root, sample_names=args.madoc_samples)
    if not madoc_samples:
        raise SystemExit("No MADOC samples found for comparison")

    madoc_metrics = [summarize_madoc_sample(sample) for sample in madoc_samples]
    sim_metrics = [summarize_simulation_run(load_simulation_run(sim_dir)) for sim_dir in sim_dirs]

    comparison_rows = build_comparison_table(madoc_metrics, sim_metrics)
    config = {
        "madoc_root": str(args.madoc_root),
        "madoc_samples": [sample.name for sample in madoc_samples],
        "sim_dirs": [str(p) for p in sim_dirs],
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.json or out_dir / f"{args.report_prefix}.json"
    csv_path = args.table_csv or out_dir / f"{args.report_prefix}_table.csv"

    payload = build_report_payload(madoc_metrics, sim_metrics, comparison_rows, config)
    write_json_report(json_path, payload)
    write_comparison_csv(csv_path, comparison_rows)

    if args.print_summary:
        for row in comparison_rows:
            logger.info(
                "%s/%s: madoc=%.3f sim=%.3f delta=%s",
                row["category"],
                row["label"],
                (row["madoc"] or {}).get("mean", float("nan")),
                (row["simulation"] or {}).get("mean", float("nan")),
                row.get("delta"),
            )


if __name__ == "__main__":
    main()
