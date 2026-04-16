from __future__ import annotations

import json
import logging
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .metrics import MadocSampleMetrics, SimulationMetrics

logger = logging.getLogger(__name__)


COMPARISON_SLOTS = [
    ("activity", "posts_total", "Total posts", "count"),
    ("activity", "comments_total", "Total comments", "count"),
    ("activity", "unique_users", "Unique users", "count"),
    ("activity", "avg_thread_length", "Avg thread length", "float"),
    ("activity", "mean_posts_per_user", "Mean interactions per user", "float"),
    ("activity", "mean_daily_active_users", "Mean daily active users", "float"),
    ("toxicity", "overall_mean", "Mean toxicity (all)", "float"),
    ("toxicity", "posts_mean", "Mean toxicity (posts)", "float"),
    ("toxicity", "comments_mean", "Mean toxicity (comments)", "float"),
]


def build_comparison_table(
    madoc_metrics: Sequence[MadocSampleMetrics],
    sim_metrics: Sequence[SimulationMetrics],
) -> List[Dict[str, Any]]:
    """Create per-metric comparison rows between MADOC samples and simulations."""
    rows: List[Dict[str, Any]] = []
    for category, key, label, value_type in COMPARISON_SLOTS:
        madoc_vals = _extract_values(madoc_metrics, category, key)
        sim_vals = _extract_values(sim_metrics, category, key)
        row = {
            "category": category,
            "metric": key,
            "label": label,
            "madoc": _aggregate_values(madoc_vals),
            "simulation": _aggregate_values(sim_vals),
            "value_type": value_type,
        }
        madoc_mean = row["madoc"]["mean"] if row["madoc"] else None
        sim_mean = row["simulation"]["mean"] if row["simulation"] else None
        if madoc_mean is not None and sim_mean is not None:
            delta = sim_mean - madoc_mean
            row["delta"] = delta
            if madoc_mean:
                row["pct_diff"] = delta / madoc_mean
        rows.append(row)
    return rows


def build_report_payload(
    madoc_metrics: Sequence[MadocSampleMetrics],
    sim_metrics: Sequence[SimulationMetrics],
    comparison_rows: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the full JSON-ready payload."""
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": timestamp,
        "config": config,
        "madoc_samples": [m.to_dict() for m in madoc_metrics],
        "simulation_runs": [s.to_dict() for s in sim_metrics],
        "comparison": list(comparison_rows),
    }
    return payload


def write_json_report(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    logger.info("Wrote comparison report to %s", path)


def write_comparison_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Persist comparison rows as a CSV table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "metric",
        "label",
        "value_type",
        "madoc_mean",
        "madoc_min",
        "madoc_max",
        "madoc_count",
        "simulation_mean",
        "simulation_min",
        "simulation_max",
        "simulation_count",
        "delta",
        "pct_diff",
    ]
    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "category": row.get("category"),
                    "metric": row.get("metric"),
                    "label": row.get("label"),
                    "value_type": row.get("value_type"),
                    "madoc_mean": _nested(row.get("madoc"), "mean"),
                    "madoc_min": _nested(row.get("madoc"), "min"),
                    "madoc_max": _nested(row.get("madoc"), "max"),
                    "madoc_count": _nested(row.get("madoc"), "count"),
                    "simulation_mean": _nested(row.get("simulation"), "mean"),
                    "simulation_min": _nested(row.get("simulation"), "min"),
                    "simulation_max": _nested(row.get("simulation"), "max"),
                    "simulation_count": _nested(row.get("simulation"), "count"),
                    "delta": row.get("delta"),
                    "pct_diff": row.get("pct_diff"),
                }
            )
    logger.info("Wrote comparison table CSV to %s", path)


def _extract_values(
    metrics: Sequence[Any], category: str, key: str
) -> List[float]:
    values: List[float] = []
    for item in metrics:
        section = getattr(item, category, {})
        if not section:
            continue
        value = section.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _aggregate_values(values: Iterable[float]) -> Dict[str, Any]:
    vals = list(values)
    if not vals:
        return {}
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
    }


def _nested(obj: Optional[Dict[str, Any]], key: str) -> Optional[Any]:
    if not obj:
        return None
    return obj.get(key)
