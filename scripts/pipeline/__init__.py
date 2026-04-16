"""Shared pipeline utilities for MADOC vs simulation comparisons."""

from .io import (
    MadocSampleData,
    SimulationRun,
    load_madoc_sample,
    load_madoc_samples,
    load_simulation_run,
    load_simulation_runs,
)
from .metrics import (
    MadocSampleMetrics,
    SimulationMetrics,
    summarize_madoc_sample,
    summarize_simulation_run,
)
from .reporting import (
    build_comparison_table,
    build_report_payload,
    write_comparison_csv,
    write_json_report,
)

__all__ = [
    "MadocSampleData",
    "SimulationRun",
    "MadocSampleMetrics",
    "SimulationMetrics",
    "load_madoc_sample",
    "load_madoc_samples",
    "load_simulation_run",
    "load_simulation_runs",
    "summarize_madoc_sample",
    "summarize_simulation_run",
    "build_comparison_table",
    "build_report_payload",
    "write_comparison_csv",
    "write_json_report",
]
