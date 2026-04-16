from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.debug("JSON not found: %s", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON at {path}") from exc


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.debug("CSV not found: %s", path)
        return None
    return pd.read_csv(path)


@dataclass
class MadocSampleData:
    """Container for MADOC sample JSON payloads."""

    name: str
    path: Path
    full_results: Dict[str, Any]
    toxicity: Dict[str, Any]
    thread_length: Dict[str, Any]


def load_madoc_sample(sample_dir: Path) -> MadocSampleData:
    """Load the main JSON artifacts for a MADOC sample directory."""
    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    name = sample_dir.name
    full_results = _read_json(sample_dir / "full_results.json")
    toxicity = _read_json(sample_dir / "toxicity.json")
    thread_length = _read_json(sample_dir / "thread_length.json")
    return MadocSampleData(
        name=name,
        path=sample_dir,
        full_results=full_results,
        toxicity=toxicity,
        thread_length=thread_length,
    )


def load_madoc_samples(
    madoc_root: Path, sample_names: Optional[Sequence[str]] = None
) -> List[MadocSampleData]:
    """Load one or more MADOC samples from a root directory."""
    madoc_root = Path(madoc_root)
    if not madoc_root.exists():
        raise FileNotFoundError(f"MADOC root does not exist: {madoc_root}")

    if sample_names:
        sample_dirs: List[Path] = []
        for name in sample_names:
            candidate = madoc_root / name
            if candidate.is_dir():
                sample_dirs.append(candidate)
            else:
                logger.warning("Requested MADOC sample missing: %s", candidate)
    else:
        sample_dirs = sorted(
            p for p in madoc_root.iterdir() if p.is_dir() and p.name.startswith("sample_")
        )

    return [load_madoc_sample(sdir) for sdir in sample_dirs]


@dataclass
class SimulationRun:
    """In-memory representation of a simulation directory."""

    name: str
    path: Path
    posts: Optional[pd.DataFrame]
    users: Optional[pd.DataFrame]
    news: Optional[pd.DataFrame]
    toxigen: Optional[pd.DataFrame]


def load_simulation_run(sim_dir: Path) -> SimulationRun:
    """Load CSV artifacts for a simulation run directory."""
    sim_dir = Path(sim_dir)
    if not sim_dir.is_dir():
        raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")
    name = sim_dir.name
    posts = _read_csv(sim_dir / "posts.csv")
    users = _read_csv(sim_dir / "users.csv")
    news = _read_csv(sim_dir / "news.csv")
    toxigen = _read_csv(sim_dir / "toxigen.csv")
    if posts is None:
        logger.warning("Simulation '%s' missing posts.csv at %s", name, sim_dir)
    return SimulationRun(
        name=name,
        path=sim_dir,
        posts=posts,
        users=users,
        news=news,
        toxigen=toxigen,
    )


def load_simulation_runs(sim_dirs: Sequence[Path]) -> List[SimulationRun]:
    """Load multiple simulation directories."""
    runs: List[SimulationRun] = []
    for entry in sim_dirs:
        sim_path = Path(entry)
        if not sim_path.exists():
            logger.warning("Simulation path does not exist: %s", sim_path)
            continue
        runs.append(load_simulation_run(sim_path))
    return runs
