from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def require_columns(df: pd.DataFrame, required: Iterable[str], *, context: str) -> None:
    """Raise a ValueError if any required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {', '.join(missing)}")
