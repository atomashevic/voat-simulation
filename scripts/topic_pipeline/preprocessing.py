"""
Text preprocessing helpers for the topic modelling pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"^[.@]\S+\s+")
WS_RE = re.compile(r"\s+")
REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
ACCOUNT_DELETED_RE = re.compile(r"\baccount\s+deleted\b", re.IGNORECASE)
NERDS_RE = re.compile(r"\bnerds?\b", re.IGNORECASE)


def clean_text(text: str, *, lowercase: bool = False) -> str:
    """Apply lightweight normalisation that works across corpora."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    original = text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = URL_RE.sub(" ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1\1", text)
    text = re.sub(r"\.([A-Z])", r". \1", text)
    text = MENTION_RE.sub("", text)
    text = ACCOUNT_DELETED_RE.sub(" ", text)
    text = NERDS_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    text = text.strip()

    if lowercase:
        text = text.lower()

    if not text and original:
        logger.debug("Dropped text after cleaning due to emptiness.")

    return text


def preprocess_documents(
    documents: pd.DataFrame,
    *,
    text_column: str = "text",
    lowercase: bool = False,
    min_chars: int = 80,
) -> pd.DataFrame:
    """
    Clean a document DataFrame in-place and drop very short entries.

    Returns a new DataFrame with index reset to keep downstream merges simple.
    """
    if text_column not in documents.columns:
        raise ValueError(f"Column '{text_column}' not found in documents")

    processed = documents.copy()
    processed[text_column] = processed[text_column].fillna("").astype(str)
    processed[text_column] = processed[text_column].map(
        lambda text: clean_text(text, lowercase=lowercase)
    )
    processed = processed[processed[text_column].str.len() >= int(min_chars)]
    processed = processed.reset_index(drop=True)

    logger.info(
        "Prepared %d documents (min_chars=%d, lowercase=%s)",
        len(processed),
        min_chars,
        lowercase,
    )

    return processed


def iter_texts(documents: pd.DataFrame, text_column: str = "text") -> Iterable[str]:
    """Yield cleaned texts in document order."""
    for text in documents[text_column].tolist():
        yield text
