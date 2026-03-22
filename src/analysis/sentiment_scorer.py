"""
src/analysis/sentiment_scorer.py

Stage 5 — Sentiment Scoring and Energy Level Classification

Reads:
    data/processed/lyrics_cleaned.csv

Produces:
    data/analysis/layer2_sentiment.csv
    data/analysis/.sentiment_complete   (sentinel)

Pipeline:
    1. Load lyrics_cleaned.csv
    2. For each song with lyrics_quality != "missing":
       a. Compute VADER compound score on lyrics_clean
       b. Compute TextBlob polarity on lyrics_clean
       c. Ensemble: weighted average (VADER 0.6, TextBlob 0.4)
       d. C21 fallback: if lyrics_clean is null, use lyrics_verse_only
       e. Map ensemble score → energy_level (1–5) via config bins
    3. Songs with lyrics_quality == "missing" → sentiment_score=NaN, energy_level=NaN
    4. Validate output against layer2_sentiment_schema
    5. Write layer2_sentiment.csv
    6. Write sentinel with config_hash

Ensemble weighting rationale:
    VADER is optimized for short social/lyric text and handles
    punctuation/capitalization signals well. TextBlob provides a
    lexicon-based second opinion. 60/40 weighting favors VADER
    while retaining TextBlob's smoothing effect on edge cases.

Energy level mapping:
    The VADER compound score range [-1.0, +1.0] is divided into 5 bins
    using the thresholds defined in config.analysis.energy_level.bins.
    Default bins [0.2, 0.4, 0.6, 0.8] produce:
        score < 0.2           → energy_level 1  (very low / dark)
        0.2 ≤ score < 0.4     → energy_level 2  (low)
        0.4 ≤ score < 0.6     → energy_level 3  (moderate)
        0.6 ≤ score < 0.8     → energy_level 4  (high)
        score ≥ 0.8           → energy_level 5  (very high / euphoric)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.pipeline.config_loader import ProjectConfig, config_hash
from src.pipeline.schemas import (
    VALID_LYRICS_QUALITY,
    layer2_sentiment_schema,
    validate,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_INPUT_PATH = Path("data/processed/lyrics_cleaned.csv")
_OUTPUT_PATH = Path("data/analysis/layer2_sentiment.csv")
_SENTINEL = Path("data/analysis/.sentiment_complete")

# Ensemble weights — must sum to 1.0
_VADER_WEIGHT = 0.6
_TEXTBLOB_WEIGHT = 0.4

# Initialise VADER analyser once at module load (avoids repeated disk reads)
_vader = SentimentIntensityAnalyzer()


# ── Public Entry Point ────────────────────────────────────────────────────────


def run(config: ProjectConfig) -> dict:
    """
    Execute Stage 5 — Sentiment Scoring and Energy Level Classification.

    Args:
        config: Loaded ProjectConfig instance.

    Returns:
        Summary dict with scored_count, skipped_count, output_path.

    Raises:
        FileNotFoundError: if lyrics_cleaned.csv does not exist.
        RuntimeError:      if scoring produces zero valid rows.
    """
    logger.info("Stage 5 — Sentiment Scoring")
    logger.info("Input:  {}", _INPUT_PATH)
    logger.info("Output: {}", _OUTPUT_PATH)

    # Ensure output directory exists
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load input
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(
            f"lyrics_cleaned.csv not found at {_INPUT_PATH}. "
            "Run TEXT_CLEANING stage first."
        )

    df = pd.read_csv(_INPUT_PATH, dtype={"song_id": str})
    logger.info("Loaded {:,} songs from {}", len(df), _INPUT_PATH)

    # Apply decade filter if configured
    df = _apply_decade_filter(df, config)

    # Score sentiment and energy
    df = _score_dataframe(df, config)

    # Select output columns
    output_cols = ["song_id", "sentiment_score", "energy_level"]
    output_df = df[output_cols].copy()

    # Re-cast energy_level to nullable integer — survives CSV round-trip
    output_df["energy_level"] = pd.array(
        output_df["energy_level"].tolist(),
        dtype=pd.Int64Dtype(),
    )

    # Validate
    output_df = validate(output_df, layer2_sentiment_schema, "SENTIMENT_SCORING")

    # Write output
    output_df.to_csv(_OUTPUT_PATH, index=False)
    scored = output_df["sentiment_score"].notna().sum()
    skipped = output_df["sentiment_score"].isna().sum()
    logger.success(
        "Sentiment scoring complete — {:,} scored, {:,} skipped (missing lyrics)",
        scored,
        skipped,
    )

    # Write sentinel
    _write_sentinel(config, scored=int(scored), total=len(output_df))

    return {
        "scored_count": int(scored),
        "skipped_count": int(skipped),
        "output_path": str(_OUTPUT_PATH),
    }


# ── Core Scoring ──────────────────────────────────────────────────────────────


def _score_dataframe(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """
    Apply sentiment scoring and energy level assignment to every row.

    Modifies df in place adding columns:
        sentiment_score  (float, nullable)
        energy_level     (int, nullable)

    Args:
        df:     DataFrame from lyrics_cleaned.csv.
        config: ProjectConfig for energy bin thresholds.

    Returns:
        Modified DataFrame.
    """
    bins = config.analysis.energy_level.bins

    scores = []
    energy_levels = []
    processed = 0
    skipped = 0

    for _, row in df.iterrows():
        quality = row.get("lyrics_quality", "missing")

        if quality == "missing":
            scores.append(np.nan)
            energy_levels.append(pd.NA)
            skipped += 1
            continue

        # C21 fallback: prefer lyrics_clean, fall back to lyrics_verse_only
        text = _select_text(row)

        if not text or not text.strip():
            scores.append(np.nan)
            energy_levels.append(pd.NA)
            skipped += 1
            continue

        score = _ensemble_sentiment(text)
        level = _compute_energy_level(score, bins)

        scores.append(score)
        energy_levels.append(level)
        processed += 1

    df["sentiment_score"] = scores
    df["energy_level"] = pd.array(energy_levels, dtype=pd.Int64Dtype())

    logger.debug(
        "Scoring pass complete — {} processed, {} skipped",
        processed,
        skipped,
    )
    return df


def _select_text(row: pd.Series) -> Optional[str]:
    """
    Select the best available lyrics text for a row (C21 fallback logic).

    Priority:
        1. lyrics_clean        (full cleaned lyrics)
        2. lyrics_verse_only   (verse-only fallback)
        3. None                (no text available)

    Args:
        row: A single DataFrame row.

    Returns:
        Text string or None.
    """
    lyrics_clean = row.get("lyrics_clean")
    if isinstance(lyrics_clean, str) and lyrics_clean.strip():
        return lyrics_clean

    lyrics_verse = row.get("lyrics_verse_only")
    if isinstance(lyrics_verse, str) and lyrics_verse.strip():
        logger.debug(
            "C21 fallback: using lyrics_verse_only for song_id={}",
            row.get("song_id", "unknown"),
        )
        return lyrics_verse

    return None


def _ensemble_sentiment(text: str) -> float:
    """
    Compute a weighted ensemble sentiment score from VADER and TextBlob.

    Args:
        text: Cleaned lyrics string.

    Returns:
        Ensemble score in [-1.0, 1.0], rounded to 4 decimal places.
    """
    vader_score = _vader.polarity_scores(text)["compound"]
    textblob_score = TextBlob(text).sentiment.polarity

    ensemble = _VADER_WEIGHT * vader_score + _TEXTBLOB_WEIGHT * textblob_score

    # Clamp to valid range (TextBlob can produce edge values outside [-1, 1])
    ensemble = max(-1.0, min(1.0, ensemble))
    return round(ensemble, 4)


def _compute_energy_level(score: float, bins: list[float]) -> int:
    """
    Map a sentiment score to an integer energy level (1–5) using bin edges.

    The bins list defines N boundary values, producing N+1 buckets.
    Default bins [0.2, 0.4, 0.6, 0.8] produce levels 1–5.

    Args:
        score: Ensemble sentiment score in [-1.0, 1.0].
        bins:  Ascending list of boundary values from config.

    Returns:
        Integer energy level in [1, len(bins)+1].
    """
    for i, threshold in enumerate(bins):
        if score < threshold:
            return i + 1
    return len(bins) + 1


# ── Decade Filter ─────────────────────────────────────────────────────────────


def _apply_decade_filter(
    df: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """
    If config.dataset.decade_filter is set, restrict df to that decade only.

    Args:
        df:     Input DataFrame with 'decade' column.
        config: ProjectConfig with optional decade_filter.

    Returns:
        Filtered (or unmodified) DataFrame.
    """
    decade_filter = config.dataset.decade_filter
    if not decade_filter:
        return df

    if "decade" not in df.columns:
        logger.warning(
            "decade_filter='{}' set but 'decade' column not found — skipping filter.",
            decade_filter,
        )
        return df

    filtered = df[df["decade"] == decade_filter].copy()
    logger.info(
        "Decade filter '{}' applied — {:,} → {:,} songs",
        decade_filter,
        len(df),
        len(filtered),
    )
    return filtered


# ── Sentinel ──────────────────────────────────────────────────────────────────


def _write_sentinel(config: ProjectConfig, scored: int, total: int) -> None:
    """
    Write the stage sentinel file with config_hash and run metadata.

    Args:
        config: ProjectConfig for hash computation.
        scored: Number of successfully scored songs.
        total:  Total number of songs processed.
    """
    _SENTINEL.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "stage": "SENTIMENT_SCORING",
        "completed_at": datetime.now(tz=timezone.utc).isoformat(),
        "config_hash": config_hash(config),
        "scored": scored,
        "total": total,
        "output_path": str(_OUTPUT_PATH),
    }

    _SENTINEL.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    logger.debug("Sentinel written: {}", _SENTINEL)
