"""
src/cultural_metrics/cmi_calculator.py

Stage 10 — Cultural Mood Index (CMI) Calculation
==================================================
Joins the master analysis DataFrame (layer4_schema) with the Jungian
classification output (layer5_schema), computes a Cultural Resonance
Score for every song, then aggregates per-decade Cultural Mood Index
(CMI) metrics and writes two output files:

  data/analysis/layer6_cultural_metrics.csv  — per-song scores
  data/analysis/decade_cmi.csv               — per-decade aggregates

Cultural Resonance Score
------------------------
A normalised float [0.0, 1.0] representing how culturally significant
a song is within its decade cohort, computed from:

  sentiment_score     (weight: config.cmi.weight_sentiment,  default 0.25)
  energy_level        (weight: config.cmi.weight_energy,     default 0.20)
  chorus_detected     (weight: config.cmi.weight_chorus,     default 0.15)
  lyrics_quality      (weight: config.cmi.weight_quality,    default 0.20)
  development_score   (weight: config.cmi.weight_jungian,    default 0.20)

Each component is normalised to [0.0, 1.0] before weighting.
Songs with missing values for a component receive the decade-cohort
mean for that component (mean imputation within decade).

Decade CMI Aggregates
---------------------
For each decade bucket the following are computed:
  CMI_sentiment       — mean sentiment_score across scored songs
  CMI_energy          — mean energy_level across scored songs
  emotional_tone      — mode emotional_tone
  dominant_jung_stage — mode jung_stage (excluding "unclassified")
  top_themes          — pipe-delimited top 3 themes by frequency
  top_resonance_songs — pipe-delimited top 3 song titles by
                        cultural_resonance_score

Coverage gate
-------------
config.cmi.min_coverage_pct (default 0.60) — the fraction of songs
in the input that must have a non-null cultural_resonance_score for
the stage to proceed. Raises CoverageError if gate is not met.

Idempotency
-----------
Sentinel .cmi_complete records config_hash. If matched, both output
files are read from disk and returned. Delete the sentinel (or change
config) to force recalculation.

Output schemas
--------------
See src/pipeline/schemas.py:
  layer6_schema      — layer6_cultural_metrics.csv
  decade_cmi_schema  — decade_cmi.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.config_loader import (
    ProjectConfig,
    load_config,
    sentinel_config_matches,
    write_sentinel,
)
from src.pipeline.schemas import (
    decade_cmi_schema,
    layer6_schema,
    validate,
)

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INPUT_LAYER4 = _PROJECT_ROOT / "data" / "analysis" / "layer2_full_analysis.csv"
_INPUT_LAYER5 = _PROJECT_ROOT / "data" / "analysis" / "layer5_jungian.csv"
_OUTPUT_LAYER6 = _PROJECT_ROOT / "data" / "analysis" / "layer6_cultural_metrics.csv"
_OUTPUT_DECADE = _PROJECT_ROOT / "data" / "analysis" / "decade_cmi.csv"
_SENTINEL = _PROJECT_ROOT / "data" / "analysis" / ".cmi_complete"

# ── Default component weights ─────────────────────────────────────────────────
_DEFAULT_WEIGHTS = {
    "sentiment": 0.25,
    "energy": 0.20,
    "chorus": 0.15,
    "quality": 0.20,
    "jungian": 0.20,
}

# ── Lyrics quality → numeric map ──────────────────────────────────────────────
_QUALITY_SCORE = {"full": 1.0, "partial": 0.5, "missing": 0.0}


# ═════════════════════════════════════════════════════════════════════════════
# Exceptions
# ═════════════════════════════════════════════════════════════════════════════


class CoverageError(RuntimeError):
    """Raised when scored song coverage falls below config.cmi.min_coverage_pct."""


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def run(config: ProjectConfig) -> dict:
    """
    Execute Stage 10 — CMI calculation.

    Parameters
    ----------
    config : ProjectConfig

    Returns
    -------
    dict with keys:
        songs_total     : int
        songs_scored    : int
        coverage_pct    : float
        decades_computed: int
        output_layer6   : Path
        output_decade   : Path
        skipped         : bool
    """
    for path, label in [
        (_INPUT_LAYER4, "layer2_full_analysis.csv"),
        (_INPUT_LAYER5, "layer5_jungian.csv"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found at {path}. Run the preceding stage first."
            )

    _OUTPUT_LAYER6.parent.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check ────────────────────────────────────────────────────
    if sentinel_config_matches(_SENTINEL, config):
        logger.info("Stage 10 [CMI] — sentinel matched, skipping.")
        layer6 = pd.read_csv(_OUTPUT_LAYER6)
        decade = pd.read_csv(_OUTPUT_DECADE)
        return {
            "songs_total": len(layer6),
            "songs_scored": int(layer6["cultural_resonance_score"].notna().sum()),
            "coverage_pct": float(layer6["cultural_resonance_score"].notna().mean()),
            "decades_computed": len(decade),
            "output_layer6": _OUTPUT_LAYER6,
            "output_decade": _OUTPUT_DECADE,
            "skipped": True,
        }

    # ── Load and join inputs ─────────────────────────────────────────────────
    layer4 = pd.read_csv(_INPUT_LAYER4)
    layer5 = pd.read_csv(_INPUT_LAYER5)
    logger.info(
        "Stage 10 [CMI] — loaded %d songs from layer4, %d from layer5.",
        len(layer4),
        len(layer5),
    )

    merged = layer4.merge(
        layer5[["song_id", "jung_stage", "development_score"]],
        on="song_id",
        how="left",
    )

    # ── Compute Cultural Resonance Score ─────────────────────────────────────
    weights = _load_weights(config)
    merged = _impute_components(merged)
    merged["cultural_resonance_score"] = _compute_resonance(merged, weights)

    # ── Coverage gate ────────────────────────────────────────────────────────
    coverage_pct = float(merged["cultural_resonance_score"].notna().mean())
    min_coverage = getattr(getattr(config, "cmi", None), "min_coverage_pct", 0.60)

    if coverage_pct < min_coverage:
        raise CoverageError(
            f"Stage 10 [CMI] coverage {coverage_pct:.1%} is below "
            f"minimum required {min_coverage:.1%}. "
            "Check upstream stage outputs for missing data."
        )

    logger.info(
        "Stage 10 [CMI] — coverage %.1f%% (threshold %.1f%%).",
        coverage_pct * 100,
        min_coverage * 100,
    )

    # ── Build layer6 (per-song) ───────────────────────────────────────────────
    layer6_df = merged[["song_id", "cultural_resonance_score"]].copy()
    layer6_df = validate(layer6_df, layer6_schema, stage_name="CMI_LAYER6")

    # ── Build decade CMI aggregates ───────────────────────────────────────────
    decade_df = _build_decade_cmi(merged)
    decade_df = validate(decade_df, decade_cmi_schema, stage_name="CMI_DECADE")

    # ── Write outputs ─────────────────────────────────────────────────────────
    layer6_df.to_csv(_OUTPUT_LAYER6, index=False)
    decade_df.to_csv(_OUTPUT_DECADE, index=False)
    write_sentinel(_SENTINEL, stage="CMI", config=config)

    songs_scored = int(layer6_df["cultural_resonance_score"].notna().sum())
    logger.info(
        "Stage 10 [CMI] — complete. scored=%d/%d coverage=%.1f%% decades=%d → %s, %s",
        songs_scored,
        len(layer6_df),
        coverage_pct * 100,
        len(decade_df),
        _OUTPUT_LAYER6,
        _OUTPUT_DECADE,
    )

    return {
        "songs_total": len(layer6_df),
        "songs_scored": songs_scored,
        "coverage_pct": coverage_pct,
        "decades_computed": len(decade_df),
        "output_layer6": _OUTPUT_LAYER6,
        "output_decade": _OUTPUT_DECADE,
        "skipped": False,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Weight loading
# ═════════════════════════════════════════════════════════════════════════════


def _load_weights(config: ProjectConfig) -> dict[str, float]:
    """
    Load CMI component weights from config, falling back to defaults.

    Weights are normalised to sum to 1.0 so mis-configured values
    never silently produce scores outside [0.0, 1.0].
    """
    cmi_cfg = getattr(config, "cmi", None)
    weights = {
        "sentiment": getattr(
            cmi_cfg, "weight_sentiment", _DEFAULT_WEIGHTS["sentiment"]
        ),
        "energy": getattr(cmi_cfg, "weight_energy", _DEFAULT_WEIGHTS["energy"]),
        "chorus": getattr(cmi_cfg, "weight_chorus", _DEFAULT_WEIGHTS["chorus"]),
        "quality": getattr(cmi_cfg, "weight_quality", _DEFAULT_WEIGHTS["quality"]),
        "jungian": getattr(cmi_cfg, "weight_jungian", _DEFAULT_WEIGHTS["jungian"]),
    }
    total = sum(weights.values())
    if total <= 0:
        return _DEFAULT_WEIGHTS.copy()
    return {k: v / total for k, v in weights.items()}


# ═════════════════════════════════════════════════════════════════════════════
# Component imputation
# ═════════════════════════════════════════════════════════════════════════════


def _impute_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing component values with within-decade cohort means.

    Operates on a copy — does not mutate the input DataFrame.
    Components imputed: sentiment_score, energy_level, development_score.
    """
    df = df.copy()

    for col in ("sentiment_score", "energy_level", "development_score"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df.groupby("decade")[col].transform(lambda s: s.fillna(s.mean()))

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Resonance score computation
# ═════════════════════════════════════════════════════════════════════════════


def _compute_resonance(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Compute the Cultural Resonance Score for each song.

    Each component is normalised to [0.0, 1.0]:
      sentiment_score  : (-1, 1) → (0, 1) via (x + 1) / 2
      energy_level     : (1, 5)  → (0, 1) via (x - 1) / 4
      chorus_detected  : bool    → 0.0 or 1.0
      lyrics_quality   : str     → mapped via _QUALITY_SCORE
      development_score: (1, 7)  → (0, 1) via (x - 1) / 6

    Returns a Series of floats clipped to [0.0, 1.0].
    Songs where all components are null receive NaN.
    """
    sentiment_norm = (
        pd.to_numeric(df["sentiment_score"], errors="coerce")
        .clip(-1.0, 1.0)
        .add(1.0)
        .div(2.0)
    )

    energy_raw = pd.to_numeric(
        df.get("energy_level", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    energy_norm = (energy_raw.clip(1.0, 5.0) - 1.0) / 4.0

    chorus_norm = (
        df.get("chorus_detected", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(float)
    )

    quality_norm = (
        df.get("lyrics_quality", pd.Series("missing", index=df.index))
        .map(_QUALITY_SCORE)
        .fillna(0.0)
    )

    jungian_raw = pd.to_numeric(
        df.get("development_score", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    jungian_norm = (jungian_raw.clip(1.0, 7.0) - 1.0) / 6.0

    score = (
        (
            sentiment_norm * weights["sentiment"]
            + energy_norm * weights["energy"]
            + chorus_norm * weights["chorus"]
            + quality_norm * weights["quality"]
            + jungian_norm * weights["jungian"]
        )
        .clip(0.0, 1.0)
        .round(6)
    )

    return score


# ═════════════════════════════════════════════════════════════════════════════
# Decade CMI aggregation
# ═════════════════════════════════════════════════════════════════════════════


def _build_decade_cmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-song data into per-decade CMI metrics.

    Returns a DataFrame matching decade_cmi_schema.
    """
    rows: list[dict] = []

    for decade_label, group in df.groupby("decade", sort=True):
        scored = group.dropna(subset=["cultural_resonance_score"])

        song_count = len(group)
        scored_count = len(scored)

        cmi_sentiment = (
            float(group["sentiment_score"].mean())
            if group["sentiment_score"].notna().any()
            else None
        )
        cmi_energy = (
            float(group["energy_level"].mean())
            if group["energy_level"].notna().any()
            else None
        )

        emotional_tone = _mode_value(group, "emotional_tone")
        dominant_jung_stage = _mode_value(
            group[
                group.get("jung_stage", pd.Series("unclassified", index=group.index))
                != "unclassified"
            ],
            "jung_stage",
        )

        top_themes = _top_pipe_themes(group, "themes", n=3)
        top_resonance_songs = _top_resonance(scored, n=3)

        rows.append(
            {
                "decade_label": decade_label,
                "song_count": song_count,
                "scored_count": scored_count,
                "CMI_sentiment": cmi_sentiment,
                "CMI_energy": cmi_energy,
                "emotional_tone": emotional_tone,
                "dominant_jung_stage": dominant_jung_stage,
                "top_themes": top_themes,
                "top_resonance_songs": top_resonance_songs,
            }
        )

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ═════════════════════════════════════════════════════════════════════════════


def _mode_value(df: pd.DataFrame, col: str) -> str | None:
    """Return the most frequent non-null value in a column, or None."""
    if col not in df.columns:
        return None
    series = df[col].dropna()
    if series.empty:
        return None
    return str(series.mode().iloc[0])


def _top_pipe_themes(df: pd.DataFrame, col: str, n: int) -> str | None:
    """
    Flatten all pipe-delimited theme strings in col, count frequencies,
    and return the top-n themes as a pipe-delimited string.
    Returns None if no themes are present.
    """
    if col not in df.columns:
        return None

    all_themes: list[str] = []
    for cell in df[col].dropna():
        all_themes.extend(t.strip() for t in str(cell).split("|") if t.strip())

    if not all_themes:
        return None

    from collections import Counter

    top = [theme for theme, _ in Counter(all_themes).most_common(n)]
    return "|".join(top)


def _top_resonance(scored: pd.DataFrame, n: int) -> str | None:
    """
    Return the top-n song titles by cultural_resonance_score as a
    pipe-delimited string. Returns None if scored is empty.
    """
    if scored.empty or "song_title" not in scored.columns:
        return None

    top = scored.nlargest(n, "cultural_resonance_score")["song_title"].tolist()
    return "|".join(str(t) for t in top) if top else None


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Stage 10 — Cultural Mood Index calculation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "config" / "project_config.yaml",
        help="Path to project_config.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete sentinel and recalculate even if cache is valid",
    )
    args = parser.parse_args()

    if args.force and _SENTINEL.exists():
        _SENTINEL.unlink()
        logger.info("Sentinel deleted — forcing recalculation.")

    cfg = load_config(args.config)
    result = run(cfg)

    print(f"\nCMI calculation complete.")
    print(f"  Total songs     : {result['songs_total']:,}")
    print(f"  Scored songs    : {result['songs_scored']:,}")
    print(f"  Coverage        : {result['coverage_pct']:.1%}")
    print(f"  Decades computed: {result['decades_computed']}")
    print(f"  Output layer6   : {result['output_layer6']}")
    print(f"  Output decade   : {result['output_decade']}")
    sys.exit(0)
