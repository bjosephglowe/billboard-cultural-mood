"""
src/pipeline/schemas.py

Pandera DataFrameSchema definitions for every CSV boundary in the
Billboard Cultural Mood Analysis pipeline.

Schema inventory:
    metadata_schema          — song_metadata.csv (Stage 1 output)
    cleaned_schema           — lyrics_cleaned.csv (Stage 3 output)
    chorus_schema            — chorus_extracted.csv (Stage 4 output)
    layer2_sentiment_schema  — layer2_sentiment.csv (Stage 5 output)
    layer2_emotion_schema    — layer2_emotion.csv (Stage 6 output)
    layer2_themes_schema     — layer2_themes.csv (Stage 7 output)
    layer4_schema            — layer2_full_analysis.csv (Stage 8 master output)
    layer5_schema            — layer5_jungian.csv (Stage 9 output)
    layer6_schema            — layer6_cultural_metrics.csv (Stage 10 output)
    decade_cmi_schema        — decade_cmi.csv (Stage 10 aggregate output)

Shared utility:
    validate()  — wraps pandera validation with loguru logging

Usage:
    from src.pipeline.schemas import layer4_schema, validate
    validate(df, layer4_schema, "contrast_metrics")
"""

from __future__ import annotations

import pandera as pa
import pandera as pd
import pandera.errors  # force submodule registration
from loguru import logger
from pandera import Check, Column, DataFrameSchema

# ── Valid Value Sets ──────────────────────────────────────────────────────────

VALID_LYRICS_QUALITY = ["full", "partial", "missing"]

VALID_NARRATIVE_PERSPECTIVES = [
    "first_person",
    "second_person",
    "third_person",
    "abstract",
]

VALID_CHORUS_METHODS = ["tag", "repetition", "llm", "none"]

VALID_EMOTIONAL_TONES = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "neutral",
]

VALID_ENERGY_LEVELS = [1, 2, 3, 4, 5]

VALID_THEME_TAXONOMY = [
    "love",
    "heartbreak",
    "party_celebration",
    "identity",
    "struggle",
    "rebellion",
    "wealth_success",
    "friendship",
    "nostalgia",
    "self_reflection",
    "empowerment",
    "unity",
]

VALID_JUNG_STAGES = [
    "shadow",
    "persona",
    "anima_animus",
    "integration",
    "transcendence",
    "unclassified",
]

VALID_JUNGIAN_QUALITY_FLAGS = ["high", "low"]

VALID_ENERGY_SHIFTS = ["increase", "decrease", "stable"]

VALID_THEME_SHIFTS = ["none", "minor", "major"]

VALID_DECADE_LABELS = [
    "1960s*",
    "1970s",
    "1980s",
    "1990s",
    "2000s",
    "2010s",
    "2020s",
]


# ── Stage 1 — Song Metadata Schema ───────────────────────────────────────────

metadata_schema = DataFrameSchema(
    columns={
        "song_id": Column(
            str,
            Check(lambda s: s.str.match(r"^[0-9a-f]{16}$"), element_wise=False),
            nullable=False,
        ),
        "song_title": Column(str, nullable=False),
        "artist": Column(str, nullable=False),
        "year": Column(
            int,
            Check.in_range(1958, 2030),
            nullable=False,
        ),
        "decade": Column(
            str,
            Check.isin(VALID_DECADE_LABELS),
            nullable=False,
        ),
        "chart_rank": Column(
            int,
            Check.in_range(1, 100),
            nullable=True,
        ),
        "chart_weeks_on": Column(
            int,
            Check.ge(0),
            nullable=True,
        ),
    },
    strict=False,
    coerce=True,
    name="metadata_schema",
)


# ── Stage 3 — Cleaned Lyrics Schema ──────────────────────────────────────────

cleaned_schema = DataFrameSchema(
    columns={
        "song_id": Column(
            str,
            Check(lambda s: s.str.match(r"^[0-9a-f]{16}$"), element_wise=False),
            nullable=False,
        ),
        "song_title": Column(str, nullable=False),
        "artist": Column(str, nullable=False),
        "year": Column(int, Check.in_range(1958, 2030), nullable=False),
        "decade": Column(str, Check.isin(VALID_DECADE_LABELS), nullable=False),
        "lyrics_clean": Column(str, nullable=True),
        "lyrics_verse_only": Column(str, nullable=True),
        "chorus_token_count": Column(int, Check.ge(0), nullable=False),
        "lyrics_quality": Column(
            str,
            Check.isin(VALID_LYRICS_QUALITY),
            nullable=False,
        ),
        "narrative_perspective": Column(
            str,
            Check.isin(VALID_NARRATIVE_PERSPECTIVES),
            nullable=True,
        ),
        "has_section_tags": Column(bool, nullable=False),
        "section_count": Column(int, Check.ge(0), nullable=False),
    },
    strict=False,
    coerce=True,
    name="cleaned_schema",
)


# ── Stage 4 — Chorus Schema ───────────────────────────────────────────────────

chorus_schema = DataFrameSchema(
    columns={
        "song_id": Column(
            str,
            Check(lambda s: s.str.match(r"^[0-9a-f]{16}$"), element_wise=False),
            nullable=False,
        ),
        "chorus_detected": Column(bool, nullable=False),
        "chorus_method": Column(
            str,
            Check.isin(VALID_CHORUS_METHODS),
            nullable=False,
        ),
        "chorus_text": Column(str, nullable=True),
        "chorus_token_count": Column(int, Check.ge(0), nullable=False),
    },
    strict=False,
    coerce=True,
    name="chorus_schema",
)


# ── Stage 5 — Sentiment Schema (Layer 2a) ────────────────────────────────────

layer2_sentiment_schema = DataFrameSchema(
    columns={
        "song_id": Column(
            str,
            Check(lambda s: s.str.match(r"^[0-9a-f]{16}$"), element_wise=False),
            nullable=False,
        ),
        "sentiment_score": Column(
            float,
            Check.in_range(-1.0, 1.0),
            nullable=True,
        ),
        "energy_level": Column(
            "Int64",
            Check.isin(VALID_ENERGY_LEVELS),
            nullable=True,
        ),
    },
    strict=False,
    coerce=True,
    name="layer2_sentiment_schema",
)


# ── Stage 6 — Emotion Schema (Layer 2b) ──────────────────────────────────────

layer2_emotion_schema = DataFrameSchema(
    columns={
        "song_id": Column(
            str,
            Check(lambda s: s.str.match(r"^[0-9a-f]{16}$"), element_wise=False),
            nullable=False,
        ),
        "emotional_tone": Column(
            str,
            Check.isin(VALID_EMOTIONAL_TONES),
            nullable=True,
        ),
        "chorus_sentiment_score": Column(
            float,
            Check.in_range(-1.0, 1.0),
            nullable=True,
        ),
        "chorus_emotional_tone": Column(
            str,
            Check.isin(VALID_EMOTIONAL_TONES),
            nullable=True,
        ),
    },
    strict=False,
    coerce=True,
    name="layer2_emotion_schema",
)


# ── Stage 7 — Themes Schema (Layer 2c) ───────────────────────────────────────

layer2_themes_schema = DataFrameSchema(
    columns={
        "song_id": Column(
            str,
            Check(lambda s: s.str.match(r"^[0-9a-f]{16}$"), element_wise=False),
            nullable=False,
        ),
        "themes": Column(
            str,
            Check(
                lambda s: (
                    s.dropna()
                    .str.split("|")
                    .apply(lambda tags: all(t in VALID_THEME_TAXONOMY for t in tags))
                    .all()
                ),
                element_wise=False,
                error="themes column contains labels not in VALID_THEME_TAXONOMY",
            ),
            nullable=True,
        ),
    },
    strict=False,
    coerce=True,
    name="layer2_themes_schema",
)


# ── Stage 8 — Master Full Analysis Schema (Layer 4) ──────────────────────────

layer4_schema = DataFrameSchema(
    columns={
        # Identity
        "song_id": Column(
            str,
            Check(lambda s: s.str.match(r"^[0-9a-f]{16}$"), element_wise=False),
            nullable=False,
        ),
        "song_title": Column(str, nullable=False),
        "artist": Column(str, nullable=False),
        "year": Column(int, Check.in_range(1958, 2030), nullable=False),
        "decade": Column(str, Check.isin(VALID_DECADE_LABELS), nullable=False),
        # Lyrics metadata
        "lyrics_quality": Column(
            str,
            Check.isin(VALID_LYRICS_QUALITY),
            nullable=False,
        ),
        "chorus_token_count": Column(int, Check.ge(0), nullable=False),
        "narrative_perspective": Column(
            str,
            Check.isin(VALID_NARRATIVE_PERSPECTIVES),
            nullable=True,
        ),
        # Layer 2 — Sentiment
        "sentiment_score": Column(
            float,
            Check.in_range(-1.0, 1.0),
            nullable=True,
        ),
        "energy_level": Column(
            "Int64",
            Check.isin(VALID_ENERGY_LEVELS),
            nullable=True,
        ),
        # Layer 2 — Emotion
        "emotional_tone": Column(
            str,
            Check.isin(VALID_EMOTIONAL_TONES),
            nullable=True,
        ),
        # Layer 2 — Themes
        "themes": Column(str, nullable=True),
        # Layer 3 — Chorus
        "chorus_detected": Column(bool, nullable=False),
        "chorus_method": Column(
            str,
            Check.isin(VALID_CHORUS_METHODS),
            nullable=False,
        ),
        "chorus_sentiment_score": Column(
            float,
            Check.in_range(-1.0, 1.0),
            nullable=True,
        ),
        "chorus_emotional_tone": Column(
            str,
            Check.isin(VALID_EMOTIONAL_TONES),
            nullable=True,
        ),
        # Layer 4 — Contrast metrics
        "contrast_sentiment_index": Column(float, nullable=True),
        "energy_shift": Column(
            str,
            Check.isin(VALID_ENERGY_SHIFTS),
            nullable=True,
        ),
        "theme_shift": Column(
            str,
            Check.isin(VALID_THEME_SHIFTS),
            nullable=True,
        ),
    },
    strict=False,
    coerce=True,
    name="layer4_schema",
)


# ── Stage 9 — Jungian Schema (Layer 5) ───────────────────────────────────────

layer5_schema = DataFrameSchema(
    columns={
        "song_id": Column(
            str,
            Check(lambda s: s.str.match(r"^[0-9a-f]{16}$"), element_wise=False),
            nullable=False,
        ),
        "jung_stage": Column(
            str,
            Check.isin(VALID_JUNG_STAGES),
            nullable=False,
        ),
        "psychological_theme": Column(str, nullable=True),
        "development_score": Column(
            int,
            Check.in_range(1, 7),
            nullable=True,
        ),
        "jungian_quality_flag": Column(
            str,
            Check.isin(VALID_JUNGIAN_QUALITY_FLAGS),
            nullable=True,
        ),
    },
    strict=False,
    coerce=True,
    name="layer5_schema",
)


# ── Stage 10 — Cultural Resonance Schema (Layer 6) ───────────────────────────

layer6_schema = DataFrameSchema(
    columns={
        "song_id": Column(
            str,
            Check(lambda s: s.str.match(r"^[0-9a-f]{16}$"), element_wise=False),
            nullable=False,
        ),
        "cultural_resonance_score": Column(
            float,
            Check.in_range(0.0, 1.0),
            nullable=True,
        ),
    },
    strict=False,
    coerce=True,
    name="layer6_schema",
)


# ── Stage 10 — Decade CMI Aggregate Schema ───────────────────────────────────

decade_cmi_schema = DataFrameSchema(
    columns={
        "decade_label": Column(
            str,
            Check.isin(VALID_DECADE_LABELS),
            nullable=False,
        ),
        "song_count": Column(int, Check.ge(0), nullable=False),
        "scored_count": Column(int, Check.ge(0), nullable=False),
        "CMI_sentiment": Column(
            float,
            Check.in_range(-1.0, 1.0),
            nullable=True,
        ),
        "CMI_energy": Column(
            float,
            Check.in_range(1.0, 5.0),
            nullable=True,
        ),
        "emotional_tone": Column(
            str,
            Check.isin(VALID_EMOTIONAL_TONES),
            nullable=True,
        ),
        "dominant_jung_stage": Column(
            str,
            Check.isin(VALID_JUNG_STAGES),
            nullable=True,
        ),
        "top_themes": Column(str, nullable=True),
        "top_resonance_songs": Column(str, nullable=True),
    },
    strict=False,
    coerce=True,
    name="decade_cmi_schema",
)


# ── Shared Validation Utility ─────────────────────────────────────────────────


def validate(
    df: "pd.DataFrame",
    schema: DataFrameSchema,
    stage_name: str = "unknown",
) -> "pd.DataFrame":
    """
    Validate a DataFrame against a Pandera schema with loguru error logging.

    Args:
        df:         DataFrame to validate.
        schema:     Pandera DataFrameSchema to validate against.
        stage_name: Human-readable stage name for log messages.

    Returns:
        The validated DataFrame (Pandera may coerce types).

    Raises:
        pa.errors.SchemaErrors: if lazy validation collects multiple failures.
        pa.errors.SchemaError:  if a single validation rule fails.
    """
    logger.debug(
        "Validating {} rows × {} cols against {} [stage: {}]",
        len(df),
        len(df.columns),
        schema.name or "unnamed_schema",
        stage_name,
    )
    try:
        validated = schema.validate(df, lazy=True)
        logger.debug("Schema validation passed [stage: {}]", stage_name)
        return validated

    except pa.errors.SchemaErrors as exc:
        logger.error(
            "Schema validation FAILED [stage: {}] — {} error(s):\n{}",
            stage_name,
            len(exc.failure_cases),
            exc.failure_cases.to_string(),
        )
        raise

    except pa.errors.SchemaError as exc:
        logger.error(
            "Schema validation FAILED [stage: {}]: {}",
            stage_name,
            exc,
        )
        raise
