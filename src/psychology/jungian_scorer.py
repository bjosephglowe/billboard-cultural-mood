"""
src/psychology/jungian_scorer.py

Stage 9 — Jungian Psychological Stage Classification
======================================================
Classifies each song in data/analysis/layer2_full_analysis.csv into one
of six Jungian psychological development stages using GPT-4o via the
OpenAI API, then writes the result to data/analysis/layer5_jungian.csv.

Jungian Stage Taxonomy
-----------------------
shadow        — unconscious negative traits; darkness, repression, conflict
persona       — social mask; conformity, performance, image management
anima_animus  — contrasexual inner figure; idealised love, yearning, duality
integration   — reconciliation of opposites; self-acceptance, healing
transcendence — movement beyond ego; unity, spiritual themes, collective
unclassified  — insufficient signal or ambiguous content

Classification approach
------------------------
Songs are batched (config.analysis.jungian_batch_size, default 10) and
sent to GPT-4o with a structured prompt. The model returns a JSON array
with one record per song containing:
  song_id             : str  — echoed back for join safety
  jung_stage          : str  — one of VALID_JUNG_STAGES
  psychological_theme : str  — 3–8 word descriptive phrase
  development_score   : int  — 1 (primitive) to 7 (highly integrated)

Quality flagging
-----------------
jungian_quality_flag is assigned post-classification:
  "high" — jung_stage != "unclassified" AND development_score is not null
  "low"  — jung_stage == "unclassified" OR development_score is null

API key gating
--------------
If OPENAI_API_KEY is absent or is a placeholder, all songs receive
jung_stage="unclassified", development_score=None,
psychological_theme=None, jungian_quality_flag="low".
No exception is raised — the pipeline continues gracefully.

Idempotency
-----------
Sentinel .jungian_complete records config_hash. If matched, the fetch
is skipped and the cached CSV is returned. Delete the sentinel (or
change config) to force re-classification.

Output schema
-------------
See src/pipeline/schemas.py → layer5_schema.

song_id              : str   — 16-char hex
jung_stage           : str   — one of VALID_JUNG_STAGES
psychological_theme  : str   — short descriptive phrase (nullable)
development_score    : int   — 1–7 Jungian development scale (nullable)
jungian_quality_flag : str   — "high" | "low" (nullable)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipeline.config_loader import (
    ProjectConfig,
    load_config,
    sentinel_config_matches,
    write_sentinel,
)
from src.pipeline.schemas import (
    VALID_JUNG_STAGES,
    layer5_schema,
    validate,
)

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INPUT_PATH = _PROJECT_ROOT / "data" / "analysis" / "layer2_full_analysis.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "analysis" / "layer5_jungian.csv"
_SENTINEL = _PROJECT_ROOT / "data" / "analysis" / ".jungian_complete"

# ── Model config ──────────────────────────────────────────────────────────────
_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_BATCH_SIZE = 10
_RETRY_SLEEP_BASE = 2.0  # seconds — multiplied by attempt number on retry

# ── Classification prompt ─────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a Jungian psychoanalyst and musicologist specialising in
the psychological symbolism of popular song lyrics.

Your task: classify each song into exactly one Jungian psychological development stage
based on its lyrical content, themes, and emotional tone.

Stage definitions:
- shadow        : unconscious negative traits; darkness, repression, rage, fear, addiction
- persona       : social mask; performance, image, conformity, status, approval-seeking
- anima_animus  : contrasexual inner figure; idealised romance, yearning, projection, duality
- integration   : reconciliation of shadow and persona; healing, self-acceptance, growth
- transcendence : beyond ego; unity, spiritual themes, collective consciousness, surrender
- unclassified  : insufficient lyrical content or genuinely ambiguous signal

Return ONLY a valid JSON array. No prose, no markdown, no code fences.
Each element must have exactly these keys:
  "song_id"             : string  — the song_id provided, echoed back unchanged
  "jung_stage"          : string  — one of the six stages above
  "psychological_theme" : string  — 3 to 8 words describing the core psychological theme
  "development_score"   : integer — 1 (primitive/reactive) to 7 (highly integrated/conscious)

If you cannot classify a song, use "unclassified" for jung_stage and null for
psychological_theme and development_score."""

_USER_PROMPT_TEMPLATE = """Classify the following {n} songs.

{songs_block}

Return a JSON array with {n} elements, one per song, in the same order."""


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def run(config: ProjectConfig) -> dict:
    """
    Execute Stage 9 — Jungian classification.

    Parameters
    ----------
    config : ProjectConfig

    Returns
    -------
    dict with keys:
        songs_total      : int
        songs_classified : int  — jung_stage != "unclassified"
        songs_skipped    : int  — jung_stage == "unclassified"
        quality_high     : int  — jungian_quality_flag == "high"
        quality_low      : int  — jungian_quality_flag == "low"
        stage_dist       : dict — {stage_label: count}
        api_available    : bool
        output_path      : Path
        skipped          : bool — True if sentinel matched
    """
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(
            f"layer2_full_analysis.csv not found at {_INPUT_PATH}. "
            "Run Stage 8 [CONTRAST_METRICS] first."
        )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check ────────────────────────────────────────────────────
    if sentinel_config_matches(_SENTINEL, config):
        logger.info("Stage 9 [JUNGIAN] — sentinel matched, skipping.")
        df = pd.read_csv(_OUTPUT_PATH)
        return _build_summary(df, skipped=True, api_available=None)

    layer4 = pd.read_csv(_INPUT_PATH)
    logger.info("Stage 9 [JUNGIAN] — classifying %d songs …", len(layer4))

    # ── API key gate ─────────────────────────────────────────────────────────
    client, api_available = _init_client()

    if not api_available:
        logger.warning(
            "OPENAI_API_KEY not set — Stage 9 [JUNGIAN] running in fallback mode. "
            "All songs will receive jung_stage='unclassified'."
        )
        df = _build_fallback_dataframe(layer4["song_id"].tolist())
    else:
        df = _classify_all(layer4, client, config)

    # ── Assign quality flags ─────────────────────────────────────────────────
    df["jungian_quality_flag"] = df.apply(_assign_quality_flag, axis=1)

    # ── Cast development_score to nullable Int64 ─────────────────────────────
    df["development_score"] = pd.array(
        df["development_score"].tolist(), dtype=pd.Int64Dtype()
    )

    # ── Validate ─────────────────────────────────────────────────────────────
    df = validate(df, layer5_schema, stage_name="JUNGIAN")

    # ── Write output ─────────────────────────────────────────────────────────
    df.to_csv(_OUTPUT_PATH, index=False)
    write_sentinel(_SENTINEL, stage="JUNGIAN", config=config)

    summary = _build_summary(df, skipped=False, api_available=api_available)
    logger.info(
        "Stage 9 [JUNGIAN] — complete. classified=%d unclassified=%d "
        "quality_high=%d → %s",
        summary["songs_classified"],
        summary["songs_skipped"],
        summary["quality_high"],
        _OUTPUT_PATH,
    )
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Client initialisation
# ═════════════════════════════════════════════════════════════════════════════


def _init_client() -> tuple[Any, bool]:
    """
    Initialise the OpenAI client.

    Returns (client, True) if API key is valid.
    Returns (None, False) if key is absent or placeholder — no exception raised.
    """
    try:
        import openai  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for Stage 9. "
            "Install with: pip install openai"
        ) from exc

    token = os.getenv("OPENAI_API_KEY", "").strip()
    if not token or token.startswith("your_"):
        return None, False

    return openai.OpenAI(api_key=token), True


# ═════════════════════════════════════════════════════════════════════════════
# Classification
# ═════════════════════════════════════════════════════════════════════════════


def _classify_all(
    layer4: pd.DataFrame,
    client: Any,
    config: ProjectConfig,
) -> pd.DataFrame:
    """
    Classify all songs in layer4 in batches. Returns a DataFrame of
    raw classification records (before quality flagging).
    """
    batch_size = getattr(config.analysis, "jungian_batch_size", _DEFAULT_BATCH_SIZE)
    model = getattr(config.analysis, "jungian_model", _DEFAULT_MODEL)
    max_retries = getattr(config.analysis, "jungian_max_retries", 3)

    song_ids = layer4["song_id"].tolist()
    all_records: list[dict] = []

    for batch_start in range(0, len(song_ids), batch_size):
        batch_ids = song_ids[batch_start : batch_start + batch_size]
        batch_rows = layer4.iloc[batch_start : batch_start + batch_size]

        batch_records = _classify_batch(
            batch_ids=batch_ids,
            batch_rows=batch_rows,
            client=client,
            model=model,
            max_retries=max_retries,
        )
        all_records.extend(batch_records)

        logger.debug(
            "Jungian batch %d–%d complete.",
            batch_start,
            min(batch_start + batch_size, len(song_ids)) - 1,
        )

    return pd.DataFrame(all_records)


def _classify_batch(
    batch_ids: list[str],
    batch_rows: pd.DataFrame,
    client: Any,
    model: str,
    max_retries: int,
) -> list[dict]:
    """
    Classify one batch of songs via GPT-4o.

    On all failure paths (API error, malformed JSON, missing song_ids),
    returns fallback records for the entire batch so the pipeline never
    stalls on a single bad batch.
    """
    songs_block = _build_songs_block(batch_ids, batch_rows)
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        n=len(batch_ids),
        songs_block=songs_block,
    )

    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # low temp for consistent staging
                max_tokens=512,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            records = _parse_response(content, batch_ids)

            if records is not None:
                return records

            logger.warning(
                "Jungian batch parse failed on attempt %d/%d — retrying.",
                attempt,
                max_retries,
            )

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Jungian API attempt %d/%d failed: %s",
                attempt,
                max_retries,
                exc,
            )

        if attempt < max_retries:
            time.sleep(_RETRY_SLEEP_BASE * attempt)

    logger.error(
        "All %d Jungian attempts failed for batch starting at song %s. "
        "Last error: %s — using fallback records.",
        max_retries,
        batch_ids[0],
        last_exc,
    )
    return _fallback_records(batch_ids)


def _build_songs_block(
    batch_ids: list[str],
    batch_rows: pd.DataFrame,
) -> str:
    """
    Build the numbered song block for the GPT-4o prompt.

    Each entry includes song_id, song_title, artist, decade, themes,
    emotional_tone, and dominant_theme (if present) to give the model
    enough signal for accurate Jungian staging.
    """
    lines: list[str] = []
    for i, (song_id, row) in enumerate(
        zip(batch_ids, batch_rows.itertuples(index=False)), start=1
    ):
        title = getattr(row, "song_title", "Unknown")
        artist = getattr(row, "artist", "Unknown")
        decade = getattr(row, "decade", "Unknown")
        themes = getattr(row, "themes", "") or ""
        emotion = getattr(row, "emotional_tone", "") or ""

        lines.append(
            f'{i}. song_id={song_id} | "{title}" by {artist} '
            f"({decade}) | themes: {themes} | emotion: {emotion}"
        )

    return "\n".join(lines)


def _parse_response(content: str, batch_ids: list[str]) -> list[dict] | None:
    """
    Parse GPT-4o JSON response into a list of validated record dicts.

    Accepts both a bare JSON array and a JSON object with a single
    array-valued key (GPT-4o sometimes wraps arrays in an object when
    response_format=json_object is set).

    Returns None if parsing fails or if any song_id is missing from
    the response — caller will retry.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.debug("JSON decode error: %s", exc)
        return None

    # Unwrap single-key object if needed
    if isinstance(parsed, dict):
        values = list(parsed.values())
        if len(values) == 1 and isinstance(values[0], list):
            parsed = values[0]

    if not isinstance(parsed, list):
        logger.debug("Parsed JSON is not a list: %s", type(parsed))
        return None

    # Build a lookup keyed by song_id — model may reorder
    response_map: dict[str, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("song_id", "")).strip()
        if sid:
            response_map[sid] = item

    # Ensure every batch song_id is accounted for
    records: list[dict] = []
    for song_id in batch_ids:
        if song_id not in response_map:
            logger.debug("song_id %s missing from GPT response.", song_id)
            return None  # trigger retry

        item = response_map[song_id]
        stage = str(item.get("jung_stage", "unclassified")).strip().lower()

        if stage not in VALID_JUNG_STAGES:
            logger.debug(
                "Invalid jung_stage '%s' for song %s — coercing to unclassified.",
                stage,
                song_id,
            )
            stage = "unclassified"

        raw_score = item.get("development_score")
        dev_score: int | None = None
        if raw_score is not None:
            try:
                dev_score = int(raw_score)
                if not (1 <= dev_score <= 7):
                    dev_score = None
            except (ValueError, TypeError):
                dev_score = None

        psych_theme = item.get("psychological_theme")
        if psych_theme is not None:
            psych_theme = str(psych_theme).strip() or None

        records.append(
            {
                "song_id": song_id,
                "jung_stage": stage,
                "psychological_theme": psych_theme,
                "development_score": dev_score,
            }
        )

    return records


# ═════════════════════════════════════════════════════════════════════════════
# Fallback helpers
# ═════════════════════════════════════════════════════════════════════════════


def _fallback_records(song_ids: list[str]) -> list[dict]:
    """Return unclassified records for a list of song_ids."""
    return [
        {
            "song_id": sid,
            "jung_stage": "unclassified",
            "psychological_theme": None,
            "development_score": None,
        }
        for sid in song_ids
    ]


def _build_fallback_dataframe(song_ids: list[str]) -> pd.DataFrame:
    """Build a complete fallback DataFrame when API is unavailable."""
    return pd.DataFrame(_fallback_records(song_ids))


# ═════════════════════════════════════════════════════════════════════════════
# Post-classification helpers
# ═════════════════════════════════════════════════════════════════════════════


def _assign_quality_flag(row: pd.Series) -> str:
    """
    Assign jungian_quality_flag based on classification completeness.

    "high" — jung_stage is not "unclassified" AND development_score is not null
    "low"  — otherwise
    """
    if row["jung_stage"] == "unclassified":
        return "low"
    if pd.isna(row.get("development_score")):
        return "low"
    return "high"


def _build_summary(
    df: pd.DataFrame,
    skipped: bool,
    api_available: bool | None,
) -> dict:
    """Build the run() return dict from the output DataFrame."""
    stage_dist = df["jung_stage"].value_counts().to_dict()
    songs_total = len(df)
    songs_classified = int(songs_total - stage_dist.get("unclassified", 0))
    songs_skipped = int(stage_dist.get("unclassified", 0))
    quality_counts = df["jungian_quality_flag"].value_counts()

    return {
        "songs_total": songs_total,
        "songs_classified": songs_classified,
        "songs_skipped": songs_skipped,
        "quality_high": int(quality_counts.get("high", 0)),
        "quality_low": int(quality_counts.get("low", 0)),
        "stage_dist": stage_dist,
        "api_available": api_available,
        "output_path": _OUTPUT_PATH,
        "skipped": skipped,
    }


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
        description="Stage 9 — Jungian psychological stage classification (GPT-4o)"
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
        help="Delete sentinel and re-classify even if cache is valid",
    )
    args = parser.parse_args()

    if args.force and _SENTINEL.exists():
        _SENTINEL.unlink()
        logger.info("Sentinel deleted — forcing re-classification.")

    cfg = load_config(args.config)
    result = run(cfg)

    print(f"\nJungian classification complete.")
    print(f"  Total songs     : {result['songs_total']:,}")
    print(f"  Classified      : {result['songs_classified']:,}")
    print(f"  Unclassified    : {result['songs_skipped']:,}")
    print(f"  Quality high    : {result['quality_high']:,}")
    print(f"  Quality low     : {result['quality_low']:,}")
    print(f"  API available   : {result['api_available']}")
    print(f"  Stage distribution:")
    for stage, count in sorted(result["stage_dist"].items(), key=lambda x: -x[1]):
        print(f"    {stage:<16}: {count:,}")
    print(f"  Output          : {result['output_path']}")
    sys.exit(0)
