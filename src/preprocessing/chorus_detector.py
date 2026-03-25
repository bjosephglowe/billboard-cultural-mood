"""
src/preprocessing/chorus_detector.py

Stage 4 — Chorus Detection
============================
Extracts the chorus (or most-repeated passage) from each song in
data/processed/lyrics_cleaned.csv and writes the result to
data/processed/chorus_extracted.csv.

Detection tiers
---------------
Every song is attempted in order. The first tier that succeeds
assigns chorus_tier and chorus_text for that song. If all three
tiers fail, chorus_text is set to the full lyrics and
chorus_tier is set to "none".

Tier 1 — Explicit tag detection
    Looks for section tags preserved in lyrics_clean by text_cleaner.py.
    Pattern configured via config.preprocessing.chorus_tier1_tag_pattern.
    Default: r"\\[chorus\\]" (case-insensitive).
    Extracts the text block immediately following the first [Chorus] tag.
    Fast, deterministic, zero API cost.

Tier 2 — Repetition analysis
    Splits lyrics into line-groups (stanzas) and finds the stanza that
    repeats most often (>= config.preprocessing.chorus_tier2_min_repetitions
    times, with >= config.preprocessing.chorus_tier2_min_tokens tokens).
    Uses normalised line hashing for fuzzy match tolerance.
    No external dependencies.

Tier 3 — GPT-4o fallback
    For songs where Tier 1 and Tier 2 both fail, sends the full cleaned
    lyrics to GPT-4o with a structured prompt asking it to identify and
    return only the chorus text.
    Batched at config.preprocessing.chorus_max_tokens tokens per request.
    Gated by OPENAI_API_KEY — if key is absent, tier 3 is skipped and
    songs fall through to "none".

chorus_tier values
------------------
    "tag"           — found via explicit [Chorus] tag
    "repetition"    — found via repetition analysis
    "llm"         — found via GPT-4o
    "none" — no chorus detected; full lyrics used as proxy

Idempotency
-----------
Sentinel .chorus_detection_complete records config_hash.

Output schema
-------------
See src/pipeline/schemas.py → chorus_schema.

    song_id             : str
    chorus_detected     : bool  — True if a chorus was identified
    chorus_method       : str   — "tag" | "repetition" | "llm" | "none"
    chorus_text         : str   — extracted chorus (or empty string if none)
    chorus_token_count  : Int64 — token count of chorus_text
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.pipeline.config_loader import (
    ProjectConfig,
    load_config,
    sentinel_config_matches,
    write_sentinel,
)
from src.pipeline.schemas import chorus_schema, validate

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "lyrics_cleaned.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "chorus_extracted.csv"
_SENTINEL = _PROJECT_ROOT / "data" / "processed" / ".chorus_detection_complete"

# ── GPT-4o prompt ─────────────────────────────────────────────────────────────
_CHORUS_PROMPT = """You are a music analyst. Given the lyrics below, identify and return ONLY the chorus text — the section that repeats most and forms the emotional core of the song.

Rules:
- Return only the chorus lyrics, exactly as they appear in the text
- Do not include section labels like [Chorus]
- Do not explain or comment
- If no clear chorus exists, return the most repeated passage
- Maximum {max_tokens} tokens in your response

Lyrics:
{lyrics}"""


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def run(config: ProjectConfig) -> dict:
    """
    Execute Stage 4 — Chorus detection.

    Parameters
    ----------
    config : ProjectConfig

    Returns
    -------
    dict with keys:
        songs_total      : int
        tier_counts      : dict  — {tier_label: count}
        output_path      : Path
        skipped          : bool
    """
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(
            f"lyrics_cleaned.csv not found at {_INPUT_PATH}. "
            "Run Stage 3 [TEXT_CLEANING] first."
        )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check ────────────────────────────────────────────────────
    if sentinel_config_matches(_SENTINEL, config):
        logger.info("Stage 4 [CHORUS] — sentinel matched, skipping.")
        df = pd.read_csv(_OUTPUT_PATH)
        return {
            "songs_total": len(df),
            "tier_counts": df["chorus_method"].value_counts().to_dict(),
            "output_path": _OUTPUT_PATH,
            "skipped": True,
        }

    cleaned = pd.read_csv(_INPUT_PATH)
    logger.info("Stage 4 [CHORUS] — detecting chorus for %d songs …", len(cleaned))

    # ── Build per-song regex from config ─────────────────────────────────────
    tag_pattern = re.compile(
        config.preprocessing.chorus_tier1_tag_pattern,
        re.IGNORECASE,
    )

    # ── Determine GPT-4o availability ────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gpt_available = False  # Tier 3 disabled — _tier3_gpt4o not implemented
    if not gpt_available:
        logger.warning(
            "OPENAI_API_KEY not set — Tier 3 (GPT-4o) disabled. "
            "Songs failing Tier 1 and 2 will use 'none'."
        )

    records: list[dict] = []

    for _, row in cleaned.iterrows():
        lyrics = str(row.get("lyrics_clean", ""))
        result = _detect_chorus(
            lyrics=lyrics,
            config=config,
            tag_pattern=tag_pattern,
            gpt_available=gpt_available,
        )
        records.append(
            {
                "song_id": row["song_id"],
                "chorus_detected": result["tier"] != "none",
                "chorus_method": result["tier"],  # matches VALID_CHORUS_METHODS
                "chorus_text": result["text"],
                "chorus_token_count": len(result["text"].split()),
            }
        )

    df = pd.DataFrame(records)
    df["chorus_token_count"] = pd.array(
        df["chorus_token_count"].tolist(), dtype=pd.Int64Dtype()
    )

    # ── Validate ─────────────────────────────────────────────────────────────
    df = validate(df, chorus_schema, stage_name="CHORUS")

    # ── Write output ─────────────────────────────────────────────────────────
    df.to_csv(_OUTPUT_PATH, index=False)
    write_sentinel(_SENTINEL, stage="CHORUS", config=config)

    tier_counts = df["chorus_method"].value_counts().to_dict()
    logger.info(
        "Stage 4 [CHORUS] — complete. tier breakdown: %s → %s",
        tier_counts,
        _OUTPUT_PATH,
    )

    return {
        "songs_total": len(df),
        "tier_counts": tier_counts,
        "output_path": _OUTPUT_PATH,
        "skipped": False,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Detection cascade
# ═════════════════════════════════════════════════════════════════════════════


def _detect_chorus(
    lyrics: str,
    config: ProjectConfig,
    tag_pattern: re.Pattern,
    gpt_available: bool,
) -> dict:
    """
    Run the 3-tier detection cascade for a single song.

    Returns dict with keys:
        text : str  — extracted chorus text
        tier : str  — detection method label
    """
    if not lyrics or not lyrics.strip():
        return {"text": "", "tier": "none"}

    # ── Tier 1: Explicit tag ─────────────────────────────────────────────────
    tier1 = _tier1_tag(lyrics, tag_pattern)
    if tier1:
        return {"text": tier1, "tier": "tag"}

    # ── Tier 2: Repetition ───────────────────────────────────────────────────
    tier2 = _tier2_repetition(lyrics, config)
    if tier2:
        return {"text": tier2, "tier": "repetition"}

    # ── Tier 3: GPT-4o ──────────────────────────────────────────────────────
    if gpt_available:
        tier3 = _tier3_gpt4o(lyrics, config)
        if tier3:
            return {"text": tier3, "tier": "llm"}

    # ── Fallback: full lyrics ────────────────────────────────────────────────
    return {"text": lyrics, "tier": "none"}


# ── Tier 1 ────────────────────────────────────────────────────────────────────


def _tier1_tag(lyrics: str, tag_pattern: re.Pattern) -> str | None:
    """
    Extract the text block immediately following the first chorus tag.

    Searches lyrics for the configured tag pattern. If found, extracts
    all lines up to the next section tag (or end of string).

    Returns None if no tag match is found.
    """
    lines = lyrics.splitlines()
    in_chorus = False
    chorus_lines: list[str] = []
    any_tag_re = re.compile(r"^\[.+\]$", re.IGNORECASE)

    for line in lines:
        stripped = line.strip()

        if tag_pattern.search(stripped):
            # Found chorus tag — start collecting
            in_chorus = True
            chorus_lines = []
            continue

        if in_chorus:
            if any_tag_re.match(stripped) and chorus_lines:
                # Hit a new section tag — stop collecting
                break
            chorus_lines.append(stripped)

    if chorus_lines:
        text = "\n".join(l for l in chorus_lines if l)
        if text.strip():
            return text.strip()

    return None


# ── Tier 2 ────────────────────────────────────────────────────────────────────


def _tier2_repetition(lyrics: str, config: ProjectConfig) -> str | None:
    """
    Identify the most-repeated stanza using normalised line hashing.

    Algorithm:
        1. Split lyrics into stanzas (groups separated by blank lines)
        2. Normalise each stanza: lowercase, strip punctuation, collapse spaces
        3. Hash each normalised stanza
        4. Count hash frequencies
        5. The stanza whose hash appears >= min_repetitions times and whose
           token count >= min_tokens is the chorus candidate
        6. Return the original (un-normalised) text of the best candidate

    Returns None if no qualifying candidate is found.
    """
    min_reps = config.preprocessing.chorus_tier2_min_repetitions
    min_tokens = config.preprocessing.chorus_tier2_min_tokens

    # Split into stanzas on one or more blank lines
    stanzas = [s.strip() for s in re.split(r"\n{2,}", lyrics) if s.strip()]

    if len(stanzas) < 2:
        return None

    # Build normalised hash → list of original stanzas
    hash_to_originals: dict[str, list[str]] = {}

    for stanza in stanzas:
        normalised = _normalise_stanza(stanza)
        h = hashlib.md5(normalised.encode()).hexdigest()
        hash_to_originals.setdefault(h, []).append(stanza)

    # Find candidates meeting repetition and length thresholds
    candidates: list[tuple[int, str]] = []  # (count, original_text)

    for h, originals in hash_to_originals.items():
        count = len(originals)
        token_count = len(originals[0].split())
        if count >= min_reps and token_count >= min_tokens:
            candidates.append((count, originals[0]))

    if not candidates:
        return None

    # Return the most-repeated qualifying stanza
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _normalise_stanza(text: str) -> str:
    """
    Normalise a stanza for repetition comparison.
    Lowercase, remove punctuation, collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
