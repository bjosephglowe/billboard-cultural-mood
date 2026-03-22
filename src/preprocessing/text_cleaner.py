"""
src/preprocessing/text_cleaner.py

Stage 3 — Text Cleaning & Normalisation
=========================================
Consumes data/processed/lyrics_raw.csv and produces
data/processed/lyrics_cleaned.csv.

Operations performed (in order)
---------------------------------
1. Filter out songs with lyrics_quality == "missing"
2. Strip Genius section tags            e.g. [Chorus], [Verse 1], [Bridge]
3. Strip embedded metadata lines        e.g. lines ending in "Lyrics", "Embed"
4. Normalise Unicode                    NFKC normalisation, curly quotes → straight
5. Normalise whitespace                 collapse runs, strip blank lines
6. Detect narrative perspective         1st / 2nd / 3rd person heuristic
7. Recompute token count                on cleaned text
8. Validate against cleaned_schema
9. Write lyrics_cleaned.csv + sentinel

Perspective detection
----------------------
Assigns one of: "first_person" | "second_person" | "third_person" | "abstract"

Heuristic:
    - Count 1st-person pronouns  : i, me, my, mine, myself, we, us, our
    - Count 2nd-person pronouns  : you, your, yours, yourself
    - Count 3rd-person pronouns  : he, she, they, him, her, his, their, them
    - Dominant category wins if its share >= PERSPECTIVE_DOMINANCE_THRESHOLD (0.60)
    - Otherwise: "abstract"

Idempotency
-----------
Sentinel .text_cleaning_complete records config_hash. Matching sentinel
skips the stage and returns the cached CSV.

Output schema
-------------
See src/pipeline/schemas.py → cleaned_schema.

      song_id               : str
      song_title            : str
      artist                : str
      year                  : int
      decade                : str
      lyrics_clean          : str
      lyrics_verse_only     : str
      token_count           : int
      lyrics_quality        : str   — "full" | "partial"
      narrative_perspective : str   — "first_person" | "second_person" |
                                      "third_person" | "abstract"
      has_section_tags      : bool
      section_count         : int
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.pipeline.config_loader import (
    ProjectConfig,
    load_config,
    sentinel_config_matches,
    write_sentinel,
)
from src.pipeline.schemas import cleaned_schema, validate

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "lyrics_raw.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "lyrics_cleaned.csv"
_SENTINEL = _PROJECT_ROOT / "data" / "processed" / ".text_cleaning_complete"

# ── Cleaning patterns ─────────────────────────────────────────────────────────
# Section tags: [Chorus], [Verse 1], [Pre-Chorus], [Hook], etc.
_SECTION_TAG_RE = re.compile(r"\[.*?\]", re.IGNORECASE)

# Genius embed footer: lines like "23 Embed", "See [Artist] LiveGet tickets..."
_EMBED_LINE_RE = re.compile(
    r"^\s*(\d+\s*embed|see .+ live|get tickets|you might also like)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Curly quotes → straight quotes
_CURLY_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": "-",
        "\u2013": "-",
    }
)

# ── Perspective detection ─────────────────────────────────────────────────────
_FIRST_PERSON = frozenset(
    ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours"]
)
_SECOND_PERSON = frozenset(["you", "your", "yours", "yourself", "yourselves"])
_THIRD_PERSON = frozenset(
    [
        "he",
        "she",
        "they",
        "him",
        "her",
        "his",
        "their",
        "them",
        "himself",
        "herself",
        "themselves",
    ]
)

_PERSPECTIVE_DOMINANCE_THRESHOLD = 0.60


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def run(config: ProjectConfig) -> dict:
    """
    Execute Stage 3 — Text cleaning and normalisation.

    Parameters
    ----------
    config : ProjectConfig

    Returns
    -------
    dict with keys:
        songs_input        : int   — rows in lyrics_raw.csv
        songs_output       : int   — rows after missing filter
        songs_dropped      : int   — songs removed (missing quality)
        perspective_counts : dict  — {perspective: count}
        output_path        : Path
        skipped            : bool
    """
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(
            f"lyrics_raw.csv not found at {_INPUT_PATH}. Run Stage 2 [LYRICS] first."
        )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check ────────────────────────────────────────────────────
    if sentinel_config_matches(_SENTINEL, config):
        logger.info("Stage 3 [TEXT_CLEANING] — sentinel matched, skipping.")
        df = pd.read_csv(_OUTPUT_PATH)
        return _build_summary(df, songs_input=len(df), skipped=True)

    raw = pd.read_csv(_INPUT_PATH)
    songs_input = len(raw)
    logger.info("Stage 3 [TEXT_CLEANING] — processing %d songs …", songs_input)

    # ── Step 1: Filter missing-quality songs ─────────────────────────────────
    df = raw[raw["lyrics_quality"].isin(["full", "partial"])].copy()
    songs_dropped = songs_input - len(df)
    if songs_dropped:
        logger.info("Dropped %d songs with lyrics_quality='missing'.", songs_dropped)

    # ── Step 2–7: Clean each song ────────────────────────────────────────────
    df["lyrics_clean"] = df["lyrics_raw"].apply(_clean_lyrics)
    df["narrative_perspective"] = df["lyrics_clean"].apply(_detect_perspective)
    df["token_count"] = df["lyrics_clean"].apply(
        lambda t: len(t.split()) if isinstance(t, str) else 0
    )
    df["token_count"] = pd.array(df["token_count"].tolist(), dtype=pd.Int64Dtype())

    # ── Add columns required by cleaned_schema ───────────────────────────────
    df["has_section_tags"] = df["lyrics_clean"].apply(
        lambda t: bool(re.search(r"\[.+?\]", t)) if isinstance(t, str) else False
    )
    df["section_count"] = df["lyrics_clean"].apply(
        lambda t: len(re.findall(r"\[.+?\]", t)) if isinstance(t, str) else 0
    )
    df["lyrics_verse_only"] = df["lyrics_clean"].apply(_extract_verse_only)

    # ── Propagate decade from metadata if missing ────────────────────────────
    # lyrics_raw.csv may have an empty decade column — join from metadata

    # Join decade AND year from song_metadata.csv
    meta_path = _PROJECT_ROOT / "data" / "processed" / "song_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)[["song_id", "decade", "year"]]
        df = df.drop(columns=["decade", "year"], errors="ignore")
        df = df.merge(meta, on="song_id", how="left")
        df["decade"] = df["decade"].fillna("")
        df["year"] = df["year"].fillna(0).astype(int)
    else:
        df["year"] = 0

    # ── Drop raw lyrics column — not needed downstream ───────────────────────
    df = df.drop(columns=["lyrics_raw"], errors="ignore")

    # ── Reorder columns to match schema ─────────────────────────────────────
    # — matches cleaned_schema column set exactly:
    df = df[
        [
            "song_id",
            "song_title",
            "artist",
            "year",
            "decade",
            "lyrics_clean",
            "lyrics_verse_only",
            "token_count",
            "lyrics_quality",
            "narrative_perspective",
            "has_section_tags",
            "section_count",
        ]
    ]

    # ── Validate ─────────────────────────────────────────────────────────────
    df = validate(df, cleaned_schema, stage_name="TEXT_CLEANING")

    # ── Write output ─────────────────────────────────────────────────────────
    df.to_csv(_OUTPUT_PATH, index=False)
    write_sentinel(_SENTINEL, stage="TEXT_CLEANING", config=config)

    summary = _build_summary(df, songs_input=songs_input, skipped=False)
    summary["songs_dropped"] = songs_dropped

    logger.info(
        "Stage 3 [TEXT_CLEANING] — complete. %d/%d songs retained → %s",
        len(df),
        songs_input,
        _OUTPUT_PATH,
    )
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Cleaning pipeline
# ═════════════════════════════════════════════════════════════════════════════


def _clean_lyrics(text: str | float) -> str:
    """
    Apply the full cleaning pipeline to a single lyrics string.

    Steps:
        1. Guard against NaN / non-string input
        2. Strip section tags              [Chorus], [Verse 1] …
        3. Strip Genius embed lines        "23 Embed", "You might also like"
        4. NFKC Unicode normalisation
        5. Curly quote → straight quote substitution
        6. Collapse whitespace and blank lines
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Strip section tags
    text = _SECTION_TAG_RE.sub("", text)

    # Strip Genius embed footer lines
    text = _EMBED_LINE_RE.sub("", text)

    # NFKC normalisation (ligatures, half-width chars, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Curly quotes and dashes → ASCII equivalents
    text = text.translate(_CURLY_QUOTE_MAP)

    # Collapse multiple spaces within lines
    lines = [" ".join(line.split()) for line in text.splitlines()]

    # Remove blank lines, strip each line
    lines = [line.strip() for line in lines if line.strip()]

    return "\n".join(lines)


def _extract_verse_only(text: str | float) -> str:
    """
    Return lyrics with chorus, bridge, hook, and non-verse sections removed.

    Strips all lines between a non-verse section tag and the next section
    tag (or end of string). Returns full text if no section tags are present.
    Non-verse tags: chorus, bridge, hook, refrain, outro, intro.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    non_verse_re = re.compile(
        r"^\[(chorus|bridge|hook|refrain|outro|intro)[^\]]*\]$",
        re.IGNORECASE,
    )
    any_tag_re = re.compile(r"^\[.+\]$", re.IGNORECASE)

    lines = text.splitlines()
    result: list[str] = []
    skip = False

    for line in lines:
        stripped = line.strip()
        if non_verse_re.match(stripped):
            skip = True
            continue
        if any_tag_re.match(stripped) and skip:
            skip = False
            continue
        if not skip:
            result.append(stripped)

    return "\n".join(l for l in result if l)


# ═════════════════════════════════════════════════════════════════════════════
# Perspective detection
# ═════════════════════════════════════════════════════════════════════════════


def _detect_perspective(text: str | float) -> str:
    """
    Classify the narrative perspective of a lyrics string.

    Returns one of: "first_person" | "second_person" | "third_person" | "abstract"

    Algorithm:
        1. Tokenise to lowercase words (letters only)
        2. Count pronoun hits per category
        3. If total pronoun count is 0 → "abstract" (indeterminate)
        4. If dominant category share >= PERSPECTIVE_DOMINANCE_THRESHOLD → that label
        5. Otherwise → "abstract"
    """
    if not isinstance(text, str) or not text.strip():
        return "abstract"

    words = re.findall(r"[a-z]+", text.lower())

    first = sum(1 for w in words if w in _FIRST_PERSON)
    second = sum(1 for w in words if w in _SECOND_PERSON)
    third = sum(1 for w in words if w in _THIRD_PERSON)
    total = first + second + third

    if total == 0:
        return "abstract"

    shares = {
        "first_person": first / total,
        "second_person": second / total,
        "third_person": third / total,
    }

    dominant, share = max(shares.items(), key=lambda x: x[1])

    if share >= _PERSPECTIVE_DOMINANCE_THRESHOLD:
        return dominant
    return "abstract"


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════


def _build_summary(
    df: pd.DataFrame,
    songs_input: int,
    skipped: bool,
) -> dict:
    perspective_counts = df["narrative_perspective"].value_counts().to_dict()
    return {
        "songs_input": songs_input,
        "songs_output": len(df),
        "songs_dropped": songs_input - len(df),
        "perspective_counts": perspective_counts,
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
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Stage 3 — Lyrics text cleaning and normalisation"
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
        help="Delete sentinel and re-clean even if cache is valid",
    )
    args = parser.parse_args()

    if args.force and _SENTINEL.exists():
        _SENTINEL.unlink()
        logger.info("Sentinel deleted — forcing re-clean.")

    cfg = load_config(args.config)
    result = run(cfg)

    print(f"\nText cleaning complete.")
    print(f"  Input songs    : {result['songs_input']:,}")
    print(f"  Output songs   : {result['songs_output']:,}")
    print(f"  Dropped        : {result['songs_dropped']:,}")
    print(f"  Perspective breakdown:")
    for label, count in sorted(result["perspective_counts"].items()):
        print(f"    {label:<20}: {count:,}")
    print(f"  Output         : {result['output_path']}")
    sys.exit(0)
