"""
src/data/lyrics_fetcher.py

Stage 2 — Lyrics Ingestion
===========================
For every song in data/processed/song_metadata.csv, fetches lyrics
from the Genius API, assigns a lyrics_quality tier, and writes
data/processed/lyrics_raw.csv.

Per-song JSON cache
-------------------
Each successful Genius response is written to:
    cache/lyrics/<song_id>.json

On subsequent runs the cache is checked first — no API call is made
for songs already cached. This means interrupted runs are safely
resumable without re-fetching already-retrieved songs.

Lyrics quality tiers
---------------------
    full      — token count >= config.lyrics.min_token_threshold
    partial   — token count >= config.lyrics.partial_token_threshold
    missing   — no lyrics found or token count below partial threshold

Only songs with quality "full" or "partial" are passed to downstream
NLP stages. Songs with quality "missing" are retained in the CSV for
traceability but filtered out by text_cleaner.py.

Rate limiting
-------------
config.lyrics.genius_sleep_time controls the sleep between API calls
(default 0.5 s). config.lyrics.genius_max_retries controls per-song
retry attempts on transient failures (default 3).

Idempotency
-----------
Sentinel .lyrics_complete records the config_hash. If matched, the
stage is skipped and the cached CSV is returned. Per-song JSON cache
is always respected regardless of sentinel state.

Output schema
-------------
No Pandera schema defined for this intermediate file.
Output is validated downstream by text_cleaner.py → cleaned_schema.

    song_id        : str    — 16-char hex
    song_title          : str
    artist         : str
    decade         : str
    lyrics_raw     : str    — raw lyrics text (empty string if missing)
    lyrics_quality : str    — "full" | "partial" | "missing"
    token_count    : Int64  — whitespace-split token count
    genius_url     : str    — Genius page URL (empty string if missing)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.pipeline.config_loader import (
    ProjectConfig,
    load_config,
    sentinel_config_matches,
    write_sentinel,
)
from src.pipeline.schemas import validate
from src.utils.identifiers import make_song_id

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "song_metadata.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "lyrics_raw.csv"
_CACHE_DIR = _PROJECT_ROOT / "cache" / "lyrics"
_SENTINEL = _PROJECT_ROOT / "data" / "processed" / ".lyrics_complete"


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def run(config: ProjectConfig) -> dict:
    """
    Execute Stage 2 — Lyrics ingestion.

    Parameters
    ----------
    config : ProjectConfig

    Returns
    -------
    dict with keys:
        songs_total     : int   — songs in input metadata
        songs_full      : int   — songs with quality "full"
        songs_partial   : int   — songs with quality "partial"
        songs_missing   : int   — songs with quality "missing"
        songs_cached    : int   — songs served from JSON cache (no API call)
        output_path     : Path
        skipped         : bool  — True if sentinel matched
    """
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(
            f"song_metadata.csv not found at {_INPUT_PATH}. "
            "Run Stage 1 [BILLBOARD] first."
        )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check ────────────────────────────────────────────────────
    if sentinel_config_matches(_SENTINEL, config):
        logger.info("Stage 2 [LYRICS] — sentinel matched, skipping fetch.")
        df = pd.read_csv(_OUTPUT_PATH)
        return _build_summary(df, skipped=True)

    metadata = pd.read_csv(_INPUT_PATH)
    logger.info("Stage 2 [LYRICS] — fetching lyrics for %d songs …", len(metadata))

    genius = _init_genius(config)
    records: list[dict] = []
    songs_cached = 0

    for _, row in metadata.iterrows():
        song_id = str(row["song_id"])
        cache_path = _CACHE_DIR / f"{song_id}.json"

        # ── Per-song cache hit ───────────────────────────────────────────────
        if cache_path.exists():
            cached = _load_cache(cache_path)
            if cached is not None:
                records.append(cached)
                songs_cached += 1
                continue

        # ── Genius API fetch ─────────────────────────────────────────────────
        result = _fetch_with_retry(
            genius=genius,
            title=str(row["song_title"]),
            artist=str(row["artist"]),
            song_id=song_id,
            config=config,
        )
        _save_cache(cache_path, result)
        records.append(result)
        time.sleep(config.lyrics.genius_sleep_time)

    df = pd.DataFrame(records)

    # ── Cast token_count to nullable Int64 ──────────────────────────────────
    # — validate call removed, lightweight column check substituted:
    df["token_count"] = pd.array(df["token_count"].tolist(), dtype=pd.Int64Dtype())

    # ── Lightweight sanity check (no Pandera schema for intermediate file) ────
    _required_cols = {
        "song_id",
        "song_title",
        "artist",
        "lyrics_raw",
        "lyrics_quality",
        "token_count",
    }
    missing_cols = _required_cols - set(df.columns)
    if missing_cols:
        raise RuntimeError(
            f"Stage 2 [LYRICS] output is missing required columns: {missing_cols}"
        )

    # ── Write output ─────────────────────────────────────────────────────────
    df.to_csv(
        _OUTPUT_PATH, index=False, quoting=1
    )  # csv.QUOTE_ALL — safe for multiline lyrics

    write_sentinel(_SENTINEL, stage="LYRICS", config=config)

    summary = _build_summary(df, skipped=False)
    summary["songs_cached"] = songs_cached

    logger.info(
        "Stage 2 [LYRICS] — complete. full=%d partial=%d missing=%d cached=%d → %s",
        summary["songs_full"],
        summary["songs_partial"],
        summary["songs_missing"],
        songs_cached,
        _OUTPUT_PATH,
    )
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════


def _init_genius(config: ProjectConfig):
    """
    Initialise and return a lyricsgenius.Genius client.

    Reads GENIUS_API_TOKEN from the environment. Raises EnvironmentError
    if the token is absent or is still the placeholder value from .env.example.
    """
    try:
        import lyricsgenius  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'lyricsgenius' package is required for Stage 2. "
            "Install with: pip install lyricsgenius"
        ) from exc

    token = os.getenv("GENIUS_API_TOKEN", "").strip()
    if not token or token.startswith("your_"):
        raise EnvironmentError(
            "GENIUS_API_TOKEN is not set or is still a placeholder. "
            "Add a valid token to your .env file."
        )

    genius = lyricsgenius.Genius(
        token,
        timeout=15,
        retries=config.lyrics.genius_max_retries,
        sleep_time=0,  # we handle sleep manually for predictability
        verbose=False,
        remove_section_headers=False,  # preserve tags for chorus detection
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)", "(Demo)", "(Instrumental)"],
    )
    return genius


def _fetch_with_retry(
    genius,
    title: str,
    artist: str,
    song_id: str,
    config: ProjectConfig,
) -> dict:
    """
    Fetch lyrics for one song with retry logic.

    Returns a record dict regardless of success/failure — quality is set
    to "missing" on all failure paths so the song is retained in the output
    for traceability.
    """
    last_exc: Exception | None = None

    for attempt in range(1, config.lyrics.genius_max_retries + 1):
        try:
            song = genius.search_song(title=title, artist=artist)

            if song is None or not song.lyrics:
                return _missing_record(song_id, title, artist)

            lyrics_text = _clean_genius_header(song.lyrics)
            token_count = len(lyrics_text.split())
            quality = _score_quality(token_count, config)

            return {
                "song_id": song_id,
                "song_title": title,
                "artist": artist,
                "decade": "",  # joined from metadata downstream
                "lyrics_raw": lyrics_text,
                "lyrics_quality": quality,
                "token_count": token_count,
                "genius_url": song.url or "",
            }

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Lyrics fetch attempt %d/%d failed for '%s' by '%s': %s",
                attempt,
                config.lyrics.genius_max_retries,
                title,
                artist,
                exc,
            )
            if attempt < config.lyrics.genius_max_retries:
                time.sleep(config.lyrics.genius_sleep_time * attempt)

    logger.error(
        "All %d attempts failed for '%s' by '%s'. Last error: %s",
        config.lyrics.genius_max_retries,
        title,
        artist,
        last_exc,
    )
    return _missing_record(song_id, title, artist)


def _missing_record(song_id: str, title: str, artist: str) -> dict:
    """Return a well-formed record with quality='missing' and empty lyrics."""
    return {
        "song_id": song_id,
        "song_title": title,
        "artist": artist,
        "decade": "",
        "lyrics_raw": "",
        "lyrics_quality": "missing",
        "token_count": 0,
        "genius_url": "",
    }


def _score_quality(token_count: int, config: ProjectConfig) -> str:
    """
    Assign a quality tier based on token count thresholds from config.

        full    : token_count >= min_token_threshold
        partial : token_count >= partial_token_threshold
        missing : below partial threshold
    """
    if token_count >= config.lyrics.min_token_threshold:
        return "full"
    if token_count >= config.lyrics.partial_token_threshold:
        return "partial"
    return "missing"


def _clean_genius_header(lyrics: str) -> str:
    """
    Remove the boilerplate header Genius prepends to every lyrics string.

    Genius format: "<N> ContributorsTitle Lyrics\\n<actual lyrics>"
    We strip everything up to and including the first newline after
    the word "Lyrics" in the header.
    """
    if not lyrics:
        return ""

    # Find the standard Genius header terminator
    marker = "Lyrics\n"
    idx = lyrics.find(marker)
    if idx != -1:
        return lyrics[idx + len(marker) :].strip()

    # Fallback: strip first line if it looks like a header
    lines = lyrics.splitlines()
    if lines and ("Contributors" in lines[0] or "Lyrics" in lines[0]):
        return "\n".join(lines[1:]).strip()

    return lyrics.strip()


def _save_cache(path: Path, record: dict) -> None:
    """Write a lyrics record to the per-song JSON cache."""
    try:
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write cache file %s: %s", path, exc)


def _load_cache(path: Path) -> dict | None:
    """
    Load a lyrics record from the per-song JSON cache.
    Returns None if the file is missing, empty, or contains invalid JSON.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read cache file %s: %s", path, exc)
        return None


def _build_summary(df: pd.DataFrame, skipped: bool) -> dict:
    """Build the run() return dict from the output DataFrame."""
    quality_counts = df["lyrics_quality"].value_counts()
    return {
        "songs_total": len(df),
        "songs_full": int(quality_counts.get("full", 0)),
        "songs_partial": int(quality_counts.get("partial", 0)),
        "songs_missing": int(quality_counts.get("missing", 0)),
        "songs_cached": 0,
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
        description="Stage 2 — Lyrics ingestion via Genius API"
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
        help="Delete sentinel and re-fetch even if cache is valid",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        dest="clear_cache",
        help="Delete all per-song JSON cache files before fetching",
    )
    args = parser.parse_args()

    if args.force and _SENTINEL.exists():
        _SENTINEL.unlink()
        logger.info("Sentinel deleted — forcing re-fetch.")

    if args.clear_cache and _CACHE_DIR.exists():
        removed = 0
        for f in _CACHE_DIR.glob("*.json"):
            f.unlink()
            removed += 1
        logger.info("Cleared %d cached lyrics files.", removed)

    cfg = load_config(args.config)
    result = run(cfg)

    print(f"\nLyrics fetch complete.")
    print(f"  Total songs    : {result['songs_total']:,}")
    print(f"  Full quality   : {result['songs_full']:,}")
    print(f"  Partial quality: {result['songs_partial']:,}")
    print(f"  Missing        : {result['songs_missing']:,}")
    if not result["skipped"]:
        print(f"  From cache     : {result['songs_cached']:,}")
    print(f"  Output         : {result['output_path']}")
    sys.exit(0)
