"""
src/data/billboard_fetcher.py

Stage 1 — Billboard Hot 100 Data Ingestion
============================================
Fetches every weekly Billboard Hot 100 chart between
config.dataset.sample_start_year and config.dataset.sample_end_year,
deduplicates to one canonical entry per (artist, title) pair,
assigns decade bucket labels, generates a deterministic song_id
per song, validates the output against metadata_schema, and
writes data/processed/song_metadata.csv.

Idempotency
-----------
A sentinel file (.billboard_complete) records the config_hash of
the run that produced the current output. If the sentinel matches
the active config, the fetch is skipped and the cached CSV is
returned. Delete the sentinel (or change config) to force a
re-fetch.

Rate limiting
-------------
The billboard.py library makes HTTP requests to the Billboard
website. A configurable sleep between weekly fetches prevents
rate-limiting. Default: 1.0 s between requests.

Error handling
--------------
Individual week failures are logged and skipped — a single bad
week does not abort the entire fetch. If the total number of
successfully fetched weeks falls below MINIMUM_WEEK_COVERAGE
(80% of expected weeks), a RuntimeError is raised to prevent
a severely truncated dataset from propagating downstream.

Output schema
-------------
See src/pipeline/schemas.py → metadata_schema.

      song_id         : str   — 16-char hex (SHA-256 of artist+title)
      song_title      : str
      artist          : str
      year            : Int64 — year of first_chart_date
      decade          : str   — e.g. "1980s"
      chart_rank      : Int64 — best (lowest) chart position seen
      chart_weeks_on  : Int64 — longest chart run seen
      first_chart_date: str   — ISO 8601 date of earliest chart appearance
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import requests

from src.pipeline.config_loader import (
    ProjectConfig,
    config_hash,
    load_config,
    sentinel_config_matches,
    write_sentinel,
)
from src.pipeline.schemas import metadata_schema, validate
from src.utils.identifiers import make_song_id

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "song_metadata.csv"
_SENTINEL = _PROJECT_ROOT / "data" / "processed" / ".billboard_complete"

# ── Tuneable constants ────────────────────────────────────────────────────────
_SLEEP_BETWEEN_WEEKS: float = 0.5  # seconds; override in tests via monkeypatch
_MINIMUM_WEEK_COVERAGE: float = 0.80  # fraction of expected weeks required
_REQUEST_TIMEOUT_SECONDS: int = 30  # hard timeout for each HTTP request
_MAX_RETRIES: int = 3  # retry attempts per week on transient errors

# ── Data source ───────────────────────────────────────────────────────────────
_GITHUB_DATA_URL = (
    "https://raw.githubusercontent.com/mhollingshead/"
    "billboard-hot-100/main/date/{date}.json"
)


# ── Billboard chart fetch interval ───────────────────────────────────────────
_CHART_DAY_OF_WEEK: int = 5  # Saturday — Billboard publishes Saturdays


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def run(config: ProjectConfig) -> dict:
    """
    Execute Stage 1 — Billboard Hot 100 data ingestion.

    Parameters
    ----------
    config : ProjectConfig
        Validated project configuration.

    Returns
    -------
    dict with keys:
        songs_fetched   : int   — total unique songs in output
        weeks_fetched   : int   — number of weeks successfully retrieved
        weeks_expected  : int   — total weeks in configured year range
        coverage_rate   : float — weeks_fetched / weeks_expected
        output_path     : Path  — path to written CSV
        skipped         : bool  — True if sentinel matched and fetch was skipped
    """
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check ────────────────────────────────────────────────────
    if sentinel_config_matches(_SENTINEL, config):
        logger.info("Stage 1 [BILLBOARD] — sentinel matched, skipping fetch.")
        df = pd.read_csv(_OUTPUT_PATH)
        return {
            "songs_fetched": len(df),
            "weeks_fetched": None,
            "weeks_expected": None,
            "coverage_rate": None,
            "output_path": _OUTPUT_PATH,
            "skipped": True,
        }

    logger.info(
        "Stage 1 [BILLBOARD] — fetching %d–%d …",
        config.dataset.sample_start_year,
        config.dataset.sample_end_year,
    )

    # ── Collect weekly charts ────────────────────────────────────────────────
    week_dates = _generate_week_dates(config)
    weeks_expected = len(week_dates)
    raw_records: list[dict] = []
    weeks_fetched = 0
    weeks_failed = 0

    for i, chart_date in enumerate(week_dates, start=1):
        try:
            records = _fetch_single_week(chart_date)
            raw_records.extend(records)
            weeks_fetched += 1
            logger.info(
                "  [%d/%d] %s — %d entries",
                i,
                weeks_expected,
                chart_date.isoformat(),
                len(records),
            )
            time.sleep(_SLEEP_BETWEEN_WEEKS)
        except Exception as exc:  # noqa: BLE001
            weeks_failed += 1
            logger.warning(
                "Failed to fetch chart for %s (%s: %s) — skipping week.",
                chart_date.isoformat(),
                type(exc).__name__,
                exc,
            )

    # ── Coverage gate ────────────────────────────────────────────────────────
    coverage = weeks_fetched / weeks_expected if weeks_expected > 0 else 0.0
    if coverage < _MINIMUM_WEEK_COVERAGE:
        raise RuntimeError(
            f"Billboard fetch coverage gate FAILED: "
            f"retrieved {weeks_fetched}/{weeks_expected} weeks "
            f"({coverage:.1%} < {_MINIMUM_WEEK_COVERAGE:.0%} required). "
            f"Check network connectivity or Billboard availability."
        )

    logger.info(
        "Fetched %d/%d weeks (%.1f%% coverage), %d weeks failed.",
        weeks_fetched,
        weeks_expected,
        coverage * 100,
        weeks_failed,
    )

    if not raw_records:
        raise RuntimeError(
            "No chart records retrieved. Cannot proceed with empty dataset."
        )

    # ── Deduplicate + enrich ─────────────────────────────────────────────────
    df = _build_canonical_dataframe(raw_records, config)

    # ── Validate ─────────────────────────────────────────────────────────────
    df = validate(df, metadata_schema, stage_name="BILLBOARD")

    # ── Write output ─────────────────────────────────────────────────────────
    df.to_csv(_OUTPUT_PATH, index=False)
    write_sentinel(_SENTINEL, stage="BILLBOARD", config=config)

    logger.info(
        "Stage 1 [BILLBOARD] — complete. %d unique songs → %s",
        len(df),
        _OUTPUT_PATH,
    )

    return {
        "songs_fetched": len(df),
        "weeks_fetched": weeks_fetched,
        "weeks_expected": weeks_expected,
        "coverage_rate": coverage,
        "output_path": _OUTPUT_PATH,
        "skipped": False,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════


def _generate_week_dates(config: ProjectConfig) -> list[date]:
    """
    Return one date per week (Saturday) for every week whose year falls
    within [sample_start_year, sample_end_year].

    Billboard publishes a new chart each Saturday. We step weekly
    from the first Saturday of January in start_year to the last
    Saturday of December in end_year.
    """
    start = date(config.dataset.sample_start_year, 1, 1)
    end = date(config.dataset.sample_end_year, 12, 31)

    # Advance start to the nearest Saturday on or after Jan 1
    days_until_saturday = (_CHART_DAY_OF_WEEK - start.weekday()) % 7
    current = start + timedelta(days=days_until_saturday)

    weeks: list[date] = []
    while current <= end:
        weeks.append(current)
        current += timedelta(weeks=1)

    logger.debug(
        "Generated %d weekly chart dates (%s → %s).",
        len(weeks),
        weeks[0].isoformat(),
        weeks[-1].isoformat(),
    )
    return weeks


def _fetch_single_week(chart_date: date) -> list[dict]:
    """
    Fetch one week of Billboard Hot 100 data from the mhollingshead/
    billboard-hot-100 GitHub repository (updated daily).

    URL pattern:
        https://raw.githubusercontent.com/mhollingshead/
        billboard-hot-100/main/date/YYYY-MM-DD.json

    Returns a list of raw record dicts with keys:
        title, artist, peak_position, weeks_on_chart, chart_date

    Raises
    ------
    requests.HTTPError  : 404 if the date has no chart (not a valid
                          Billboard release date); other 4xx/5xx on
                          GitHub outage
    requests.Timeout    : if request exceeds _REQUEST_TIMEOUT_SECONDS
    RuntimeError        : if JSON is valid but contains no entries
    """
    url = _GITHUB_DATA_URL.format(date=chart_date.isoformat())

    # ── Fetch with retry on transient errors ─────────────────────────────
    response = None
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            break
        except requests.exceptions.HTTPError as exc:
            # 404 = date not a valid Billboard release date — not retryable
            if exc.response is not None and exc.response.status_code == 404:
                raise
            last_exc = exc
            logger.warning(
                "HTTP error on attempt %d/%d for %s: %s — retrying …",
                attempt,
                _MAX_RETRIES,
                chart_date.isoformat(),
                exc,
            )
            time.sleep(attempt * 2)
        except (
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as exc:
            last_exc = exc
            logger.warning(
                "Request error on attempt %d/%d for %s: %s — retrying …",
                attempt,
                _MAX_RETRIES,
                chart_date.isoformat(),
                exc,
            )
            time.sleep(attempt * 2)

    if response is None:
        raise RuntimeError(
            f"All {_MAX_RETRIES} attempts failed for {chart_date.isoformat()}. "
            f"Last error: {last_exc}"
        )

    # ── Parse JSON ────────────────────────────────────────────────────────
    payload = response.json()  # {"date": "...", "data": [...]}
    entries = payload.get("data", [])

    if not entries:
        raise RuntimeError(
            f"Zero chart entries in JSON payload for {chart_date.isoformat()}."
        )

    # ── Map to internal schema ────────────────────────────────────────────
    records: list[dict] = []
    for item in entries:
        title = _normalise_text(item.get("song", ""))
        artist = _normalise_text(item.get("artist", ""))
        if not title or not artist:
            continue
        records.append(
            {
                "title": title,
                "artist": artist,
                "peak_position": item.get("peak_position"),
                "weeks_on_chart": item.get("weeks_on_chart"),
                "chart_date": chart_date.isoformat(),
            }
        )

    return records


def _build_canonical_dataframe(
    raw_records: list[dict],
    config: ProjectConfig,
) -> pd.DataFrame:
    """
    Deduplicate raw weekly chart records into one canonical row per song.

    Deduplication key : (artist, title)  — case-normalised
    peak_position     : minimum (best) position seen across all weeks
    weeks_on_chart    : maximum (longest run) seen across all weeks
    first_chart_date  : earliest chart_date seen for this (artist, title)
    chart_year        : year extracted from first_chart_date
    decade            : assigned from config decade_buckets

    Parameters
    ----------
    raw_records : list[dict]
        All records from _fetch_single_week calls, potentially duplicated.
    config : ProjectConfig
        Used for decade bucket assignment.

    Returns
    -------
    pd.DataFrame conforming to metadata_schema (before Pandera validation).
    """
    df = pd.DataFrame(raw_records)

    # ── Normalise join key ───────────────────────────────────────────────────
    df["_key"] = (
        df["artist"].str.lower().str.strip()
        + "||"
        + df["title"].str.lower().str.strip()
    )

    # ── Aggregate per key ────────────────────────────────────────────────────
    df["chart_date"] = pd.to_datetime(df["chart_date"])

    agg = (
        df.groupby("_key", sort=False)
        .agg(
            song_title=("title", "first"),
            artist=("artist", "first"),
            chart_rank=("peak_position", "min"),
            chart_weeks_on=("weeks_on_chart", "max"),
            first_chart_date=("chart_date", "min"),
        )
        .reset_index(drop=True)
    )

    agg["first_chart_date"] = agg["first_chart_date"].dt.date.astype(str)
    agg["year"] = agg["first_chart_date"].str[:4].astype(int)

    # ── Decade assignment ────────────────────────────────────────────────────
    agg["decade"] = agg["year"].apply(lambda y: _assign_decade(y, config))

    # Drop songs that fall outside all configured decade buckets
    outside = agg["decade"].isna()
    if outside.any():
        logger.warning(
            "%d songs fell outside all configured decade buckets and will be dropped.",
            outside.sum(),
        )
        agg = agg[~outside].copy()

    # ── Apply decade filter if configured ───────────────────────────────────
    if config.dataset.decade_filter:
        agg = agg[agg["decade"] == config.dataset.decade_filter].copy()
        logger.info(
            "Decade filter '%s' applied — %d songs retained.",
            config.dataset.decade_filter,
            len(agg),
        )

    # ── Generate song IDs ────────────────────────────────────────────────────
    agg["song_id"] = agg.apply(
        lambda row: make_song_id(row["artist"], row["song_title"]),
        axis=1,
    )

    # ── Cast numeric columns to nullable Int64 ───────────────────────────────
    for col in ("chart_rank", "chart_weeks_on", "year"):
        agg[col] = pd.array(agg[col].tolist(), dtype=pd.Int64Dtype())

    # ── Reorder columns to match schema ─────────────────────────────────────
    agg = agg[
        [
            "song_id",
            "song_title",
            "artist",
            "year",
            "decade",
            "chart_rank",
            "chart_weeks_on",
            "first_chart_date",
        ]
    ]

    logger.info(
        "Deduplicated %d raw chart entries → %d unique songs.",
        len(df),
        len(agg),
    )
    return agg


def _assign_decade(year: int, config: ProjectConfig) -> str | None:
    """
    Return the decade label for a given year using config.dataset.decade_buckets.
    Returns None if the year falls outside all configured buckets.
    """
    for bucket in config.dataset.decade_buckets:
        if bucket.start <= year <= bucket.end:
            return bucket.label
    return None


def _normalise_text(text: str | None) -> str:
    """
    Strip leading/trailing whitespace and normalise internal spacing.
    Returns empty string for None or empty input.
    """
    if not text:
        return ""
    return " ".join(text.strip().split())


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
        description="Stage 1 — Billboard Hot 100 data ingestion"
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
    args = parser.parse_args()

    if args.force and _SENTINEL.exists():
        _SENTINEL.unlink()
        logger.info("Sentinel deleted — forcing re-fetch.")

    cfg = load_config(args.config)
    result = run(cfg)

    print(f"\nBillboard fetch complete.")
    print(f"  Unique songs   : {result['songs_fetched']:,}")
    if not result["skipped"]:
        print(
            f"  Weeks fetched  : {result['weeks_fetched']:,} / {result['weeks_expected']:,}"
        )
        print(f"  Coverage       : {result['coverage_rate']:.1%}")
    print(f"  Output         : {result['output_path']}")
    sys.exit(0)
