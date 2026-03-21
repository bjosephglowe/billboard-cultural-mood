"""
main.py

Billboard Cultural Mood Analysis — Pipeline Orchestrator

Single entry point for all pipeline execution modes.

Usage:
    python -m main --dry-run
    python -m main
    python -m main --stage SENTIMENT_SCORING
    python -m main --stages SENTIMENT_SCORING EMOTION_CLASSIFICATION
    python -m main --sample-years 1985-1989 --decade-filter 1980s
    python -m main --force
    python -m main --log-level DEBUG

Exit codes:
    0 — success
    1 — stage failure
    2 — argument / configuration error
    3 — fatal environment error (missing API key, unrecoverable IO)
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

# ── Stage Registry ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageDefinition:
    """Immutable descriptor for a single pipeline stage."""

    name: str  # CLI name — used with --stage / --stages
    module_path: str  # Dotted import path; module must expose run(config)
    sentinel: Path  # Sentinel file written by the stage on success
    description: str  # Short human-readable description (used in help + logs)


STAGE_REGISTRY: list[StageDefinition] = [
    StageDefinition(
        name="BILLBOARD_FETCH",
        module_path="src.data.billboard_fetcher",
        sentinel=Path("data/processed/.billboard_complete"),
        description="Fetch Billboard Hot 100 chart data → song_metadata.csv",
    ),
    StageDefinition(
        name="LYRICS_FETCH",
        module_path="src.data.lyrics_fetcher",
        sentinel=Path("data/processed/.lyrics_complete"),
        description="Fetch and cache lyrics via Genius → cache/lyrics/",
    ),
    StageDefinition(
        name="TEXT_CLEANING",
        module_path="src.preprocessing.text_cleaner",
        sentinel=Path("data/processed/.cleaning_complete"),
        description="Clean lyrics, classify narrative perspective → lyrics_cleaned.csv",
    ),
    StageDefinition(
        name="CHORUS_DETECTION",
        module_path="src.preprocessing.chorus_detector",
        sentinel=Path("data/processed/.chorus_complete"),
        description="Extract chorus via 3-tier detection cascade → chorus_extracted.csv",
    ),
    StageDefinition(
        name="SENTIMENT_SCORING",
        module_path="src.analysis.sentiment_scorer",
        sentinel=Path("data/analysis/.sentiment_complete"),
        description="Score sentiment + energy level (VADER/TextBlob/NRC) → layer2_sentiment.csv",
    ),
    StageDefinition(
        name="EMOTION_CLASSIFICATION",
        module_path="src.analysis.emotion_classifier",
        sentinel=Path("data/analysis/.emotion_complete"),
        description="Classify emotional tone (DistilRoBERTa Ekman 7-class) → layer2_emotion.csv",
    ),
    StageDefinition(
        name="THEME_CLASSIFICATION",
        module_path="src.analysis.theme_classifier",
        sentinel=Path("data/analysis/.themes_complete"),
        description="Zero-shot theme classification (BART-MNLI 12-class) → layer2_themes.csv",
    ),
    StageDefinition(
        name="CONTRAST_METRICS",
        module_path="src.analysis.contrast_metrics",
        sentinel=Path("data/analysis/.contrast_complete"),
        description="Compute Layer 4 contrast metrics, assemble master CSV → layer2_full_analysis.csv",
    ),
    StageDefinition(
        name="JUNGIAN_SCORING",
        module_path="src.psychology.jungian_scorer",
        sentinel=Path("data/analysis/.jungian_complete"),
        description="Jungian psychological scoring via GPT-4o → layer5_jungian.csv",
    ),
    StageDefinition(
        name="CULTURAL_METRICS",
        module_path="src.cultural_metrics.cmi_calculator",
        sentinel=Path("data/analysis/.cultural_metrics_complete"),
        description="Compute Cultural Mood Index and resonance scores → decade_cmi.csv",
    ),
    StageDefinition(
        name="VISUALIZATION",
        module_path="src.visualizations.trend_charts",
        sentinel=Path("outputs/.visualization_complete"),
        description="Generate 8-chart suite and HTML report → outputs/",
    ),
]

# O(1) lookup by name
_STAGE_BY_NAME: dict[str, StageDefinition] = {s.name: s for s in STAGE_REGISTRY}

ALL_STAGE_NAMES: list[str] = [s.name for s in STAGE_REGISTRY]


# ── Logging Setup ─────────────────────────────────────────────────────────────


def _configure_logging(log_level: str = "INFO") -> None:
    """
    Configure loguru with a colorized console sink and a rotating file sink.

    Args:
        log_level: Console verbosity — DEBUG, INFO, WARNING, or ERROR.
    """
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"

    logger.remove()

    # Console — colorized, human-readable
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
        ),
        colorize=True,
    )

    # File — full detail with source location
    logger.add(
        str(log_file),
        level="DEBUG",
        format=("{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}"),
        rotation="50 MB",
        retention="30 days",
        encoding="utf-8",
    )

    logger.debug("Log file: {}", log_file)


# ── Config Loading ────────────────────────────────────────────────────────────


def _load_config(args: argparse.Namespace):
    """
    Load ProjectConfig and apply CLI overrides for year range and decade filter.

    Args:
        args: Parsed argparse namespace.

    Returns:
        ProjectConfig instance with CLI overrides applied.
    """
    from src.utils.config_loader import load_config

    config = load_config()

    if args.sample_years:
        start, end = _parse_year_range(args.sample_years)
        config.dataset.sample_start_year = start
        config.dataset.sample_end_year = end
        logger.info("CLI override — sample years: {}–{}", start, end)

    if args.decade_filter:
        config.dataset.decade_filter = args.decade_filter
        logger.info("CLI override — decade filter: {}", args.decade_filter)

    if args.log_level:
        config.output.log_level = args.log_level.upper()

    return config


def _parse_year_range(s: str) -> tuple[int, int]:
    """
    Parse a 'YYYY-YYYY' string into (start, end) integers.

    Args:
        s: Year range string, e.g. '1985-1989'.

    Returns:
        Tuple of (start_year, end_year).

    Exits with code 2 on parse failure.
    """
    try:
        parts = s.split("-")
        if len(parts) != 2:
            raise ValueError
        start, end = int(parts[0]), int(parts[1])
        if end < start:
            raise ValueError
        return start, end
    except (ValueError, IndexError):
        logger.error(
            "Invalid --sample-years value: '{}'. Expected format: YYYY-YYYY (e.g. 1985-1989).",
            s,
        )
        sys.exit(2)


# ── Dry Run ───────────────────────────────────────────────────────────────────


def _dry_run() -> int:
    """
    Validate that all 11 stage modules are importable and expose a run() function.

    Makes no API calls and performs no file I/O beyond logging.

    Returns:
        0 if all stages pass, 1 if any stage fails.
    """
    logger.info("DRY RUN — validating all {} stage imports...", len(STAGE_REGISTRY))
    failures: list[str] = []

    for stage in STAGE_REGISTRY:
        try:
            mod = importlib.import_module(stage.module_path)
            if not hasattr(mod, "run"):
                msg = (
                    f"{stage.name}: module '{stage.module_path}' has no run() function"
                )
                failures.append(msg)
                logger.error("  ✗ {}", msg)
            else:
                logger.info(
                    "  ✓ {:<30} → {}",
                    stage.name,
                    stage.module_path,
                )
        except ImportError as exc:
            msg = f"{stage.name}: ImportError — {exc}"
            failures.append(msg)
            logger.error("  ✗ {}", msg)
        except Exception as exc:
            msg = f"{stage.name}: unexpected error — {exc}"
            failures.append(msg)
            logger.error("  ✗ {}", msg)

    if failures:
        logger.error(
            "Dry run FAILED — {}/{} stage(s) have import errors.",
            len(failures),
            len(STAGE_REGISTRY),
        )
        return 1

    logger.success(
        "Dry run PASSED — all {} stages importable and have run().",
        len(STAGE_REGISTRY),
    )
    return 0


# ── Stage Resolution ──────────────────────────────────────────────────────────


def _resolve_stages(args: argparse.Namespace) -> list[StageDefinition]:
    """
    Determine which stages to run based on CLI arguments.

    Resolution priority:
        1. --stage STAGE_NAME   → exactly one stage
        2. --stages S1 S2 ...   → specific ordered subset
        3. (default)            → all stages in registry order

    Args:
        args: Parsed argparse namespace.

    Returns:
        Ordered list of StageDefinition objects to run.

    Exits with code 2 on unknown stage name.
    """
    if args.stage:
        name = args.stage.upper()
        if name not in _STAGE_BY_NAME:
            logger.error(
                "Unknown stage: '{}'\n\nValid stage names:\n  {}",
                name,
                "\n  ".join(ALL_STAGE_NAMES),
            )
            sys.exit(2)
        return [_STAGE_BY_NAME[name]]

    if args.stages:
        resolved = []
        for raw in args.stages:
            name = raw.upper()
            if name not in _STAGE_BY_NAME:
                logger.error(
                    "Unknown stage: '{}'\n\nValid stage names:\n  {}",
                    name,
                    "\n  ".join(ALL_STAGE_NAMES),
                )
                sys.exit(2)
            resolved.append(_STAGE_BY_NAME[name])
        return resolved

    return list(STAGE_REGISTRY)


# ── Single Stage Execution ────────────────────────────────────────────────────


def _run_stage(
    stage: StageDefinition,
    config,
    force: bool = False,
    skip_sentinel: bool = False,
) -> bool:
    """
    Import and execute a single pipeline stage.

    Sentinel-based skip logic:
        - If the sentinel file exists AND force=False, the stage is skipped.
        - If config_hash in the sentinel differs from the current config,
          the stage re-runs regardless of force.

    Args:
        stage:          StageDefinition to execute.
        config:         Loaded ProjectConfig instance.
        force:          If True, ignore sentinel and always re-run.
        skip_sentinel:  If True, bypass sentinel check entirely.

    Returns:
        True on success, False on recoverable failure.

    Exits with code 3 on EnvironmentError (unrecoverable).
    """
    from src.pipeline.config_loader import sentinel_config_matches

    # Sentinel-based skip
    if not force and not skip_sentinel:
        if sentinel_config_matches(stage.sentinel, config):
            logger.info(
                "SKIP {:<30} — sentinel current: {}",
                stage.name,
                stage.sentinel,
            )
            return True
        elif stage.sentinel.exists():
            logger.warning(
                "Re-running {} — sentinel exists but config has changed.",
                stage.name,
            )

    _log_stage_header(stage)
    stage_start = time.monotonic()

    try:
        mod = importlib.import_module(stage.module_path)
        result = mod.run(config)
        elapsed = time.monotonic() - stage_start

        logger.success(
            "✓ {} completed in {:.1f}s",
            stage.name,
            elapsed,
        )
        if result:
            logger.debug("  Stage result: {}", result)

        return True

    except FileNotFoundError as exc:
        elapsed = time.monotonic() - stage_start
        logger.error(
            "✗ {} FAILED ({:.1f}s) — missing input file: {}",
            stage.name,
            elapsed,
            exc,
        )
        return False

    except EnvironmentError as exc:
        elapsed = time.monotonic() - stage_start
        logger.error(
            "✗ {} FATAL ({:.1f}s) — environment error: {}",
            stage.name,
            elapsed,
            exc,
        )
        sys.exit(3)

    except RuntimeError as exc:
        elapsed = time.monotonic() - stage_start
        logger.error(
            "✗ {} FAILED ({:.1f}s) — runtime error: {}",
            stage.name,
            elapsed,
            exc,
        )
        return False

    except Exception as exc:
        elapsed = time.monotonic() - stage_start
        logger.exception(
            "✗ {} FAILED ({:.1f}s) — unexpected error: {}",
            stage.name,
            elapsed,
            exc,
        )
        return False


def _log_stage_header(stage: StageDefinition) -> None:
    """Print a visual separator for a stage execution block."""
    logger.info("")
    logger.info("━" * 54)
    logger.info("  STAGE  {}", stage.name)
    logger.info("  {}", stage.description)
    logger.info("━" * 54)


# ── Pipeline Execution ────────────────────────────────────────────────────────


def _run_pipeline(
    stages: list[StageDefinition],
    config,
    force: bool = False,
) -> int:
    """
    Execute a list of stages sequentially, halting on first failure.

    Args:
        stages: Ordered list of stages to run.
        config: Loaded ProjectConfig.
        force:  If True, ignore sentinels and re-run all stages.

    Returns:
        Exit code — 0 on full success, 1 on any stage failure.
    """
    run_start = time.monotonic()
    results: list[tuple[str, bool, float]] = []

    _log_pipeline_header(stages)

    for stage in stages:
        stage_start = time.monotonic()
        success = _run_stage(stage, config, force=force)
        elapsed = time.monotonic() - stage_start
        results.append((stage.name, success, elapsed))

        if not success:
            logger.error(
                "Pipeline halted at stage {} after {:.1f}s.",
                stage.name,
                elapsed,
            )
            _print_summary(results, time.monotonic() - run_start)
            return 1

    _print_summary(results, time.monotonic() - run_start)
    return 0


def _log_pipeline_header(stages: list[StageDefinition]) -> None:
    """Print pipeline run header."""
    logger.info("")
    logger.info("╔" + "═" * 52 + "╗")
    logger.info("║  Billboard Cultural Mood Analysis Pipeline       ║")
    logger.info("║  Stages queued : {:<34}║", len(stages))
    logger.info(
        "║  Started       : {:<34}║",
        datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    logger.info("╚" + "═" * 52 + "╝")


def _print_summary(
    results: list[tuple[str, bool, float]],
    total_elapsed: float,
) -> None:
    """Print a per-stage result table and overall pipeline outcome."""
    logger.info("")
    logger.info("━" * 54)
    logger.info("  PIPELINE SUMMARY")
    logger.info("━" * 54)

    for name, success, elapsed in results:
        status = "✓" if success else "✗"
        logger.info("  {} {:<35} {:>6.1f}s", status, name, elapsed)

    passed = sum(1 for _, s, _ in results if s)
    failed = len(results) - passed

    logger.info("━" * 54)
    logger.info(
        "  {} passed  |  {} failed  |  {:.1f}s total",
        passed,
        failed,
        total_elapsed,
    )

    if failed == 0:
        logger.success("  ✓ Pipeline completed successfully.")
    else:
        logger.error("  ✗ Pipeline completed with {} failure(s).", failed)


# ── Argument Parser ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""

    stage_list = "\n".join(f"  {s.name:<35} {s.description}" for s in STAGE_REGISTRY)

    parser = argparse.ArgumentParser(
        prog="main",
        description="Billboard Cultural Mood Analysis — Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
available stages:
{stage_list}

examples:
  python -m main --dry-run
  python -m main
  python -m main --stage JUNGIAN_SCORING
  python -m main --stages SENTIMENT_SCORING EMOTION_CLASSIFICATION CONTRAST_METRICS
  python -m main --sample-years 1985-1989 --decade-filter 1980s
  python -m main --force
  python -m main --log-level DEBUG
        """,
    )

    # ── Execution mode (mutually exclusive) ───────────────────────────────────
    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate all 11 stage imports without executing any pipeline logic. "
            "Exits 0 if all imports succeed, 1 otherwise."
        ),
    )
    mode.add_argument(
        "--stage",
        type=str,
        metavar="STAGE_NAME",
        help="Run exactly one named stage.",
    )
    mode.add_argument(
        "--stages",
        nargs="+",
        type=str,
        metavar="STAGE_NAME",
        help="Run a specific ordered sequence of named stages.",
    )

    # ── Dataset filters ────────────────────────────────────────────────────────
    parser.add_argument(
        "--sample-years",
        type=str,
        metavar="YYYY-YYYY",
        help=(
            "Restrict all pipeline stages to songs in the given year range. "
            "Example: --sample-years 1985-1989"
        ),
    )
    parser.add_argument(
        "--decade-filter",
        type=str,
        metavar="DECADE_LABEL",
        help=(
            "Restrict all pipeline stages to a single decade bucket. "
            "Example: --decade-filter 1980s"
        ),
    )

    # ── Execution control ──────────────────────────────────────────────────────
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore existing sentinel files and re-run all queued stages. "
            "Useful after config changes or data corrections."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Console log verbosity: DEBUG, INFO, WARNING, ERROR. Default: INFO.",
    )

    return parser


# ── Entry Point ───────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    """
    Main entry point. Parses arguments, configures logging, and dispatches
    to dry-run validation or pipeline execution.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Integer exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    # ── Dry run — no config loading needed ────────────────────────────────────
    if args.dry_run:
        return _dry_run()

    # ── Full or partial pipeline run ──────────────────────────────────────────
    try:
        config = _load_config(args)
    except FileNotFoundError as exc:
        logger.error("Config file not found: {}", exc)
        sys.exit(2)
    except Exception as exc:
        logger.error("Failed to load config: {}", exc)
        sys.exit(2)

    stages = _resolve_stages(args)
    return _run_pipeline(stages, config, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
