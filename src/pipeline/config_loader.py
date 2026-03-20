"""
src/pipeline/config_loader.py

Project configuration loader and models for the Billboard Cultural Mood Analysis
pipeline.

Responsibilities:
    - Define the typed ProjectConfig model hierarchy (Pydantic v2).
    - Load config/project_config.yaml into ProjectConfig.
    - Provide config_hash() for sentinel files.
    - Provide sentinel_config_matches() for config-change detection.

Usage:
    from src.pipeline.config_loader import ProjectConfig, load_config, config_hash

    config = load_config()
    print(config.dataset.sample_start_year)

All pipeline stages should accept a ProjectConfig instance as their only
configuration argument:

    def run(config: ProjectConfig) -> dict:
        ...

This centralizes configuration and avoids scattered environment-variable reads
throughout the codebase.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "project_config.yaml"


# ── Model Definitions ─────────────────────────────────────────────────────────


class DecadeBucket(BaseModel):
    """Single decade bucket definition."""

    label: str = Field(..., description="Decade label (e.g. '1980s', '1960s*')")
    start: int = Field(..., description="Inclusive start year")
    end: int = Field(..., description="Inclusive end year")
    non_standard: bool = Field(
        False,
        description="True if this bucket is not a standard 10-year span",
    )

    @field_validator("label")
    @classmethod
    def label_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Decade label must be a non-empty string")
        return v.strip()

    @field_validator("end")
    @classmethod
    def end_not_before_start(cls, v: int, info):
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"end ({v}) must be >= start ({start})")
        return v


class DatasetConfig(BaseModel):
    """Dataset-wide configuration, including decade buckets and filters."""

    sample_start_year: int = Field(..., ge=1900)
    sample_end_year: int = Field(..., ge=1900)
    decade_filter: Optional[str] = Field(
        default=None,
        description="If set, restricts analysis to a single decade label.",
    )
    decade_buckets: List[DecadeBucket]

    @field_validator("sample_end_year")
    @classmethod
    def end_not_before_start(cls, v: int, info):
        start = info.data.get("sample_start_year")
        if start is not None and v < start:
            raise ValueError(
                f"dataset.sample_end_year ({v}) must be >= sample_start_year ({start})"
            )
        return v

    @field_validator("decade_buckets")
    @classmethod
    def buckets_non_empty(cls, v: List[DecadeBucket]) -> List[DecadeBucket]:
        if not v:
            raise ValueError("dataset.decade_buckets must contain at least one bucket")
        return v


class LyricsConfig(BaseModel):
    """Lyrics ingestion and quality thresholds."""

    cache_dir: str = Field(
        "cache/lyrics", description="Directory for per-song JSON cache"
    )
    min_token_threshold: int = Field(50, ge=1)
    partial_token_threshold: int = Field(150, ge=1)
    genius_sleep_time: float = Field(0.5, ge=0.0)
    genius_max_retries: int = Field(3, ge=0)


class PreprocessingConfig(BaseModel):
    """Text preprocessing and chorus detection configuration."""

    chorus_tier1_tag_pattern: str
    chorus_tier2_min_repetitions: int = Field(3, ge=1)
    chorus_tier2_min_tokens: int = Field(6, ge=1)
    chorus_tier3_model: str = Field("gpt-4o")
    chorus_max_tokens: int = Field(200, ge=1)


class EnergyLevelConfig(BaseModel):
    """Energy level bin edges."""

    bins: List[float] = Field(..., description="Ascending list of bin edges in (0,1)")

    @field_validator("bins")
    @classmethod
    def bins_must_be_sorted_and_in_range(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError(
                "analysis.energy_level.bins must contain at least one value"
            )
        if sorted(v) != v:
            raise ValueError("analysis.energy_level.bins must be sorted ascending")
        if any(not (0.0 < b < 1.0) for b in v):
            raise ValueError("analysis.energy_level.bins values must be in (0, 1)")
        return v


class AnalysisConfig(BaseModel):
    """Configuration for sentiment, emotion, themes, coverage gate, and Jungian scoring."""

    energy_level: EnergyLevelConfig

    emotion_model: str = Field(
        "j-hartmann/emotion-english-distilroberta-base",
        description="Hugging Face model ID for emotion classifier",
    )
    emotion_batch_size: int = Field(16, ge=1)

    theme_model: str = Field(
        "facebook/bart-large-mnli",
        description="Hugging Face model ID for zero-shot theme classifier",
    )
    theme_threshold: float = Field(
        0.40,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for a theme to be included",
    )
    theme_batch_size: int = Field(8, ge=1)
    theme_taxonomy: List[str]

    coverage_gate_threshold: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="Minimum coverage of full-quality songs required",
    )

    jungian_model: str = Field("gpt-4o", description="OpenAI model for Jungian scoring")
    jungian_batch_size: int = Field(5, ge=1)
    jungian_inter_batch_sleep: float = Field(1.0, ge=0.0)
    jungian_max_retries: int = Field(3, ge=0)
    jungian_valid_stages: List[str]
    jungian_dev_score_min: int = Field(1)
    jungian_dev_score_max: int = Field(7)

    @field_validator("theme_taxonomy")
    @classmethod
    def taxonomy_non_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError(
                "analysis.theme_taxonomy must contain at least one theme label"
            )
        return v

    @field_validator("jungian_dev_score_max")
    @classmethod
    def dev_score_range_valid(cls, v: int, info):
        min_val = info.data.get("jungian_dev_score_min")
        if min_val is not None and v < min_val:
            raise ValueError(
                f"analysis.jungian_dev_score_max ({v}) must be >= jungian_dev_score_min ({min_val})"
            )
        return v


class CulturalMetricsConfig(BaseModel):
    """Configuration for cultural resonance and CMI computation."""

    normalization: str = Field(
        "minmax_within_decade",
        description="Normalization strategy for cultural_resonance_score",
    )
    feature_vector: List[str] = Field(
        ...,
        description="Columns included in the resonance feature vector",
    )
    top_resonance_songs_per_decade: int = Field(5, ge=1)


class OutputConfig(BaseModel):
    """Output and logging configuration."""

    output_dir: str = Field("outputs")
    save_visualizations: bool = Field(True)
    visualization_format: str = Field("png")
    visualization_scale: float = Field(2.0, ge=0.1)
    report_format: str = Field("html")
    log_level: str = Field(
        "INFO",
        description="Console log level: DEBUG, INFO, WARNING, ERROR",
    )


class ProjectConfig(BaseModel):
    """Top-level project configuration object."""

    dataset: DatasetConfig
    lyrics: LyricsConfig
    preprocessing: PreprocessingConfig
    analysis: AnalysisConfig
    cultural_metrics: CulturalMetricsConfig
    output: OutputConfig


# ── Load / Hash Utilities ─────────────────────────────────────────────────────


def load_config(path: Path | str | None = None) -> ProjectConfig:
    """
    Load the project configuration from YAML into a ProjectConfig object.

    Args:
        path: Optional override for the config file path. If None, the default
              config/project_config.yaml relative to project root is used.

    Returns:
        ProjectConfig instance.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValidationError:   if the YAML cannot be parsed into ProjectConfig.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.debug("Loading config from {}", config_path)

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        cfg = ProjectConfig.model_validate(raw)
    except ValidationError as exc:
        logger.error("Failed to validate ProjectConfig from {}", config_path)
        raise

    logger.info(
        "Config loaded: {}–{} ({} decade buckets)",
        cfg.dataset.sample_start_year,
        cfg.dataset.sample_end_year,
        len(cfg.dataset.decade_buckets),
    )
    return cfg


def config_hash(config: ProjectConfig) -> str:
    """
    Compute an 8-character MD5 hash of the serialized ProjectConfig.

    This hash is stored in each stage's sentinel file to detect config changes
    between runs. If the hash differs from the current config, the stage
    should be re-run even if its sentinel exists.

    Args:
        config: ProjectConfig instance

    Returns:
        8-character lowercase hex string
    """
    serialized = json.dumps(
        config.model_dump(),
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:8]


def sentinel_config_matches(sentinel_path: Path, current_config: ProjectConfig) -> bool:
    """
    Return True if the sentinel file exists AND its stored config_hash matches
    the hash of the current ProjectConfig.

    Intended usage in a stage's run() function:

        from src.pipeline.config_loader import sentinel_config_matches

        if sentinel_config_matches(SENTINEL_PATH, config) and not force:
            logger.info("Skipping stage — sentinel up-to-date: {}", SENTINEL_PATH)
            return {...}

    Args:
        sentinel_path: Path to the stage's sentinel JSON file.
        current_config: ProjectConfig object for the current run.

    Returns:
        True if sentinel exists and hashes match, False otherwise.
    """
    if not sentinel_path.exists():
        return False

    try:
        data = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Sentinel file is missing or corrupted: {}", sentinel_path)
        return False

    stored_hash = data.get("config_hash")
    current_hash = config_hash(current_config)

    if stored_hash is None:
        logger.warning(
            "Sentinel file {} has no config_hash field; treating as outdated.",
            sentinel_path,
        )
        return False

    if stored_hash != current_hash:
        logger.warning(
            "Config changed since last run (stored: {}, current: {}) for {}",
            stored_hash,
            current_hash,
            sentinel_path,
        )
        return False

    return True
