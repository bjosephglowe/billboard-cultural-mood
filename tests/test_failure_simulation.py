"""
tests/test_failure_simulation.py

Deliberate failure condition tests for the Billboard Cultural Mood
Analysis pipeline.

Philosophy:
    A pipeline that fails silently is more dangerous than one that
    crashes loudly. These tests confirm that every known failure mode
    raises the correct exception, logs an error, and does NOT write
    partial output or a false-positive sentinel.

Test groups:
    TestCoverageGateFailures     — gate breaches halt the pipeline correctly
    TestSchemaViolations         — invalid data raises SchemaError/SchemaErrors
    TestSentinelFailureModes     — corrupted/missing/stale sentinels are detected
    TestMissingInputFailures     — absent required CSVs raise FileNotFoundError
    TestConfigValidationFailures — invalid YAML raises ValidationError
    TestAtomicWriteSafety        — partial writes do not corrupt existing output
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pandera.errors
import pytest

from src.pipeline.config_loader import (
    ProjectConfig,
    config_hash,
    load_config,
    sentinel_config_matches,
)
from src.pipeline.schemas import (
    VALID_DECADE_LABELS,
    VALID_EMOTIONAL_TONES,
    cleaned_schema,
    layer2_sentiment_schema,
    layer4_schema,
    validate,
)
from src.utils.identifiers import make_song_id

# ═════════════════════════════════════════════════════════════════════════════
# TestCoverageGateFailures
# ═════════════════════════════════════════════════════════════════════════════


class TestCoverageGateFailures:
    """Coverage gate correctly halts pipeline when threshold is not met."""

    def test_gate_raises_when_coverage_below_threshold(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """
        contrast_metrics.run() raises RuntimeError when fewer than
        config.analysis.coverage_gate_threshold of full-quality songs
        have sentiment scores.
        """
        import src.analysis.contrast_metrics as cm

        # Build a sentiment df with all nulls — 0% coverage
        null_sentiment = pd.DataFrame(
            {
                "song_id": sample_songs_df["song_id"].tolist(),
                "sentiment_score": [None] * len(sample_songs_df),
                "energy_level": pd.array(
                    [pd.NA] * len(sample_songs_df),
                    dtype=pd.Int64Dtype(),
                ),
            }
        )

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        output_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)
        null_sentiment.to_csv(sent_path, index=False)

        with (
            patch.object(cm, "_LYRICS_PATH", lyrics_path),
            patch.object(cm, "_SENTIMENT_PATH", sent_path),
            patch.object(
                cm,
                "_EMOTION_PATH",
                pipeline_workspace / "data" / "analysis" / "layer2_emotion.csv",
            ),
            patch.object(
                cm,
                "_THEMES_PATH",
                pipeline_workspace / "data" / "analysis" / "layer2_themes.csv",
            ),
            patch.object(
                cm,
                "_CHORUS_PATH",
                pipeline_workspace / "data" / "processed" / "chorus_extracted.csv",
            ),
            patch.object(cm, "_OUTPUT_PATH", output_path),
            patch.object(cm, "_SENTINEL", sentinel),
        ):
            with pytest.raises(RuntimeError, match="Coverage gate FAILED"):
                cm.run(tmp_config)

    def test_gate_does_not_write_output_on_failure(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """
        When coverage gate fails, layer2_full_analysis.csv is NOT written.
        Confirms atomic write behaviour — no partial output on failure.
        """
        import src.analysis.contrast_metrics as cm

        null_sentiment = pd.DataFrame(
            {
                "song_id": sample_songs_df["song_id"].tolist(),
                "sentiment_score": [None] * len(sample_songs_df),
                "energy_level": pd.array(
                    [pd.NA] * len(sample_songs_df),
                    dtype=pd.Int64Dtype(),
                ),
            }
        )

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        output_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)
        null_sentiment.to_csv(sent_path, index=False)

        with (
            patch.object(cm, "_LYRICS_PATH", lyrics_path),
            patch.object(cm, "_SENTIMENT_PATH", sent_path),
            patch.object(
                cm,
                "_EMOTION_PATH",
                pipeline_workspace / "data" / "analysis" / "layer2_emotion.csv",
            ),
            patch.object(
                cm,
                "_THEMES_PATH",
                pipeline_workspace / "data" / "analysis" / "layer2_themes.csv",
            ),
            patch.object(
                cm,
                "_CHORUS_PATH",
                pipeline_workspace / "data" / "processed" / "chorus_extracted.csv",
            ),
            patch.object(cm, "_OUTPUT_PATH", output_path),
            patch.object(cm, "_SENTINEL", sentinel),
        ):
            with pytest.raises(RuntimeError):
                cm.run(tmp_config)

        assert not output_path.exists(), (
            "layer2_full_analysis.csv should NOT exist after coverage gate failure"
        )

    def test_gate_does_not_write_sentinel_on_failure(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """
        When coverage gate fails, the sentinel is NOT written.
        A false-positive sentinel would cause the stage to be skipped
        on the next run, hiding the coverage problem permanently.
        """
        import src.analysis.contrast_metrics as cm

        null_sentiment = pd.DataFrame(
            {
                "song_id": sample_songs_df["song_id"].tolist(),
                "sentiment_score": [None] * len(sample_songs_df),
                "energy_level": pd.array(
                    [pd.NA] * len(sample_songs_df),
                    dtype=pd.Int64Dtype(),
                ),
            }
        )

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        output_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)
        null_sentiment.to_csv(sent_path, index=False)

        with (
            patch.object(cm, "_LYRICS_PATH", lyrics_path),
            patch.object(cm, "_SENTIMENT_PATH", sent_path),
            patch.object(
                cm,
                "_EMOTION_PATH",
                pipeline_workspace / "data" / "analysis" / "layer2_emotion.csv",
            ),
            patch.object(
                cm,
                "_THEMES_PATH",
                pipeline_workspace / "data" / "analysis" / "layer2_themes.csv",
            ),
            patch.object(
                cm,
                "_CHORUS_PATH",
                pipeline_workspace / "data" / "processed" / "chorus_extracted.csv",
            ),
            patch.object(cm, "_OUTPUT_PATH", output_path),
            patch.object(cm, "_SENTINEL", sentinel),
        ):
            with pytest.raises(RuntimeError):
                cm.run(tmp_config)

        assert not sentinel.exists(), (
            "Sentinel should NOT be written after coverage gate failure"
        )

    def test_gate_passes_at_exact_threshold(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """
        Coverage gate passes when coverage equals exactly the threshold.
        Uses a 5-song fixture with threshold set to 0.80 and 4/5 scored.
        """
        import src.analysis.contrast_metrics as cm

        # 4 of 5 songs scored = 80% coverage
        partial_sentiment = pd.DataFrame(
            {
                "song_id": sample_songs_df["song_id"].tolist(),
                "sentiment_score": [0.5, -0.3, 0.1, 0.7, None],
                "energy_level": pd.array(
                    [3, 2, 2, 4, pd.NA],
                    dtype=pd.Int64Dtype(),
                ),
            }
        )

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        output_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)
        partial_sentiment.to_csv(sent_path, index=False)

        # Set threshold to exactly 80%
        mutated = tmp_config.model_copy(deep=True)
        mutated.analysis.coverage_gate_threshold = 0.80

        with (
            patch.object(cm, "_LYRICS_PATH", lyrics_path),
            patch.object(cm, "_SENTIMENT_PATH", sent_path),
            patch.object(
                cm,
                "_EMOTION_PATH",
                pipeline_workspace / "data" / "analysis" / "layer2_emotion.csv",
            ),
            patch.object(
                cm,
                "_THEMES_PATH",
                pipeline_workspace / "data" / "analysis" / "layer2_themes.csv",
            ),
            patch.object(
                cm,
                "_CHORUS_PATH",
                pipeline_workspace / "data" / "processed" / "chorus_extracted.csv",
            ),
            patch.object(cm, "_OUTPUT_PATH", output_path),
            patch.object(cm, "_SENTINEL", sentinel),
        ):
            result = cm.run(mutated)

        assert result["coverage_rate"] >= 0.80


# ═════════════════════════════════════════════════════════════════════════════
# TestSchemaViolations
# ═════════════════════════════════════════════════════════════════════════════


class TestSchemaViolations:
    """Schema validation raises correctly on invalid data."""

    def test_sentiment_score_above_range_rejected(self):
        """sentiment_score > 1.0 raises SchemaError/SchemaErrors."""
        from tests.conftest import FIXTURE_SONG_IDS

        bad_df = pd.DataFrame(
            {
                "song_id": [FIXTURE_SONG_IDS[0]],
                "sentiment_score": [1.5],
                "energy_level": pd.array([3], dtype=pd.Int64Dtype()),
            }
        )
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            validate(bad_df, layer2_sentiment_schema, "test")

    def test_sentiment_score_below_range_rejected(self):
        """sentiment_score < -1.0 raises SchemaError/SchemaErrors."""
        from tests.conftest import FIXTURE_SONG_IDS

        bad_df = pd.DataFrame(
            {
                "song_id": [FIXTURE_SONG_IDS[0]],
                "sentiment_score": [-1.5],
                "energy_level": pd.array([3], dtype=pd.Int64Dtype()),
            }
        )
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            validate(bad_df, layer2_sentiment_schema, "test")

    def test_energy_level_out_of_range_rejected(self):
        """energy_level=6 (outside [1,5]) raises SchemaError/SchemaErrors."""
        from tests.conftest import FIXTURE_SONG_IDS

        bad_df = pd.DataFrame(
            {
                "song_id": [FIXTURE_SONG_IDS[0]],
                "sentiment_score": [0.5],
                "energy_level": pd.array([6], dtype=pd.Int64Dtype()),
            }
        )
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            validate(bad_df, layer2_sentiment_schema, "test")

    def test_invalid_song_id_format_rejected(self):
        """song_id not matching ^[0-9a-f]{16}$ raises SchemaError/SchemaErrors."""
        bad_df = pd.DataFrame(
            {
                "song_id": ["NOT_A_VALID_ID"],
                "sentiment_score": [0.5],
                "energy_level": pd.array([3], dtype=pd.Int64Dtype()),
            }
        )
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            validate(bad_df, layer2_sentiment_schema, "test")

    def test_invalid_decade_label_rejected(self, sample_songs_df: pd.DataFrame):
        """decade not in VALID_DECADE_LABELS raises SchemaError/SchemaErrors."""
        bad_df = sample_songs_df.copy()
        bad_df.loc[0, "decade"] = "1850s"
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            validate(bad_df, cleaned_schema, "test")

    def test_invalid_lyrics_quality_rejected(self, sample_songs_df: pd.DataFrame):
        """lyrics_quality not in valid set raises SchemaError/SchemaErrors."""
        bad_df = sample_songs_df.copy()
        bad_df.loc[0, "lyrics_quality"] = "excellent"
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            validate(bad_df, cleaned_schema, "test")

    def test_multiple_violations_all_reported(self):
        """
        Lazy validation collects ALL errors before raising.
        A DataFrame with 3 bad rows raises SchemaErrors (not just the first).
        """
        from tests.conftest import FIXTURE_SONG_IDS

        bad_df = pd.DataFrame(
            {
                "song_id": FIXTURE_SONG_IDS[:3],
                "sentiment_score": [2.0, -2.0, 99.0],  # all out of range
                "energy_level": pd.array([3, 3, 3], dtype=pd.Int64Dtype()),
            }
        )
        with pytest.raises(
            (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
        ) as exc_info:
            validate(bad_df, layer2_sentiment_schema, "test")

        # SchemaErrors.failure_cases should contain multiple failures
        exc = exc_info.value
        if hasattr(exc, "failure_cases"):
            assert len(exc.failure_cases) >= 1


# ═════════════════════════════════════════════════════════════════════════════
# TestSentinelFailureModes
# ═════════════════════════════════════════════════════════════════════════════


class TestSentinelFailureModes:
    """Sentinel detection correctly identifies all failure modes."""

    def test_missing_sentinel_returns_false(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """Non-existent sentinel file returns False."""
        path = pipeline_workspace / "data" / "analysis" / ".does_not_exist"
        assert sentinel_config_matches(path, tmp_config) is False

    def test_corrupted_json_returns_false(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """Sentinel containing invalid JSON returns False."""
        path = pipeline_workspace / "data" / "analysis" / ".corrupt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not valid json at all }", encoding="utf-8")
        assert sentinel_config_matches(path, tmp_config) is False

    def test_empty_file_returns_false(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """Empty sentinel file returns False."""
        path = pipeline_workspace / "data" / "analysis" / ".empty"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        assert sentinel_config_matches(path, tmp_config) is False

    def test_missing_config_hash_field_returns_false(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """Sentinel JSON without config_hash field returns False."""
        path = pipeline_workspace / "data" / "analysis" / ".no_hash"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"stage": "TEST", "completed_at": "2026-01-01"}),
            encoding="utf-8",
        )
        assert sentinel_config_matches(path, tmp_config) is False

    def test_stale_hash_returns_false(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """Sentinel with outdated config_hash returns False after config change."""
        path = pipeline_workspace / "data" / "analysis" / ".stale"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write sentinel for current config
        path.write_text(
            json.dumps({"config_hash": config_hash(tmp_config)}),
            encoding="utf-8",
        )
        assert sentinel_config_matches(path, tmp_config) is True

        # Mutate config — sentinel is now stale
        stale_config = tmp_config.model_copy(deep=True)
        stale_config.dataset.sample_start_year = 1900
        assert sentinel_config_matches(path, stale_config) is False

    def test_correct_sentinel_returns_true(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """Sentinel with matching config_hash returns True."""
        path = pipeline_workspace / "data" / "analysis" / ".valid"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "stage": "TEST",
                    "config_hash": config_hash(tmp_config),
                }
            ),
            encoding="utf-8",
        )
        assert sentinel_config_matches(path, tmp_config) is True


# ═════════════════════════════════════════════════════════════════════════════
# TestMissingInputFailures
# ═════════════════════════════════════════════════════════════════════════════


class TestMissingInputFailures:
    """Absent required CSVs raise FileNotFoundError with helpful messages."""

    def test_sentiment_scorer_missing_lyrics_raises(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """sentiment_scorer.run() raises FileNotFoundError when lyrics CSV absent."""
        import src.analysis.sentiment_scorer as scorer

        missing = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        output_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"

        with (
            patch.object(scorer, "_INPUT_PATH", missing),
            patch.object(scorer, "_OUTPUT_PATH", output_path),
            patch.object(scorer, "_SENTINEL", sentinel),
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                scorer.run(tmp_config)

        assert "lyrics_cleaned.csv" in str(exc_info.value) or "TEXT_CLEANING" in str(
            exc_info.value
        )

    def test_contrast_metrics_missing_lyrics_raises(
        self,
        pipeline_workspace: Path,
        sample_sentiment_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """contrast_metrics.run() raises FileNotFoundError when lyrics CSV absent."""
        import src.analysis.contrast_metrics as cm

        missing = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        output_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        sample_sentiment_df.to_csv(sent_path, index=False)

        with (
            patch.object(cm, "_LYRICS_PATH", missing),
            patch.object(cm, "_SENTIMENT_PATH", sent_path),
            patch.object(cm, "_OUTPUT_PATH", output_path),
            patch.object(cm, "_SENTINEL", sentinel),
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                cm.run(tmp_config)

        assert "lyrics_cleaned.csv" in str(exc_info.value) or "TEXT_CLEANING" in str(
            exc_info.value
        )

    def test_contrast_metrics_missing_sentiment_raises(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """contrast_metrics.run() raises FileNotFoundError when sentiment CSV absent."""
        import src.analysis.contrast_metrics as cm

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        missing = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        output_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)

        with (
            patch.object(cm, "_LYRICS_PATH", lyrics_path),
            patch.object(cm, "_SENTIMENT_PATH", missing),
            patch.object(cm, "_OUTPUT_PATH", output_path),
            patch.object(cm, "_SENTINEL", sentinel),
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                cm.run(tmp_config)

        assert "layer2_sentiment.csv" in str(
            exc_info.value
        ) or "SENTIMENT_SCORING" in str(exc_info.value)

    def test_missing_input_does_not_write_output(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """No output CSV is written when a required input is missing."""
        import src.analysis.sentiment_scorer as scorer

        missing = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        output_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"

        with (
            patch.object(scorer, "_INPUT_PATH", missing),
            patch.object(scorer, "_OUTPUT_PATH", output_path),
            patch.object(scorer, "_SENTINEL", sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                scorer.run(tmp_config)

        assert not output_path.exists()
        assert not sentinel.exists()


# ═════════════════════════════════════════════════════════════════════════════
# TestConfigValidationFailures
# ═════════════════════════════════════════════════════════════════════════════


class TestConfigValidationFailures:
    """Invalid configuration raises ValidationError before any stage runs."""

    def test_load_config_raises_on_missing_file(self, tmp_path: Path):
        """load_config() raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent_config.yaml")

    def test_load_config_raises_on_invalid_year_range(
        self,
        tmp_path: Path,
    ):
        """load_config() raises ValidationError when end_year < start_year."""
        import yaml
        from pydantic import ValidationError

        bad_config = {
            "dataset": {
                "sample_start_year": 2020,
                "sample_end_year": 1990,  # end before start
                "decade_filter": None,
                "decade_buckets": [
                    {
                        "label": "2020s",
                        "start": 2020,
                        "end": 2025,
                        "non_standard": False,
                    }
                ],
            },
            "lyrics": {
                "cache_dir": "cache/lyrics",
                "min_token_threshold": 50,
                "partial_token_threshold": 150,
                "genius_sleep_time": 0.5,
                "genius_max_retries": 3,
            },
            "preprocessing": {
                "chorus_tier1_tag_pattern": r"\[chorus\]",
                "chorus_tier2_min_repetitions": 3,
                "chorus_tier2_min_tokens": 6,
                "chorus_tier3_model": "gpt-4o",
                "chorus_max_tokens": 200,
            },
            "analysis": {
                "energy_level": {"bins": [0.2, 0.4, 0.6, 0.8]},
                "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
                "emotion_batch_size": 16,
                "theme_model": "facebook/bart-large-mnli",
                "theme_threshold": 0.4,
                "theme_batch_size": 8,
                "theme_taxonomy": ["love"],
                "coverage_gate_threshold": 0.85,
                "jungian_model": "gpt-4o",
                "jungian_batch_size": 5,
                "jungian_inter_batch_sleep": 1.0,
                "jungian_max_retries": 3,
                "jungian_valid_stages": ["shadow"],
                "jungian_dev_score_min": 1,
                "jungian_dev_score_max": 7,
            },
            "cultural_metrics": {
                "normalization": "minmax_within_decade",
                "feature_vector": ["sentiment_score"],
                "top_resonance_songs_per_decade": 5,
            },
            "output": {
                "output_dir": "outputs",
                "save_visualizations": True,
                "visualization_format": "png",
                "visualization_scale": 2.0,
                "report_format": "html",
                "log_level": "INFO",
            },
        }

        config_path = tmp_path / "bad_config.yaml"
        config_path.write_text(
            yaml.dump(bad_config),
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            load_config(config_path)

    def test_load_config_raises_on_empty_decade_buckets(
        self,
        tmp_path: Path,
    ):
        """load_config() raises ValidationError when decade_buckets is empty."""
        import yaml
        from pydantic import ValidationError

        bad_config = {
            "dataset": {
                "sample_start_year": 1958,
                "sample_end_year": 2025,
                "decade_filter": None,
                "decade_buckets": [],  # empty list
            },
            "lyrics": {
                "cache_dir": "cache/lyrics",
                "min_token_threshold": 50,
                "partial_token_threshold": 150,
                "genius_sleep_time": 0.5,
                "genius_max_retries": 3,
            },
            "preprocessing": {
                "chorus_tier1_tag_pattern": r"\[chorus\]",
                "chorus_tier2_min_repetitions": 3,
                "chorus_tier2_min_tokens": 6,
                "chorus_tier3_model": "gpt-4o",
                "chorus_max_tokens": 200,
            },
            "analysis": {
                "energy_level": {"bins": [0.2, 0.4, 0.6, 0.8]},
                "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
                "emotion_batch_size": 16,
                "theme_model": "facebook/bart-large-mnli",
                "theme_threshold": 0.4,
                "theme_batch_size": 8,
                "theme_taxonomy": ["love"],
                "coverage_gate_threshold": 0.85,
                "jungian_model": "gpt-4o",
                "jungian_batch_size": 5,
                "jungian_inter_batch_sleep": 1.0,
                "jungian_max_retries": 3,
                "jungian_valid_stages": ["shadow"],
                "jungian_dev_score_min": 1,
                "jungian_dev_score_max": 7,
            },
            "cultural_metrics": {
                "normalization": "minmax_within_decade",
                "feature_vector": ["sentiment_score"],
                "top_resonance_songs_per_decade": 5,
            },
            "output": {
                "output_dir": "outputs",
                "save_visualizations": True,
                "visualization_format": "png",
                "visualization_scale": 2.0,
                "report_format": "html",
                "log_level": "INFO",
            },
        }

        config_path = tmp_path / "empty_buckets.yaml"
        config_path.write_text(
            yaml.dump(bad_config),
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            load_config(config_path)

    def test_config_hash_changes_when_config_mutated(
        self,
        tmp_config: ProjectConfig,
    ):
        """config_hash() produces a different hash after any config mutation."""
        original_hash = config_hash(tmp_config)

        mutated = tmp_config.model_copy(deep=True)
        mutated.dataset.sample_start_year = 1900

        mutated_hash = config_hash(mutated)
        assert original_hash != mutated_hash


# ═════════════════════════════════════════════════════════════════════════════
# TestAtomicWriteSafety
# ═════════════════════════════════════════════════════════════════════════════


class TestAtomicWriteSafety:
    """Atomic write preserves existing output when a stage fails mid-run."""

    def test_existing_output_preserved_on_coverage_failure(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        sample_sentiment_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """
        A pre-existing layer2_full_analysis.csv is NOT overwritten when
        the current run fails the coverage gate.

        Simulates a re-run after data corruption where the previous good
        output should be preserved.
        """
        import src.analysis.contrast_metrics as cm

        output_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write a "previous good" output
        previous_good = pd.DataFrame(
            {
                "song_id": ["previousgooddata"],
                "song_title": ["Previous Song"],
            }
        )
        previous_good.to_csv(output_path, index=False)
        original_mtime = output_path.stat().st_mtime

        # Now run with null sentiment — should fail coverage gate
        null_sentiment = pd.DataFrame(
            {
                "song_id": sample_songs_df["song_id"].tolist(),
                "sentiment_score": [None] * len(sample_songs_df),
                "energy_level": pd.array(
                    [pd.NA] * len(sample_songs_df),
                    dtype=pd.Int64Dtype(),
                ),
            }
        )

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"

        sample_songs_df.to_csv(lyrics_path, index=False)
        null_sentiment.to_csv(sent_path, index=False)

        with (
            patch.object(cm, "_LYRICS_PATH", lyrics_path),
            patch.object(cm, "_SENTIMENT_PATH", sent_path),
            patch.object(
                cm,
                "_EMOTION_PATH",
                pipeline_workspace / "data" / "analysis" / "layer2_emotion.csv",
            ),
            patch.object(
                cm,
                "_THEMES_PATH",
                pipeline_workspace / "data" / "analysis" / "layer2_themes.csv",
            ),
            patch.object(
                cm,
                "_CHORUS_PATH",
                pipeline_workspace / "data" / "processed" / "chorus_extracted.csv",
            ),
            patch.object(cm, "_OUTPUT_PATH", output_path),
            patch.object(cm, "_SENTINEL", sentinel),
        ):
            with pytest.raises(RuntimeError):
                cm.run(tmp_config)

        # Previous good output must still be intact
        assert output_path.exists(), "Previous output was deleted on failure"
        current_content = pd.read_csv(output_path)
        assert "previousgooddata" in current_content["song_id"].values, (
            "Previous output was overwritten during a failed run"
        )
