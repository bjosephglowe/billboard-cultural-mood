"""
tests/test_integration.py

Primary smoke test for the Billboard Cultural Mood Analysis pipeline.

Tests the full critical path using fixture data — no external API calls,
no real Billboard or Genius requests.

Test groups:
    TestConfigLoader      — ProjectConfig loads and validates correctly
    TestIdentifiers       — song_id generation and validation
    TestSchemas           — Pandera schema validation accepts valid fixtures
    TestSentimentScorer   — Stage 5 run() produces valid output end-to-end
    TestContrastMetrics   — Stage 8 run() produces valid master CSV end-to-end
    TestSentinelLogic     — Sentinels are written, read, and matched correctly
    TestMainDryRun        — main.py --dry-run exits with expected code
    TestDesignSystem      — _design_system constants are internally consistent

All tests use tmp_path isolation — no writes to real data/ or outputs/ dirs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
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
    VALID_JUNG_STAGES,
    cleaned_schema,
    decade_cmi_schema,
    layer2_sentiment_schema,
    layer4_schema,
    layer5_schema,
    validate,
)
from src.utils.identifiers import make_song_id, validate_song_id

# ═════════════════════════════════════════════════════════════════════════════
# TestConfigLoader
# ═════════════════════════════════════════════════════════════════════════════


class TestConfigLoader:
    """Verify ProjectConfig loads cleanly and all fields are accessible."""

    def test_config_loads(self, project_config: ProjectConfig):
        """load_config() returns a ProjectConfig instance."""
        assert isinstance(project_config, ProjectConfig)

    def test_dataset_year_range(self, project_config: ProjectConfig):
        """Dataset year range is within expected historical bounds."""
        assert project_config.dataset.sample_start_year >= 1958
        assert project_config.dataset.sample_end_year <= 2030
        assert (
            project_config.dataset.sample_end_year
            >= project_config.dataset.sample_start_year
        )

    def test_decade_buckets_non_empty(self, project_config: ProjectConfig):
        """At least one decade bucket is defined."""
        assert len(project_config.dataset.decade_buckets) >= 1

    def test_decade_bucket_labels_match_schema(self, project_config: ProjectConfig):
        """All decade bucket labels in config exist in VALID_DECADE_LABELS."""
        config_labels = {b.label for b in project_config.dataset.decade_buckets}
        schema_labels = set(VALID_DECADE_LABELS)
        unknown = config_labels - schema_labels
        assert not unknown, f"Config decade labels not in schema: {unknown}"

    def test_coverage_gate_threshold_valid(self, project_config: ProjectConfig):
        """Coverage gate threshold is a float in (0, 1]."""
        t = project_config.analysis.coverage_gate_threshold
        assert 0.0 < t <= 1.0

    def test_energy_bins_sorted(self, project_config: ProjectConfig):
        """Energy level bins are sorted ascending."""
        bins = project_config.analysis.energy_level.bins
        assert bins == sorted(bins)

    def test_config_hash_is_8_chars(self, project_config: ProjectConfig):
        """config_hash() returns an 8-character lowercase hex string."""
        h = config_hash(project_config)
        assert len(h) == 8
        assert h == h.lower()
        assert all(c in "0123456789abcdef" for c in h)

    def test_config_hash_is_deterministic(self, project_config: ProjectConfig):
        """Same config always produces the same hash."""
        h1 = config_hash(project_config)
        h2 = config_hash(project_config)
        assert h1 == h2

    def test_theme_taxonomy_non_empty(self, project_config: ProjectConfig):
        """Theme taxonomy contains at least one label."""
        assert len(project_config.analysis.theme_taxonomy) >= 1

    def test_jungian_valid_stages_match_schema(self, project_config: ProjectConfig):
        """All Jungian valid stages in config exist in VALID_JUNG_STAGES."""
        config_stages = set(project_config.analysis.jungian_valid_stages)
        schema_stages = set(VALID_JUNG_STAGES)
        unknown = config_stages - schema_stages
        assert not unknown, f"Config Jungian stages not in schema: {unknown}"


# ═════════════════════════════════════════════════════════════════════════════
# TestIdentifiers
# ═════════════════════════════════════════════════════════════════════════════


class TestIdentifiers:
    """Verify song_id generation and validation."""

    def test_make_song_id_returns_16_char_hex(self):
        """make_song_id() returns a 16-character lowercase hex string."""
        sid = make_song_id("The Beatles", "Hey Jude", 1968)
        assert len(sid) == 16
        assert sid == sid.lower()
        assert all(c in "0123456789abcdef" for c in sid)

    def test_make_song_id_is_deterministic(self):
        """Same inputs always produce the same song_id."""
        sid1 = make_song_id("The Beatles", "Hey Jude", 1968)
        sid2 = make_song_id("The Beatles", "Hey Jude", 1968)
        assert sid1 == sid2

    def test_make_song_id_differs_by_artist(self):
        """Different artists produce different song_ids for same title/year."""
        sid1 = make_song_id("Artist A", "Song", 2000)
        sid2 = make_song_id("Artist B", "Song", 2000)
        assert sid1 != sid2

    def test_make_song_id_differs_by_year(self):
        """Different years produce different song_ids for same artist/title."""
        sid1 = make_song_id("Artist", "Song", 1990)
        sid2 = make_song_id("Artist", "Song", 1991)
        assert sid1 != sid2

    def test_validate_song_id_accepts_valid(self):
        """validate_song_id() returns True for a valid make_song_id() output."""
        sid = make_song_id("Beyonce", "Crazy in Love", 2003)
        assert validate_song_id(sid) is True

    def test_validate_song_id_rejects_short(self):
        """validate_song_id() returns False for a string shorter than 16 chars."""
        assert validate_song_id("abc123") is False

    def test_validate_song_id_rejects_non_hex(self):
        """validate_song_id() returns False for a non-hex string."""
        assert validate_song_id("zzzzzzzzzzzzzzzz") is False

    def test_fixture_song_ids_all_valid(self):
        """All fixture song_ids from conftest pass validate_song_id()."""
        from tests.conftest import FIXTURE_SONG_IDS

        for sid in FIXTURE_SONG_IDS:
            assert validate_song_id(sid), f"Invalid fixture song_id: {sid}"


# ═════════════════════════════════════════════════════════════════════════════
# TestSchemas
# ═════════════════════════════════════════════════════════════════════════════


class TestSchemas:
    """Verify Pandera schemas accept valid fixture DataFrames."""

    def test_cleaned_schema_accepts_fixture(self, sample_songs_df: pd.DataFrame):
        """cleaned_schema validates sample_songs_df without errors."""
        result = validate(sample_songs_df, cleaned_schema, "test_cleaned")
        assert len(result) == 5

    def test_sentiment_schema_accepts_fixture(self, sample_sentiment_df: pd.DataFrame):
        """layer2_sentiment_schema validates sample_sentiment_df without errors."""
        result = validate(
            sample_sentiment_df, layer2_sentiment_schema, "test_sentiment"
        )
        assert len(result) == 5

    def test_layer4_schema_accepts_fixture(self, sample_layer4_df: pd.DataFrame):
        """layer4_schema validates sample_layer4_df without errors."""
        result = validate(sample_layer4_df, layer4_schema, "test_layer4")
        assert len(result) == 5

    def test_decade_cmi_schema_accepts_fixture(self, sample_cmi_df: pd.DataFrame):
        """decade_cmi_schema validates sample_cmi_df without errors."""
        result = validate(sample_cmi_df, decade_cmi_schema, "test_cmi")
        assert len(result) == 7

    def test_schema_rejects_invalid_sentiment_range(self):
        """layer2_sentiment_schema rejects sentiment_score outside [-1, 1]."""
        import pandera.errors

        from tests.conftest import FIXTURE_SONG_IDS

        bad_df = pd.DataFrame(
            {
                "song_id": [FIXTURE_SONG_IDS[0]],
                "sentiment_score": [99.0],
                "energy_level": pd.array([3], dtype=pd.Int64Dtype()),
            }
        )
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            validate(bad_df, layer2_sentiment_schema, "test_bad_sentiment")

    def test_schema_rejects_invalid_decade_label(self):
        """cleaned_schema rejects an unknown decade label."""
        import pandera.errors

        from tests.conftest import FIXTURE_SONG_IDS

        bad_df = pd.DataFrame(
            {
                "song_id": [FIXTURE_SONG_IDS[0]],
                "song_title": ["Test Song"],
                "artist": ["Test Artist"],
                "year": [2000],
                "decade": ["9999s"],  # invalid
                "lyrics_clean": ["test lyrics"],
                "lyrics_verse_only": ["test lyrics"],
                "token_count": [2],
                "lyrics_quality": ["full"],
                "narrative_perspective": ["first_person"],
                "has_section_tags": [False],
                "section_count": [0],
            }
        )
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            validate(bad_df, cleaned_schema, "test_bad_decade")


# ═════════════════════════════════════════════════════════════════════════════
# TestSentimentScorer
# ═════════════════════════════════════════════════════════════════════════════


class TestSentimentScorer:
    """End-to-end tests for Stage 5 — Sentiment Scoring."""

    def test_run_produces_output_csv(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """sentiment_scorer.run() writes layer2_sentiment.csv to expected path."""
        import src.analysis.sentiment_scorer as scorer

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        output_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)

        with (
            patch.object(scorer, "_INPUT_PATH", lyrics_path),
            patch.object(scorer, "_OUTPUT_PATH", output_path),
            patch.object(scorer, "_SENTINEL", sentinel),
        ):
            result = scorer.run(tmp_config)

        assert output_path.exists(), "layer2_sentiment.csv was not written"
        assert result["scored_count"] > 0

    def test_run_output_passes_schema(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """sentiment_scorer.run() output passes layer2_sentiment_schema."""
        import src.analysis.sentiment_scorer as scorer

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        output_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)

        with (
            patch.object(scorer, "_INPUT_PATH", lyrics_path),
            patch.object(scorer, "_OUTPUT_PATH", output_path),
            patch.object(scorer, "_SENTINEL", sentinel),
        ):
            scorer.run(tmp_config)

        df = pd.read_csv(output_path, dtype={"song_id": str})
        validated = validate(df, layer2_sentiment_schema, "smoke_sentiment")
        assert len(validated) == 5

    def test_run_writes_sentinel(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """sentiment_scorer.run() writes a sentinel with correct config_hash."""
        import src.analysis.sentiment_scorer as scorer

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        output_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)

        with (
            patch.object(scorer, "_INPUT_PATH", lyrics_path),
            patch.object(scorer, "_OUTPUT_PATH", output_path),
            patch.object(scorer, "_SENTINEL", sentinel),
        ):
            scorer.run(tmp_config)

        assert sentinel.exists(), "Sentinel was not written"
        data = json.loads(sentinel.read_text())
        assert data["stage"] == "SENTIMENT_SCORING"
        assert data["config_hash"] == config_hash(tmp_config)
        assert data["scored"] > 0

    def test_run_scores_all_full_quality_songs(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """All 5 full-quality fixture songs receive non-null sentiment scores."""
        import src.analysis.sentiment_scorer as scorer

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        output_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)

        with (
            patch.object(scorer, "_INPUT_PATH", lyrics_path),
            patch.object(scorer, "_OUTPUT_PATH", output_path),
            patch.object(scorer, "_SENTINEL", sentinel),
        ):
            result = scorer.run(tmp_config)

        assert result["skipped_count"] == 0
        assert result["scored_count"] == 5

    def test_run_raises_on_missing_input(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """sentiment_scorer.run() raises FileNotFoundError if input CSV absent."""
        import src.analysis.sentiment_scorer as scorer

        missing_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        output_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"

        with (
            patch.object(scorer, "_INPUT_PATH", missing_path),
            patch.object(scorer, "_OUTPUT_PATH", output_path),
            patch.object(scorer, "_SENTINEL", sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                scorer.run(tmp_config)

    def test_missing_lyrics_song_gets_null_score(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """Songs with lyrics_quality='missing' receive null sentiment_score."""
        import src.analysis.sentiment_scorer as scorer

        df_with_missing = sample_songs_df.copy()
        df_with_missing.loc[0, "lyrics_quality"] = "missing"
        df_with_missing.loc[0, "lyrics_clean"] = None

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        output_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"

        df_with_missing.to_csv(lyrics_path, index=False)

        with (
            patch.object(scorer, "_INPUT_PATH", lyrics_path),
            patch.object(scorer, "_OUTPUT_PATH", output_path),
            patch.object(scorer, "_SENTINEL", sentinel),
        ):
            result = scorer.run(tmp_config)

        df = pd.read_csv(output_path, dtype={"song_id": str})
        missing_row = df[df["song_id"] == sample_songs_df.loc[0, "song_id"]]
        assert missing_row["sentiment_score"].isna().all()
        assert result["skipped_count"] == 1


# ═════════════════════════════════════════════════════════════════════════════
# TestContrastMetrics
# ═════════════════════════════════════════════════════════════════════════════


class TestContrastMetrics:
    """End-to-end tests for Stage 8 — Contrast Metrics."""

    def test_run_produces_master_csv(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        sample_sentiment_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """contrast_metrics.run() writes layer2_full_analysis.csv."""
        import src.analysis.contrast_metrics as cm

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        output_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)
        sample_sentiment_df.to_csv(sent_path, index=False)

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
            result = cm.run(tmp_config)

        assert output_path.exists(), "layer2_full_analysis.csv was not written"
        assert result["total_songs"] == 5

    def test_run_output_passes_layer4_schema(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        sample_sentiment_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """contrast_metrics.run() output passes layer4_schema validation."""
        import src.analysis.contrast_metrics as cm

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        output_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)
        sample_sentiment_df.to_csv(sent_path, index=False)

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
            cm.run(tmp_config)

        df = pd.read_csv(output_path, dtype={"song_id": str})
        validated = validate(df, layer4_schema, "smoke_layer4")
        assert len(validated) == 5

    def test_run_writes_sentinel_with_correct_hash(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        sample_sentiment_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """contrast_metrics.run() sentinel contains correct config_hash."""
        import src.analysis.contrast_metrics as cm

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        output_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)
        sample_sentiment_df.to_csv(sent_path, index=False)

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
            cm.run(tmp_config)

        assert sentinel.exists()
        data = json.loads(sentinel.read_text())
        assert data["stage"] == "CONTRAST_METRICS"
        assert data["config_hash"] == config_hash(tmp_config)
        assert data["total_songs"] == 5

    def test_run_raises_on_missing_lyrics(
        self,
        pipeline_workspace: Path,
        sample_sentiment_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """contrast_metrics.run() raises FileNotFoundError if lyrics CSV absent."""
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
            with pytest.raises(FileNotFoundError):
                cm.run(tmp_config)

    def test_coverage_gate_passes_with_full_data(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        sample_sentiment_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """Coverage gate does not raise when all full-quality songs are scored."""
        import src.analysis.contrast_metrics as cm

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        output_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)
        sample_sentiment_df.to_csv(sent_path, index=False)

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
            result = cm.run(tmp_config)

        assert result["coverage_rate"] >= tmp_config.analysis.coverage_gate_threshold


# ═════════════════════════════════════════════════════════════════════════════
# TestSentinelLogic
# ═════════════════════════════════════════════════════════════════════════════


class TestSentinelLogic:
    """Verify sentinel read/write and config-hash matching."""

    def test_sentinel_matches_after_write(
        self,
        pipeline_workspace: Path,
        sample_songs_df: pd.DataFrame,
        tmp_config: ProjectConfig,
    ):
        """sentinel_config_matches() returns True immediately after a stage run."""
        import src.analysis.sentiment_scorer as scorer

        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        output_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"

        sample_songs_df.to_csv(lyrics_path, index=False)

        with (
            patch.object(scorer, "_INPUT_PATH", lyrics_path),
            patch.object(scorer, "_OUTPUT_PATH", output_path),
            patch.object(scorer, "_SENTINEL", sentinel),
        ):
            scorer.run(tmp_config)

        assert sentinel_config_matches(sentinel, tmp_config) is True

    def test_sentinel_does_not_match_missing_file(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """sentinel_config_matches() returns False when sentinel file absent."""
        missing = pipeline_workspace / "data" / "analysis" / ".nonexistent_sentinel"
        assert sentinel_config_matches(missing, tmp_config) is False

    def test_sentinel_does_not_match_after_config_change(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """sentinel_config_matches() returns False when config changes."""
        sentinel = pipeline_workspace / "data" / "analysis" / ".test_sentinel"
        sentinel.parent.mkdir(parents=True, exist_ok=True)

        # Write sentinel for original config
        payload = {
            "stage": "TEST",
            "config_hash": config_hash(tmp_config),
        }
        sentinel.write_text(json.dumps(payload), encoding="utf-8")
        assert sentinel_config_matches(sentinel, tmp_config) is True

        # Mutate config — hash must change
        mutated = tmp_config.model_copy(deep=True)
        mutated.dataset.sample_start_year = 1900
        assert sentinel_config_matches(sentinel, mutated) is False

    def test_sentinel_does_not_match_corrupted_file(
        self,
        pipeline_workspace: Path,
        tmp_config: ProjectConfig,
    ):
        """sentinel_config_matches() returns False for a corrupted sentinel."""
        sentinel = pipeline_workspace / "data" / "analysis" / ".corrupt_sentinel"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("this is not valid json {{{", encoding="utf-8")
        assert sentinel_config_matches(sentinel, tmp_config) is False


# ═════════════════════════════════════════════════════════════════════════════
# TestMainDryRun
# ═════════════════════════════════════════════════════════════════════════════


class TestMainDryRun:
    """Verify main.py CLI behaviour."""

    def test_dry_run_exits_nonzero_before_stages_built(self):
        """
        --dry-run exits 1 when stage modules are not yet built.

        This is the EXPECTED and CORRECT behaviour at this build stage.
        The test confirms that main.py runs, parses args, attempts imports,
        and exits with code 1 — not an unexpected crash.
        """
        result = subprocess.run(
            [sys.executable, "-m", "main", "--dry-run"],
            capture_output=True,
            text=True,
        )
        # Exit 1 = dry run ran correctly and reported import failures
        # Exit 2 = argument/config error (unexpected)
        # Exit 3 = fatal environment error (unexpected)
        assert result.returncode in (0, 1), (
            f"Unexpected exit code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_help_flag_exits_zero(self):
        """main.py --help exits 0 and prints usage information."""
        result = subprocess.run(
            [sys.executable, "-m", "main", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Billboard" in result.stdout or "stage" in result.stdout.lower()

    def test_unknown_stage_exits_two(self):
        """main.py --stage INVALID exits 2."""
        result = subprocess.run(
            [sys.executable, "-m", "main", "--stage", "NONEXISTENT_STAGE_XYZ"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2

    def test_bad_year_range_exits_two(self):
        """main.py --sample-years with bad format exits 2."""
        result = subprocess.run(
            [sys.executable, "-m", "main", "--sample-years", "not-a-range"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2


# ═════════════════════════════════════════════════════════════════════════════
# TestDesignSystem
# ═════════════════════════════════════════════════════════════════════════════


class TestDesignSystem:
    """Verify _design_system constants are internally consistent."""

    def test_decade_order_matches_valid_decade_labels(self):
        """DECADE_ORDER is identical to VALID_DECADE_LABELS from schemas."""
        from src.visualizations._design_system import DECADE_ORDER

        assert DECADE_ORDER == VALID_DECADE_LABELS

    def test_all_decades_have_colors(self):
        """Every decade in DECADE_ORDER has an entry in DECADE_COLORS."""
        from src.visualizations._design_system import DECADE_COLORS, DECADE_ORDER

        for label in DECADE_ORDER:
            assert label in DECADE_COLORS, f"Missing color for decade: {label}"

    def test_all_emotions_have_colors(self):
        """Every emotion in VALID_EMOTIONAL_TONES has an entry in EMOTION_COLORS."""
        from src.visualizations._design_system import EMOTION_COLORS

        for tone in VALID_EMOTIONAL_TONES:
            assert tone in EMOTION_COLORS, f"Missing color for emotion: {tone}"

    def test_all_jung_stages_have_colors(self):
        """Every stage in VALID_JUNG_STAGES has an entry in JUNG_COLORS."""
        from src.visualizations._design_system import JUNG_COLORS

        for stage in VALID_JUNG_STAGES:
            assert stage in JUNG_COLORS, f"Missing color for Jung stage: {stage}"

    def test_hex_with_alpha_correct_output(self):
        """hex_with_alpha() returns correct rgba() string."""
        from src.visualizations._design_system import hex_with_alpha

        assert hex_with_alpha("#E8A838", 0.6) == "rgba(232, 168, 56, 0.6)"
        assert hex_with_alpha("#000000", 0.0) == "rgba(0, 0, 0, 0.0)"
        assert hex_with_alpha("#FFFFFF", 1.0) == "rgba(255, 255, 255, 1.0)"

    def test_hex_with_alpha_rejects_bad_alpha(self):
        """hex_with_alpha() raises ValueError for alpha outside [0, 1]."""
        from src.visualizations._design_system import hex_with_alpha

        with pytest.raises(ValueError):
            hex_with_alpha("#E8A838", 1.5)
        with pytest.raises(ValueError):
            hex_with_alpha("#E8A838", -0.1)

    def test_decade_color_with_alpha_correct(self):
        """decade_color_with_alpha() returns correct rgba() for known decade."""
        from src.visualizations._design_system import decade_color_with_alpha

        result = decade_color_with_alpha("1980s", 0.4)
        assert result.startswith("rgba(")
        assert result.endswith("0.4)")

    def test_base_layout_has_required_keys(self):
        """BASE_LAYOUT contains all keys required by Plotly figure layout."""
        from src.visualizations._design_system import BASE_LAYOUT

        required = {"paper_bgcolor", "plot_bgcolor", "font", "margin", "legend"}
        assert required.issubset(set(BASE_LAYOUT.keys()))
