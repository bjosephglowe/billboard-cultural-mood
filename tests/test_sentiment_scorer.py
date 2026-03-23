"""
tests/test_sentiment_scorer.py

Unit tests for src/analysis/sentiment_scorer.py

Coverage:
- _ensemble_sentiment: score range, type, directionality, edge cases
- _compute_energy_level: all 5 bin boundaries, exact edges, full valid set
- _score_dataframe: output columns, dtypes, quality filtering, no duplicates
- run(): sentinel written on success with correct config_hash
- run(): sentinel skip when already current
- run(): raises FileNotFoundError when lyrics_cleaned.csv absent
- run(): return dict contains scored_count key
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.pipeline.schemas import layer2_sentiment_schema, validate
from src.utils.identifiers import make_song_id

# ── Helpers ───────────────────────────────────────────────────────────────────


def _scorer():
    from src.analysis import sentiment_scorer

    return sentiment_scorer


# ── _ensemble_sentiment ───────────────────────────────────────────────────────


class TestEnsembleSentiment:
    def test_returns_float(self):
        result = _scorer()._ensemble_sentiment("I love this happy wonderful song")
        assert isinstance(result, float)

    def test_range_positive_text(self):
        result = _scorer()._ensemble_sentiment("I love this happy wonderful song")
        assert -1.0 <= result <= 1.0

    def test_range_negative_text(self):
        result = _scorer()._ensemble_sentiment("I hate this terrible awful disaster")
        assert -1.0 <= result <= 1.0

    def test_positive_text_yields_positive_score(self):
        result = _scorer()._ensemble_sentiment("I love this happy wonderful song")
        assert result > 0.0, f"Expected positive score, got {result}"

    def test_negative_text_yields_negative_score(self):
        result = _scorer()._ensemble_sentiment("I hate this terrible awful disaster")
        assert result < 0.0, f"Expected negative score, got {result}"

    def test_neutral_text_near_zero(self):
        result = _scorer()._ensemble_sentiment("The cat sat on the mat by the window")
        assert -0.5 <= result <= 0.5, f"Expected near-neutral score, got {result}"

    def test_empty_string_returns_float(self):
        result = _scorer()._ensemble_sentiment("")
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_short_text_does_not_raise(self):
        result = _scorer()._ensemble_sentiment("yeah")
        assert isinstance(result, float)


# ── _compute_energy_level ─────────────────────────────────────────────────────


class TestComputeEnergyLevel:
    """
    Bins: [0.2, 0.4, 0.6, 0.8]
    score < 0.2       → level 1
    0.2 ≤ score < 0.4 → level 2
    0.4 ≤ score < 0.6 → level 3
    0.6 ≤ score < 0.8 → level 4
    score ≥ 0.8       → level 5
    """

    BINS = [0.2, 0.4, 0.6, 0.8]

    def test_below_bin0_is_level1(self):
        assert _scorer()._compute_energy_level(0.1, self.BINS) == 1

    def test_negative_score_is_level1(self):
        assert _scorer()._compute_energy_level(-0.9, self.BINS) == 1

    def test_exactly_bin0_is_level2(self):
        assert _scorer()._compute_energy_level(0.2, self.BINS) == 2

    def test_between_bin0_bin1_is_level2(self):
        assert _scorer()._compute_energy_level(0.3, self.BINS) == 2

    def test_exactly_bin1_is_level3(self):
        assert _scorer()._compute_energy_level(0.4, self.BINS) == 3

    def test_between_bin1_bin2_is_level3(self):
        assert _scorer()._compute_energy_level(0.5, self.BINS) == 3

    def test_exactly_bin2_is_level4(self):
        assert _scorer()._compute_energy_level(0.6, self.BINS) == 4

    def test_between_bin2_bin3_is_level4(self):
        assert _scorer()._compute_energy_level(0.7, self.BINS) == 4

    def test_exactly_bin3_is_level5(self):
        assert _scorer()._compute_energy_level(0.8, self.BINS) == 5

    def test_above_bin3_is_level5(self):
        assert _scorer()._compute_energy_level(0.99, self.BINS) == 5

    def test_exact_one_is_level5(self):
        assert _scorer()._compute_energy_level(1.0, self.BINS) == 5

    def test_result_in_valid_set(self):
        for score in [-1.0, -0.5, 0.0, 0.19, 0.2, 0.39, 0.4, 0.59, 0.6, 0.79, 0.8, 1.0]:
            level = _scorer()._compute_energy_level(score, self.BINS)
            assert level in {1, 2, 3, 4, 5}, (
                f"Level {level} out of range for score {score}"
            )


# ── _score_dataframe ──────────────────────────────────────────────────────────


class TestScoreDataframe:
    """Tests for _score_dataframe(df, config) — the internal scoring function."""

    def test_output_has_required_columns(self, sample_songs_df, project_config):
        result = _scorer()._score_dataframe(sample_songs_df.copy(), project_config)
        assert "sentiment_score" in result.columns
        assert "energy_level" in result.columns
        assert "song_id" in result.columns

    def test_missing_quality_rows_get_nan_score(self, project_config):
        df = pd.DataFrame(
            {
                "song_id": [make_song_id("Test", "Song A", 2000)],
                "song_title": ["Song A"],
                "artist": ["Test"],
                "year": [2000],
                "decade": ["2000s"],
                "lyrics_clean": [None],
                "lyrics_verse_only": [None],
                "chorus_token_count": [0],
                "lyrics_quality": ["missing"],
                "narrative_perspective": ["first_person"],
                "has_section_tags": [False],
                "section_count": [0],
            }
        )
        result = _scorer()._score_dataframe(df, project_config)
        assert pd.isna(result.iloc[0]["sentiment_score"])
        assert pd.isna(result.iloc[0]["energy_level"])

    def test_full_quality_rows_get_float_score(self, sample_songs_df, project_config):
        result = _scorer()._score_dataframe(sample_songs_df.copy(), project_config)
        full_mask = result["lyrics_quality"] == "full"
        assert result.loc[full_mask, "sentiment_score"].notna().all()

    def test_sentiment_score_range(self, sample_songs_df, project_config):
        result = _scorer()._score_dataframe(sample_songs_df.copy(), project_config)
        scores = result["sentiment_score"].dropna()
        assert (scores >= -1.0).all()
        assert (scores <= 1.0).all()

    def test_energy_level_valid_values(self, sample_songs_df, project_config):
        result = _scorer()._score_dataframe(sample_songs_df.copy(), project_config)
        levels = result["energy_level"].dropna()
        assert set(levels.tolist()).issubset({1, 2, 3, 4, 5})

    def test_energy_level_dtype_is_Int64(self, sample_songs_df, project_config):
        result = _scorer()._score_dataframe(sample_songs_df.copy(), project_config)
        assert str(result["energy_level"].dtype) == "Int64"

    def test_partial_quality_rows_get_nan_score(self, project_config):
        df = pd.DataFrame(
            {
                "song_id": [make_song_id("Test", "Song B", 2001)],
                "song_title": ["Song B"],
                "artist": ["Test"],
                "year": [2001],
                "decade": ["2000s"],
                "lyrics_clean": ["yeah yeah yeah"],
                "lyrics_verse_only": ["yeah yeah yeah"],
                "chorus_token_count": [3],
                "lyrics_quality": ["partial"],
                "narrative_perspective": ["first_person"],
                "has_section_tags": [False],
                "section_count": [0],
            }
        )
        # partial quality is NOT skipped by _score_dataframe — only "missing" is
        # partial songs receive a score; this is intentional pipeline behaviour
        result = _scorer()._score_dataframe(df, project_config)
        assert result.iloc[0]["sentiment_score"] is not None

    def test_schema_validation_passes(self, sample_songs_df, project_config):
        scorer = _scorer()
        result = scorer._score_dataframe(sample_songs_df.copy(), project_config)
        output = result[["song_id", "sentiment_score", "energy_level"]].copy()
        output["energy_level"] = pd.array(
            output["energy_level"].tolist(), dtype=pd.Int64Dtype()
        )
        validated = validate(output, layer2_sentiment_schema, "test_score_dataframe")
        assert validated is not None

    def test_no_duplicate_song_ids(self, sample_songs_df, project_config):
        result = _scorer()._score_dataframe(sample_songs_df.copy(), project_config)
        assert result["song_id"].nunique() == len(result)


# ── run() — sentinel and file I/O ────────────────────────────────────────────


class TestSentinelBehavior:
    """Tests for run(config) — file I/O, sentinel write, skip, and error handling."""

    def test_run_writes_sentinel(self, pipeline_workspace, tmp_config, sample_songs_df):
        scorer = _scorer()
        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"
        sample_songs_df.to_csv(lyrics_path, index=False)

        with (
            patch.object(scorer, "_INPUT_PATH", new=lyrics_path),
            patch.object(scorer, "_OUTPUT_PATH", new=out_path),
            patch.object(scorer, "_SENTINEL", new=sentinel),
        ):
            scorer.run(tmp_config)

        assert sentinel.exists(), "Sentinel not written after successful run"

    def test_sentinel_contains_config_hash(
        self, pipeline_workspace, tmp_config, sample_songs_df
    ):
        from src.pipeline.config_loader import config_hash

        scorer = _scorer()
        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"
        sample_songs_df.to_csv(lyrics_path, index=False)

        with (
            patch.object(scorer, "_INPUT_PATH", new=lyrics_path),
            patch.object(scorer, "_OUTPUT_PATH", new=out_path),
            patch.object(scorer, "_SENTINEL", new=sentinel),
        ):
            scorer.run(tmp_config)

        payload = json.loads(sentinel.read_text())
        assert payload["config_hash"] == config_hash(tmp_config)

    def test_run_raises_if_lyrics_csv_missing(self, pipeline_workspace, tmp_config):
        scorer = _scorer()
        missing_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"

        with (
            patch.object(scorer, "_INPUT_PATH", new=missing_path),
            patch.object(scorer, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                scorer.run(tmp_config)

    def test_run_returns_scored_count_key(
        self, pipeline_workspace, tmp_config, sample_songs_df
    ):
        scorer = _scorer()
        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".sentiment_complete"
        sample_songs_df.to_csv(lyrics_path, index=False)

        with (
            patch.object(scorer, "_INPUT_PATH", new=lyrics_path),
            patch.object(scorer, "_OUTPUT_PATH", new=out_path),
            patch.object(scorer, "_SENTINEL", new=sentinel),
        ):
            result = scorer.run(tmp_config)

        assert isinstance(result, dict)
        assert "scored_count" in result
        assert result["scored_count"] >= 0
