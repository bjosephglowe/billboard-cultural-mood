"""
tests/test_contrast_metrics.py

Unit tests for src/analysis/contrast_metrics.py

Coverage:
- _jaccard_distance: identical sets, disjoint sets, partial overlap, empty sets
- _jaccard_bucket: boundary values for none / minor / major
- _theme_shift: null themes, no chorus, matching emotions, diverging emotions
- _energy_shift: increase / decrease / stable, null guards
- _contrast_sentiment: correct delta, null guards, no-chorus guard
- _compute_coverage_rate: full coverage, partial, zero full-quality songs
- _enforce_coverage_gate: passes at threshold, raises below threshold
- _merge_layers: all optional layers None, all layers provided, column presence
- run(): raises FileNotFoundError for missing lyrics CSV
- run(): raises FileNotFoundError for missing sentiment CSV
- run(): raises RuntimeError when coverage gate fails
- run(): returns correct keys on success
- run(): writes output CSV and sentinel
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.pipeline.schemas import layer4_schema, validate
from src.utils.identifiers import make_song_id

# ── Helpers ───────────────────────────────────────────────────────────────────


def _cm():
    from src.analysis import contrast_metrics

    return contrast_metrics


def _row(**kwargs) -> pd.Series:
    """Build a minimal pd.Series representing one song row."""
    defaults = {
        "song_id": make_song_id("Test", "Song", 2000),
        "lyrics_quality": "full",
        "sentiment_score": 0.5,
        "energy_level": 3,
        "chorus_detected": False,
        "chorus_sentiment_score": None,
        "chorus_emotional_tone": None,
        "emotional_tone": None,
        "themes": None,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


BINS = [0.2, 0.4, 0.6, 0.8]


# ── _jaccard_distance ─────────────────────────────────────────────────────────


class TestJaccardDistance:
    def test_identical_sets_is_zero(self):
        result = _cm()._jaccard_distance({"love", "joy"}, {"love", "joy"})
        assert result == 0.0

    def test_disjoint_sets_is_one(self):
        result = _cm()._jaccard_distance({"love"}, {"rebellion"})
        assert result == 1.0

    def test_partial_overlap(self):
        # |A ∩ B| = 1, |A ∪ B| = 3 → distance = 1 - 1/3 ≈ 0.6667
        result = _cm()._jaccard_distance({"love", "joy"}, {"love", "sadness"})
        assert 0.6 < result < 0.7

    def test_both_empty_is_zero(self):
        result = _cm()._jaccard_distance(set(), set())
        assert result == 0.0

    def test_one_empty_is_one(self):
        result = _cm()._jaccard_distance({"love"}, set())
        assert result == 1.0

    def test_returns_float(self):
        result = _cm()._jaccard_distance({"a"}, {"b"})
        assert isinstance(result, float)

    def test_result_in_range(self):
        result = _cm()._jaccard_distance({"love", "joy", "peace"}, {"joy", "hope"})
        assert 0.0 <= result <= 1.0


# ── _jaccard_bucket ───────────────────────────────────────────────────────────


class TestJaccardBucket:
    def test_zero_is_none(self):
        assert _cm()._jaccard_bucket(0.0) == "none"

    def test_above_zero_below_half_is_minor(self):
        assert _cm()._jaccard_bucket(0.25) == "minor"

    def test_exactly_half_is_minor(self):
        assert _cm()._jaccard_bucket(0.5) == "minor"

    def test_above_half_is_major(self):
        assert _cm()._jaccard_bucket(0.75) == "major"

    def test_one_is_major(self):
        assert _cm()._jaccard_bucket(1.0) == "major"


# ── _theme_shift ──────────────────────────────────────────────────────────────


class TestThemeShift:
    def test_returns_none_when_chorus_not_detected(self):
        row = _row(chorus_detected=False, themes="love|joy")
        result = _cm()._theme_shift(row)
        assert result is None

    def test_returns_none_when_themes_null(self):
        row = _row(chorus_detected=True, themes=None)
        result = _cm()._theme_shift(row)
        assert result is None

    def test_returns_none_when_themes_nan(self):
        row = _row(chorus_detected=True, themes=float("nan"))
        result = _cm()._theme_shift(row)
        assert result is None

    def test_same_emotion_returns_none(self):
        row = _row(
            chorus_detected=True,
            themes="love|nostalgia",
            emotional_tone="joy",
            chorus_emotional_tone="joy",
        )
        result = _cm()._theme_shift(row)
        assert result == "none"

    def test_different_emotion_returns_minor(self):
        row = _row(
            chorus_detected=True,
            themes="love|nostalgia",
            emotional_tone="joy",
            chorus_emotional_tone="sadness",
        )
        result = _cm()._theme_shift(row)
        assert result == "minor"

    def test_missing_chorus_emotion_returns_none(self):
        row = _row(
            chorus_detected=True,
            themes="love|nostalgia",
            emotional_tone="joy",
            chorus_emotional_tone=None,
        )
        result = _cm()._theme_shift(row)
        assert result == "none"

    def test_result_in_valid_set(self):
        from src.pipeline.schemas import VALID_THEME_SHIFTS

        row = _row(
            chorus_detected=True,
            themes="love",
            emotional_tone="joy",
            chorus_emotional_tone="anger",
        )
        result = _cm()._theme_shift(row)
        assert result in VALID_THEME_SHIFTS


# ── _energy_shift ─────────────────────────────────────────────────────────────


class TestEnergyShift:
    def test_returns_none_when_chorus_not_detected(self):
        row = _row(chorus_detected=False, energy_level=3, chorus_sentiment_score=0.9)
        result = _cm()._energy_shift(row, BINS)
        assert result is None

    def test_returns_none_when_energy_level_null(self):
        row = _row(chorus_detected=True, energy_level=None, chorus_sentiment_score=0.9)
        result = _cm()._energy_shift(row, BINS)
        assert result is None

    def test_returns_none_when_chorus_score_null(self):
        row = _row(chorus_detected=True, energy_level=3, chorus_sentiment_score=None)
        result = _cm()._energy_shift(row, BINS)
        assert result is None

    def test_increase_when_chorus_energy_higher(self):
        # song energy_level=2 (score ~0.3), chorus score=0.9 → chorus level=5
        row = _row(chorus_detected=True, energy_level=2, chorus_sentiment_score=0.9)
        result = _cm()._energy_shift(row, BINS)
        assert result == "increase"

    def test_decrease_when_chorus_energy_lower(self):
        # song energy_level=5, chorus score=-0.5 → chorus level=1
        row = _row(chorus_detected=True, energy_level=5, chorus_sentiment_score=-0.5)
        result = _cm()._energy_shift(row, BINS)
        assert result == "decrease"

    def test_stable_when_chorus_energy_equal(self):
        # song energy_level=3 (score ~0.5), chorus score=0.5 → chorus level=3
        row = _row(chorus_detected=True, energy_level=3, chorus_sentiment_score=0.5)
        result = _cm()._energy_shift(row, BINS)
        assert result == "stable"

    def test_result_in_valid_set(self):
        from src.pipeline.schemas import VALID_ENERGY_SHIFTS

        row = _row(chorus_detected=True, energy_level=3, chorus_sentiment_score=0.9)
        result = _cm()._energy_shift(row, BINS)
        assert result in VALID_ENERGY_SHIFTS


# ── _contrast_sentiment ───────────────────────────────────────────────────────


class TestContrastSentiment:
    def test_returns_none_when_chorus_not_detected(self):
        row = _row(
            chorus_detected=False, sentiment_score=0.5, chorus_sentiment_score=0.2
        )
        result = _cm()._contrast_sentiment(row)
        assert result is None

    def test_returns_none_when_sentiment_null(self):
        row = _row(
            chorus_detected=True, sentiment_score=None, chorus_sentiment_score=0.2
        )
        result = _cm()._contrast_sentiment(row)
        assert result is None

    def test_returns_none_when_chorus_score_null(self):
        row = _row(
            chorus_detected=True, sentiment_score=0.5, chorus_sentiment_score=None
        )
        result = _cm()._contrast_sentiment(row)
        assert result is None

    def test_correct_positive_delta(self):
        row = _row(
            chorus_detected=True, sentiment_score=0.5, chorus_sentiment_score=0.3
        )
        result = _cm()._contrast_sentiment(row)
        assert result == pytest.approx(0.2, abs=0.001)

    def test_correct_negative_delta(self):
        row = _row(
            chorus_detected=True, sentiment_score=0.2, chorus_sentiment_score=0.6
        )
        result = _cm()._contrast_sentiment(row)
        assert result == pytest.approx(-0.4, abs=0.001)

    def test_zero_delta_when_equal(self):
        row = _row(
            chorus_detected=True, sentiment_score=0.5, chorus_sentiment_score=0.5
        )
        result = _cm()._contrast_sentiment(row)
        assert result == pytest.approx(0.0, abs=0.001)


# ── _compute_coverage_rate ────────────────────────────────────────────────────


class TestComputeCoverageRate:
    def test_full_coverage(self):
        df = pd.DataFrame(
            {
                "lyrics_quality": ["full", "full", "full"],
                "sentiment_score": [0.5, -0.1, 0.8],
            }
        )
        rate = _cm()._compute_coverage_rate(df)
        assert rate == pytest.approx(1.0)

    def test_partial_coverage(self):
        df = pd.DataFrame(
            {
                "lyrics_quality": ["full", "full", "full", "full"],
                "sentiment_score": [0.5, None, 0.8, None],
            }
        )
        rate = _cm()._compute_coverage_rate(df)
        assert rate == pytest.approx(0.5)

    def test_no_full_quality_songs_returns_one(self):
        df = pd.DataFrame(
            {
                "lyrics_quality": ["missing", "partial"],
                "sentiment_score": [None, None],
            }
        )
        rate = _cm()._compute_coverage_rate(df)
        assert rate == 1.0

    def test_mixed_quality_only_full_counted(self):
        df = pd.DataFrame(
            {
                "lyrics_quality": ["full", "missing", "partial", "full"],
                "sentiment_score": [0.5, None, None, None],
            }
        )
        rate = _cm()._compute_coverage_rate(df)
        assert rate == pytest.approx(0.5)


# ── _enforce_coverage_gate ────────────────────────────────────────────────────


class TestEnforceCoverageGate:
    def test_passes_at_full_coverage(self, project_config):
        df = pd.DataFrame(
            {
                "lyrics_quality": ["full"] * 5,
                "sentiment_score": [0.5, 0.3, -0.1, 0.8, 0.2],
            }
        )
        # Should not raise
        _cm()._enforce_coverage_gate(df, project_config)

    def test_passes_at_exact_threshold(self, project_config):
        # threshold=0.85; 17/20 = 0.85 exactly
        scored = [0.5] * 17 + [None] * 3
        df = pd.DataFrame(
            {
                "lyrics_quality": ["full"] * 20,
                "sentiment_score": scored,
            }
        )
        _cm()._enforce_coverage_gate(df, project_config)

    def test_raises_below_threshold(self, project_config):
        # threshold=0.85; 16/20 = 0.80 — below threshold
        scored = [0.5] * 16 + [None] * 4
        df = pd.DataFrame(
            {
                "lyrics_quality": ["full"] * 20,
                "sentiment_score": scored,
            }
        )
        with pytest.raises(RuntimeError, match="Coverage gate FAILED"):
            _cm()._enforce_coverage_gate(df, project_config)

    def test_raises_with_zero_scored(self, project_config):
        df = pd.DataFrame(
            {
                "lyrics_quality": ["full"] * 5,
                "sentiment_score": [None] * 5,
            }
        )
        with pytest.raises(RuntimeError):
            _cm()._enforce_coverage_gate(df, project_config)


# ── _merge_layers ─────────────────────────────────────────────────────────────


class TestMergeLayers:
    def test_all_optional_none_produces_null_columns(
        self, sample_songs_df, sample_sentiment_df
    ):
        cm = _cm()
        df = cm._merge_layers(sample_songs_df, sample_sentiment_df, None, None, None)
        assert "emotional_tone" in df.columns
        assert "themes" in df.columns
        assert "chorus_detected" in df.columns
        assert df["chorus_detected"].all() == False

    def test_chorus_detected_false_when_chorus_none(
        self, sample_songs_df, sample_sentiment_df
    ):
        cm = _cm()
        df = cm._merge_layers(sample_songs_df, sample_sentiment_df, None, None, None)
        assert (df["chorus_detected"] == False).all()

    def test_chorus_method_none_when_chorus_absent(
        self, sample_songs_df, sample_sentiment_df
    ):
        cm = _cm()
        df = cm._merge_layers(sample_songs_df, sample_sentiment_df, None, None, None)
        assert (df["chorus_method"] == "none").all()

    def test_row_count_preserved(self, sample_songs_df, sample_sentiment_df):
        cm = _cm()
        df = cm._merge_layers(sample_songs_df, sample_sentiment_df, None, None, None)
        assert len(df) == len(sample_songs_df)

    def test_sentiment_columns_merged(self, sample_songs_df, sample_sentiment_df):
        cm = _cm()
        df = cm._merge_layers(sample_songs_df, sample_sentiment_df, None, None, None)
        assert "sentiment_score" in df.columns
        assert "energy_level" in df.columns

    def test_all_layers_merged(
        self,
        sample_songs_df,
        sample_sentiment_df,
        sample_emotion_df,
        sample_themes_df,
        sample_chorus_df,
    ):
        cm = _cm()
        df = cm._merge_layers(
            sample_songs_df,
            sample_sentiment_df,
            sample_emotion_df,
            sample_themes_df,
            sample_chorus_df,
        )
        assert "emotional_tone" in df.columns
        assert "themes" in df.columns
        assert "chorus_detected" in df.columns
        assert len(df) == len(sample_songs_df)


# ── run() ─────────────────────────────────────────────────────────────────────


class TestRun:
    def _write_csv(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def test_raises_if_lyrics_csv_missing(self, pipeline_workspace, tmp_config):
        cm = _cm()
        missing = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        with (
            patch.object(cm, "_LYRICS_PATH", new=missing),
            patch.object(cm, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                cm.run(tmp_config)

    def test_raises_if_sentiment_csv_missing(
        self, pipeline_workspace, tmp_config, sample_songs_df
    ):
        cm = _cm()
        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        missing_sent = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"
        self._write_csv(lyrics_path, sample_songs_df)

        with (
            patch.object(cm, "_LYRICS_PATH", new=lyrics_path),
            patch.object(cm, "_SENTIMENT_PATH", new=missing_sent),
            patch.object(
                cm, "_EMOTION_PATH", new=missing_sent.parent / "no_emotion.csv"
            ),
            patch.object(cm, "_THEMES_PATH", new=missing_sent.parent / "no_themes.csv"),
            patch.object(cm, "_CHORUS_PATH", new=missing_sent.parent / "no_chorus.csv"),
            patch.object(cm, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                cm.run(tmp_config)

    def test_raises_runtime_error_when_coverage_gate_fails(
        self, pipeline_workspace, tmp_config, sample_songs_df
    ):
        cm = _cm()
        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"

        # All songs have null sentiment — 0% coverage, well below 0.85
        low_coverage_sent = pd.DataFrame(
            {
                "song_id": sample_songs_df["song_id"].tolist(),
                "sentiment_score": [None] * len(sample_songs_df),
                "energy_level": pd.array(
                    [None] * len(sample_songs_df), dtype=pd.Int64Dtype()
                ),
            }
        )
        self._write_csv(lyrics_path, sample_songs_df)
        self._write_csv(sent_path, low_coverage_sent)

        with (
            patch.object(cm, "_LYRICS_PATH", new=lyrics_path),
            patch.object(cm, "_SENTIMENT_PATH", new=sent_path),
            patch.object(
                cm,
                "_EMOTION_PATH",
                new=pipeline_workspace / "data" / "analysis" / "no_emotion.csv",
            ),
            patch.object(
                cm,
                "_THEMES_PATH",
                new=pipeline_workspace / "data" / "analysis" / "no_themes.csv",
            ),
            patch.object(
                cm,
                "_CHORUS_PATH",
                new=pipeline_workspace / "data" / "processed" / "no_chorus.csv",
            ),
            patch.object(cm, "_OUTPUT_PATH", new=out_path),
            patch.object(cm, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(RuntimeError, match="Coverage gate FAILED"):
                cm.run(tmp_config)

    def test_run_returns_correct_keys(
        self, pipeline_workspace, tmp_config, sample_songs_df, sample_sentiment_df
    ):
        cm = _cm()
        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"
        self._write_csv(lyrics_path, sample_songs_df)
        self._write_csv(sent_path, sample_sentiment_df)

        with (
            patch.object(cm, "_LYRICS_PATH", new=lyrics_path),
            patch.object(cm, "_SENTIMENT_PATH", new=sent_path),
            patch.object(
                cm,
                "_EMOTION_PATH",
                new=pipeline_workspace / "data" / "analysis" / "no_emotion.csv",
            ),
            patch.object(
                cm,
                "_THEMES_PATH",
                new=pipeline_workspace / "data" / "analysis" / "no_themes.csv",
            ),
            patch.object(
                cm,
                "_CHORUS_PATH",
                new=pipeline_workspace / "data" / "processed" / "no_chorus.csv",
            ),
            patch.object(cm, "_OUTPUT_PATH", new=out_path),
            patch.object(cm, "_SENTINEL", new=sentinel),
        ):
            result = cm.run(tmp_config)

        assert "total_songs" in result
        assert "scored_songs" in result
        assert "coverage_rate" in result
        assert "output_path" in result

    def test_run_writes_output_csv(
        self, pipeline_workspace, tmp_config, sample_songs_df, sample_sentiment_df
    ):
        cm = _cm()
        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"
        self._write_csv(lyrics_path, sample_songs_df)
        self._write_csv(sent_path, sample_sentiment_df)

        with (
            patch.object(cm, "_LYRICS_PATH", new=lyrics_path),
            patch.object(cm, "_SENTIMENT_PATH", new=sent_path),
            patch.object(
                cm,
                "_EMOTION_PATH",
                new=pipeline_workspace / "data" / "analysis" / "no_emotion.csv",
            ),
            patch.object(
                cm,
                "_THEMES_PATH",
                new=pipeline_workspace / "data" / "analysis" / "no_themes.csv",
            ),
            patch.object(
                cm,
                "_CHORUS_PATH",
                new=pipeline_workspace / "data" / "processed" / "no_chorus.csv",
            ),
            patch.object(cm, "_OUTPUT_PATH", new=out_path),
            patch.object(cm, "_SENTINEL", new=sentinel),
        ):
            cm.run(tmp_config)

        assert out_path.exists(), "layer2_full_analysis.csv was not written"

    def test_run_writes_sentinel(
        self, pipeline_workspace, tmp_config, sample_songs_df, sample_sentiment_df
    ):
        cm = _cm()
        lyrics_path = pipeline_workspace / "data" / "processed" / "lyrics_cleaned.csv"
        sent_path = pipeline_workspace / "data" / "analysis" / "layer2_sentiment.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".contrast_complete"
        self._write_csv(lyrics_path, sample_songs_df)
        self._write_csv(sent_path, sample_sentiment_df)

        with (
            patch.object(cm, "_LYRICS_PATH", new=lyrics_path),
            patch.object(cm, "_SENTIMENT_PATH", new=sent_path),
            patch.object(
                cm,
                "_EMOTION_PATH",
                new=pipeline_workspace / "data" / "analysis" / "no_emotion.csv",
            ),
            patch.object(
                cm,
                "_THEMES_PATH",
                new=pipeline_workspace / "data" / "analysis" / "no_themes.csv",
            ),
            patch.object(
                cm,
                "_CHORUS_PATH",
                new=pipeline_workspace / "data" / "processed" / "no_chorus.csv",
            ),
            patch.object(cm, "_OUTPUT_PATH", new=out_path),
            patch.object(cm, "_SENTINEL", new=sentinel),
        ):
            cm.run(tmp_config)

        assert sentinel.exists(), "Sentinel not written after successful run"
        payload = json.loads(sentinel.read_text())
        assert "config_hash" in payload
        assert payload["stage"] == "CONTRAST_METRICS"
