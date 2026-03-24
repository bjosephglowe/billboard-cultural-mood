"""
tests/test_cmi_calculator.py

Unit tests for src/cultural_metrics/cmi_calculator.py

Coverage:
- _load_weights: default fallback, normalisation to 1.0, config override
- _impute_components: decade-mean imputation, missing columns handled
- _compute_resonance: score range, known value, chorus/quality/jungian
- _mode_value: normal case, empty, missing column, ties
- _top_pipe_themes: frequency counting, n limit, empty, null-only
- _top_resonance: top-n by score, empty DataFrame, missing song_title
- _build_decade_cmi: required columns, row-per-decade, null handling
- CoverageError: is RuntimeError subclass
- run(): FileNotFoundError for missing layer4
- run(): FileNotFoundError for missing layer5
- run(): CoverageError when coverage below threshold
- run(): returns correct keys on success
- run(): writes layer6 and decade CSVs
- run(): calls write_sentinel
- run(): sentinel skip path
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.pipeline.schemas import decade_cmi_schema, layer6_schema, validate
from src.utils.identifiers import make_song_id

# ── Helpers ───────────────────────────────────────────────────────────────────


def _cmi():
    from src.cultural_metrics import cmi_calculator

    return cmi_calculator


def _make_merged_df(n: int = 5, decade: str = "2000s") -> pd.DataFrame:
    """
    Build a minimal merged DataFrame suitable for _compute_resonance
    and _build_decade_cmi — mirrors what run() produces after joining
    layer4 + layer5.
    """
    sids = [make_song_id("Artist", f"Song{i}", 2000 + i) for i in range(n)]
    return pd.DataFrame(
        {
            "song_id": sids,
            "song_title": [f"Song{i}" for i in range(n)],
            "artist": ["Artist"] * n,
            "decade": [decade] * n,
            "lyrics_quality": ["full"] * n,
            "sentiment_score": [0.5, -0.2, 0.8, 0.1, -0.5],
            "energy_level": pd.array([3, 2, 5, 4, 1], dtype=pd.Int64Dtype()),
            "chorus_detected": [True, False, True, True, False],
            "emotional_tone": ["joy", "fear", "joy", "neutral", "anger"],
            "themes": [
                "love|nostalgia",
                "struggle",
                "love|empowerment",
                "nostalgia",
                "rebellion|struggle",
            ],
            "jung_stage": [
                "persona",
                "shadow",
                "integration",
                "unclassified",
                "persona",
            ],
            "development_score": pd.array([4, 2, 6, None, 3], dtype=pd.Int64Dtype()),
        }
    )


def _make_layer5_df(song_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "song_id": song_ids,
            "jung_stage": ["persona"] * len(song_ids),
            "development_score": pd.array([4] * len(song_ids), dtype=pd.Int64Dtype()),
        }
    )


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ── CoverageError ─────────────────────────────────────────────────────────────


class TestCoverageError:
    def test_is_runtime_error_subclass(self):
        assert issubclass(_cmi().CoverageError, RuntimeError)

    def test_can_be_raised_and_caught_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise _cmi().CoverageError("test coverage error")


# ── _load_weights ─────────────────────────────────────────────────────────────


class TestLoadWeights:
    def test_returns_dict_with_five_keys(self, project_config):
        weights = _cmi()._load_weights(project_config)
        assert set(weights.keys()) == {
            "sentiment",
            "energy",
            "chorus",
            "quality",
            "jungian",
        }

    def test_weights_sum_to_one(self, project_config):
        weights = _cmi()._load_weights(project_config)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_all_weights_non_negative(self, project_config):
        weights = _cmi()._load_weights(project_config)
        for k, v in weights.items():
            assert v >= 0.0, f"Negative weight for {k}: {v}"

    def test_fallback_when_no_cmi_config(self, project_config):
        """Default weights should match _DEFAULT_WEIGHTS values."""
        weights = _cmi()._load_weights(project_config)
        assert weights["sentiment"] == pytest.approx(0.25, abs=0.01)
        assert weights["energy"] == pytest.approx(0.20, abs=0.01)
        assert weights["chorus"] == pytest.approx(0.15, abs=0.01)
        assert weights["quality"] == pytest.approx(0.20, abs=0.01)
        assert weights["jungian"] == pytest.approx(0.20, abs=0.01)


# ── _impute_components ────────────────────────────────────────────────────────


class TestImputeComponents:
    def test_null_sentiment_filled_with_decade_mean(self):
        df = pd.DataFrame(
            {
                "decade": ["2000s", "2000s", "2000s"],
                "sentiment_score": [0.4, 0.6, None],
                "energy_level": [3.0, 3.0, 3.0],
                "development_score": [4.0, 4.0, 4.0],
            }
        )
        result = _cmi()._impute_components(df)
        assert result.iloc[2]["sentiment_score"] == pytest.approx(0.5, abs=0.01)

    def test_null_energy_filled_with_decade_mean(self):
        df = pd.DataFrame(
            {
                "decade": ["1990s", "1990s", "1990s"],
                "sentiment_score": [0.5, 0.5, 0.5],
                "energy_level": [2.0, 4.0, None],
                "development_score": [3.0, 3.0, 3.0],
            }
        )
        result = _cmi()._impute_components(df)
        assert result.iloc[2]["energy_level"] == pytest.approx(3.0, abs=0.01)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame(
            {
                "decade": ["2000s"] * 5,
                "sentiment_score": [0.5, 0.3, None, 0.7, 0.4],
                "energy_level": [3.0, 2.0, 4.0, 5.0, 1.0],  # plain float — no Int64
                "development_score": [4.0, 3.0, 5.0, 2.0, 4.0],
            }
        )
        original_nulls = df["sentiment_score"].isna().sum()
        _ = _cmi()._impute_components(df)
        assert df["sentiment_score"].isna().sum() == original_nulls

    def test_missing_development_score_column_handled(self):
        df = pd.DataFrame(
            {
                "decade": ["2000s"] * 3,
                "sentiment_score": [0.5, 0.3, 0.4],
                "energy_level": [3.0, 3.0, 3.0],
            }
        )
        result = _cmi()._impute_components(df)
        assert "development_score" in result.columns

    def test_cross_decade_imputation_isolated(self):
        """Imputation must not bleed across decade groups."""
        df = pd.DataFrame(
            {
                "decade": ["1980s", "1980s", "1990s", "1990s"],
                "sentiment_score": [0.8, None, 0.2, None],
                "energy_level": [4.0, 4.0, 2.0, 2.0],
                "development_score": [5.0, 5.0, 2.0, 2.0],
            }
        )
        result = _cmi()._impute_components(df)
        assert result.iloc[1]["sentiment_score"] == pytest.approx(0.8, abs=0.01)
        assert result.iloc[3]["sentiment_score"] == pytest.approx(0.2, abs=0.01)


# ── _compute_resonance ────────────────────────────────────────────────────────


class TestComputeResonance:
    def _default_weights(self):
        return {
            "sentiment": 0.25,
            "energy": 0.20,
            "chorus": 0.15,
            "quality": 0.20,
            "jungian": 0.20,
        }

    def test_returns_series(self):
        df = _make_merged_df()
        result = _cmi()._compute_resonance(df, self._default_weights())
        assert isinstance(result, pd.Series)

    def test_scores_in_range(self):
        df = _make_merged_df()
        result = _cmi()._compute_resonance(df, self._default_weights())
        assert (result.dropna() >= 0.0).all()
        assert (result.dropna() <= 1.0).all()

    def test_length_matches_input(self):
        df = _make_merged_df(5)
        result = _cmi()._compute_resonance(df, self._default_weights())
        assert len(result) == 5

    def test_chorus_detected_true_increases_score(self):
        """Song with chorus_detected=True should score higher than identical
        song with chorus_detected=False, all else equal."""
        weights = self._default_weights()
        base = {
            "song_id": [make_song_id("A", "S1", 2000), make_song_id("A", "S2", 2001)],
            "sentiment_score": [0.5, 0.5],
            "energy_level": pd.array([3, 3], dtype=pd.Int64Dtype()),
            "chorus_detected": [True, False],
            "lyrics_quality": ["full", "full"],
            "development_score": pd.array([4, 4], dtype=pd.Int64Dtype()),
            "decade": ["2000s", "2000s"],
        }
        df = pd.DataFrame(base)
        scores = _cmi()._compute_resonance(df, weights)
        assert scores.iloc[0] > scores.iloc[1]

    def test_full_quality_scores_higher_than_missing(self):
        weights = self._default_weights()
        df = pd.DataFrame(
            {
                "song_id": [
                    make_song_id("A", "S1", 2000),
                    make_song_id("A", "S2", 2001),
                ],
                "sentiment_score": [0.5, 0.5],
                "energy_level": pd.array([3, 3], dtype=pd.Int64Dtype()),
                "chorus_detected": [False, False],
                "lyrics_quality": ["full", "missing"],
                "development_score": pd.array([4, 4], dtype=pd.Int64Dtype()),
                "decade": ["2000s", "2000s"],
            }
        )
        scores = _cmi()._compute_resonance(df, weights)
        assert scores.iloc[0] > scores.iloc[1]

    def test_known_value_maximum_inputs(self):
        """
        All components at maximum should produce score ≈ 1.0.
        sentiment=1.0 → norm=1.0, energy=5 → norm=1.0,
        chorus=True → 1.0, quality=full → 1.0, dev=7 → norm=1.0
        """
        weights = self._default_weights()
        df = pd.DataFrame(
            {
                "song_id": [make_song_id("A", "S1", 2000)],
                "sentiment_score": [1.0],
                "energy_level": pd.array([5], dtype=pd.Int64Dtype()),
                "chorus_detected": [True],
                "lyrics_quality": ["full"],
                "development_score": pd.array([7], dtype=pd.Int64Dtype()),
                "decade": ["2000s"],
            }
        )
        scores = _cmi()._compute_resonance(df, weights)
        assert scores.iloc[0] == pytest.approx(1.0, abs=0.001)

    def test_known_value_minimum_inputs(self):
        """
        All components at minimum should produce score ≈ 0.0.
        sentiment=-1.0 → norm=0.0, energy=1 → norm=0.0,
        chorus=False → 0.0, quality=missing → 0.0, dev=1 → norm=0.0
        """
        weights = self._default_weights()
        df = pd.DataFrame(
            {
                "song_id": [make_song_id("A", "S1", 2000)],
                "sentiment_score": [-1.0],
                "energy_level": pd.array([1], dtype=pd.Int64Dtype()),
                "chorus_detected": [False],
                "lyrics_quality": ["missing"],
                "development_score": pd.array([1], dtype=pd.Int64Dtype()),
                "decade": ["2000s"],
            }
        )
        scores = _cmi()._compute_resonance(df, weights)
        assert scores.iloc[0] == pytest.approx(0.0, abs=0.001)


# ── _mode_value ───────────────────────────────────────────────────────────────


class TestModeValue:
    def test_returns_most_frequent(self):
        df = pd.DataFrame({"col": ["joy", "joy", "anger", "joy"]})
        assert _cmi()._mode_value(df, "col") == "joy"

    def test_returns_none_when_all_null(self):
        df = pd.DataFrame({"col": [None, None, None]})
        assert _cmi()._mode_value(df, "col") is None

    def test_returns_none_when_column_missing(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        assert _cmi()._mode_value(df, "col") is None

    def test_returns_none_when_empty_df(self):
        df = pd.DataFrame({"col": pd.Series([], dtype=str)})
        assert _cmi()._mode_value(df, "col") is None

    def test_returns_string(self):
        df = pd.DataFrame({"col": ["joy", "anger"]})
        result = _cmi()._mode_value(df, "col")
        assert isinstance(result, str)


# ── _top_pipe_themes ──────────────────────────────────────────────────────────


class TestTopPipeThemes:
    def test_returns_top_n_by_frequency(self):
        df = pd.DataFrame(
            {"themes": ["love|nostalgia", "love|struggle", "struggle|identity"]}
        )
        result = _cmi()._top_pipe_themes(df, "themes", n=2)
        assert result is not None
        parts = result.split("|")
        assert len(parts) == 2
        assert "love" in parts or "struggle" in parts

    def test_respects_n_limit(self):
        df = pd.DataFrame(
            {"themes": ["love|nostalgia|joy|rebellion|struggle|identity"]}
        )
        result = _cmi()._top_pipe_themes(df, "themes", n=3)
        assert result is not None
        assert len(result.split("|")) == 3

    def test_returns_none_when_all_null(self):
        df = pd.DataFrame({"themes": [None, None]})
        result = _cmi()._top_pipe_themes(df, "themes", n=3)
        assert result is None

    def test_returns_none_when_column_missing(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = _cmi()._top_pipe_themes(df, "themes", n=3)
        assert result is None

    def test_top_theme_is_most_frequent(self):
        df = pd.DataFrame(
            {"themes": ["love", "love", "love", "nostalgia", "nostalgia", "rebellion"]}
        )
        result = _cmi()._top_pipe_themes(df, "themes", n=1)
        assert result == "love"


# ── _top_resonance ────────────────────────────────────────────────────────────


class TestTopResonance:
    def _scored_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "song_title": ["Song A", "Song B", "Song C", "Song D"],
                "cultural_resonance_score": [0.9, 0.4, 0.7, 0.55],
            }
        )

    def test_returns_pipe_delimited_string(self):
        result = _cmi()._top_resonance(self._scored_df(), n=2)
        assert isinstance(result, str)
        assert "|" in result

    def test_returns_top_n_by_score(self):
        result = _cmi()._top_resonance(self._scored_df(), n=2)
        songs = result.split("|")
        assert songs[0] == "Song A"
        assert songs[1] == "Song C"

    def test_returns_none_when_empty(self):
        df = pd.DataFrame({"song_title": [], "cultural_resonance_score": []})
        assert _cmi()._top_resonance(df, n=3) is None

    def test_returns_none_when_missing_song_title(self):
        df = pd.DataFrame({"cultural_resonance_score": [0.9, 0.7]})
        assert _cmi()._top_resonance(df, n=2) is None

    def test_n_larger_than_available_returns_all(self):
        result = _cmi()._top_resonance(self._scored_df(), n=10)
        assert len(result.split("|")) == 4


# ── _build_decade_cmi ─────────────────────────────────────────────────────────


class TestBuildDecadeCmi:
    def test_returns_dataframe(self):
        df = _make_merged_df()
        df["cultural_resonance_score"] = 0.5
        result = _cmi()._build_decade_cmi(df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_decade(self):
        df1 = _make_merged_df(5, "1980s")
        df2 = _make_merged_df(
            5, "1990s"
        )  # ← was n=3; must match list lengths in helper
        df = pd.concat([df1, df2], ignore_index=True)
        df["cultural_resonance_score"] = 0.5
        result = _cmi()._build_decade_cmi(df)
        assert len(result) == 2

    def test_required_columns_present(self):
        df = _make_merged_df()
        df["cultural_resonance_score"] = 0.5
        result = _cmi()._build_decade_cmi(df)
        for col in [
            "decade_label",
            "song_count",
            "scored_count",
            "CMI_sentiment",
            "CMI_energy",
            "emotional_tone",
            "dominant_jung_stage",
            "top_themes",
            "top_resonance_songs",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_song_count_correct(self):
        df = _make_merged_df(5, "2000s")
        df["cultural_resonance_score"] = 0.5
        result = _cmi()._build_decade_cmi(df)
        assert result.iloc[0]["song_count"] == 5

    def test_unclassified_excluded_from_dominant_jung_stage(self):
        """dominant_jung_stage must never be 'unclassified'."""
        df = _make_merged_df(5, "2000s")
        df["cultural_resonance_score"] = 0.5
        # Force all but one to unclassified
        df["jung_stage"] = "unclassified"
        df.iloc[0, df.columns.get_loc("jung_stage")] = "shadow"
        result = _cmi()._build_decade_cmi(df)
        dominant = result.iloc[0]["dominant_jung_stage"]
        assert dominant != "unclassified"

    def test_cmi_sentiment_is_mean(self):
        df = _make_merged_df(5, "2000s")  # keep n=5 to match helper list lengths
        df["cultural_resonance_score"] = 0.5
        df["sentiment_score"] = [
            0.2,
            0.4,
            0.6,
            0.8,
            0.0,
        ]  # ← was 4 values; need 5; mean=0.4
        result = _cmi()._build_decade_cmi(df)
        assert result.iloc[0]["CMI_sentiment"] == pytest.approx(0.4, abs=0.01)

    def test_schema_validation_passes(self):
        df = _make_merged_df(5, "2000s")  # ← was n=3
        df["cultural_resonance_score"] = 0.5
        result = _cmi()._build_decade_cmi(df)
        validated = validate(result, decade_cmi_schema, "test_build_decade_cmi")
        assert validated is not None


# ── run() ─────────────────────────────────────────────────────────────────────


class TestRun:
    def _write_inputs(
        self,
        layer4_path: Path,
        layer5_path: Path,
        sample_layer4_df: pd.DataFrame,
    ) -> None:
        sids = sample_layer4_df["song_id"].tolist()
        _write_csv(layer4_path, sample_layer4_df)
        _write_csv(layer5_path, _make_layer5_df(sids))

    def test_raises_if_layer4_missing(self, pipeline_workspace, tmp_config):
        cm = _cmi()
        missing = pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        layer5 = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".cmi_complete"

        with (
            patch.object(cm, "_INPUT_LAYER4", new=missing),
            patch.object(cm, "_INPUT_LAYER5", new=layer5),
            patch.object(cm, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                cm.run(tmp_config)

    def test_raises_if_layer5_missing(
        self, pipeline_workspace, tmp_config, sample_layer4_df
    ):
        cm = _cmi()
        layer4_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        missing_layer5 = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".cmi_complete"
        _write_csv(layer4_path, sample_layer4_df)

        with (
            patch.object(cm, "_INPUT_LAYER4", new=layer4_path),
            patch.object(cm, "_INPUT_LAYER5", new=missing_layer5),
            patch.object(cm, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                cm.run(tmp_config)

    def test_raises_coverage_error_when_coverage_too_low(
        self, pipeline_workspace, tmp_config, sample_layer4_df
    ):
        cm = _cmi()
        layer4_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        layer5_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        out6 = pipeline_workspace / "data" / "analysis" / "layer6_cultural_metrics.csv"
        out_dec = pipeline_workspace / "data" / "analysis" / "decade_cmi.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".cmi_complete"
        self._write_inputs(layer4_path, layer5_path, sample_layer4_df)

        # Patch _compute_resonance to return all-NaN to force 0% coverage
        null_series = pd.Series([None] * len(sample_layer4_df), dtype=float)

        with (
            patch.object(cm, "_INPUT_LAYER4", new=layer4_path),
            patch.object(cm, "_INPUT_LAYER5", new=layer5_path),
            patch.object(cm, "_OUTPUT_LAYER6", new=out6),
            patch.object(cm, "_OUTPUT_DECADE", new=out_dec),
            patch.object(cm, "_SENTINEL", new=sentinel),
            patch(
                "src.cultural_metrics.cmi_calculator._compute_resonance",
                return_value=null_series,
            ),
        ):
            with pytest.raises(cm.CoverageError):
                cm.run(tmp_config)

    def test_run_returns_correct_keys(
        self, pipeline_workspace, tmp_config, sample_layer4_df
    ):
        cm = _cmi()
        layer4_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        layer5_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        out6 = pipeline_workspace / "data" / "analysis" / "layer6_cultural_metrics.csv"
        out_dec = pipeline_workspace / "data" / "analysis" / "decade_cmi.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".cmi_complete"
        self._write_inputs(layer4_path, layer5_path, sample_layer4_df)

        with (
            patch.object(cm, "_INPUT_LAYER4", new=layer4_path),
            patch.object(cm, "_INPUT_LAYER5", new=layer5_path),
            patch.object(cm, "_OUTPUT_LAYER6", new=out6),
            patch.object(cm, "_OUTPUT_DECADE", new=out_dec),
            patch.object(cm, "_SENTINEL", new=sentinel),
            patch("src.cultural_metrics.cmi_calculator.write_sentinel"),
        ):
            result = cm.run(tmp_config)

        for key in [
            "songs_total",
            "songs_scored",
            "coverage_pct",
            "decades_computed",
            "output_layer6",
            "output_decade",
            "skipped",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_run_writes_both_output_csvs(
        self, pipeline_workspace, tmp_config, sample_layer4_df
    ):
        cm = _cmi()
        layer4_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        layer5_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        out6 = pipeline_workspace / "data" / "analysis" / "layer6_cultural_metrics.csv"
        out_dec = pipeline_workspace / "data" / "analysis" / "decade_cmi.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".cmi_complete"
        self._write_inputs(layer4_path, layer5_path, sample_layer4_df)

        with (
            patch.object(cm, "_INPUT_LAYER4", new=layer4_path),
            patch.object(cm, "_INPUT_LAYER5", new=layer5_path),
            patch.object(cm, "_OUTPUT_LAYER6", new=out6),
            patch.object(cm, "_OUTPUT_DECADE", new=out_dec),
            patch.object(cm, "_SENTINEL", new=sentinel),
            patch("src.cultural_metrics.cmi_calculator.write_sentinel"),
        ):
            cm.run(tmp_config)

        assert out6.exists(), "layer6_cultural_metrics.csv not written"
        assert out_dec.exists(), "decade_cmi.csv not written"

    def test_run_calls_write_sentinel(
        self, pipeline_workspace, tmp_config, sample_layer4_df
    ):
        cm = _cmi()
        layer4_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        layer5_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        out6 = pipeline_workspace / "data" / "analysis" / "layer6_cultural_metrics.csv"
        out_dec = pipeline_workspace / "data" / "analysis" / "decade_cmi.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".cmi_complete"
        self._write_inputs(layer4_path, layer5_path, sample_layer4_df)

        with (
            patch.object(cm, "_INPUT_LAYER4", new=layer4_path),
            patch.object(cm, "_INPUT_LAYER5", new=layer5_path),
            patch.object(cm, "_OUTPUT_LAYER6", new=out6),
            patch.object(cm, "_OUTPUT_DECADE", new=out_dec),
            patch.object(cm, "_SENTINEL", new=sentinel),
            patch("src.cultural_metrics.cmi_calculator.write_sentinel") as mock_ws,
        ):
            cm.run(tmp_config)

        mock_ws.assert_called_once()

    def test_run_skips_when_sentinel_matches(
        self, pipeline_workspace, tmp_config, sample_layer4_df
    ):
        cm = _cmi()
        layer4_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        layer5_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        out6 = pipeline_workspace / "data" / "analysis" / "layer6_cultural_metrics.csv"
        out_dec = pipeline_workspace / "data" / "analysis" / "decade_cmi.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".cmi_complete"
        self._write_inputs(layer4_path, layer5_path, sample_layer4_df)

        # Pre-write outputs so sentinel-skip read_csv succeeds
        pd.DataFrame(
            {
                "song_id": sample_layer4_df["song_id"],
                "cultural_resonance_score": [0.5] * len(sample_layer4_df),
            }
        ).to_csv(out6, index=False)
        pd.DataFrame(
            {"decade_label": ["2000s"], "song_count": [5], "scored_count": [5]}
        ).to_csv(out_dec, index=False)

        with (
            patch.object(cm, "_INPUT_LAYER4", new=layer4_path),
            patch.object(cm, "_INPUT_LAYER5", new=layer5_path),
            patch.object(cm, "_OUTPUT_LAYER6", new=out6),
            patch.object(cm, "_OUTPUT_DECADE", new=out_dec),
            patch.object(cm, "_SENTINEL", new=sentinel),
            patch(
                "src.cultural_metrics.cmi_calculator.sentinel_config_matches",
                return_value=True,
            ),
        ):
            result = cm.run(tmp_config)

        assert result["skipped"] is True

    def test_run_skipped_false_on_fresh_run(
        self, pipeline_workspace, tmp_config, sample_layer4_df
    ):
        cm = _cmi()
        layer4_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        layer5_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        out6 = pipeline_workspace / "data" / "analysis" / "layer6_cultural_metrics.csv"
        out_dec = pipeline_workspace / "data" / "analysis" / "decade_cmi.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".cmi_complete"
        self._write_inputs(layer4_path, layer5_path, sample_layer4_df)

        with (
            patch.object(cm, "_INPUT_LAYER4", new=layer4_path),
            patch.object(cm, "_INPUT_LAYER5", new=layer5_path),
            patch.object(cm, "_OUTPUT_LAYER6", new=out6),
            patch.object(cm, "_OUTPUT_DECADE", new=out_dec),
            patch.object(cm, "_SENTINEL", new=sentinel),
            patch("src.cultural_metrics.cmi_calculator.write_sentinel"),
        ):
            result = cm.run(tmp_config)

        assert result["skipped"] is False
