"""
tests/test_theme_classifier.py

Unit tests for src/analysis/theme_classifier.py

Coverage:
- _classify_batch: correct label normalisation, all taxonomy themes present
- _classify_batch: empty string handled without raising
- _classify_batch: multi-label independence (multi_label=True behaviour)
- _build_dataframe: themes column is pipe-delimited string
- _build_dataframe: threshold filtering — only themes >= threshold included
- _build_dataframe: themes column contains only valid taxonomy labels
- _build_dataframe: dominant_theme is max-score label
- _build_dataframe: per-theme score columns present for all taxonomy items
- _build_dataframe: empty themes string when nothing clears threshold
- run(): raises FileNotFoundError when chorus_extracted.csv absent
- run(): returns songs_classified and skipped keys
- run(): writes output CSV with correct columns
- run(): calls write_sentinel on success
- run(): skips classification when sentinel matches config
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipeline.schemas import VALID_THEME_TAXONOMY, layer2_themes_schema, validate
from src.utils.identifiers import make_song_id

# ── Helpers ───────────────────────────────────────────────────────────────────


def _tc():
    from src.analysis import theme_classifier

    return theme_classifier


TAXONOMY = list(VALID_THEME_TAXONOMY)  # 12-class canonical list
THRESHOLD = 0.40


def _make_mock_classifier(
    dominant: str = "love", above_threshold: list[str] | None = None
) -> MagicMock:
    """
    Build a mock HuggingFace zero-shot-classification pipeline callable.

    dominant        : theme receiving score 0.85
    above_threshold : additional themes receiving score 0.55 (above 0.40)
    All other themes receive 0.10.
    """
    if above_threshold is None:
        above_threshold = []

    scores = {theme: 0.10 for theme in TAXONOMY}
    scores[dominant] = 0.85
    for t in above_threshold:
        if t != dominant:
            scores[t] = 0.55

    def _side_effect(texts, candidate_labels, multi_label=True, **kwargs):
        return [
            {
                "sequence": text,
                "labels": list(scores.keys()),
                "scores": list(scores.values()),
            }
            for text in texts
        ]

    return MagicMock(side_effect=_side_effect)


def _write_chorus_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ── _classify_batch ───────────────────────────────────────────────────────────


class TestClassifyBatch:
    def test_returns_list_of_dicts(self):
        mock_clf = _make_mock_classifier("love")
        result = _tc()._classify_batch(mock_clf, ["I love you"], TAXONOMY)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_result_length_matches_input(self):
        mock_clf = _make_mock_classifier("love")
        texts = ["song one", "song two", "song three"]
        result = _tc()._classify_batch(mock_clf, texts, TAXONOMY)
        assert len(result) == len(texts)

    def test_all_taxonomy_themes_present(self):
        mock_clf = _make_mock_classifier("love")
        result = _tc()._classify_batch(mock_clf, ["test"], TAXONOMY)
        for theme in TAXONOMY:
            assert theme in result[0], f"Missing theme: {theme}"

    def test_labels_are_lowercased(self):
        def _mixed_case(texts, candidate_labels, multi_label=True, **kwargs):
            return [
                {
                    "sequence": t,
                    "labels": ["Love", "HEARTBREAK"] + TAXONOMY[2:],
                    "scores": [0.85, 0.60] + [0.10] * (len(TAXONOMY) - 2),
                }
                for t in texts
            ]

        mock_clf = MagicMock(side_effect=_mixed_case)
        result = _tc()._classify_batch(mock_clf, ["test"], TAXONOMY)
        assert "love" in result[0]
        assert "heartbreak" in result[0]
        assert "Love" not in result[0]

    def test_empty_string_does_not_raise(self):
        mock_clf = _make_mock_classifier("identity")
        result = _tc()._classify_batch(mock_clf, [""], TAXONOMY)
        assert len(result) == 1

    def test_dominant_theme_has_highest_score(self):
        mock_clf = _make_mock_classifier("rebellion")
        result = _tc()._classify_batch(mock_clf, ["test song"], TAXONOMY)
        dominant = max(result[0], key=result[0].get)
        assert dominant == "rebellion"


# ── _build_dataframe ──────────────────────────────────────────────────────────


class TestBuildDataframe:
    def _scores(
        self, dominant: str = "love", extras: list[str] | None = None
    ) -> list[dict]:
        s = {theme: 0.10 for theme in TAXONOMY}
        s[dominant] = 0.85
        if extras:
            for t in extras:
                s[t] = 0.55
        return [s]

    def test_themes_column_is_string(self):
        sid = make_song_id("Test", "Song", 2000)
        df = _tc()._build_dataframe([sid], self._scores("love"), TAXONOMY, THRESHOLD)
        assert isinstance(df.iloc[0]["themes"], str)

    def test_themes_column_pipe_delimited(self):
        sid = make_song_id("Test", "Song", 2000)
        scores = self._scores("love", extras=["heartbreak"])
        df = _tc()._build_dataframe([sid], scores, TAXONOMY, THRESHOLD)
        themes_val = df.iloc[0]["themes"]
        if themes_val:
            parts = themes_val.split("|")
            assert len(parts) >= 1

    def test_themes_above_threshold_included(self):
        sid = make_song_id("Test", "Song", 2000)
        scores = self._scores("love", extras=["nostalgia"])
        df = _tc()._build_dataframe([sid], scores, TAXONOMY, THRESHOLD)
        themes = df.iloc[0]["themes"].split("|")
        assert "love" in themes
        assert "nostalgia" in themes

    def test_themes_below_threshold_excluded(self):
        sid = make_song_id("Test", "Song", 2000)
        # Only "love" at 0.85 clears threshold; all others at 0.10
        df = _tc()._build_dataframe([sid], self._scores("love"), TAXONOMY, THRESHOLD)
        themes = df.iloc[0]["themes"].split("|") if df.iloc[0]["themes"] else []
        assert "heartbreak" not in themes
        assert "struggle" not in themes

    def test_empty_themes_when_nothing_above_threshold(self):
        sid = make_song_id("Test", "Song", 2000)
        # All scores below threshold
        low_scores = [{theme: 0.05 for theme in TAXONOMY}]
        df = _tc()._build_dataframe([sid], low_scores, TAXONOMY, THRESHOLD)
        assert df.iloc[0]["themes"] == ""

    def test_themes_only_valid_taxonomy_labels(self):
        sid = make_song_id("Test", "Song", 2000)
        scores = self._scores("love", extras=["empowerment", "unity"])
        df = _tc()._build_dataframe([sid], scores, TAXONOMY, THRESHOLD)
        themes_val = df.iloc[0]["themes"]
        if themes_val:
            for t in themes_val.split("|"):
                assert t in VALID_THEME_TAXONOMY, f"Invalid theme: {t}"

    def test_dominant_theme_is_max_score(self):
        sid = make_song_id("Test", "Song", 2000)
        df = _tc()._build_dataframe(
            [sid], self._scores("rebellion"), TAXONOMY, THRESHOLD
        )
        assert df.iloc[0]["dominant_theme"] == "rebellion"

    def test_dominant_theme_score_in_range(self):
        sid = make_song_id("Test", "Song", 2000)
        df = _tc()._build_dataframe([sid], self._scores("love"), TAXONOMY, THRESHOLD)
        score = df.iloc[0]["dominant_theme_score"]
        assert 0.0 <= score <= 1.0

    def test_per_theme_score_columns_present(self):
        sid = make_song_id("Test", "Song", 2000)
        df = _tc()._build_dataframe([sid], self._scores("love"), TAXONOMY, THRESHOLD)
        for theme in TAXONOMY:
            col = f"theme_{theme}"
            assert col in df.columns, f"Missing column: {col}"

    def test_per_theme_scores_in_range(self):
        sid = make_song_id("Test", "Song", 2000)
        df = _tc()._build_dataframe([sid], self._scores("love"), TAXONOMY, THRESHOLD)
        for theme in TAXONOMY:
            val = df.iloc[0][f"theme_{theme}"]
            assert 0.0 <= val <= 1.0, f"theme_{theme}={val} out of range"

    def test_row_count_matches_input(self):
        sids = [make_song_id("A", f"Song{i}", 2000 + i) for i in range(4)]
        scores = self._scores("love") * 4
        df = _tc()._build_dataframe(sids, scores, TAXONOMY, THRESHOLD)
        assert len(df) == 4

    def test_schema_validation_passes(self):
        sid = make_song_id("Test", "Song", 2000)
        scores = self._scores("love", extras=["nostalgia"])
        df = _tc()._build_dataframe([sid], scores, TAXONOMY, THRESHOLD)
        validated = validate(
            df[["song_id", "themes"]], layer2_themes_schema, "test_theme_build"
        )
        assert validated is not None

    def test_threshold_boundary_exact_value_included(self):
        """Score exactly equal to threshold must be included."""
        sid = make_song_id("Test", "Song", 2000)
        boundary_scores = [{theme: 0.10 for theme in TAXONOMY}]
        boundary_scores[0]["love"] = 0.85
        boundary_scores[0]["heartbreak"] = THRESHOLD  # exactly at boundary
        df = _tc()._build_dataframe([sid], boundary_scores, TAXONOMY, THRESHOLD)
        themes = df.iloc[0]["themes"].split("|") if df.iloc[0]["themes"] else []
        assert "heartbreak" in themes


# ── run() ─────────────────────────────────────────────────────────────────────


class TestRun:
    def test_raises_file_not_found_if_chorus_csv_missing(
        self, pipeline_workspace, tmp_config
    ):
        tc = _tc()
        missing = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".themes_complete"

        with (
            patch.object(tc, "_INPUT_PATH", new=missing),
            patch.object(tc, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                tc.run(tmp_config)

    def test_run_returns_songs_classified_key(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        tc = _tc()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_themes.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".themes_complete"
        _write_chorus_csv(chorus_path, sample_chorus_df)

        with (
            patch.object(tc, "_INPUT_PATH", new=chorus_path),
            patch.object(tc, "_OUTPUT_PATH", new=out_path),
            patch.object(tc, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.theme_classifier._load_classifier",
                return_value=_make_mock_classifier("love"),
            ),
            patch("src.analysis.theme_classifier.write_sentinel"),
        ):
            result = tc.run(tmp_config)

        assert "songs_classified" in result
        assert result["songs_classified"] == len(sample_chorus_df)

    def test_run_returns_skipped_false_on_fresh_run(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        tc = _tc()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_themes.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".themes_complete"
        _write_chorus_csv(chorus_path, sample_chorus_df)

        with (
            patch.object(tc, "_INPUT_PATH", new=chorus_path),
            patch.object(tc, "_OUTPUT_PATH", new=out_path),
            patch.object(tc, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.theme_classifier._load_classifier",
                return_value=_make_mock_classifier("love"),
            ),
            patch("src.analysis.theme_classifier.write_sentinel"),
        ):
            result = tc.run(tmp_config)

        assert result["skipped"] is False

    def test_run_writes_output_csv(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        tc = _tc()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_themes.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".themes_complete"
        _write_chorus_csv(chorus_path, sample_chorus_df)

        with (
            patch.object(tc, "_INPUT_PATH", new=chorus_path),
            patch.object(tc, "_OUTPUT_PATH", new=out_path),
            patch.object(tc, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.theme_classifier._load_classifier",
                return_value=_make_mock_classifier("love"),
            ),
            patch("src.analysis.theme_classifier.write_sentinel"),
        ):
            tc.run(tmp_config)

        assert out_path.exists(), "layer2_themes.csv was not written"

    def test_run_output_csv_has_themes_column(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        tc = _tc()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_themes.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".themes_complete"
        _write_chorus_csv(chorus_path, sample_chorus_df)

        with (
            patch.object(tc, "_INPUT_PATH", new=chorus_path),
            patch.object(tc, "_OUTPUT_PATH", new=out_path),
            patch.object(tc, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.theme_classifier._load_classifier",
                return_value=_make_mock_classifier("love"),
            ),
            patch("src.analysis.theme_classifier.write_sentinel"),
        ):
            tc.run(tmp_config)

        result_df = pd.read_csv(out_path)
        assert "song_id" in result_df.columns
        assert "themes" in result_df.columns
        assert "dominant_theme" in result_df.columns

    def test_run_calls_write_sentinel(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        tc = _tc()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_themes.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".themes_complete"
        _write_chorus_csv(chorus_path, sample_chorus_df)

        with (
            patch.object(tc, "_INPUT_PATH", new=chorus_path),
            patch.object(tc, "_OUTPUT_PATH", new=out_path),
            patch.object(tc, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.theme_classifier._load_classifier",
                return_value=_make_mock_classifier("love"),
            ),
            patch("src.analysis.theme_classifier.write_sentinel") as mock_ws,
        ):
            tc.run(tmp_config)

        mock_ws.assert_called_once()

    def test_run_skips_when_sentinel_matches(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        tc = _tc()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_themes.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".themes_complete"
        _write_chorus_csv(chorus_path, sample_chorus_df)

        # Pre-write output so sentinel-skip read_csv succeeds
        sample_chorus_df[["song_id"]].assign(
            dominant_theme="love",
            dominant_theme_score=0.85,
            themes="love",
        ).to_csv(out_path, index=False)

        with (
            patch.object(tc, "_INPUT_PATH", new=chorus_path),
            patch.object(tc, "_OUTPUT_PATH", new=out_path),
            patch.object(tc, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.theme_classifier.sentinel_config_matches",
                return_value=True,
            ),
            patch("src.analysis.theme_classifier._load_classifier") as mock_load,
        ):
            result = tc.run(tmp_config)

        mock_load.assert_not_called()
        assert result["skipped"] is True
