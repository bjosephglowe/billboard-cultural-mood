"""
tests/test_emotion_classifier.py

Unit tests for src/analysis/emotion_classifier.py

Coverage:
- _classify_batch: correct label normalisation, all 7 Ekman labels present
- _classify_batch: empty string handled without raising
- _classify_batch: dominant label is max-score label
- _build_dataframe: output columns, emotional_tone, score range
- _build_dataframe: all 7 emotion_* columns present
- run(): raises FileNotFoundError when chorus_extracted.csv absent
- run(): calls _load_classifier and _classify_batch, writes output CSV
- run(): return dict contains songs_classified and skipped keys
- run(): sentinel written via write_sentinel after successful run
- run(): skips classification when sentinel matches config
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.pipeline.schemas import VALID_EMOTIONAL_TONES, layer2_emotion_schema, validate
from src.utils.identifiers import make_song_id

# ── Helpers ───────────────────────────────────────────────────────────────────


def _ec():
    from src.analysis import emotion_classifier

    return emotion_classifier


EKMAN_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def _make_mock_classifier(dominant: str = "joy") -> MagicMock:
    """
    Build a mock HuggingFace pipeline callable that returns deterministic
    Ekman scores for any input batch.

    dominant: the label that receives score 0.80; others share 0.20/6.
    """
    other_score = round(0.20 / 6, 6)
    base_scores = {label: other_score for label in EKMAN_LABELS}
    base_scores[dominant] = 0.80

    def _side_effect(texts, **kwargs):
        # HuggingFace pipeline returns list[list[dict]] when top_k=None
        return [
            [{"label": lbl, "score": score} for lbl, score in base_scores.items()]
            for _ in texts
        ]

    mock = MagicMock(side_effect=_side_effect)
    return mock


def _make_chorus_df(song_ids: list[str], texts: list[str]) -> pd.DataFrame:
    """Build a minimal chorus_extracted.csv fixture."""
    return pd.DataFrame(
        {
            "song_id": song_ids,
            "chorus_detected": [True] * len(song_ids),
            "chorus_method": ["tag"] * len(song_ids),
            "chorus_text": texts,
            "chorus_token_count": [len(t.split()) if t else 0 for t in texts],
        }
    )


# ── _classify_batch ───────────────────────────────────────────────────────────


class TestClassifyBatch:
    def test_returns_list_of_dicts(self):
        mock_clf = _make_mock_classifier("joy")
        result = _ec()._classify_batch(mock_clf, ["I love this song"])
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_result_length_matches_input(self):
        mock_clf = _make_mock_classifier("joy")
        texts = ["song one", "song two", "song three"]
        result = _ec()._classify_batch(mock_clf, texts)
        assert len(result) == len(texts)

    def test_all_seven_labels_present(self):
        mock_clf = _make_mock_classifier("joy")
        result = _ec()._classify_batch(mock_clf, ["test text"])
        for label in EKMAN_LABELS:
            assert label in result[0], f"Missing label: {label}"

    def test_labels_are_lowercased(self):
        # Simulate model returning mixed-case labels
        def _mixed_case(texts, **kwargs):
            return [
                [
                    {"label": "Joy", "score": 0.8},
                    {"label": "ANGER", "score": 0.03},
                    {"label": "Disgust", "score": 0.03},
                    {"label": "Fear", "score": 0.03},
                    {"label": "Neutral", "score": 0.03},
                    {"label": "Sadness", "score": 0.04},
                    {"label": "Surprise", "score": 0.04},
                ]
                for _ in texts
            ]

        mock_clf = MagicMock(side_effect=_mixed_case)
        result = _ec()._classify_batch(mock_clf, ["test"])
        assert "joy" in result[0]
        assert "anger" in result[0]
        assert "Joy" not in result[0]

    def test_empty_string_does_not_raise(self):
        mock_clf = _make_mock_classifier("neutral")
        result = _ec()._classify_batch(mock_clf, [""])
        assert len(result) == 1
        assert "neutral" in result[0]

    def test_scores_sum_approximately_one(self):
        mock_clf = _make_mock_classifier("joy")
        result = _ec()._classify_batch(mock_clf, ["happy song"])
        total = sum(result[0].values())
        assert abs(total - 1.0) < 0.01, f"Scores sum to {total}, expected ~1.0"

    def test_dominant_label_has_highest_score(self):
        mock_clf = _make_mock_classifier("sadness")
        result = _ec()._classify_batch(mock_clf, ["sad song"])
        dominant = max(result[0], key=result[0].get)
        assert dominant == "sadness"


# ── _build_dataframe ──────────────────────────────────────────────────────────


class TestBuildDataframe:
    def _scores(self, dominant: str = "joy") -> list[dict]:
        other = round(0.20 / 6, 6)
        base = {label: other for label in EKMAN_LABELS}
        base[dominant] = 0.80
        return [base]

    def test_output_has_song_id_column(self):
        sid = make_song_id("Test", "Song", 2000)
        df = _ec()._build_dataframe([sid], self._scores("joy"))
        assert "song_id" in df.columns

    def test_dominant_emotion_column_present(self):
        sid = make_song_id("Test", "Song", 2000)
        df = _ec()._build_dataframe([sid], self._scores("anger"))
        assert "emotional_tone" in df.columns

    def test_dominant_emotion_is_max_label(self):
        sid = make_song_id("Test", "Song", 2000)
        df = _ec()._build_dataframe([sid], self._scores("fear"))
        assert df.iloc[0]["emotional_tone"] == "fear"

    def test_dominant_emotion_score_range(self):
        sid = make_song_id("Test", "Song", 2000)
        df = _ec()._build_dataframe([sid], self._scores("joy"))
        score = df.iloc[0]["dominant_emotion_score"]
        assert 0.0 <= score <= 1.0

    def test_all_emotion_columns_present(self):
        sid = make_song_id("Test", "Song", 2000)
        df = _ec()._build_dataframe([sid], self._scores("joy"))
        for label in EKMAN_LABELS:
            assert f"emotion_{label}" in df.columns, f"Missing column: emotion_{label}"

    def test_per_emotion_scores_in_range(self):
        sid = make_song_id("Test", "Song", 2000)
        df = _ec()._build_dataframe([sid], self._scores("joy"))
        for label in EKMAN_LABELS:
            val = df.iloc[0][f"emotion_{label}"]
            assert 0.0 <= val <= 1.0, f"emotion_{label}={val} out of range"

    def test_row_count_matches_input(self):
        sids = [make_song_id("A", f"Song{i}", 2000 + i) for i in range(3)]
        scores = self._scores("joy") * 3
        df = _ec()._build_dataframe(sids, scores)
        assert len(df) == 3

    def test_dominant_emotion_label_in_ekman_set(self):
        sid = make_song_id("Test", "Song", 2000)
        for dominant in EKMAN_LABELS:
            df = _ec()._build_dataframe([sid], self._scores(dominant))
            assert df.iloc[0]["emotional_tone"] in EKMAN_LABELS


# ── run() ─────────────────────────────────────────────────────────────────────


class TestRun:
    def _write_chorus_csv(self, path: Path, sample_chorus_df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sample_chorus_df.to_csv(path, index=False)

    def test_raises_file_not_found_if_chorus_csv_missing(
        self, pipeline_workspace, tmp_config
    ):
        ec = _ec()
        missing = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".emotion_complete"

        with (
            patch.object(ec, "_INPUT_PATH", new=missing),
            patch.object(ec, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                ec.run(tmp_config)

    def test_run_returns_songs_classified_key(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        ec = _ec()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_emotion.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".emotion_complete"
        self._write_chorus_csv(chorus_path, sample_chorus_df)

        mock_clf = _make_mock_classifier("joy")

        with (
            patch.object(ec, "_INPUT_PATH", new=chorus_path),
            patch.object(ec, "_OUTPUT_PATH", new=out_path),
            patch.object(ec, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.emotion_classifier._load_classifier",
                return_value=mock_clf,
            ),
            patch("src.analysis.emotion_classifier.write_sentinel"),
        ):
            result = ec.run(tmp_config)

        assert "songs_classified" in result
        assert result["songs_classified"] == len(sample_chorus_df)

    def test_run_returns_skipped_false_on_fresh_run(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        ec = _ec()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_emotion.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".emotion_complete"
        self._write_chorus_csv(chorus_path, sample_chorus_df)

        with (
            patch.object(ec, "_INPUT_PATH", new=chorus_path),
            patch.object(ec, "_OUTPUT_PATH", new=out_path),
            patch.object(ec, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.emotion_classifier._load_classifier",
                return_value=_make_mock_classifier("joy"),
            ),
            patch("src.analysis.emotion_classifier.write_sentinel"),
        ):
            result = ec.run(tmp_config)

        assert result["skipped"] is False

    def test_run_writes_output_csv(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        ec = _ec()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_emotion.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".emotion_complete"
        self._write_chorus_csv(chorus_path, sample_chorus_df)

        with (
            patch.object(ec, "_INPUT_PATH", new=chorus_path),
            patch.object(ec, "_OUTPUT_PATH", new=out_path),
            patch.object(ec, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.emotion_classifier._load_classifier",
                return_value=_make_mock_classifier("neutral"),
            ),
            patch("src.analysis.emotion_classifier.write_sentinel"),
        ):
            ec.run(tmp_config)

        assert out_path.exists(), "layer2_emotion.csv was not written"

    def test_run_output_csv_has_correct_columns(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        ec = _ec()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_emotion.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".emotion_complete"
        self._write_chorus_csv(chorus_path, sample_chorus_df)

        with (
            patch.object(ec, "_INPUT_PATH", new=chorus_path),
            patch.object(ec, "_OUTPUT_PATH", new=out_path),
            patch.object(ec, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.emotion_classifier._load_classifier",
                return_value=_make_mock_classifier("joy"),
            ),
            patch("src.analysis.emotion_classifier.write_sentinel"),
        ):
            ec.run(tmp_config)

        result_df = pd.read_csv(out_path)
        assert "song_id" in result_df.columns
        assert "emotional_tone" in result_df.columns
        assert "dominant_emotion_score" in result_df.columns
        for label in EKMAN_LABELS:
            assert f"emotion_{label}" in result_df.columns

    def test_run_calls_write_sentinel(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        ec = _ec()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_emotion.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".emotion_complete"
        self._write_chorus_csv(chorus_path, sample_chorus_df)

        with (
            patch.object(ec, "_INPUT_PATH", new=chorus_path),
            patch.object(ec, "_OUTPUT_PATH", new=out_path),
            patch.object(ec, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.emotion_classifier._load_classifier",
                return_value=_make_mock_classifier("joy"),
            ),
            patch("src.analysis.emotion_classifier.write_sentinel") as mock_ws,
        ):
            ec.run(tmp_config)

        mock_ws.assert_called_once()

    def test_run_skips_when_sentinel_matches(
        self, pipeline_workspace, tmp_config, sample_chorus_df
    ):
        ec = _ec()
        chorus_path = pipeline_workspace / "data" / "processed" / "chorus_extracted.csv"
        out_path = pipeline_workspace / "data" / "analysis" / "layer2_emotion.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".emotion_complete"
        self._write_chorus_csv(chorus_path, sample_chorus_df)

        # Pre-write output so the sentinel-skip read_csv succeeds
        sample_chorus_df[["song_id"]].assign(
            emotional_tone="joy",
            dominant_emotion_score=0.8,
        ).to_csv(out_path, index=False)

        with (
            patch.object(ec, "_INPUT_PATH", new=chorus_path),
            patch.object(ec, "_OUTPUT_PATH", new=out_path),
            patch.object(ec, "_SENTINEL", new=sentinel),
            patch(
                "src.analysis.emotion_classifier.sentinel_config_matches",
                return_value=True,
            ),
            patch("src.analysis.emotion_classifier._load_classifier") as mock_load,
        ):
            result = ec.run(tmp_config)

        mock_load.assert_not_called()
        assert result["skipped"] is True
