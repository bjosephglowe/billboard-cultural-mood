"""
tests/test_jungian_scorer.py

Unit tests for src/psychology/jungian_scorer.py

Coverage:
- _assign_quality_flag: high vs low flag logic
- _fallback_records: structure, jung_stage, null fields
- _build_fallback_dataframe: columns, length, all unclassified
- _build_songs_block: line count, song_id present in output
- _parse_response: valid array, wrapped object, missing song_id,
  invalid jung_stage coercion, out-of-range dev score, null theme
- run(): FileNotFoundError when input CSV absent
- run(): fallback path when API unavailable (no key)
- run(): returns correct keys on success
- run(): writes output CSV and sentinel
- run(): sentinel skip path
- run(): api_available=False in fallback summary
- run(): all songs unclassified in fallback mode
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipeline.schemas import VALID_JUNG_STAGES, layer5_schema, validate
from src.utils.identifiers import make_song_id

# ── Helpers ───────────────────────────────────────────────────────────────────


def _js():
    from src.psychology import jungian_scorer

    return jungian_scorer


VALID_STAGES = list(VALID_JUNG_STAGES)


def _make_layer4_df(n: int = 3) -> pd.DataFrame:
    """Build a minimal layer2_full_analysis.csv fixture."""
    sids = [make_song_id("Artist", f"Song{i}", 2000 + i) for i in range(n)]
    return pd.DataFrame(
        {
            "song_id": sids,
            "song_title": [f"Song{i}" for i in range(n)],
            "artist": ["Artist"] * n,
            "decade": ["2000s"] * n,
            "lyrics_quality": ["full"] * n,
            "sentiment_score": [0.5] * n,
            "energy_level": [3] * n,
            "emotional_tone": ["joy"] * n,
            "themes": ["love|nostalgia"] * n,
            "dominant_theme": ["love"] * n,
            "chorus_detected": [True] * n,
            "contrast_sentiment_index": [0.1] * n,
            "energy_shift": ["stable"] * n,
            "theme_shift": ["none"] * n,
        }
    )


def _make_gpt_response(song_ids: list[str], stage: str = "shadow") -> str:
    """Build a valid JSON string mimicking a GPT-4o response."""
    records = [
        {
            "song_id": sid,
            "jung_stage": stage,
            "psychological_theme": "darkness and repression",
            "development_score": 2,
        }
        for sid in song_ids
    ]
    return json.dumps(records)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ── _assign_quality_flag ──────────────────────────────────────────────────────


class TestAssignQualityFlag:
    def test_high_when_classified_with_score(self):
        row = pd.Series({"jung_stage": "shadow", "development_score": 3})
        assert _js()._assign_quality_flag(row) == "high"

    def test_low_when_unclassified(self):
        row = pd.Series({"jung_stage": "unclassified", "development_score": 3})
        assert _js()._assign_quality_flag(row) == "low"

    def test_low_when_score_null(self):
        row = pd.Series({"jung_stage": "shadow", "development_score": None})
        assert _js()._assign_quality_flag(row) == "low"

    def test_low_when_unclassified_and_score_null(self):
        row = pd.Series({"jung_stage": "unclassified", "development_score": None})
        assert _js()._assign_quality_flag(row) == "low"

    def test_high_for_every_valid_non_unclassified_stage(self):
        for stage in VALID_STAGES:
            if stage == "unclassified":
                continue
            row = pd.Series({"jung_stage": stage, "development_score": 4})
            assert _js()._assign_quality_flag(row) == "high", (
                f"Failed for stage: {stage}"
            )


# ── _fallback_records ─────────────────────────────────────────────────────────


class TestFallbackRecords:
    def test_length_matches_input(self):
        sids = [make_song_id("A", f"S{i}", 2000 + i) for i in range(4)]
        result = _js()._fallback_records(sids)
        assert len(result) == 4

    def test_all_unclassified(self):
        sids = [make_song_id("A", f"S{i}", 2000 + i) for i in range(3)]
        result = _js()._fallback_records(sids)
        for rec in result:
            assert rec["jung_stage"] == "unclassified"

    def test_development_score_is_none(self):
        sids = [make_song_id("A", "S0", 2000)]
        result = _js()._fallback_records(sids)
        assert result[0]["development_score"] is None

    def test_psychological_theme_is_none(self):
        sids = [make_song_id("A", "S0", 2000)]
        result = _js()._fallback_records(sids)
        assert result[0]["psychological_theme"] is None

    def test_song_ids_preserved(self):
        sids = [make_song_id("A", f"S{i}", 2000 + i) for i in range(3)]
        result = _js()._fallback_records(sids)
        assert [r["song_id"] for r in result] == sids


# ── _build_fallback_dataframe ─────────────────────────────────────────────────


class TestBuildFallbackDataframe:
    def test_returns_dataframe(self):
        sids = [make_song_id("A", f"S{i}", 2000 + i) for i in range(3)]
        df = _js()._build_fallback_dataframe(sids)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches(self):
        sids = [make_song_id("A", f"S{i}", 2000 + i) for i in range(5)]
        df = _js()._build_fallback_dataframe(sids)
        assert len(df) == 5

    def test_required_columns_present(self):
        sids = [make_song_id("A", "S0", 2000)]
        df = _js()._build_fallback_dataframe(sids)
        for col in [
            "song_id",
            "jung_stage",
            "psychological_theme",
            "development_score",
        ]:
            assert col in df.columns

    def test_all_unclassified(self):
        sids = [make_song_id("A", f"S{i}", 2000 + i) for i in range(3)]
        df = _js()._build_fallback_dataframe(sids)
        assert (df["jung_stage"] == "unclassified").all()


# ── _build_songs_block ────────────────────────────────────────────────────────


class TestBuildSongsBlock:
    def test_line_count_matches_batch_size(self):
        layer4 = _make_layer4_df(4)
        sids = layer4["song_id"].tolist()
        block = _js()._build_songs_block(sids, layer4)
        lines = [l for l in block.splitlines() if l.strip()]
        assert len(lines) == 4

    def test_song_ids_present_in_output(self):
        layer4 = _make_layer4_df(3)
        sids = layer4["song_id"].tolist()
        block = _js()._build_songs_block(sids, layer4)
        for sid in sids:
            assert sid in block

    def test_returns_string(self):
        layer4 = _make_layer4_df(2)
        sids = layer4["song_id"].tolist()
        block = _js()._build_songs_block(sids, layer4)
        assert isinstance(block, str)


# ── _parse_response ───────────────────────────────────────────────────────────


class TestParseResponse:
    def test_valid_array_returns_records(self):
        sids = [make_song_id("A", f"S{i}", 2000 + i) for i in range(2)]
        content = _make_gpt_response(sids, "shadow")
        result = _js()._parse_response(content, sids)
        assert result is not None
        assert len(result) == 2

    def test_wrapped_object_unwrapped_correctly(self):
        sids = [make_song_id("A", "S0", 2000)]
        records = [
            {
                "song_id": sids[0],
                "jung_stage": "persona",
                "psychological_theme": "social mask",
                "development_score": 3,
            }
        ]
        wrapped = json.dumps({"classifications": records})
        result = _js()._parse_response(wrapped, sids)
        assert result is not None
        assert result[0]["jung_stage"] == "persona"

    def test_missing_song_id_returns_none(self):
        sids = [make_song_id("A", "S0", 2000), make_song_id("A", "S1", 2001)]
        # Only return one of the two songs
        partial = json.dumps(
            [
                {
                    "song_id": sids[0],
                    "jung_stage": "shadow",
                    "psychological_theme": "darkness",
                    "development_score": 2,
                }
            ]
        )
        result = _js()._parse_response(partial, sids)
        assert result is None

    def test_invalid_json_returns_none(self):
        sids = [make_song_id("A", "S0", 2000)]
        result = _js()._parse_response("not valid json {{", sids)
        assert result is None

    def test_invalid_jung_stage_coerced_to_unclassified(self):
        sids = [make_song_id("A", "S0", 2000)]
        content = json.dumps(
            [
                {
                    "song_id": sids[0],
                    "jung_stage": "totally_invalid_stage",
                    "psychological_theme": "something",
                    "development_score": 3,
                }
            ]
        )
        result = _js()._parse_response(content, sids)
        assert result is not None
        assert result[0]["jung_stage"] == "unclassified"

    def test_out_of_range_dev_score_becomes_none(self):
        sids = [make_song_id("A", "S0", 2000)]
        content = json.dumps(
            [
                {
                    "song_id": sids[0],
                    "jung_stage": "shadow",
                    "psychological_theme": "darkness",
                    "development_score": 99,  # out of 1-7 range
                }
            ]
        )
        result = _js()._parse_response(content, sids)
        assert result is not None
        assert result[0]["development_score"] is None

    def test_null_theme_preserved_as_none(self):
        sids = [make_song_id("A", "S0", 2000)]
        content = json.dumps(
            [
                {
                    "song_id": sids[0],
                    "jung_stage": "unclassified",
                    "psychological_theme": None,
                    "development_score": None,
                }
            ]
        )
        result = _js()._parse_response(content, sids)
        assert result is not None
        assert result[0]["psychological_theme"] is None

    def test_returned_stage_in_valid_set(self):
        sids = [make_song_id("A", "S0", 2000)]
        content = _make_gpt_response(sids, "integration")
        result = _js()._parse_response(content, sids)
        assert result is not None
        assert result[0]["jung_stage"] in VALID_JUNG_STAGES


# ── run() ─────────────────────────────────────────────────────────────────────


class TestRun:
    def test_raises_file_not_found_when_input_missing(
        self, pipeline_workspace, tmp_config
    ):
        js = _js()
        missing = pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".jungian_complete"

        with (
            patch.object(js, "_INPUT_PATH", new=missing),
            patch.object(js, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                js.run(tmp_config)

    def test_fallback_mode_when_no_api_key(self, pipeline_workspace, tmp_config):
        js = _js()
        layer4 = _make_layer4_df(3)
        input_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        out_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".jungian_complete"
        _write_csv(input_path, layer4)

        with (
            patch.object(js, "_INPUT_PATH", new=input_path),
            patch.object(js, "_OUTPUT_PATH", new=out_path),
            patch.object(js, "_SENTINEL", new=sentinel),
            patch(
                "src.psychology.jungian_scorer._init_client", return_value=(None, False)
            ),
            patch("src.psychology.jungian_scorer.write_sentinel"),
        ):
            result = js.run(tmp_config)

        assert result["api_available"] is False

    def test_fallback_all_songs_unclassified(self, pipeline_workspace, tmp_config):
        js = _js()
        layer4 = _make_layer4_df(3)
        input_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        out_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".jungian_complete"
        _write_csv(input_path, layer4)

        with (
            patch.object(js, "_INPUT_PATH", new=input_path),
            patch.object(js, "_OUTPUT_PATH", new=out_path),
            patch.object(js, "_SENTINEL", new=sentinel),
            patch(
                "src.psychology.jungian_scorer._init_client", return_value=(None, False)
            ),
            patch("src.psychology.jungian_scorer.write_sentinel"),
        ):
            result = js.run(tmp_config)

        assert result["songs_classified"] == 0
        assert result["songs_skipped"] == len(layer4)

    def test_run_returns_correct_keys(self, pipeline_workspace, tmp_config):
        js = _js()
        layer4 = _make_layer4_df(3)
        input_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        out_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".jungian_complete"
        _write_csv(input_path, layer4)

        with (
            patch.object(js, "_INPUT_PATH", new=input_path),
            patch.object(js, "_OUTPUT_PATH", new=out_path),
            patch.object(js, "_SENTINEL", new=sentinel),
            patch(
                "src.psychology.jungian_scorer._init_client", return_value=(None, False)
            ),
            patch("src.psychology.jungian_scorer.write_sentinel"),
        ):
            result = js.run(tmp_config)

        for key in [
            "songs_total",
            "songs_classified",
            "songs_skipped",
            "quality_high",
            "quality_low",
            "stage_dist",
            "api_available",
            "output_path",
            "skipped",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_run_writes_output_csv(self, pipeline_workspace, tmp_config):
        js = _js()
        layer4 = _make_layer4_df(3)
        input_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        out_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".jungian_complete"
        _write_csv(input_path, layer4)

        with (
            patch.object(js, "_INPUT_PATH", new=input_path),
            patch.object(js, "_OUTPUT_PATH", new=out_path),
            patch.object(js, "_SENTINEL", new=sentinel),
            patch(
                "src.psychology.jungian_scorer._init_client", return_value=(None, False)
            ),
            patch("src.psychology.jungian_scorer.write_sentinel"),
        ):
            js.run(tmp_config)

        assert out_path.exists(), "layer5_jungian.csv was not written"

    def test_run_calls_write_sentinel(self, pipeline_workspace, tmp_config):
        js = _js()
        layer4 = _make_layer4_df(3)
        input_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        out_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".jungian_complete"
        _write_csv(input_path, layer4)

        with (
            patch.object(js, "_INPUT_PATH", new=input_path),
            patch.object(js, "_OUTPUT_PATH", new=out_path),
            patch.object(js, "_SENTINEL", new=sentinel),
            patch(
                "src.psychology.jungian_scorer._init_client", return_value=(None, False)
            ),
            patch("src.psychology.jungian_scorer.write_sentinel") as mock_ws,
        ):
            js.run(tmp_config)

        mock_ws.assert_called_once()

    def test_run_skips_when_sentinel_matches(self, pipeline_workspace, tmp_config):
        js = _js()
        layer4 = _make_layer4_df(3)
        input_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        out_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".jungian_complete"
        _write_csv(input_path, layer4)

        # Pre-write output so sentinel-skip read_csv succeeds
        layer4[["song_id"]].assign(
            jung_stage="unclassified",
            psychological_theme=None,
            development_score=None,
            jungian_quality_flag="low",
        ).to_csv(out_path, index=False)

        with (
            patch.object(js, "_INPUT_PATH", new=input_path),
            patch.object(js, "_OUTPUT_PATH", new=out_path),
            patch.object(js, "_SENTINEL", new=sentinel),
            patch(
                "src.psychology.jungian_scorer.sentinel_config_matches",
                return_value=True,
            ),
            patch("src.psychology.jungian_scorer._init_client") as mock_init,
        ):
            result = js.run(tmp_config)

        mock_init.assert_not_called()
        assert result["skipped"] is True

    def test_run_with_live_api_classifies_songs(self, pipeline_workspace, tmp_config):
        """Verify run() uses _classify_all when API is available."""
        js = _js()
        layer4 = _make_layer4_df(3)
        input_path = (
            pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        )
        out_path = pipeline_workspace / "data" / "analysis" / "layer5_jungian.csv"
        sentinel = pipeline_workspace / "data" / "analysis" / ".jungian_complete"
        _write_csv(input_path, layer4)

        sids = layer4["song_id"].tolist()
        classified_df = pd.DataFrame(
            [
                {
                    "song_id": sid,
                    "jung_stage": "shadow",
                    "psychological_theme": "darkness and repression",
                    "development_score": 2,
                }
                for sid in sids
            ]
        )

        mock_client = MagicMock()

        with (
            patch.object(js, "_INPUT_PATH", new=input_path),
            patch.object(js, "_OUTPUT_PATH", new=out_path),
            patch.object(js, "_SENTINEL", new=sentinel),
            patch(
                "src.psychology.jungian_scorer._init_client",
                return_value=(mock_client, True),
            ),
            patch(
                "src.psychology.jungian_scorer._classify_all",
                return_value=classified_df,
            ),
            patch("src.psychology.jungian_scorer.write_sentinel"),
        ):
            result = js.run(tmp_config)

        assert result["api_available"] is True
        assert result["songs_classified"] == 3
        assert result["songs_skipped"] == 0
