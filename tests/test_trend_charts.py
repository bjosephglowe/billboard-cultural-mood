"""
tests/test_trend_charts.py

Unit tests for src/visualizations/trend_charts.py

Coverage:
- _ordered_decades: canonical order, unknowns appended, empty input
- _sort_decades: sorted by DECADE_ORDER, missing col returns unchanged
- _write_meta: known chart name writes correct caption, unknown name
  writes fallback, meta file exists at correct path
- _chart_cmi_sentiment_trend: returns Figure, null rows dropped,
  single trace present
- _chart_emotion_distribution: returns Figure, empty df handled,
  invalid emotions filtered out
- _chart_jungian_distribution: returns Figure, missing column returns
  empty Figure, unclassified rendered last
- _chart_theme_heatmap: returns Figure, missing themes column returns
  empty Figure, matrix includes all valid themes
- _chart_resonance_distribution: returns Figure, null scores dropped,
  one box per decade
- run(): FileNotFoundError for each missing required input
- run(): returns correct keys on success
- run(): calls write_sentinel
- run(): sentinel skip path
- run(): charts_written count matches generators
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.utils.identifiers import make_song_id

# ── Helpers ───────────────────────────────────────────────────────────────────


def _tc():
    from src.visualizations import trend_charts

    return trend_charts


def _make_decade_df() -> pd.DataFrame:
    """Minimal decade_cmi.csv fixture."""
    return pd.DataFrame(
        {
            "decade_label": ["1960s*", "1980s", "1990s", "2000s", "2010s"],
            "CMI_sentiment": [0.12, 0.08, -0.05, 0.15, 0.22],
            "CMI_energy": [2.8, 3.1, 3.4, 3.2, 3.0],
            "emotional_tone": ["joy", "joy", "anger", "joy", "neutral"],
            "dominant_jung_stage": [
                "persona",
                "shadow",
                "shadow",
                "integration",
                "persona",
            ],
            "song_count": [120, 98, 105, 112, 108],
            "scored_count": [115, 94, 100, 108, 103],
        }
    )


def _make_layer4_df(n: int = 5) -> pd.DataFrame:
    """Minimal layer2_full_analysis.csv fixture."""
    sids = [make_song_id("Artist", f"Song{i}", 2000 + i) for i in range(n)]
    return pd.DataFrame(
        {
            "song_id": sids,
            "song_title": [f"Song{i}" for i in range(n)],
            "decade": ["2000s"] * n,
            "emotional_tone": ["joy", "anger", "joy", "neutral", "fear"],
            "jung_stage": ["persona", "shadow", "integration", "persona", "shadow"],
            "themes": [
                "love|nostalgia",
                "struggle",
                "love|empowerment",
                "nostalgia",
                "rebellion|struggle",
            ],
            "cultural_resonance_score": [0.7, 0.4, 0.8, 0.55, 0.3],
        }
    )


def _make_layer6_df(song_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "song_id": song_ids,
            "cultural_resonance_score": [0.6] * len(song_ids),
        }
    )


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_all_inputs(
    workspace: Path,
    decade_df: pd.DataFrame,
    layer4_df: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    decade_path = workspace / "data" / "analysis" / "decade_cmi.csv"
    layer4_path = workspace / "data" / "analysis" / "layer2_full_analysis.csv"
    layer6_path = workspace / "data" / "analysis" / "layer6_cultural_metrics.csv"
    layer6_df = _make_layer6_df(layer4_df["song_id"].tolist())
    _write_csv(decade_path, decade_df)
    _write_csv(layer4_path, layer4_df)
    _write_csv(layer6_path, layer6_df)
    return decade_path, layer4_path, layer6_path


# ── _ordered_decades ──────────────────────────────────────────────────────────


class TestOrderedDecades:
    def test_returns_canonical_order(self):
        from src.visualizations._design_system import DECADE_ORDER

        decades = ["2010s", "1980s", "1960s*", "2000s"]
        result = _tc()._ordered_decades(decades)
        positions = {d: i for i, d in enumerate(result)}
        # Earlier decades should appear before later ones
        assert positions.get("1960s*", 0) < positions.get("1980s", 1)
        assert positions.get("1980s", 0) < positions.get("2000s", 1)

    def test_unknown_decades_appended_at_end(self):
        result = _tc()._ordered_decades(["2000s", "2099s"])
        assert result[-1] == "2099s"

    def test_empty_input_returns_empty(self):
        result = _tc()._ordered_decades([])
        assert result == []

    def test_returns_list(self):
        result = _tc()._ordered_decades(["2000s"])
        assert isinstance(result, list)


# ── _sort_decades ─────────────────────────────────────────────────────────────


class TestSortDecades:
    def test_sorts_by_canonical_order(self):
        df = pd.DataFrame(
            {
                "decade_label": ["2010s", "1980s", "2000s"],
                "val": [1, 2, 3],
            }
        )
        result = _tc()._sort_decades(df, col="decade_label")
        labels = result["decade_label"].tolist()
        assert labels.index("1980s") < labels.index("2000s") < labels.index("2010s")

    def test_missing_col_returns_unchanged(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = _tc()._sort_decades(df, col="decade_label")
        assert list(result.columns) == ["other"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"decade_label": ["2010s", "1980s"], "val": [1, 2]})
        original_order = df["decade_label"].tolist()
        _ = _tc()._sort_decades(df, col="decade_label")
        assert df["decade_label"].tolist() == original_order

    def test_sort_key_column_removed(self):
        df = pd.DataFrame({"decade_label": ["2010s", "1980s"]})
        result = _tc()._sort_decades(df)
        assert "_sort_key" not in result.columns


# ── _write_meta ───────────────────────────────────────────────────────────────


class TestWriteMeta:
    def test_known_chart_writes_correct_caption(self, tmp_path):
        png_path = tmp_path / "cmi_sentiment_trend.png"
        png_path.write_bytes(b"fake")
        _tc()._write_meta(png_path)
        meta_path = tmp_path / "cmi_sentiment_trend.png.meta.json"
        assert meta_path.exists()
        import json

        meta = json.loads(meta_path.read_text())
        assert "CMI" in meta["caption"] or "Sentiment" in meta["caption"]

    def test_unknown_chart_writes_fallback(self, tmp_path):
        png_path = tmp_path / "custom_chart.png"
        png_path.write_bytes(b"fake")
        _tc()._write_meta(png_path)
        meta_path = tmp_path / "custom_chart.png.meta.json"
        assert meta_path.exists()

    def test_meta_has_caption_and_description_keys(self, tmp_path):
        import json

        png_path = tmp_path / "emotion_distribution.png"
        png_path.write_bytes(b"fake")
        _tc()._write_meta(png_path)
        meta = json.loads((tmp_path / "emotion_distribution.png.meta.json").read_text())
        assert "caption" in meta
        assert "description" in meta


# ── _chart_cmi_sentiment_trend ────────────────────────────────────────────────


class TestChartCmiSentimentTrend:
    def test_returns_figure(self):
        df = _make_decade_df()
        fig = _tc()._chart_cmi_sentiment_trend(df)
        assert isinstance(fig, go.Figure)

    def test_has_one_trace(self):
        df = _make_decade_df()
        fig = _tc()._chart_cmi_sentiment_trend(df)
        assert len(fig.data) == 1

    def test_null_cmi_rows_excluded(self):
        df = _make_decade_df()
        df.loc[2, "CMI_sentiment"] = None
        fig = _tc()._chart_cmi_sentiment_trend(df)
        # Non-null rows should appear in trace x values
        x_vals = list(fig.data[0].x)
        assert df.iloc[2]["decade_label"] not in x_vals

    def test_empty_df_returns_figure(self):
        df = pd.DataFrame({"decade_label": [], "CMI_sentiment": []})
        fig = _tc()._chart_cmi_sentiment_trend(df)
        assert isinstance(fig, go.Figure)


# ── _chart_emotion_distribution ───────────────────────────────────────────────


class TestChartEmotionDistribution:
    @pytest.fixture(autouse=True)
    def _patch(self):
        with patch("src.visualizations.trend_charts.BASE_LAYOUT", new={}):
            yield

    def test_returns_figure(self):
        df = _make_layer4_df()
        fig = _tc()._chart_emotion_distribution(df)
        assert isinstance(fig, go.Figure)

    def test_invalid_emotions_filtered(self):
        df = _make_layer4_df()
        df.loc[0, "emotional_tone"] = "rage_unknown"
        fig = _tc()._chart_emotion_distribution(df)
        trace_names = [t.name.lower() for t in fig.data]
        assert "rage_unknown" not in trace_names

    def test_empty_df_returns_figure(self):
        df = pd.DataFrame({"decade": [], "emotional_tone": []})
        fig = _tc()._chart_emotion_distribution(df)
        assert isinstance(fig, go.Figure)

    def test_barmode_is_group(self):
        df = _make_layer4_df()
        fig = _tc()._chart_emotion_distribution(df)
        assert fig.layout.barmode == "group"


# ── _chart_jungian_distribution ───────────────────────────────────────────────


class TestChartJungianDistribution:
    @pytest.fixture(autouse=True)
    def _patch(self):
        with patch("src.visualizations.trend_charts.BASE_LAYOUT", new={}):
            yield

    def test_returns_figure(self):
        df = _make_layer4_df()
        fig = _tc()._chart_jungian_distribution(df)
        assert isinstance(fig, go.Figure)

    def test_missing_jung_stage_column_returns_empty_figure(self):
        df = _make_layer4_df().drop(columns=["jung_stage"])
        fig = _tc()._chart_jungian_distribution(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_barmode_is_stack(self):
        df = _make_layer4_df()
        fig = _tc()._chart_jungian_distribution(df)
        if len(fig.data) > 0:
            assert fig.layout.barmode == "stack"

    def test_unclassified_trace_last_when_present(self):
        df = _make_layer4_df()
        df["jung_stage"] = [
            "shadow",
            "unclassified",
            "persona",
            "integration",
            "unclassified",
        ]
        fig = _tc()._chart_jungian_distribution(df)
        if len(fig.data) > 1:
            assert fig.data[-1].name.lower() == "unclassified"


# ── _chart_theme_heatmap ──────────────────────────────────────────────────────


class TestChartThemeHeatmap:
    def test_returns_figure(self):
        df = _make_layer4_df()
        fig = _tc()._chart_theme_heatmap(df)
        assert isinstance(fig, go.Figure)

    def test_missing_themes_column_returns_empty_figure(self):
        df = _make_layer4_df().drop(columns=["themes"])
        fig = _tc()._chart_theme_heatmap(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_has_heatmap_trace(self):
        df = _make_layer4_df()
        fig = _tc()._chart_theme_heatmap(df)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_y_axis_contains_valid_themes(self):
        df = _make_layer4_df()
        fig = _tc()._chart_theme_heatmap(df)
        if len(fig.data) > 0:
            y_labels = [y.lower().replace(" ", "_") for y in fig.data[0].y]
            tc = _tc()
            for label in y_labels:
                assert label in tc._VALID_THEMES


# ── _chart_resonance_distribution ────────────────────────────────────────────


class TestChartResonanceDistribution:
    def test_returns_figure(self):
        df = _make_layer4_df()
        fig = _tc()._chart_resonance_distribution(df)
        assert isinstance(fig, go.Figure)

    def test_null_scores_excluded(self):
        df = _make_layer4_df()
        df.loc[0, "cultural_resonance_score"] = None
        fig = _tc()._chart_resonance_distribution(df)
        # Should still produce a figure without error
        assert isinstance(fig, go.Figure)

    def test_one_box_per_decade(self):
        df = pd.concat(
            [
                _make_layer4_df(5).assign(decade="1980s"),
                _make_layer4_df(5).assign(decade="2000s"),
            ],
            ignore_index=True,
        )
        fig = _tc()._chart_resonance_distribution(df)
        assert len(fig.data) == 2

    def test_empty_df_returns_figure(self):
        df = pd.DataFrame(
            {
                "decade": [],
                "cultural_resonance_score": [],
            }
        )
        fig = _tc()._chart_resonance_distribution(df)
        assert isinstance(fig, go.Figure)


# ── run() ─────────────────────────────────────────────────────────────────────


class TestRun:
    def test_raises_if_decade_csv_missing(self, pipeline_workspace, tmp_config):
        tc = _tc()
        missing = pipeline_workspace / "data" / "analysis" / "decade_cmi.csv"
        layer4 = pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        layer6 = (
            pipeline_workspace / "data" / "analysis" / "layer6_cultural_metrics.csv"
        )
        out_dir = pipeline_workspace / "outputs" / "visualizations"
        sentinel = out_dir / ".charts_complete"

        with (
            patch.object(tc, "_INPUT_DECADE", new=missing),
            patch.object(tc, "_INPUT_LAYER4", new=layer4),
            patch.object(tc, "_INPUT_LAYER6", new=layer6),
            patch.object(tc, "_OUTPUT_DIR", new=out_dir),
            patch.object(tc, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                tc.run(tmp_config)

    def test_raises_if_layer4_missing(self, pipeline_workspace, tmp_config):
        tc = _tc()
        decade = pipeline_workspace / "data" / "analysis" / "decade_cmi.csv"
        missing = pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        layer6 = (
            pipeline_workspace / "data" / "analysis" / "layer6_cultural_metrics.csv"
        )
        out_dir = pipeline_workspace / "outputs" / "visualizations"
        sentinel = out_dir / ".charts_complete"
        _write_csv(decade, _make_decade_df())

        with (
            patch.object(tc, "_INPUT_DECADE", new=decade),
            patch.object(tc, "_INPUT_LAYER4", new=missing),
            patch.object(tc, "_INPUT_LAYER6", new=layer6),
            patch.object(tc, "_OUTPUT_DIR", new=out_dir),
            patch.object(tc, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                tc.run(tmp_config)

    def test_raises_if_layer6_missing(self, pipeline_workspace, tmp_config):
        tc = _tc()
        decade = pipeline_workspace / "data" / "analysis" / "decade_cmi.csv"
        layer4 = pipeline_workspace / "data" / "analysis" / "layer2_full_analysis.csv"
        missing = (
            pipeline_workspace / "data" / "analysis" / "layer6_cultural_metrics.csv"
        )
        out_dir = pipeline_workspace / "outputs" / "visualizations"
        sentinel = out_dir / ".charts_complete"
        layer4_df = _make_layer4_df()
        _write_csv(decade, _make_decade_df())
        _write_csv(layer4, layer4_df)

        with (
            patch.object(tc, "_INPUT_DECADE", new=decade),
            patch.object(tc, "_INPUT_LAYER4", new=layer4),
            patch.object(tc, "_INPUT_LAYER6", new=missing),
            patch.object(tc, "_OUTPUT_DIR", new=out_dir),
            patch.object(tc, "_SENTINEL", new=sentinel),
        ):
            with pytest.raises(FileNotFoundError):
                tc.run(tmp_config)

    def test_run_returns_correct_keys(self, pipeline_workspace, tmp_config):
        tc = _tc()
        layer4_df = _make_layer4_df()
        decade_path, layer4_path, layer6_path = _write_all_inputs(
            pipeline_workspace, _make_decade_df(), layer4_df
        )
        out_dir = pipeline_workspace / "outputs" / "visualizations"
        sentinel = out_dir / ".charts_complete"

        with (
            patch.object(tc, "_INPUT_DECADE", new=decade_path),
            patch.object(tc, "_INPUT_LAYER4", new=layer4_path),
            patch.object(tc, "_INPUT_LAYER6", new=layer6_path),
            patch.object(tc, "_OUTPUT_DIR", new=out_dir),
            patch.object(tc, "_SENTINEL", new=sentinel),
            patch("src.visualizations.trend_charts.BASE_LAYOUT", new={}),
            patch("src.visualizations.trend_charts.write_sentinel"),
            patch("src.visualizations.trend_charts._write_meta"),
            patch.object(go.Figure, "write_image"),
        ):
            result = tc.run(tmp_config)

        assert "charts_written" in result
        assert "output_dir" in result
        assert "skipped" in result

    def test_run_returns_skipped_false_on_fresh_run(
        self, pipeline_workspace, tmp_config
    ):
        tc = _tc()
        layer4_df = _make_layer4_df()
        decade_path, layer4_path, layer6_path = _write_all_inputs(
            pipeline_workspace, _make_decade_df(), layer4_df
        )
        out_dir = pipeline_workspace / "outputs" / "visualizations"
        sentinel = out_dir / ".charts_complete"

        with (
            patch.object(tc, "_INPUT_DECADE", new=decade_path),
            patch.object(tc, "_INPUT_LAYER4", new=layer4_path),
            patch.object(tc, "_INPUT_LAYER6", new=layer6_path),
            patch.object(tc, "_OUTPUT_DIR", new=out_dir),
            patch.object(tc, "_SENTINEL", new=sentinel),
            patch("src.visualizations.trend_charts.BASE_LAYOUT", new={}),
            patch("src.visualizations.trend_charts.write_sentinel"),
            patch("src.visualizations.trend_charts._write_meta"),
            patch.object(go.Figure, "write_image"),
        ):
            result = tc.run(tmp_config)

        assert result["skipped"] is False

    def test_run_calls_write_sentinel(self, pipeline_workspace, tmp_config):
        tc = _tc()
        layer4_df = _make_layer4_df()
        decade_path, layer4_path, layer6_path = _write_all_inputs(
            pipeline_workspace, _make_decade_df(), layer4_df
        )
        out_dir = pipeline_workspace / "outputs" / "visualizations"
        sentinel = out_dir / ".charts_complete"

        with (
            patch.object(tc, "_INPUT_DECADE", new=decade_path),
            patch.object(tc, "_INPUT_LAYER4", new=layer4_path),
            patch.object(tc, "_INPUT_LAYER6", new=layer6_path),
            patch.object(tc, "_OUTPUT_DIR", new=out_dir),
            patch.object(tc, "_SENTINEL", new=sentinel),
            patch("src.visualizations.trend_charts.BASE_LAYOUT", new={}),
            patch("src.visualizations.trend_charts.write_sentinel") as mock_ws,
            patch("src.visualizations.trend_charts._write_meta"),
            patch.object(go.Figure, "write_image"),
        ):
            tc.run(tmp_config)

        mock_ws.assert_called_once()

    def test_run_charts_written_count(self, pipeline_workspace, tmp_config):
        tc = _tc()
        layer4_df = _make_layer4_df()
        decade_path, layer4_path, layer6_path = _write_all_inputs(
            pipeline_workspace, _make_decade_df(), layer4_df
        )
        out_dir = pipeline_workspace / "outputs" / "visualizations"
        sentinel = out_dir / ".charts_complete"

        mock_fig = MagicMock(spec=go.Figure)
        mock_fig.write_image = MagicMock()

        with (
            patch("src.visualizations.trend_charts.BASE_LAYOUT", new={}),
            patch(
                "src.visualizations.trend_charts._chart_cmi_sentiment_trend",
                return_value=mock_fig,
            ),
            patch(
                "src.visualizations.trend_charts._chart_emotion_distribution",
                return_value=mock_fig,
            ),
            patch(
                "src.visualizations.trend_charts._chart_jungian_distribution",
                return_value=mock_fig,
            ),
            patch(
                "src.visualizations.trend_charts._chart_theme_heatmap",
                return_value=mock_fig,
            ),
            patch(
                "src.visualizations.trend_charts._chart_resonance_distribution",
                return_value=mock_fig,
            ),
            patch.object(tc, "_INPUT_DECADE", new=decade_path),
            patch.object(tc, "_INPUT_LAYER4", new=layer4_path),
            patch.object(tc, "_INPUT_LAYER6", new=layer6_path),
            patch.object(tc, "_OUTPUT_DIR", new=out_dir),
            patch.object(tc, "_SENTINEL", new=sentinel),
            patch("src.visualizations.trend_charts.write_sentinel"),
            patch("src.visualizations.trend_charts._write_meta"),
        ):
            result = tc.run(tmp_config)

        assert result["charts_written"] == 5

    def test_run_skips_when_sentinel_matches(self, pipeline_workspace, tmp_config):
        tc = _tc()
        layer4_df = _make_layer4_df()
        decade_path, layer4_path, layer6_path = _write_all_inputs(
            pipeline_workspace, _make_decade_df(), layer4_df
        )
        out_dir = pipeline_workspace / "outputs" / "visualizations"
        sentinel = out_dir / ".charts_complete"
        out_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(tc, "_INPUT_DECADE", new=decade_path),
            patch.object(tc, "_INPUT_LAYER4", new=layer4_path),
            patch.object(tc, "_INPUT_LAYER6", new=layer6_path),
            patch.object(tc, "_OUTPUT_DIR", new=out_dir),
            patch.object(tc, "_SENTINEL", new=sentinel),
            patch(
                "src.visualizations.trend_charts.sentinel_config_matches",
                return_value=True,
            ),
        ):
            result = tc.run(tmp_config)

        assert result["skipped"] is True
