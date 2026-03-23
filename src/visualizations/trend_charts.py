"""
src/visualizations/trend_charts.py

Stage 11a — Trend Visualisation
=================================
Generates all pipeline visualisations from the Stage 10 CMI outputs
and writes PNG files to outputs/visualizations/.

Charts produced
---------------
1. cmi_sentiment_trend.png
   Line chart — CMI_sentiment by decade with fill under curve.

2. emotion_distribution.png
   Grouped bar chart — emotional_tone frequency per decade.

3. jungian_stage_distribution.png
   Stacked bar chart — jung_stage proportions per decade.

4. theme_heatmap.png
   Heatmap — theme frequency (%) per decade × theme matrix.

5. resonance_distribution.png
   Box plot — cultural_resonance_score distribution per decade.

Design system
-------------
All charts use constants from src.visualizations._design_system:
  PLOTLY_THEME    — registered Plotly template name
  DECADE_ORDER    — canonical left-to-right decade sort order
  EMOTION_COLORS — per-emotion hex colour map
  JUNG_COLORS    — per-stage hex colour map
  THEME_COLORS   — per-theme hex colour map

Idempotency
-----------
Sentinel .charts_complete records config_hash. If matched, the
output directory is returned without regenerating charts. Delete
the sentinel (or change config) to force regeneration.

Output
------
outputs/visualizations/
  cmi_sentiment_trend.png       + .meta.json
  emotion_distribution.png      + .meta.json
  jungian_stage_distribution.png + .meta.json
  theme_heatmap.png             + .meta.json
  resonance_distribution.png    + .meta.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from src.pipeline.config_loader import (
    ProjectConfig,
    load_config,
    sentinel_config_matches,
    write_sentinel,
)
from src.visualizations._design_system import (
    BASE_LAYOUT,
    CHART_SIZE,
    CHART_SIZE_HEATMAP,
    DECADE_ORDER,
    EMOTION_COLORS,
    JUNG_COLORS,
    THEME_COLORS,
)

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INPUT_DECADE = _PROJECT_ROOT / "data" / "analysis" / "decade_cmi.csv"
_INPUT_LAYER4 = _PROJECT_ROOT / "data" / "analysis" / "layer2_full_analysis.csv"
_INPUT_LAYER6 = _PROJECT_ROOT / "data" / "analysis" / "layer6_cultural_metrics.csv"
_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "visualizations"
_SENTINEL = _PROJECT_ROOT / "outputs" / "visualizations" / ".charts_complete"

# ── Valid value sets (duplicated here to avoid circular schema import) ────────
_VALID_EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
_VALID_JUNG_STAGES = [
    "shadow",
    "persona",
    "anima_animus",
    "integration",
    "transcendence",
    "unclassified",
]
_VALID_THEMES = [
    "love",
    "heartbreak",
    "party_celebration",
    "identity",
    "struggle",
    "rebellion",
    "wealth_success",
    "friendship",
    "nostalgia",
    "self_reflection",
    "empowerment",
    "unity",
]


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def run(config: ProjectConfig) -> dict:
    """
    Execute Stage 11a — Trend chart generation.

    Parameters
    ----------
    config : ProjectConfig

    Returns
    -------
    dict with keys:
        charts_written : int
        output_dir     : Path
        skipped        : bool
    """
    for path, label in [
        (_INPUT_DECADE, "decade_cmi.csv"),
        (_INPUT_LAYER4, "layer2_full_analysis.csv"),
        (_INPUT_LAYER6, "layer6_cultural_metrics.csv"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found at {path}. Run Stage 10 [CMI] first."
            )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check ────────────────────────────────────────────────────
    if sentinel_config_matches(_SENTINEL, config):
        logger.info("Stage 11a [CHARTS] — sentinel matched, skipping.")
        existing = list(_OUTPUT_DIR.glob("*.png"))
        return {
            "charts_written": len(existing),
            "output_dir": _OUTPUT_DIR,
            "skipped": True,
        }

    # ── Load inputs ──────────────────────────────────────────────────────────
    decade_df = pd.read_csv(_INPUT_DECADE)
    layer4_df = pd.read_csv(_INPUT_LAYER4)
    layer6_df = pd.read_csv(_INPUT_LAYER6)

    # Merge resonance scores into layer4 for box plot
    analysis_df = layer4_df.merge(
        layer6_df[["song_id", "cultural_resonance_score"]],
        on="song_id",
        how="left",
    )

    # Sort decades canonically
    decade_df = _sort_decades(decade_df)
    analysis_df = _sort_decades(analysis_df, col="decade")

    logger.info(
        "Stage 11a [CHARTS] — generating charts for %d decades, %d songs …",
        len(decade_df),
        len(analysis_df),
    )

    # ── Generate all charts ───────────────────────────────────────────────────
    charts_written = 0
    generators = [
        (_chart_cmi_sentiment_trend, (decade_df,), "cmi_sentiment_trend"),
        (_chart_emotion_distribution, (layer4_df,), "emotion_distribution"),
        (_chart_jungian_distribution, (layer4_df,), "jungian_stage_distribution"),
        (_chart_theme_heatmap, (layer4_df,), "theme_heatmap"),
        (_chart_resonance_distribution, (analysis_df,), "resonance_distribution"),
    ]

    for fn, args, name in generators:
        try:
            fig = fn(*args)
            path = _OUTPUT_DIR / f"{name}.png"
            fig.write_image(str(path))
            _write_meta(path)
            charts_written += 1
            logger.info("Chart written: %s", path.name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to generate chart '%s': %s", name, exc)

    write_sentinel(_SENTINEL, stage="CHARTS", config=config)

    logger.info(
        "Stage 11a [CHARTS] — complete. %d charts → %s",
        charts_written,
        _OUTPUT_DIR,
    )
    return {
        "charts_written": charts_written,
        "output_dir": _OUTPUT_DIR,
        "skipped": False,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Chart generators
# ═════════════════════════════════════════════════════════════════════════════


def _chart_cmi_sentiment_trend(decade_df: pd.DataFrame) -> go.Figure:
    """
    Line chart — CMI_sentiment by decade with light fill under curve.
    Null CMI values are dropped; gap is shown naturally by Plotly.
    """
    df = decade_df.dropna(subset=["CMI_sentiment"]).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["decade_label"],
            y=df["CMI_sentiment"],
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(99,110,250,0.15)",
            line=dict(width=2.5),
            marker=dict(size=8),
            name="CMI Sentiment",
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        **BASE_LAYOUT,
        width=CHART_SIZE["width"],
        height=CHART_SIZE["height"],
        showlegend=False,
    )
    fig.update_layout(
        title=dict(
            text=(
                "Billboard Sentiment Trend by Decade<br>"
                "<span style='font-size:16px;font-weight:normal;'>"
                "Source: Billboard Hot 100 | Mean VADER sentiment per decade"
                "</span>"
            )
        ),
    )

    fig.update_xaxes(title_text="Decade")
    fig.update_yaxes(title_text="CMI Sentiment", range=[-1, 1])
    return fig


def _chart_emotion_distribution(layer4_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart — emotional_tone frequency (%) per decade.
    Only valid Ekman emotions are shown; unknowns are dropped.
    """
    df = layer4_df[layer4_df["emotional_tone"].isin(_VALID_EMOTIONS)].copy()

    counts = df.groupby(["decade", "emotional_tone"]).size().reset_index(name="count")
    totals = counts.groupby("decade")["count"].transform("sum")
    counts["pct"] = (counts["count"] / totals * 100).round(1)
    counts = _sort_decades(counts, col="decade")

    fig = go.Figure()
    for emotion in _VALID_EMOTIONS:
        sub = counts[counts["emotional_tone"] == emotion]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=sub["decade"],
                y=sub["pct"],
                name=emotion.capitalize(),
                marker_color=EMOTION_COLORS.get(emotion),
                hovertemplate="%{x} — " + emotion + ": %{y:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        **BASE_LAYOUT,
        width=CHART_SIZE["width"],
        height=CHART_SIZE["height"],
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_layout(
        title=dict(
            text=(
                "Emotional Tone Distribution by Decade<br>"
                "<span style='font-size:16px;font-weight:normal;'>"
                "Source: Billboard Hot 100 | % of songs per emotion"
                "</span>"
            )
        ),
    )

    fig.update_xaxes(title_text="Decade")
    fig.update_yaxes(title_text="% of Songs")
    fig.update_traces(cliponaxis=False)
    return fig


def _chart_jungian_distribution(layer4_df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart — jung_stage proportions (%) per decade.
    "unclassified" is rendered last in a neutral grey.
    """
    if "jung_stage" not in layer4_df.columns:
        logger.warning(
            "jung_stage column not found in layer4 — "
            "skipping jungian_stage_distribution chart."
        )
        return go.Figure()

    df = layer4_df[layer4_df["jung_stage"].isin(_VALID_JUNG_STAGES)].copy()

    counts = df.groupby(["decade", "jung_stage"]).size().reset_index(name="count")
    totals = counts.groupby("decade")["count"].transform("sum")
    counts["pct"] = (counts["count"] / totals * 100).round(1)
    counts = _sort_decades(counts, col="decade")

    fig = go.Figure()
    stage_order = [s for s in _VALID_JUNG_STAGES if s != "unclassified"]
    stage_order.append("unclassified")

    for stage in stage_order:
        sub = counts[counts["jung_stage"] == stage]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=sub["decade"],
                y=sub["pct"],
                name=stage.replace("_", " ").title(),
                marker_color=JUNG_COLORS.get(stage),
                hovertemplate=(
                    "%{x} — " + stage.replace("_", " ") + ": %{y:.1f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **BASE_LAYOUT,
        width=CHART_SIZE["width"],
        height=CHART_SIZE["height"],
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_layout(
        title=dict(
            text=(
                "Jungian Stage Distribution by Decade<br>"
                "<span style='font-size:16px;font-weight:normal;'>"
                "Source: Billboard Hot 100 | % of songs per stage"
                "</span>"
            )
        ),
    )

    fig.update_xaxes(title_text="Decade")
    fig.update_yaxes(title_text="% of Songs", range=[0, 100])
    fig.update_traces(cliponaxis=False)
    return fig


def _chart_theme_heatmap(layer4_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap — theme penetration (% of decade songs) per decade × theme.
    Each cell = percentage of songs in that decade containing that theme.
    """
    if "themes" not in layer4_df.columns:
        logger.warning(
            "themes column not found in layer4 — skipping theme_heatmap chart."
        )
        return go.Figure()

    decade_totals = layer4_df.groupby("decade").size().to_dict()
    matrix: dict[str, dict[str, float]] = {t: {} for t in _VALID_THEMES}

    for decade, group in layer4_df.groupby("decade"):
        total = decade_totals.get(decade, 1)
        theme_counts: dict[str, int] = {t: 0 for t in _VALID_THEMES}
        for cell in group["themes"].dropna():
            for tag in str(cell).split("|"):
                tag = tag.strip()
                if tag in theme_counts:
                    theme_counts[tag] += 1
        for theme in _VALID_THEMES:
            matrix[theme][decade] = round(theme_counts[theme] / total * 100, 1)

    decades = _ordered_decades(layer4_df["decade"].unique())
    z_vals = [[matrix[t].get(d, 0.0) for d in decades] for t in _VALID_THEMES]
    labels = [[f"{v:.1f}%" for v in row] for row in z_vals]

    fig = go.Figure(
        go.Heatmap(
            z=z_vals,
            x=decades,
            y=[t.replace("_", " ").title() for t in _VALID_THEMES],
            text=labels,
            texttemplate="%{text}",
            colorscale="Blues",
            hovertemplate="Decade: %{x}<br>Theme: %{y}<br>Songs: %{z:.1f}%<extra></extra>",
            showscale=True,
            colorbar=dict(title=dict(text="% Songs")),
        )
    )

    fig.update_layout(
        **BASE_LAYOUT,
        width=CHART_SIZE_HEATMAP["width"],
        height=CHART_SIZE_HEATMAP["height"],
    )
    fig.update_layout(
        title=dict(
            text=(
                "Theme Penetration by Decade (%)<br>"
                "<span style='font-size:16px;font-weight:normal;'>"
                "Source: Billboard Hot 100 | % of decade songs per theme"
                "</span>"
            )
        ),
    )

    fig.update_xaxes(title_text="Decade")
    fig.update_yaxes(title_text="Theme")
    return fig


def _chart_resonance_distribution(analysis_df: pd.DataFrame) -> go.Figure:
    """
    Box plot — cultural_resonance_score distribution per decade.
    Songs with null resonance scores are excluded.
    """
    df = analysis_df.dropna(subset=["cultural_resonance_score"]).copy()
    decades = _ordered_decades(df["decade"].unique())

    fig = go.Figure()
    for decade in decades:
        sub = df[df["decade"] == decade]["cultural_resonance_score"]
        if sub.empty:
            continue
        fig.add_trace(
            go.Box(
                y=sub,
                name=decade,
                boxpoints="outliers",
                marker=dict(size=4, opacity=0.5),
                hovertemplate=(f"{decade}<br>Score: %{{y:.3f}}<extra></extra>"),
            )
        )

    fig.update_layout(
        **BASE_LAYOUT,
        width=CHART_SIZE["width"],
        height=CHART_SIZE["height"],
        showlegend=False,
    )
    fig.update_layout(
        title=dict(
            text=(
                "Cultural Resonance Score Distribution by Decade<br>"
                "<span style='font-size:16px;font-weight:normal;'>"
                "Source: Billboard Hot 100 | Score range 0–1"
                "</span>"
            )
        ),
    )

    fig.update_xaxes(title_text="Decade")
    fig.update_yaxes(title_text="Resonance Score", range=[0, 1])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _ordered_decades(decades: "pd.Series | list") -> list[str]:
    """Return decades sorted by DECADE_ORDER; unknowns appended at end."""
    present = set(str(d) for d in decades)
    ordered = [d for d in DECADE_ORDER if d in present]
    extras = sorted(present - set(DECADE_ORDER))
    return ordered + extras


def _sort_decades(df: pd.DataFrame, col: str = "decade_label") -> pd.DataFrame:
    """Sort a DataFrame by canonical decade order."""
    if col not in df.columns:
        return df
    order_map = {d: i for i, d in enumerate(DECADE_ORDER)}
    df = df.copy()
    df["_sort_key"] = df[col].map(order_map).fillna(999)
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"])
    return df.reset_index(drop=True)


def _write_meta(png_path: Path) -> None:
    """Write a sidecar .meta.json file for a chart PNG."""
    _META = {
        "cmi_sentiment_trend.png": {
            "caption": "Billboard CMI Sentiment Trend by Decade",
            "description": (
                "Line chart showing mean VADER sentiment score (CMI_sentiment) "
                "per decade across the Billboard Hot 100 dataset."
            ),
        },
        "emotion_distribution.png": {
            "caption": "Emotional Tone Distribution by Decade",
            "description": (
                "Grouped bar chart showing percentage of Billboard songs "
                "classified per Ekman emotion for each decade."
            ),
        },
        "jungian_stage_distribution.png": {
            "caption": "Jungian Stage Distribution by Decade",
            "description": (
                "Stacked bar chart showing percentage of songs classified into "
                "each Jungian psychological development stage per decade."
            ),
        },
        "theme_heatmap.png": {
            "caption": "Theme Penetration Heatmap by Decade",
            "description": (
                "Heatmap showing percentage of songs in each decade that contain "
                "each theme label from the 12-class theme taxonomy."
            ),
        },
        "resonance_distribution.png": {
            "caption": "Cultural Resonance Score Distribution by Decade",
            "description": (
                "Box plot showing the distribution of Cultural Resonance Scores "
                "across songs within each decade."
            ),
        },
    }

    meta = _META.get(
        png_path.name,
        {"caption": png_path.stem.replace("_", " ").title(), "description": ""},
    )
    meta_path = png_path.with_suffix(".png.meta.json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Stage 11a — Trend chart generation")
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "config" / "project_config.yaml",
        help="Path to project_config.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete sentinel and regenerate even if cache is valid",
    )
    args = parser.parse_args()

    if args.force and _SENTINEL.exists():
        _SENTINEL.unlink()
        logger.info("Sentinel deleted — forcing chart regeneration.")

    cfg = load_config(args.config)
    result = run(cfg)

    print(f"\nChart generation complete.")
    print(f"  Charts written : {result['charts_written']}")
    print(f"  Output dir     : {result['output_dir']}")
    sys.exit(0)
