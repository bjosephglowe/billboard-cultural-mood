"""
src/visualizations/report_builder.py

Stage 11b — HTML Report Generation
=====================================
Assembles a self-contained HTML report from the Stage 10 CMI outputs,
Stage 9 Jungian classifications, and the Stage 11a chart PNGs, and
writes it to outputs/reports/billboard_cultural_mood_report.html.

The report is fully self-contained — all chart images are embedded as
base64 data URIs so the file can be shared without the PNG assets.

Report sections
---------------
1. Executive Summary    — headline CMI stats across all decades
2. Sentiment Arc        — decade-by-decade CMI_sentiment narrative
3. Emotional Landscape  — dominant emotion per decade with counts
4. Jungian Analysis     — stage distribution summary per decade
5. Theme Penetration    — top 3 themes per decade
6. Chart Gallery        — all 5 Stage 11a PNGs embedded inline
7. Methodology          — data sources, scoring model, caveats

Design
------
- Dark theme matching _design_system.COLORS_BG palette
- Typography from _design_system.FONT constants
- Decade accent colours from _design_system.DECADE_COLORS
- No external CSS/JS dependencies — single file, zero network calls

Idempotency
-----------
Sentinel .report_complete records config_hash. If matched, the
existing report path is returned without regeneration. Delete the
sentinel (or change config) to force regeneration.

Output
------
outputs/reports/billboard_cultural_mood_report.html
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.pipeline.config_loader import (
    ProjectConfig,
    load_config,
    sentinel_config_matches,
    write_sentinel,
)
from src.visualizations._design_system import (
    COLORS_BG,
    DECADE_COLORS,
    DECADE_FOOTNOTE,
    DECADE_ORDER,
    FONT,
)

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INPUT_DECADE = _PROJECT_ROOT / "data" / "analysis" / "decade_cmi.csv"
_INPUT_LAYER4 = _PROJECT_ROOT / "data" / "analysis" / "layer2_full_analysis.csv"
_INPUT_LAYER5 = _PROJECT_ROOT / "data" / "analysis" / "layer5_jungian.csv"
_CHARTS_DIR = _PROJECT_ROOT / "outputs" / "visualizations"
_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "reports"
_OUTPUT_REPORT = _OUTPUT_DIR / "billboard_cultural_mood_report.html"
_SENTINEL = _OUTPUT_DIR / ".report_complete"

# ── Chart filenames in display order ─────────────────────────────────────────
_CHART_FILES = [
    "cmi_sentiment_trend.png",
    "emotion_distribution.png",
    "jungian_stage_distribution.png",
    "theme_heatmap.png",
    "resonance_distribution.png",
]


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def run(config: ProjectConfig) -> dict:
    """
    Execute Stage 11b — HTML report generation.

    Parameters
    ----------
    config : ProjectConfig

    Returns
    -------
    dict with keys:
        output_path  : Path
        file_size_kb : float
        skipped      : bool
    """
    for path, label in [
        (_INPUT_DECADE, "decade_cmi.csv"),
        (_INPUT_LAYER4, "layer2_full_analysis.csv"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found at {path}. Run Stage 10 [CMI] first."
            )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check ────────────────────────────────────────────────────
    if sentinel_config_matches(_SENTINEL, config):
        logger.info("Stage 11b [REPORT] — sentinel matched, skipping.")
        size_kb = _OUTPUT_REPORT.stat().st_size / 1024 if _OUTPUT_REPORT.exists() else 0
        return {
            "output_path": _OUTPUT_REPORT,
            "file_size_kb": round(size_kb, 1),
            "skipped": True,
        }

    # ── Load inputs ──────────────────────────────────────────────────────────
    decade_df = pd.read_csv(_INPUT_DECADE)
    layer4_df = pd.read_csv(_INPUT_LAYER4)
    layer5_df = pd.read_csv(_INPUT_LAYER5) if _INPUT_LAYER5.exists() else pd.DataFrame()

    decade_df = _sort_decades(decade_df)

    logger.info(
        "Stage 11b [REPORT] — building report for %d decades, %d songs …",
        len(decade_df),
        len(layer4_df),
    )

    # ── Build HTML sections ───────────────────────────────────────────────────
    html = _assemble_report(decade_df, layer4_df, layer5_df, config)

    # ── Write output ─────────────────────────────────────────────────────────
    _OUTPUT_REPORT.write_text(html, encoding="utf-8")
    write_sentinel(_SENTINEL, stage="REPORT", config=config)

    size_kb = round(_OUTPUT_REPORT.stat().st_size / 1024, 1)
    logger.info(
        "Stage 11b [REPORT] — complete. %.1f KB → %s",
        size_kb,
        _OUTPUT_REPORT,
    )

    return {
        "output_path": _OUTPUT_REPORT,
        "file_size_kb": size_kb,
        "skipped": False,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Report assembly
# ═════════════════════════════════════════════════════════════════════════════


def _assemble_report(
    decade_df: pd.DataFrame,
    layer4_df: pd.DataFrame,
    layer5_df: pd.DataFrame,
    config: ProjectConfig,
) -> str:
    """Assemble and return the full HTML report string."""
    generated_at = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

    sections = [
        _section_executive_summary(decade_df, layer4_df),
        _section_sentiment_arc(decade_df),
        _section_emotional_landscape(decade_df, layer4_df),
        _section_jungian_analysis(decade_df, layer5_df),
        _section_theme_penetration(decade_df),
        _section_chart_gallery(),
        _section_methodology(generated_at),
    ]

    return _wrap_html(
        title="Billboard Cultural Mood Analysis",
        body="\n".join(sections),
        generated_at=generated_at,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Section builders
# ═════════════════════════════════════════════════════════════════════════════


def _section_executive_summary(
    decade_df: pd.DataFrame,
    layer4_df: pd.DataFrame,
) -> str:
    total_songs = len(layer4_df)
    total_decades = len(decade_df)
    scored = (
        int(decade_df["scored_count"].sum())
        if "scored_count" in decade_df.columns
        else 0
    )

    sentiment_vals = decade_df["CMI_sentiment"].dropna()
    peak_decade = ""
    peak_val = ""
    low_decade = ""
    low_val = ""

    if not sentiment_vals.empty:
        peak_idx = sentiment_vals.idxmax()
        low_idx = sentiment_vals.idxmin()
        peak_decade = str(decade_df.loc[peak_idx, "decade_label"])
        peak_val = f"{sentiment_vals[peak_idx]:+.3f}"
        low_decade = str(decade_df.loc[low_idx, "decade_label"])
        low_val = f"{sentiment_vals[low_idx]:+.3f}"

    stats = [
        ("Total Songs Analysed", f"{total_songs:,}"),
        ("Decades Covered", str(total_decades)),
        ("Songs Fully Scored", f"{scored:,}"),
        ("Most Positive Decade", f"{peak_decade} ({peak_val})"),
        ("Most Negative Decade", f"{low_decade} ({low_val})"),
    ]

    cards = "\n".join(
        f"""<div class="stat-card">
               <div class="stat-value">{v}</div>
               <div class="stat-label">{k}</div>
           </div>"""
        for k, v in stats
    )

    return f"""
<section id="executive-summary">
  <h2>Executive Summary</h2>
  <div class="stat-grid">{cards}</div>
</section>"""


def _section_sentiment_arc(decade_df: pd.DataFrame) -> str:
    rows = ""
    for _, row in decade_df.iterrows():
        decade = str(row.get("decade_label", ""))
        val = row.get("CMI_sentiment")
        energy = row.get("CMI_energy")
        emotion = str(row.get("emotional_tone", "—") or "—")

        sentiment_str = f"{val:+.3f}" if pd.notna(val) else "—"
        energy_str = f"{energy:.1f}" if pd.notna(energy) else "—"
        bar_width = int(abs(val) * 100) if pd.notna(val) else 0
        bar_color = "#4A90D9" if (pd.notna(val) and val < 0) else "#F4C430"
        decade_color = DECADE_COLORS.get(decade, "#9B9EA4")

        rows += f"""
<tr>
  <td><span class="decade-badge" style="background:{decade_color}">{decade}</span></td>
  <td>
    <div class="bar-cell">
      <div class="bar" style="width:{bar_width}%;background:{bar_color}"></div>
      <span>{sentiment_str}</span>
    </div>
  </td>
  <td>{energy_str}</td>
  <td>{emotion.capitalize()}</td>
</tr>"""

    return f"""
<section id="sentiment-arc">
  <h2>Decade Sentiment Arc</h2>
  <p>Mean VADER sentiment score and average energy level per decade.
     Positive scores indicate net optimistic lyrical tone;
     negative scores indicate net pessimistic tone.</p>
  <table class="data-table">
    <thead>
      <tr>
        <th>Decade</th>
        <th>CMI Sentiment</th>
        <th>Avg Energy</th>
        <th>Dominant Emotion</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</section>"""


def _section_emotional_landscape(
    decade_df: pd.DataFrame,
    layer4_df: pd.DataFrame,
) -> str:
    rows = ""
    for _, row in decade_df.iterrows():
        decade = str(row.get("decade_label", ""))
        dominant = str(row.get("emotional_tone", "—") or "—")
        jung_stage = str(row.get("dominant_jung_stage", "—") or "—")
        song_count = int(row.get("song_count", 0))
        decade_color = DECADE_COLORS.get(decade, "#9B9EA4")

        rows += f"""
<tr>
  <td><span class="decade-badge" style="background:{decade_color}">{decade}</span></td>
  <td>{dominant.capitalize()}</td>
  <td>{jung_stage.replace("_", " ").title()}</td>
  <td>{song_count:,}</td>
</tr>"""

    return f"""
<section id="emotional-landscape">
  <h2>Emotional Landscape</h2>
  <p>Dominant emotional tone and Jungian development stage per decade,
     based on GPT-4o classification of Billboard Hot 100 lyrics.</p>
  <table class="data-table">
    <thead>
      <tr>
        <th>Decade</th>
        <th>Dominant Emotion</th>
        <th>Jungian Stage</th>
        <th>Songs</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</section>"""


def _section_jungian_analysis(
    decade_df: pd.DataFrame,
    layer5_df: pd.DataFrame,
) -> str:
    if layer5_df.empty or "jung_stage" not in layer5_df.columns:
        classified = unclassified = quality_high = 0
    else:
        classified = int((layer5_df["jung_stage"] != "unclassified").sum())
        unclassified = int((layer5_df["jung_stage"] == "unclassified").sum())
        quality_high = int(
            (layer5_df.get("jungian_quality_flag", pd.Series()) == "high").sum()
        )

    stage_rows = ""
    if not layer5_df.empty and "jung_stage" in layer5_df.columns:
        dist = (
            layer5_df[layer5_df["jung_stage"] != "unclassified"]["jung_stage"]
            .value_counts()
            .head(6)
        )
        for stage, count in dist.items():
            pct = count / max(len(layer5_df), 1) * 100
            stage_rows += f"""
<tr>
  <td>{str(stage).replace("_", " ").title()}</td>
  <td>{count:,}</td>
  <td>{pct:.1f}%</td>
</tr>"""

    return f"""
<section id="jungian-analysis">
  <h2>Jungian Psychological Analysis</h2>
  <p>Each song was classified into one of six Jungian development stages
     using GPT-4o analysis of lyrical themes, emotional tone, and symbolic content.</p>
  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-value">{classified:,}</div>
      <div class="stat-label">Songs Classified</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{unclassified:,}</div>
      <div class="stat-label">Unclassified</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{quality_high:,}</div>
      <div class="stat-label">High Quality</div>
    </div>
  </div>
  {
        f'''<table class="data-table">
    <thead>
      <tr><th>Stage</th><th>Songs</th><th>% of Total</th></tr>
    </thead>
    <tbody>{stage_rows}</tbody>
  </table>'''
        if stage_rows
        else ""
    }
</section>"""


def _section_theme_penetration(decade_df: pd.DataFrame) -> str:
    rows = ""
    for _, row in decade_df.iterrows():
        decade = str(row.get("decade_label", ""))
        top_themes = str(row.get("top_themes", "—") or "—")
        top_songs = str(row.get("top_resonance_songs", "—") or "—")
        decade_color = DECADE_COLORS.get(decade, "#9B9EA4")

        theme_badges = ""
        if top_themes != "—":
            for theme in top_themes.split("|"):
                theme_badges += f'<span class="theme-badge">{theme.strip().replace("_", " ").title()}</span>'

        rows += f"""
<tr>
  <td><span class="decade-badge" style="background:{decade_color}">{decade}</span></td>
  <td>{theme_badges if theme_badges else "—"}</td>
  <td class="small-text">{top_songs.replace("|", " · ")}</td>
</tr>"""

    return f"""
<section id="theme-penetration">
  <h2>Theme Penetration by Decade</h2>
  <p>Top 3 lyrical themes and highest cultural-resonance songs per decade.</p>
  <table class="data-table">
    <thead>
      <tr>
        <th>Decade</th>
        <th>Top Themes</th>
        <th>Top Resonance Songs</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</section>"""


def _section_chart_gallery() -> str:
    """Embed all Stage 11a PNGs as base64 data URIs."""
    items = ""
    for filename in _CHART_FILES:
        png_path = _CHARTS_DIR / filename
        meta_path = _CHARTS_DIR / f"{filename}.meta.json"

        if not png_path.exists():
            logger.warning("Chart not found for gallery: %s — skipping.", filename)
            continue

        b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")

        caption = filename.replace(".png", "").replace("_", " ").title()
        if meta_path.exists():
            try:
                import json

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                caption = meta.get("caption", caption)
            except Exception:  # noqa: BLE001
                pass

        items += f"""
<figure class="chart-figure">
  <img src="data:image/png;base64,{b64}" alt="{caption}" loading="lazy">
  <figcaption>{caption}</figcaption>
</figure>"""

    return f"""
<section id="chart-gallery">
  <h2>Chart Gallery</h2>
  <div class="chart-grid">{items}</div>
</section>"""


def _section_methodology(generated_at: str) -> str:
    return f"""
<section id="methodology">
  <h2>Methodology</h2>
  <div class="methodology-grid">
    <div class="method-block">
      <h3>Data Source</h3>
      <p>Billboard Hot 100 chart data spanning 1958–present.
         Lyrics sourced via Genius API with LRU caching.
         Songs without retrievable lyrics are retained with
         <code>lyrics_quality = "missing"</code>.</p>
    </div>
    <div class="method-block">
      <h3>Sentiment Scoring</h3>
      <p>VADER (Valence Aware Dictionary and sEntiment Reasoner)
         applied to cleaned verse-only lyrics. Scores normalised
         to [−1.0, 1.0]. Energy level (1–5) derived from
         lexical density and punctuation density heuristics.</p>
    </div>
    <div class="method-block">
      <h3>Emotion Classification</h3>
      <p>Ekman 7-class model (joy, sadness, anger, fear, disgust,
         surprise, neutral) via zero-shot transformer inference.
         Chorus-specific classification applied separately where
         chorus was detected.</p>
    </div>
    <div class="method-block">
      <h3>Jungian Classification</h3>
      <p>GPT-4o classifies each song into one of six Jungian
         development stages: shadow, persona, anima/animus,
         integration, transcendence, or unclassified.
         Development score (1–7) reflects psychological
         integration level.</p>
    </div>
    <div class="method-block">
      <h3>Cultural Resonance Score</h3>
      <p>Weighted composite of normalised sentiment, energy,
         chorus detection, lyrics quality, and Jungian development
         score. Component weights configurable via
         <code>config.cmi.*</code>. Scores clipped to [0.0, 1.0].</p>
    </div>
    <div class="method-block">
      <h3>Decade Buckets</h3>
      <p>{DECADE_FOOTNOTE}</p>
    </div>
  </div>
  <p class="generated-note">Report generated {generated_at}</p>
</section>"""


# ═════════════════════════════════════════════════════════════════════════════
# HTML wrapper
# ═════════════════════════════════════════════════════════════════════════════


def _wrap_html(title: str, body: str, generated_at: str) -> str:
    """Wrap body sections in the full HTML document with inline CSS."""

    bg_paper = COLORS_BG["paper"]
    bg_plot = COLORS_BG["plot"]
    bg_grid = COLORS_BG["grid"]
    font_fam = FONT["family"]
    col_pri = FONT["color_primary"]
    col_mut = FONT["color_muted"]

    css = f"""
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: {font_fam};
      background: {bg_paper};
      color: {col_pri};
      line-height: 1.6;
      padding: 2rem;
    }}

    header {{
      text-align: center;
      padding: 3rem 1rem 2rem;
      border-bottom: 1px solid {bg_grid};
      margin-bottom: 3rem;
    }}
    header h1 {{
      font-size: 2.2rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    header p {{
      color: {col_mut};
      margin-top: 0.5rem;
      font-size: 0.95rem;
    }}

    nav {{
      display: flex;
      gap: 1.5rem;
      justify-content: center;
      flex-wrap: wrap;
      margin-bottom: 3rem;
    }}
    nav a {{
      color: {col_mut};
      text-decoration: none;
      font-size: 0.85rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}
    nav a:hover {{ color: {col_pri}; }}

    section {{
      max-width: 1100px;
      margin: 0 auto 4rem;
    }}
    h2 {{
      font-size: 1.4rem;
      font-weight: 600;
      margin-bottom: 1.25rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid {bg_grid};
    }}
    h3 {{
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: {col_pri};
    }}
    p {{
      color: {col_mut};
      margin-bottom: 1rem;
      font-size: 0.95rem;
    }}

    /* Stat cards */
    .stat-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1.5rem;
    }}
    .stat-card {{
      background: {bg_plot};
      border: 1px solid {bg_grid};
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      min-width: 160px;
      flex: 1;
      text-align: center;
    }}
    .stat-value {{
      font-size: 1.5rem;
      font-weight: 700;
      color: {col_pri};
    }}
    .stat-label {{
      font-size: 0.78rem;
      color: {col_mut};
      margin-top: 0.25rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}

    /* Tables */
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    .data-table th {{
      background: {bg_plot};
      color: {col_mut};
      font-weight: 600;
      text-align: left;
      padding: 0.6rem 0.8rem;
      border-bottom: 1px solid {bg_grid};
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
    }}
    .data-table td {{
      padding: 0.6rem 0.8rem;
      border-bottom: 1px solid {bg_grid};
      vertical-align: middle;
    }}
    .data-table tr:last-child td {{ border-bottom: none; }}
    .data-table tr:hover td {{ background: {bg_plot}; }}

    /* Decade badge */
    .decade-badge {{
      display: inline-block;
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
      font-size: 0.78rem;
      font-weight: 600;
      color: #1A1A2E;
    }}

    /* Theme badge */
    .theme-badge {{
      display: inline-block;
      background: {bg_grid};
      border-radius: 4px;
      padding: 0.15rem 0.4rem;
      font-size: 0.75rem;
      margin-right: 0.3rem;
      color: {col_pri};
    }}

    /* Bar cell */
    .bar-cell {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .bar {{
      height: 10px;
      border-radius: 3px;
      min-width: 2px;
      max-width: 120px;
    }}

    /* Chart gallery */
    .chart-grid {{
      display: flex;
      flex-direction: column;
      gap: 2.5rem;
    }}
    .chart-figure {{
      background: {bg_plot};
      border: 1px solid {bg_grid};
      border-radius: 8px;
      overflow: hidden;
      padding: 1rem;
    }}
    .chart-figure img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 4px;
    }}
    figcaption {{
      margin-top: 0.6rem;
      color: {col_mut};
      font-size: 0.82rem;
      text-align: center;
    }}

    /* Methodology */
    .methodology-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.25rem;
      margin-bottom: 1.5rem;
    }}
    .method-block {{
      background: {bg_plot};
      border: 1px solid {bg_grid};
      border-radius: 8px;
      padding: 1.25rem;
    }}
    .method-block p {{ margin-bottom: 0; }}
    code {{
      background: {bg_grid};
      border-radius: 3px;
      padding: 0.1rem 0.35rem;
      font-size: 0.82rem;
      font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }}
    .generated-note {{
      font-size: 0.78rem;
      color: {col_mut};
      text-align: center;
      margin-top: 1rem;
    }}
    .small-text {{ font-size: 0.82rem; color: {col_mut}; }}
    """

    nav_links = "\n".join(
        f'<a href="#{s}">{s.replace("-", " ").title()}</a>'
        for s in [
            "executive-summary",
            "sentiment-arc",
            "emotional-landscape",
            "jungian-analysis",
            "theme-penetration",
            "chart-gallery",
            "methodology",
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>

<header>
  <h1>{title}</h1>
  <p>A decade-by-decade psychological and cultural analysis of the Billboard Hot 100</p>
</header>

<nav>{nav_links}</nav>

{body}

</body>
</html>"""


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _sort_decades(df: pd.DataFrame, col: str = "decade_label") -> pd.DataFrame:
    """Sort a DataFrame by canonical DECADE_ORDER."""
    if col not in df.columns:
        return df
    order_map = {d: i for i, d in enumerate(DECADE_ORDER)}
    df = df.copy()
    df["_sort_key"] = df[col].map(order_map).fillna(999)
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"])
    return df.reset_index(drop=True)


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

    parser = argparse.ArgumentParser(description="Stage 11b — HTML report generation")
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
        logger.info("Sentinel deleted — forcing report regeneration.")

    cfg = load_config(args.config)
    result = run(cfg)

    print(f"\nReport generation complete.")
    print(f"  Output  : {result['output_path']}")
    print(f"  Size    : {result['file_size_kb']:.1f} KB")
    sys.exit(0)
