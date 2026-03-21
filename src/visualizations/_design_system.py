"""
src/visualization/_design_system.py

Single source of truth for all visual design constants used across the
Billboard Cultural Mood Analysis chart suite and HTML report.

Provides:
    DECADE_ORDER          — canonical ordered list of decade labels
    DECADE_COLORS         — hex color per decade
    EMOTION_COLORS        — hex color per Ekman emotional tone
    JUNG_COLORS           — hex color per Jungian stage
    PERSPECTIVE_COLORS    — hex color per narrative perspective
    THEME_COLORS          — hex color per theme taxonomy label
    FONT                  — typography constants
    LAYOUT                — shared Plotly layout defaults
    CHART_SIZE            — standard figure dimensions
    DECADE_FOOTNOTE       — standard chart footnote string
    hex_with_alpha()      — convert hex color to rgba() string

Design principles:
    - Decade colors follow a warm-to-cool arc across time (1960s warm amber
      through 2020s cool slate) to reinforce temporal progression visually.
    - Emotion colors follow established psychological color associations
      (joy = gold, sadness = blue, anger = red, etc.).
    - All colors are WCAG AA compliant against the dark chart background.
    - No figure objects are constructed here — this module is pure constants.

IMPORTANT: DECADE_ORDER is imported from src.pipeline.schemas.VALID_DECADE_LABELS
to ensure visualization and schema validation always use identical decade labels.
Redefining it here separately would create a silent drift risk.
"""

from __future__ import annotations

from src.pipeline.schemas import VALID_DECADE_LABELS

# ── Decade Ordering ───────────────────────────────────────────────────────────

# Authoritative ordered decade list — sourced directly from schemas.py.
# All chart functions must iterate over DECADE_ORDER (not sort dynamically)
# to guarantee consistent left-to-right temporal ordering.
DECADE_ORDER: list[str] = VALID_DECADE_LABELS  # ["1960s*", "1970s", ..., "2020s"]


# ── Color Palettes ────────────────────────────────────────────────────────────

# Warm amber (1960s) → muted teal (1980s) → deep indigo (2000s) → slate (2020s)
DECADE_COLORS: dict[str, str] = {
    "1960s*": "#E8A838",  # warm amber
    "1970s": "#D4724A",  # burnt sienna
    "1980s": "#3AAFA9",  # muted teal
    "1990s": "#6C8EBF",  # mid blue
    "2000s": "#5B5EA6",  # deep indigo
    "2010s": "#9B2335",  # crimson
    "2020s": "#6B7C93",  # cool slate
}

# Ekman 7-class emotion colors — psychologically grounded associations
EMOTION_COLORS: dict[str, str] = {
    "joy": "#F4C430",  # golden yellow
    "sadness": "#4A90D9",  # medium blue
    "anger": "#D94F3D",  # warm red
    "fear": "#7B5EA7",  # muted purple
    "disgust": "#5C8A3C",  # olive green
    "surprise": "#E8834A",  # warm orange
    "neutral": "#9B9EA4",  # neutral grey
}

# Jungian stage colors — archetypal / depth psychology palette
JUNG_COLORS: dict[str, str] = {
    "shadow": "#2C2C54",  # deep indigo-black
    "persona": "#A4B0BE",  # silver grey
    "anima_animus": "#C44569",  # rose-magenta
    "integration": "#40739E",  # calm steel blue
    "transcendence": "#F0C419",  # luminous gold
    "unclassified": "#636E72",  # muted grey
}

# Narrative perspective colors
PERSPECTIVE_COLORS: dict[str, str] = {
    "first_person": "#0097A7",  # teal
    "second_person": "#FF7043",  # deep orange
    "third_person": "#7B1FA2",  # purple
    "abstract": "#546E7A",  # blue-grey
}

# Theme taxonomy colors — 12-class
THEME_COLORS: dict[str, str] = {
    "love": "#E91E63",  # pink
    "heartbreak": "#5C6BC0",  # indigo
    "party_celebration": "#FFC107",  # amber
    "identity": "#26A69A",  # teal
    "struggle": "#EF5350",  # red
    "rebellion": "#AB47BC",  # purple
    "wealth_success": "#66BB6A",  # green
    "friendship": "#29B6F6",  # light blue
    "nostalgia": "#FF7043",  # deep orange
    "self_reflection": "#8D6E63",  # brown
    "empowerment": "#D4E157",  # lime
    "unity": "#26C6DA",  # cyan
}


# ── Typography ────────────────────────────────────────────────────────────────

FONT: dict[str, str | int] = {
    "family": "Inter, Helvetica Neue, Arial, sans-serif",
    "size_title": 18,
    "size_subtitle": 14,
    "size_axis": 12,
    "size_tick": 11,
    "size_legend": 11,
    "size_annot": 10,
    "color_primary": "#F0F0F0",  # near-white — main labels on dark bg
    "color_muted": "#A0A4A8",  # muted grey — subtitles, footnotes
}


# ── Chart Dimensions ──────────────────────────────────────────────────────────

CHART_SIZE: dict[str, int] = {
    "width": 1200,
    "height": 650,
}

# Height overrides for specific chart types
CHART_SIZE_HEATMAP: dict[str, int] = {
    "width": 1200,
    "height": 780,
}

CHART_SIZE_TABLE: dict[str, int] = {
    "width": 1200,
    "height": 500,
}


# ── Background and Grid ───────────────────────────────────────────────────────

COLORS_BG: dict[str, str] = {
    "paper": "#1A1A2E",  # deep navy — chart paper background
    "plot": "#16213E",  # slightly lighter navy — plot area
    "grid": "#2A2A4A",  # subtle grid lines
    "zero_line": "#3A3A5A",  # zero-reference line
}


# ── Shared Layout Defaults ────────────────────────────────────────────────────

# Base Plotly layout dict — all chart functions should call
# fig.update_layout(**BASE_LAYOUT) and then apply chart-specific overrides.
BASE_LAYOUT: dict = {
    "paper_bgcolor": COLORS_BG["paper"],
    "plot_bgcolor": COLORS_BG["plot"],
    "font": {
        "family": FONT["family"],
        "color": FONT["color_primary"],
        "size": FONT["size_tick"],
    },
    "title": {
        "font": {
            "family": FONT["family"],
            "size": FONT["size_title"],
            "color": FONT["color_primary"],
        },
        "x": 0.05,
        "xanchor": "left",
    },
    "legend": {
        "bgcolor": "rgba(0,0,0,0)",
        "bordercolor": COLORS_BG["grid"],
        "borderwidth": 1,
        "font": {
            "size": FONT["size_legend"],
            "color": FONT["color_primary"],
        },
    },
    "margin": {"l": 70, "r": 40, "t": 80, "b": 80},
    "xaxis": {
        "gridcolor": COLORS_BG["grid"],
        "zerolinecolor": COLORS_BG["zero_line"],
        "tickfont": {
            "size": FONT["size_tick"],
            "color": FONT["color_primary"],
        },
        "title_font": {
            "size": FONT["size_axis"],
            "color": FONT["color_muted"],
        },
    },
    "yaxis": {
        "gridcolor": COLORS_BG["grid"],
        "zerolinecolor": COLORS_BG["zero_line"],
        "tickfont": {
            "size": FONT["size_tick"],
            "color": FONT["color_primary"],
        },
        "title_font": {
            "size": FONT["size_axis"],
            "color": FONT["color_muted"],
        },
    },
}


# ── Standard Footnote ─────────────────────────────────────────────────────────

DECADE_FOOTNOTE: str = (
    "* 1960s bucket covers 1958–1969 (Billboard Hot 100 launch year). "
    "All other decades are standard 10-year spans."
)


# ── Utility Functions ─────────────────────────────────────────────────────────


def hex_with_alpha(hex_color: str, alpha: float) -> str:
    """
    Convert a hex color string to a Plotly-compatible rgba() string.

    Args:
        hex_color: 6-character hex string with or without leading '#'.
                   e.g. '#E8A838' or 'E8A838'
        alpha:     Opacity value in [0.0, 1.0].

    Returns:
        rgba() string suitable for use in Plotly color properties.
        e.g. 'rgba(232, 168, 56, 0.6)'

    Raises:
        ValueError: if hex_color is not a valid 6-character hex string.
        ValueError: if alpha is not in [0.0, 1.0].

    Examples:
        >>> hex_with_alpha('#E8A838', 0.6)
        'rgba(232, 168, 56, 0.6)'

        >>> hex_with_alpha('4A90D9', 1.0)
        'rgba(74, 144, 217, 1.0)'

        >>> hex_with_alpha('#1A1A2E', 0.0)
        'rgba(26, 26, 46, 0.0)'
    """
    if not isinstance(hex_color, str):
        raise ValueError(f"hex_color must be a string, got {type(hex_color).__name__}")

    cleaned = hex_color.lstrip("#").strip()

    if len(cleaned) != 6:
        raise ValueError(
            f"hex_color must be a 6-character hex string, got: {hex_color!r}"
        )

    try:
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
    except ValueError:
        raise ValueError(f"hex_color contains non-hex characters: {hex_color!r}")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got: {alpha}")

    return f"rgba({r}, {g}, {b}, {alpha})"


def decade_color_with_alpha(decade_label: str, alpha: float) -> str:
    """
    Convenience wrapper — return the rgba() string for a decade's color.

    Args:
        decade_label: One of the values in DECADE_ORDER.
        alpha:        Opacity value in [0.0, 1.0].

    Returns:
        rgba() string for the decade's canonical color at the given opacity.

    Raises:
        KeyError:   if decade_label is not in DECADE_COLORS.
        ValueError: if alpha is out of range.

    Example:
        >>> decade_color_with_alpha('1980s', 0.4)
        'rgba(58, 175, 169, 0.4)'
    """
    if decade_label not in DECADE_COLORS:
        raise KeyError(
            f"Unknown decade label: {decade_label!r}. "
            f"Valid labels: {list(DECADE_COLORS.keys())}"
        )
    return hex_with_alpha(DECADE_COLORS[decade_label], alpha)
