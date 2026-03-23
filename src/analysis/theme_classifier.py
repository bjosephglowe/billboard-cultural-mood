"""
src/analysis/theme_classifier.py

Stage 7 — Theme Classification
================================
Runs zero-shot BART-MNLI multi-label theme classification on the
chorus text of every song in data/processed/chorus_extracted.csv.

Zero-shot classification
------------------------
Uses facebook/bart-large-mnli via the HuggingFace zero-shot-classification
pipeline. No fine-tuning required — the model is prompted with each
theme label as a candidate hypothesis and scores the probability that
the song's chorus entails that theme.

Theme taxonomy
--------------
Defined in config.analysis.theme_taxonomy. Default 12 classes:

    — matches VALID_THEME_TAXONOMY in schemas.py:
        love            heartbreak      party_celebration
        identity        struggle        rebellion
        wealth_success  friendship      nostalgia
        self_reflection empowerment     unity

Per-song outputs
----------------
    - A probability score for each theme (0.0 – 1.0)
    - dominant_theme     : highest-scoring theme label
    - dominant_theme_score: probability of dominant theme
    - theme_labels       : pipe-delimited string of themes above threshold
                           e.g. "love|heartbreak|longing"
                           Empty string if no theme exceeds threshold.

Batching strategy
-----------------
Zero-shot classification is inherently slower than standard
classification because each (text, candidate_label) pair is scored
as a separate NLI inference. For N songs and T themes that is N×T
forward passes per batch.

We batch at the song level: config.analysis.theme_batch_size songs
per batch (default 8). Within each batch all theme candidates are
scored in one pipeline call using multi_label=True.

Threshold
---------
config.analysis.theme_threshold (default 0.40) — themes scoring
above this are included in theme_labels. The threshold is intentionally
low to allow multi-label assignment.

Idempotency
-----------
Sentinel .themes_complete records config_hash.

Output schema
-------------
See src/pipeline/schemas.py → layer2_themes_schema.

    song_id              : str
    themes               : str   — pipe-delimited active themes (required by layer2_themes_schema)
    dominant_theme       : str   — highest-scoring theme label (extra col)
    dominant_theme_score : float — score of dominant theme (extra col)
    theme_labels         : str   — alias of themes; retained for compatibility (extra col)
    theme_<name>         : float — one column per taxonomy theme (extra cols)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.pipeline.config_loader import (
    ProjectConfig,
    load_config,
    sentinel_config_matches,
    write_sentinel,
)
from src.pipeline.schemas import (
    VALID_THEME_TAXONOMY as _VALID_THEMES,
)
from src.pipeline.schemas import (
    layer2_themes_schema,
    validate,
)

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "chorus_extracted.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "analysis" / "layer2_themes.csv"
_SENTINEL = _PROJECT_ROOT / "data" / "analysis" / ".themes_complete"

# ── Model identifier ──────────────────────────────────────────────────────────
_DEFAULT_MODEL = "facebook/bart-large-mnli"

# ── Column prefix for per-theme score columns ─────────────────────────────────
_THEME_COL_PREFIX = "theme_"


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def run(config: ProjectConfig) -> dict:
    """
    Execute Stage 7 — Theme classification.

    Parameters
    ----------
    config : ProjectConfig

    Returns
    -------
    dict with keys:
        songs_classified    : int
        dominant_theme_dist : dict  — {theme: count}
        theme_taxonomy      : list  — themes used
        output_path         : Path
        skipped             : bool
    """
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(
            f"chorus_extracted.csv not found at {_INPUT_PATH}. "
            "Run Stage 4 [CHORUS] first."
        )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check ────────────────────────────────────────────────────
    if sentinel_config_matches(_SENTINEL, config):
        logger.info("Stage 7 [THEMES] — sentinel matched, skipping.")
        df = pd.read_csv(_OUTPUT_PATH)
        return {
            "songs_classified": len(df),
            "dominant_theme_dist": df["dominant_theme"].value_counts().to_dict(),
            "theme_taxonomy": config.analysis.theme_taxonomy,
            "output_path": _OUTPUT_PATH,
            "skipped": True,
        }

    chorus_df = pd.read_csv(_INPUT_PATH)
    logger.info(
        "Stage 7 [THEMES] — classifying %d songs across %d themes …",
        len(chorus_df),
        len(config.analysis.theme_taxonomy),
    )

    # ── Load model ───────────────────────────────────────────────────────────
    classifier = _load_classifier(config)
    taxonomy = config.analysis.theme_taxonomy
    threshold = config.analysis.theme_threshold
    batch_size = config.analysis.theme_batch_size

    # ── Batch inference ──────────────────────────────────────────────────────
    texts = chorus_df["chorus_text"].fillna("").tolist()
    song_ids = chorus_df["song_id"].tolist()
    all_scores: list[dict] = []

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        scores = _classify_batch(classifier, batch_texts, taxonomy)
        all_scores.extend(scores)
        logger.debug(
            "Theme batch %d–%d complete.",
            batch_start,
            min(batch_start + batch_size, len(texts)) - 1,
        )

    # ── Build output DataFrame ───────────────────────────────────────────────
    df = _build_dataframe(song_ids, all_scores, taxonomy, threshold)

    # ── Validate ─────────────────────────────────────────────────────────────
    df = validate(df, layer2_themes_schema, stage_name="THEMES")

    # ── Write output ─────────────────────────────────────────────────────────
    df.to_csv(_OUTPUT_PATH, index=False)
    write_sentinel(_SENTINEL, stage="THEMES", config=config)

    dist = df["dominant_theme"].value_counts().to_dict()
    logger.info(
        "Stage 7 [THEMES] — complete. dominant distribution: %s → %s",
        dist,
        _OUTPUT_PATH,
    )

    return {
        "songs_classified": len(df),
        "dominant_theme_dist": dist,
        "theme_taxonomy": taxonomy,
        "output_path": _OUTPUT_PATH,
        "skipped": False,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════


def _load_classifier(config: ProjectConfig):
    """
    Load the HuggingFace zero-shot-classification pipeline.

    Selects best available device: CUDA → MPS → CPU.
    Returns a transformers.pipeline object.
    """
    try:
        import torch
        from transformers import pipeline  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for Stage 7. "
            "Install with: pip install transformers torch"
        ) from exc

    if torch.cuda.is_available():
        device = 0
        device_label = "CUDA"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_label = "MPS (Apple Silicon)"
    else:
        device = -1
        device_label = "CPU"

    model_name = config.analysis.theme_model or _DEFAULT_MODEL
    logger.info("Loading theme model '%s' on %s …", model_name, device_label)

    classifier = pipeline(
        task="zero-shot-classification",
        model=model_name,
        device=device,
    )

    logger.info("Theme model loaded.")
    return classifier


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════


def _classify_batch(
    classifier,
    texts: list[str],
    taxonomy: list[str],
) -> list[dict]:
    """
    Run zero-shot classification on a batch of texts against the full taxonomy.

    Uses multi_label=True so that themes are scored independently rather
    than as a mutually exclusive softmax — essential for multi-label output.

    Returns a list of score dicts, one per input text:
        {"love": 0.82, "heartbreak": 0.61, "struggle": 0.38, …}

    Empty strings are replaced with a single space to prevent errors.
    """
    safe_texts = [t if t.strip() else " " for t in texts]

    results = classifier(
        safe_texts,
        candidate_labels=taxonomy,
        multi_label=True,
    )

    # results is a list of dicts when input is a list
    # Each dict: {"labels": [...], "scores": [...], "sequence": str}
    scores_list: list[dict] = []

    for result in results:
        scores = {
            label.lower(): round(score, 6)
            for label, score in zip(result["labels"], result["scores"])
        }
        # Ensure all taxonomy themes present
        for theme in taxonomy:
            scores.setdefault(theme.lower(), 0.0)
        scores_list.append(scores)

    return scores_list


def _build_dataframe(
    song_ids: list[str],
    scores_list: list[dict],
    taxonomy: list[str],
    threshold: float,
) -> pd.DataFrame:
    """
    Build the output DataFrame from song IDs and per-song theme score dicts.

    Adds:
        dominant_theme       — theme with highest score
        dominant_theme_score — that theme's score
        theme_labels         — pipe-delimited themes above threshold
        theme_<name>         — one float column per taxonomy theme
    """
    rows = []
    for song_id, scores in zip(song_ids, scores_list):
        dominant = max(scores, key=scores.get)
        dominant_score = scores[dominant]

        from src.pipeline.schemas import (
            VALID_THEME_TAXONOMY as _VALID_THEMES,  # top-level import preferred — see note below
        )

        active_themes = sorted(
            [t for t, s in scores.items() if s >= threshold and t in _VALID_THEMES],
            key=lambda t: -scores[t],
        )
        theme_labels_str = "|".join(active_themes)

        row = {
            # ── Required by layer2_themes_schema ───────────────────────────────
            "song_id": song_id,
            "themes": theme_labels_str,  # renamed from theme_labels
            # ── Extra cols — pass through strict=False ──────────────────────────
            "dominant_theme": dominant,
            "dominant_theme_score": round(dominant_score, 6),
            "theme_labels": theme_labels_str,  # retained for downstream compat
        }

        # Add per-theme score columns
        for theme in taxonomy:
            col = f"{_THEME_COL_PREFIX}{theme.lower()}"
            row[col] = scores.get(theme.lower(), 0.0)

        rows.append(row)

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Stage 7 — Theme classification (BART-MNLI zero-shot)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "config" / "project_config.yaml",
        help="Path to project_config.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete sentinel and re-classify even if cache is valid",
    )
    args = parser.parse_args()

    if args.force and _SENTINEL.exists():
        _SENTINEL.unlink()
        logger.info("Sentinel deleted — forcing re-classification.")

    cfg = load_config(args.config)
    result = run(cfg)

    print(f"\nTheme classification complete.")
    print(f"  Songs classified : {result['songs_classified']:,}")
    print(f"  Taxonomy         : {', '.join(result['theme_taxonomy'])}")
    print(f"  Dominant theme distribution:")
    for theme, count in sorted(
        result["dominant_theme_dist"].items(), key=lambda x: -x[1]
    ):
        print(f"    {theme:<22}: {count:,}")
    print(f"  Output           : {result['output_path']}")
    sys.exit(0)
