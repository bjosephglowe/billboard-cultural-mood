"""
src/analysis/emotion_classifier.py

Stage 6 — Emotion Classification
===================================
Runs batched Ekman 7-class emotion classification on the chorus text
of every song in data/processed/chorus_extracted.csv using the
j-hartmann/emotion-english-distilroberta-base model from HuggingFace.

Ekman emotion classes
---------------------
    anger | disgust | fear | joy | neutral | sadness | surprise

For each song the classifier produces:
    - A probability score for each of the 7 classes (0.0 – 1.0)
    - A dominant_emotion label (highest-probability class)
    - A dominant_emotion_score (probability of dominant class)

Batching
--------
Songs are processed in batches of config.analysis.emotion_batch_size
(default 16). Batching is critical for GPU utilisation — processing
one song at a time is ~10x slower.

Long-text truncation
--------------------
The model has a 512-token limit. Chorus texts exceeding this are
truncated by the tokeniser (truncation=True). For the typical chorus
(30–80 tokens) this is never triggered.

Device selection
----------------
    CUDA  → used automatically if torch.cuda.is_available()
    MPS   → used on Apple Silicon if torch.backends.mps.is_available()
    CPU   → fallback

Model caching
-------------
HuggingFace downloads the model to ~/.cache/huggingface on first run.
Subsequent runs load from the local cache — no internet required.

Idempotency
-----------
Sentinel .emotion_complete records config_hash.

Output schema
-------------
See src/pipeline/schemas.py → layer2_emotion_schema.

    song_id                : str
    emotional_tone         : str   — dominant Ekman label (required by layer2_emotion_schema)
    chorus_sentiment_score : float — nullable; populated by contrast_metrics join
    chorus_emotional_tone  : str   — proxy from emotional_tone; nullable
    dominant_emotion_score : float — probability of dominant class [0.0, 1.0]
    emotion_anger          : float
    emotion_disgust        : float
    emotion_fear           : float
    emotion_joy            : float
    emotion_neutral        : float
    emotion_sadness        : float
    emotion_surprise       : float
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
from src.pipeline.schemas import layer2_emotion_schema, validate

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "chorus_extracted.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "analysis" / "layer2_emotion.csv"
_SENTINEL = _PROJECT_ROOT / "data" / "analysis" / ".emotion_complete"

# ── Model identifier ──────────────────────────────────────────────────────────
_DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# ── Ekman label set (model output order may vary — we normalise by label) ─────
_EKMAN_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def run(config: ProjectConfig) -> dict:
    """
    Execute Stage 6 — Emotion classification.

    Parameters
    ----------
    config : ProjectConfig

    Returns
    -------
    dict with keys:
        songs_classified     : int
        dominant_emotion_dist: dict  — {emotion: count}
        output_path          : Path
        skipped              : bool
    """
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(
            f"chorus_extracted.csv not found at {_INPUT_PATH}. "
            "Run Stage 4 [CHORUS] first."
        )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check ────────────────────────────────────────────────────
    if sentinel_config_matches(_SENTINEL, config):
        logger.info("Stage 6 [EMOTION] — sentinel matched, skipping.")
        df = pd.read_csv(_OUTPUT_PATH)
        return {
            "songs_classified": len(df),
            "dominant_emotion_dist": df["emotional_tone"].value_counts().to_dict(),
            "output_path": _OUTPUT_PATH,
            "skipped": True,
        }

    chorus_df = pd.read_csv(_INPUT_PATH)
    logger.info("Stage 6 [EMOTION] — classifying %d songs …", len(chorus_df))

    # ── Load model ───────────────────────────────────────────────────────────
    classifier = _load_classifier(config)

    # ── Batch inference ──────────────────────────────────────────────────────
    texts = chorus_df["chorus_text"].fillna("").tolist()
    batch_size = config.analysis.emotion_batch_size
    all_scores: list[dict] = []

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        scores = _classify_batch(classifier, batch)
        all_scores.extend(scores)
        logger.debug(
            "Emotion batch %d–%d complete.",
            batch_start,
            min(batch_start + batch_size, len(texts)) - 1,
        )

    # ── Build output DataFrame ───────────────────────────────────────────────
    df = _build_dataframe(chorus_df["song_id"].tolist(), all_scores)

    # ── Validate ─────────────────────────────────────────────────────────────
    df = validate(df, layer2_emotion_schema, stage_name="EMOTION")

    # ── Write output ─────────────────────────────────────────────────────────
    df.to_csv(_OUTPUT_PATH, index=False)
    write_sentinel(_SENTINEL, stage="EMOTION", config=config)

    dist = df["emotional_tone"].value_counts().to_dict()
    logger.info(
        "Stage 6 [EMOTION] — complete. distribution: %s → %s",
        dist,
        _OUTPUT_PATH,
    )
    return {
        "songs_classified": len(df),
        "dominant_emotion_dist": dist,
        "output_path": _OUTPUT_PATH,
        "skipped": False,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════


def _load_classifier(config: ProjectConfig):
    """
    Load the HuggingFace text-classification pipeline.

    Selects the best available device automatically:
        CUDA → MPS → CPU

    Returns a transformers.pipeline object.
    """
    try:
        import torch
        from transformers import pipeline  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for Stage 6. "
            "Install with: pip install transformers torch"
        ) from exc

    # Device selection
    if torch.cuda.is_available():
        device = 0  # first CUDA GPU
        device_label = "CUDA"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_label = "MPS (Apple Silicon)"
    else:
        device = -1  # CPU
        device_label = "CPU"

    model_name = config.analysis.emotion_model or _DEFAULT_MODEL
    logger.info("Loading emotion model '%s' on %s …", model_name, device_label)

    classifier = pipeline(
        task="text-classification",
        model=model_name,
        top_k=None,  # return scores for ALL labels
        device=device,
        truncation=True,
        max_length=512,
    )

    logger.info("Emotion model loaded.")
    return classifier


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════


def _classify_batch(classifier, texts: list[str]) -> list[dict]:
    """
    Run classifier on a batch of texts.

    Returns a list of score dicts, one per input text:
        {
            "anger": 0.12, "disgust": 0.03, "fear": 0.05,
            "joy": 0.60, "neutral": 0.08, "sadness": 0.07,
            "surprise": 0.05
        }

    Empty strings are handled gracefully — classifier receives a
    single space to avoid tokeniser errors on empty input.
    """
    safe_texts = [t if t.strip() else " " for t in texts]

    raw_results = classifier(safe_texts)

    scores_list: list[dict] = []
    for result in raw_results:
        # result is a list of {"label": str, "score": float} dicts
        scores = {item["label"].lower(): round(item["score"], 6) for item in result}
        # Ensure all 7 labels present even if model omits any
        for label in _EKMAN_LABELS:
            scores.setdefault(label, 0.0)
        scores_list.append(scores)

    return scores_list


def _build_dataframe(song_ids: list[str], scores_list: list[dict]) -> pd.DataFrame:
    """
    Combine song_ids and per-song score dicts into the output DataFrame.

    Adds dominant_emotion and dominant_emotion_score columns.
    """
    rows = []
    for song_id, scores in zip(song_ids, scores_list):
        dominant = max(scores, key=scores.get)
        dominant_score = scores[dominant]
        rows.append(
            {
                # -- Required by layer2_emotion_schema
                "song_id": song_id,
                "emotional_tone": dominant,  # renamed from dominant_emotion
                "chorus_sentiment_score": None,  # populated by contrast_metrics
                # join; nullable=True in schema
                "chorus_emotional_tone": dominant,  # best available proxy at this
                # stage; overwritten downstream
                # ── Extra cols — pass through strict=False ─────────────────────
                "dominant_emotion_score": round(dominant_score, 6),
                "emotion_anger": scores.get("anger", 0.0),
                "emotion_disgust": scores.get("disgust", 0.0),
                "emotion_fear": scores.get("fear", 0.0),
                "emotion_joy": scores.get("joy", 0.0),
                "emotion_neutral": scores.get("neutral", 0.0),
                "emotion_sadness": scores.get("sadness", 0.0),
                "emotion_surprise": scores.get("surprise", 0.0),
            }
        )

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
        description="Stage 6 — Emotion classification (DistilRoBERTa)"
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

    print(f"\nEmotion classification complete.")
    print(f"  Songs classified : {result['songs_classified']:,}")
    print(f"  Dominant emotion distribution:")
    for emotion, count in sorted(
        result["dominant_emotion_dist"].items(), key=lambda x: -x[1]
    ):
        print(f"    {emotion:<12}: {count:,}")
    print(f"  Output           : {result['output_path']}")
    sys.exit(0)
