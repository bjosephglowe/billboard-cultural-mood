"""
src/utils/identifiers.py

Shared identifier utilities for the Billboard Cultural Mood Analysis pipeline.

Provides:
    make_song_id()        — deterministic 16-char hex ID from artist + title + year
    make_cache_filename() — expected lyrics cache filename for a given song_id
    validate_song_id()    — format check for song_id strings

The song_id is the canonical primary key used across all pipeline stages,
CSV files, JSON cache entries, and test fixtures. It must be stable across
runs, case-insensitive, and robust to minor formatting differences in
artist/title strings sourced from Billboard vs Genius.

Normalization rules applied before hashing:
    - Lowercased
    - Leading/trailing whitespace stripped
    - Non-alphanumeric characters (except spaces) replaced with a space
    - Runs of multiple spaces collapsed to a single space
    - Year cast to int before encoding

Example:
    make_song_id("The Beatles", "Hey Jude", 1968)
    → "a3f2c1d8e9b04712"   (illustrative — actual value is MD5-derived)
"""

from __future__ import annotations

import hashlib
import re

# ── Public API ────────────────────────────────────────────────────────────────


def make_song_id(artist: str, title: str, year: int) -> str:
    """
    Generate a deterministic 16-character lowercase hex song identifier.

    The ID is computed as the first 16 characters of the MD5 hash of the
    normalized canonical string:
        "{normalized_artist}|{normalized_title}|{year}"

    Args:
        artist: Artist name string (any casing, any punctuation)
        title:  Song title string (any casing, any punctuation)
        year:   4-digit release year as an integer

    Returns:
        16-character lowercase hex string (e.g. "a3f2c1d8e9b04712")

    Raises:
        TypeError:  if artist or title are not strings, or year is not int-like
        ValueError: if year cannot be cast to int

    Examples:
        >>> make_song_id("The Beatles", "Hey Jude", 1968)
        'a3f2c1d8e9b04712'   # illustrative

        >>> make_song_id("the beatles", "hey jude", 1968)
        'a3f2c1d8e9b04712'   # same result — normalization is applied

        >>> make_song_id("Beyoncé", "Crazy in Love", 2003)
        '...'                 # deterministic 16-char hex
    """
    if not isinstance(artist, str):
        raise TypeError(f"artist must be a string, got {type(artist).__name__}")
    if not isinstance(title, str):
        raise TypeError(f"title must be a string, got {type(title).__name__}")

    raw = f"{_normalize(artist)}|{_normalize(title)}|{int(year)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def make_cache_filename(song_id: str) -> str:
    """
    Return the expected lyrics cache filename for a given song_id.

    The lyrics cache stores one JSON file per song at:
        cache/lyrics/{song_id}.json

    Args:
        song_id: 16-character hex song identifier from make_song_id()

    Returns:
        Filename string, e.g. "a3f2c1d8e9b04712.json"

    Raises:
        ValueError: if song_id does not match the expected 16-char hex format
    """
    if not validate_song_id(song_id):
        raise ValueError(
            f"Invalid song_id format: {song_id!r}. "
            f"Expected 16 lowercase hex characters."
        )
    return f"{song_id}.json"


def validate_song_id(song_id: str) -> bool:
    """
    Return True if song_id matches the canonical 16-character lowercase hex format.

    This is a format check only — it does not verify the ID corresponds to
    any known song in the dataset.

    Args:
        song_id: Value to check

    Returns:
        True if valid format, False otherwise

    Examples:
        >>> validate_song_id("a3f2c1d8e9b04712")
        True

        >>> validate_song_id("A3F2C1D8E9B04712")  # uppercase
        False

        >>> validate_song_id("short")
        False

        >>> validate_song_id("")
        False
    """
    if not isinstance(song_id, str):
        return False
    return bool(re.fullmatch(r"[0-9a-f]{16}", song_id))


# ── Internal Helpers ──────────────────────────────────────────────────────────


def _normalize(s: str) -> str:
    """
    Normalize a string for use in song ID generation.

    Steps:
        1. Strip leading/trailing whitespace
        2. Lowercase
        3. Replace any non-alphanumeric character (except space) with a space
        4. Collapse multiple consecutive spaces to one
        5. Strip again after collapse

    This normalization ensures that minor differences in punctuation,
    accents (after ASCII transliteration), or casing do not produce
    different IDs for the same logical song.

    Args:
        s: Raw input string

    Returns:
        Normalized string suitable for ID hashing
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()
