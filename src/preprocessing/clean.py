"""Text cleaning utilities (Phase 1.4)."""

from __future__ import annotations

import re
import unicodedata

# Map common look-alikes to canonical Romanian diacritics.
# (s-cedilla → s-comma, t-cedilla → t-comma, plus a-breve / a-circumflex / i-circumflex
#  variants that sometimes appear pre-decomposed.)
_DIACRITIC_MAP = {
    "\u015F": "\u0219",  # ş -> ș
    "\u015E": "\u0218",  # Ş -> Ș
    "\u0163": "\u021B",  # ţ -> ț
    "\u0162": "\u021A",  # Ţ -> Ț
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_DIGIT_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


def normalize_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for bad, good in _DIACRITIC_MAP.items():
        text = text.replace(bad, good)
    return text


def lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return _PUNCT_RE.sub(" ", text)


def remove_digits(text: str) -> str:
    return _DIGIT_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Default cleaning pipeline used by both LDA and BERTopic."""
    text = normalize_diacritics(text)
    text = lowercase(text)
    text = remove_punctuation(text)
    text = remove_digits(text)
    text = collapse_whitespace(text)
    return text
