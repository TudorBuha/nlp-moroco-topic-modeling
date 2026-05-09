"""Unit tests for `src.preprocessing.clean`."""

from __future__ import annotations

from src.preprocessing.clean import (
    clean_text,
    collapse_whitespace,
    lowercase,
    normalize_diacritics,
    remove_digits,
    remove_punctuation,
)


class TestNormalizeDiacritics:
    def test_s_cedilla_to_s_comma(self):
        assert normalize_diacritics("\u015fcoala") == "\u0219coala"

    def test_t_cedilla_to_t_comma(self):
        assert normalize_diacritics("\u0163ara") == "\u021bara"

    def test_uppercase_variants(self):
        assert normalize_diacritics("\u015eTI\u021aA") == "\u0218TI\u021aA"

    def test_already_canonical_unchanged(self):
        assert normalize_diacritics("\u0219tii") == "\u0219tii"

    def test_ascii_unchanged(self):
        assert normalize_diacritics("hello") == "hello"


class TestRemoveDigits:
    def test_strips_numbers(self):
        assert remove_digits("anul 2024 a fost").strip().split() == ["anul", "a", "fost"]

    def test_no_digits_unchanged(self):
        out = remove_digits("fara cifre")
        assert out == "fara cifre"


class TestRemovePunctuation:
    def test_strips_period_and_comma(self):
        assert remove_punctuation("salut, lume.").split() == ["salut", "lume"]

    def test_keeps_diacritics(self):
        out = remove_punctuation("\u0219tia\u021bi")
        assert "\u0219" in out and "\u021b" in out


class TestCollapseWhitespace:
    def test_multiple_spaces(self):
        assert collapse_whitespace("a   b\t\tc\n\nd") == "a b c d"

    def test_trims_edges(self):
        assert collapse_whitespace("   x   ") == "x"


class TestCleanTextPipeline:
    def test_full_pipeline(self):
        out = clean_text("Anul 2024 a fost interesant!\n\u015ftia\u0163i?")
        assert out == "anul a fost interesant \u0219tia\u021bi"

    def test_idempotent_on_clean_input(self):
        cleaned_once = clean_text("test simplu")
        cleaned_twice = clean_text(cleaned_once)
        assert cleaned_once == cleaned_twice


class TestLowercase:
    def test_basic(self):
        assert lowercase("ABC") == "abc"

    def test_keeps_diacritics(self):
        assert lowercase("\u0218TIRE") == "\u0219tire"
