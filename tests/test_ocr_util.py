"""Tests for shared OCR text utilities."""

from scripts.ocr.ocr_util import normalize_date, strip_thinking_blocks, extract_final_answer


# ---------- normalize_date ----------


def test_normalize_year_first():
    assert normalize_date("'94 6 22") == "1994-06-22"


def test_normalize_month_day_year_tight():
    assert normalize_date("10 3'99") == "1999-10-03"


def test_normalize_month_day_year_spaced_apostrophe():
    assert normalize_date("5 22 '95") == "1995-05-22"


def test_normalize_rejects_nonsense():
    assert normalize_date("not a date") is None


def test_normalize_two_digit_year_split_1950_boundary():
    # 49 -> 2049, 50 -> 1950
    assert normalize_date("1 1'49") == "2049-01-01"
    assert normalize_date("1 1'50") == "1950-01-01"


# ---------- strip_thinking_blocks ----------


def test_strip_think_tag():
    raw = "<think>reasoning here</think>\n10 3'99"
    assert strip_thinking_blocks(raw).strip() == "10 3'99"


def test_strip_thinking_tag_variant():
    raw = "<thinking>foo</thinking>\n5 17'94"
    assert strip_thinking_blocks(raw).strip() == "5 17'94"


def test_strip_multiline_think():
    raw = "<think>line one\nline two\nline three</think>\n\nfinal: 7 8'02"
    assert strip_thinking_blocks(raw).strip() == "final: 7 8'02"


def test_strip_noop_when_no_tags():
    raw = "10 3'99"
    assert strip_thinking_blocks(raw) == "10 3'99"


def test_strip_empty_after_strip():
    raw = "<think>only thinking</think>"
    assert strip_thinking_blocks(raw).strip() == ""


# ---------- extract_final_answer ----------


def test_extract_final_answer_uses_stripped_when_non_empty():
    raw = "<think>...</think>\n10 3'99"
    assert extract_final_answer(raw) == "10 3'99"


def test_extract_final_answer_falls_back_to_last_nonempty_line():
    raw = "<think>whole response was thinking</think>\n\n"
    # Stripped is empty; should fall back to last non-empty line of *raw*,
    # which here is the closing </think> line. Confirm that behavior.
    out = extract_final_answer(raw)
    assert out == "<think>whole response was thinking</think>"


def test_extract_final_answer_plain():
    raw = "10 3'99"
    assert extract_final_answer(raw) == "10 3'99"


def test_extract_final_answer_picks_last_line_when_multi_line():
    raw = "I see a date stamp.\nThe date is:\n10 3'99"
    assert extract_final_answer(raw) == "10 3'99"
