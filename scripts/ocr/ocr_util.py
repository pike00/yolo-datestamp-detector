"""Shared OCR text utilities used by ocr_gemma.py, bench_vlm_ocr.py,
seed_bench_ground_truth.py, and report_vlm_bench.py.

No external dependencies beyond the stdlib.
"""

from __future__ import annotations

import re

_THINK_TAG_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)


def strip_thinking_blocks(raw: str) -> str:
    """Remove <think>...</think> and <thinking>...</thinking> blocks.

    Multiline and DOTALL so embedded newlines are matched. Nested tags are
    not supported (they should not appear in practice, and Ollama's chat
    API does not emit them).
    """
    return _THINK_TAG_RE.sub("", raw)


def extract_final_answer(raw: str) -> str:
    """Return the model's best-guess final answer.

    Strips thinking blocks first. If the stripped text has any non-empty
    lines, returns the last non-empty line. If the stripped text is fully
    empty, falls back to the last non-empty line of the unstripped raw.
    """
    stripped = strip_thinking_blocks(raw)
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    fallback_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if fallback_lines:
        return fallback_lines[-1]
    return ""


def normalize_date(raw: str) -> str | None:
    """Parse raw OCR text into 'YYYY-MM-DD' or return None if not parseable.

    Accepts the three observed ScanMyPhotos stamp formats:
      - Year-first:      "'94 6 22"   -> 1994-06-22
      - Tight M D'YY:    "10 3'99"    -> 1999-10-03
      - Spaced M D 'YY:  "5 22 '95"   -> 1995-05-22
    And one partial shape (month + year only):
      - "9'95"  -> 1995-09-00   (day unknown; caller decides how to store this)

    Two-digit years split at 50: 00-49 -> 20xx, 50-99 -> 19xx.
    """
    text = raw.strip().replace(":", " ").replace(".", " ").replace("-", " ")
    text = re.sub(r"['\u2018\u2019]", "'", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Year-first: 'YY M D
    m = re.match(r"'?(\d{2})\s+(\d{1,2})\s+(\d{1,2})$", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = 1900 + y if y >= 50 else 2000 + y
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{year}-{mo:02d}-{d:02d}"

    # Tight: M D'YY  (space between day and apostrophe omitted)
    m = re.match(r"(\d{1,2})\s+(\d{1,2})'?(\d{2})$", text)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = 1900 + y if y >= 50 else 2000 + y
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{year}-{mo:02d}-{d:02d}"

    # Spaced apostrophe: M D 'YY
    m = re.match(r"(\d{1,2})\s+(\d{1,2})\s+'?(\d{2})$", text)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = 1900 + y if y >= 50 else 2000 + y
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{year}-{mo:02d}-{d:02d}"

    # Partial: M'YY or M YY (day missing)
    m = re.match(r"(\d{1,2})\s*'?(\d{2})$", text)
    if m:
        mo, y = int(m.group(1)), int(m.group(2))
        year = 1900 + y if y >= 50 else 2000 + y
        if 1 <= mo <= 12:
            return f"{year}-{mo:02d}-00"

    return None
