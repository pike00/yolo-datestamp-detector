# VLM OCR Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark six OSS Ollama vision models (kimi-vl, qwen2.5-vl:7b, minicpm-v, gemma3, llama3.2-vision:11b, internvl3:8b) against a regenerated Sonnet ground truth on 200 stratified ScanMyPhotos date-stamp crops, producing a Pareto-frontier report across (normalized-date agreement, throughput, model size, host).

**Architecture:** Three pipeline stages. Stage A (`build_bench_corpus.py`) selects 200 stems stratified by YOLO confidence, pre-crops them, and writes a portable `state/bench/` directory. Stage B (`seed_bench_ground_truth.py` + a Claude Code orchestrator session) dispatches Sonnet subagents over the crops and writes `model='sonnet'` rows into `stamp_ocr`; a human spot-check HTML gate must freeze ground truth before moving on. Stage C (`bench_vlm_ocr.py`) runs per-model Ollama inference on a target host, writing rows under `model='<ollama-tag>@<host-label>'` so the composite PK `(stem, model)` stays collision-free across hosts. `report_vlm_bench.py` joins everything and emits markdown + PNG.

**Tech Stack:** Python 3.14 via `uv` with PEP 723 inline script headers; `psycopg[binary]>=3.1.0` for Postgres; `Pillow>=12.0` for image ops; `requests` for Ollama HTTP; `matplotlib` for the Pareto PNG. Claude Code Task tool for Sonnet subagent dispatch (no direct Anthropic API). Follows existing `scripts/ocr/orchestrate_ocr.py` pattern for orchestrator/subagent split.

**Prerequisite decision locked:** `stamp_ocr` PK stays `(stem, model)`. `host_label` column is added for readable grouping. Multi-host runs of the same Ollama model are disambiguated by suffixing `model` with `@<host-label>` (e.g., `kimi-vl:latest@ares-cpu`). `report_vlm_bench.py` splits on `@` to derive host at report time. No composite-PK migration.

---

## Task 0: Fix corrupted `stamp_predictions` bboxes

**Context:** `infer_all.py:69-73` writes normalized-float bboxes (0.0-1.0) into integer columns, so Postgres truncates them. All 2,132 current rows have `x∈{0,1}, y∈{0,1}, w=0, h=0`. This must be fixed before the corpus builder can run.

**Files:**
- Create: `schema/stamp_predictions_float_bbox.sql`
- Modify: none (existing `infer_all.py` already emits floats correctly; only the schema is wrong)
- Verify via: `scripts/infer/infer_all.py`

- [ ] **Step 1: Write the migration SQL**

Create `schema/stamp_predictions_float_bbox.sql`:

```sql
-- Migrate stamp_predictions bbox columns from integer to real.
-- Integer columns silently truncate the normalized-float YOLO outputs
-- to 0 or 1, corrupting every bbox. Existing rows are truncated garbage
-- and must be re-generated from YOLO inference after this migration.

BEGIN;

-- Drop the rows first so the type change has no data to cast.
TRUNCATE TABLE stamp_predictions;

-- Change column types. `real` matches the 6-decimal float output of
-- infer_all.py (see extract_best_prediction()).
ALTER TABLE stamp_predictions
    ALTER COLUMN x TYPE real,
    ALTER COLUMN y TYPE real,
    ALTER COLUMN w TYPE real,
    ALTER COLUMN h TYPE real;

COMMIT;
```

- [ ] **Step 2: Apply the migration**

Run:

```bash
cd /home/will/photo_project
psql "postgresql://dedup:dedup_local_dev@localhost:5432/dedup" \
  -f schema/stamp_predictions_float_bbox.sql
```

If `psql` is unavailable on the path, use this Python one-liner instead:

```bash
source .venv/bin/activate && python -c "
import psycopg
sql = open('schema/stamp_predictions_float_bbox.sql').read()
with psycopg.connect('postgresql://dedup:dedup_local_dev@localhost:5432/dedup', autocommit=True) as conn:
    conn.execute(sql)
print('migration applied')
"
```

Expected output: `migration applied` (or no error from psql).

- [ ] **Step 3: Verify the schema change**

Run:

```bash
source .venv/bin/activate && python -c "
import psycopg
with psycopg.connect('postgresql://dedup:dedup_local_dev@localhost:5432/dedup') as conn:
    cols = conn.execute(\"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='stamp_predictions' AND column_name IN ('x','y','w','h') ORDER BY column_name\").fetchall()
    for c in cols: print(c)
    n = conn.execute('SELECT COUNT(*) FROM stamp_predictions').fetchone()[0]
    print(f'rows: {n}')
"
```

Expected output:
```
('h', 'real')
('w', 'real')
('x', 'real')
('y', 'real')
rows: 0
```

- [ ] **Step 4: Re-run YOLO inference**

Run:

```bash
cd /home/will/photo_project
just infer
```

Expected: prints `Running inference on N pending files...` then progress updates. Takes ~30-60 min on CPU. When done prints `Done. N new/updated, N total rows in stamp_predictions`.

- [ ] **Step 5: Verify bboxes are sane floats**

Run:

```bash
source .venv/bin/activate && python -c "
import psycopg
with psycopg.connect('postgresql://dedup:dedup_local_dev@localhost:5432/dedup') as conn:
    rows = conn.execute('SELECT stem, x, y, w, h, confidence FROM stamp_predictions WHERE confidence > 0.85 ORDER BY confidence DESC LIMIT 5').fetchall()
    for r in rows: print(r)
    bad = conn.execute('SELECT COUNT(*) FROM stamp_predictions WHERE x NOT BETWEEN 0 AND 1 OR y NOT BETWEEN 0 AND 1 OR w <= 0 OR h <= 0').fetchone()[0]
    print(f'bad rows: {bad}')
"
```

Expected: `x`, `y`, `w`, `h` are floats in [0, 1]; `w` and `h` strictly positive; `bad rows: 0`.

- [ ] **Step 6: Commit**

```bash
cd /home/will/photo_project
git add schema/stamp_predictions_float_bbox.sql
git commit -m "schema: migrate stamp_predictions bbox columns to real

Integer columns silently truncated YOLO's normalized-float bboxes to 0/1,
making every row unusable for downstream OCR. Truncate + retype to real,
then re-run infer_all.py to repopulate."
```

---

## Task 1: Add `host_label` column to `stamp_ocr`

**Files:**
- Create: `schema/stamp_ocr_host_label.sql`

- [ ] **Step 1: Write the migration SQL**

Create `schema/stamp_ocr_host_label.sql`:

```sql
-- Add host_label to stamp_ocr so multi-host bench runs of the same model
-- can be grouped in reports. Nullable; existing rows (currently none, post
-- data-loss) get NULL.

ALTER TABLE stamp_ocr ADD COLUMN IF NOT EXISTS host_label TEXT;

CREATE INDEX IF NOT EXISTS stamp_ocr_host_label_idx ON stamp_ocr(host_label);
```

- [ ] **Step 2: Apply the migration**

```bash
source .venv/bin/activate && python -c "
import psycopg
sql = open('schema/stamp_ocr_host_label.sql').read()
with psycopg.connect('postgresql://dedup:dedup_local_dev@localhost:5432/dedup', autocommit=True) as conn:
    conn.execute(sql)
print('migration applied')
"
```

Expected: `migration applied`.

- [ ] **Step 3: Verify**

```bash
source .venv/bin/activate && python -c "
import psycopg
with psycopg.connect('postgresql://dedup:dedup_local_dev@localhost:5432/dedup') as conn:
    cols = conn.execute(\"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='stamp_ocr' ORDER BY ordinal_position\").fetchall()
    for c in cols: print(c)
"
```

Expected: `('host_label', 'text')` appears in the list.

- [ ] **Step 4: Commit**

```bash
git add schema/stamp_ocr_host_label.sql
git commit -m "schema: add host_label column to stamp_ocr

Supports grouping VLM bench results by host (ares-cpu, m2pro, cloud-gpu)
without changing the (stem, model) composite PK."
```

---

## Task 2: Add `state/bench/` to `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Check current `.gitignore`**

Run:

```bash
grep -n "^state" .gitignore
```

Expected: existing lines for `state/*.json`, `state/shards/`, etc.

- [ ] **Step 2: Add bench entries**

Append to `.gitignore`:

```
state/bench/
output/vlm_bench_*
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore VLM bench artifacts"
```

---

## Task 3: Extract shared OCR utilities into `scripts/ocr/ocr_util.py`

**Context:** `normalize_date()` lives in `ocr_gemma.py` and will be reused by `bench_vlm_ocr.py`, `seed_bench_ground_truth.py`, and `report_vlm_bench.py`. `strip_thinking_blocks()` is new. Putting both in a shared module avoids cross-script imports of `ocr_gemma.py`.

**Files:**
- Create: `scripts/ocr/ocr_util.py`
- Modify: `scripts/ocr/ocr_gemma.py` (replace its `normalize_date` with an import)
- Test: `tests/test_ocr_util.py`

- [ ] **Step 1: Write failing tests for `ocr_util.py`**

Create `tests/test_ocr_util.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && pytest tests/test_ocr_util.py -v
```

Expected: all tests FAIL with `ModuleNotFoundError: No module named 'scripts.ocr.ocr_util'` or similar.

- [ ] **Step 3: Create `scripts/ocr/ocr_util.py`**

```python
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
```

Also create `scripts/ocr/__init__.py` if it does not already exist:

```bash
touch scripts/ocr/__init__.py scripts/__init__.py
```

And ensure the `tests/` invocation can import from `scripts/`. Create `tests/conftest.py` if absent:

```python
"""Pytest configuration: make `scripts/` importable as `scripts.*`."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate && pytest tests/test_ocr_util.py -v
```

Expected: all 13 tests PASS.

- [ ] **Step 5: Replace `normalize_date` in `ocr_gemma.py` with an import**

In `scripts/ocr/ocr_gemma.py`:

Remove the entire `def normalize_date(...)` function body (lines ~199-256 in the current file — verify with `grep -n "^def normalize_date" scripts/ocr/ocr_gemma.py`).

Add near the top imports:

```python
from scripts.ocr.ocr_util import normalize_date
```

If that import path fails when `ocr_gemma.py` is run directly (not via pytest), use the same `sys.path` hack the other scripts already use. Check by running:

```bash
source .venv/bin/activate && python scripts/ocr/ocr_gemma.py --limit 0 2>&1 | head -5
```

If it errors on the import, switch to:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.ocr.ocr_util import normalize_date
```

- [ ] **Step 6: Smoke-check `ocr_gemma.py` still imports cleanly**

Run:

```bash
source .venv/bin/activate && python -c "import scripts.ocr.ocr_gemma; print('import ok')"
```

Expected: `import ok`.

- [ ] **Step 7: Commit**

```bash
git add scripts/ocr/ocr_util.py scripts/ocr/ocr_gemma.py scripts/ocr/__init__.py scripts/__init__.py tests/conftest.py tests/test_ocr_util.py
git commit -m "ocr: extract normalize_date + strip_thinking_blocks into ocr_util

Shared module for the upcoming VLM bench scripts. ocr_gemma.py now
imports instead of defining its own copy."
```

---

## Task 4: `build_bench_corpus.py` stratified sampler

**Files:**
- Create: `scripts/ocr/build_bench_corpus.py`
- Test: `tests/test_build_bench_corpus.py`

- [ ] **Step 1: Write failing tests for the sampler**

Create `tests/test_build_bench_corpus.py`:

```python
"""Tests for the bench corpus builder's stratification logic."""

import pytest

from scripts.ocr.build_bench_corpus import (
    CONFIDENCE_BUCKETS,
    assign_bucket,
    stratified_sample,
)


def test_assign_bucket_boundaries():
    assert assign_bucket(0.0) == "[0.0, 0.3)"
    assert assign_bucket(0.29) == "[0.0, 0.3)"
    assert assign_bucket(0.3) == "[0.3, 0.6)"
    assert assign_bucket(0.59) == "[0.3, 0.6)"
    assert assign_bucket(0.6) == "[0.6, 0.85)"
    assert assign_bucket(0.84) == "[0.6, 0.85)"
    assert assign_bucket(0.85) == "[0.85, 1.0]"
    assert assign_bucket(1.0) == "[0.85, 1.0]"


def test_stratified_sample_equal_buckets():
    # 100 items in each bucket, request 50 per bucket, expect 200 total.
    rows = []
    for conf, bucket in [(0.1, "[0.0, 0.3)"), (0.4, "[0.3, 0.6)"), (0.7, "[0.6, 0.85)"), (0.9, "[0.85, 1.0]")]:
        for i in range(100):
            rows.append((f"stem_{bucket}_{i}", conf))
    sampled, skew = stratified_sample(rows, per_bucket=50, seed=42)
    assert len(sampled) == 200
    counts = {b: 0 for b in CONFIDENCE_BUCKETS}
    for stem, conf in sampled:
        counts[assign_bucket(conf)] += 1
    assert all(c == 50 for c in counts.values())
    assert skew == {}


def test_stratified_sample_underfilled_bucket_reports_skew():
    # [0.0, 0.3) has only 20 items; other buckets have 100 each.
    # With per_bucket=50, we take 20 + 50 + 50 + 50 = 170, then redistribute
    # the deficit (30) across over-filled buckets, capping at available rows.
    rows = []
    for i in range(20):
        rows.append((f"low_{i}", 0.1))
    for conf, bucket in [(0.4, "[0.3, 0.6)"), (0.7, "[0.6, 0.85)"), (0.9, "[0.85, 1.0]")]:
        for i in range(100):
            rows.append((f"stem_{bucket}_{i}", conf))
    sampled, skew = stratified_sample(rows, per_bucket=50, seed=42)
    assert len(sampled) == 200
    counts = {b: 0 for b in CONFIDENCE_BUCKETS}
    for stem, conf in sampled:
        counts[assign_bucket(conf)] += 1
    assert counts["[0.0, 0.3)"] == 20
    # Deficit of 30 should be spread across the 3 over-filled buckets.
    # Exact distribution isn't important; total must be 200.
    assert sum(counts.values()) == 200
    assert "[0.0, 0.3)" in skew
    assert skew["[0.0, 0.3)"]["target"] == 50
    assert skew["[0.0, 0.3)"]["actual"] == 20


def test_stratified_sample_deterministic_with_seed():
    rows = [(f"s{i}", 0.4) for i in range(100)]
    a, _ = stratified_sample(rows, per_bucket=10, seed=42)
    b, _ = stratified_sample(rows, per_bucket=10, seed=42)
    assert [s for s, _ in a] == [s for s, _ in b]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && pytest tests/test_build_bench_corpus.py -v
```

Expected: all tests FAIL with `ModuleNotFoundError` on `build_bench_corpus`.

- [ ] **Step 3: Implement `build_bench_corpus.py`**

Create `scripts/ocr/build_bench_corpus.py`:

```python
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pillow>=12.0",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Build the portable VLM bench corpus: 200 stems stratified by YOLO
confidence bucket, pre-cropped to the stamp region.

Writes:
    state/bench/corpus/<stem>.jpg      - full source images (copied)
    state/bench/crops/<stem>.jpg       - padded stamp crops (JPEG)
    state/bench/manifest.json          - stem metadata + stratification info

Usage:
    uv run scripts/ocr/build_bench_corpus.py [--per-bucket 50] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402

SCANMYPHOTOS_DIR = BASE_DIR / "scanmyphotos"
BENCH_DIR = BASE_DIR / "state" / "bench"
CORPUS_DIR = BENCH_DIR / "corpus"
CROPS_DIR = BENCH_DIR / "crops"
MANIFEST_PATH = BENCH_DIR / "manifest.json"

PAD_FACTOR = 0.5
CROP_MAX_SIDE = 512

CONFIDENCE_BUCKETS = ["[0.0, 0.3)", "[0.3, 0.6)", "[0.6, 0.85)", "[0.85, 1.0]"]


def assign_bucket(confidence: float) -> str:
    if confidence < 0.3:
        return "[0.0, 0.3)"
    if confidence < 0.6:
        return "[0.3, 0.6)"
    if confidence < 0.85:
        return "[0.6, 0.85)"
    return "[0.85, 1.0]"


def stratified_sample(
    rows: list[tuple[str, float]],
    per_bucket: int,
    seed: int,
) -> tuple[list[tuple[str, float]], dict]:
    """Return (sampled_rows, skew_report).

    `rows` is a list of (stem, confidence). Target is `per_bucket` stems per
    confidence bucket. If a bucket is underfilled, take all it has and
    redistribute the deficit across over-filled buckets round-robin.

    `skew_report` maps bucket name -> {"target": int, "actual": int} for
    every underfilled bucket; empty dict if all buckets hit target.
    """
    rng = random.Random(seed)
    by_bucket: dict[str, list[tuple[str, float]]] = {b: [] for b in CONFIDENCE_BUCKETS}
    for stem, conf in rows:
        by_bucket[assign_bucket(conf)].append((stem, conf))
    for b in CONFIDENCE_BUCKETS:
        rng.shuffle(by_bucket[b])

    target = per_bucket
    picked: dict[str, list[tuple[str, float]]] = {b: [] for b in CONFIDENCE_BUCKETS}
    skew: dict[str, dict] = {}
    deficit = 0
    for b in CONFIDENCE_BUCKETS:
        available = by_bucket[b]
        if len(available) >= target:
            picked[b] = available[:target]
        else:
            picked[b] = available[:]
            deficit += target - len(available)
            skew[b] = {"target": target, "actual": len(available)}

    # Redistribute deficit round-robin across over-filled buckets.
    over_filled = [b for b in CONFIDENCE_BUCKETS if len(by_bucket[b]) > len(picked[b])]
    idx = 0
    while deficit > 0 and over_filled:
        b = over_filled[idx % len(over_filled)]
        remaining_in_bucket = by_bucket[b][len(picked[b]):]
        if remaining_in_bucket:
            picked[b].append(remaining_in_bucket[0])
            deficit -= 1
        else:
            over_filled.remove(b)
            continue
        idx += 1

    flat = [item for b in CONFIDENCE_BUCKETS for item in picked[b]]
    return flat, skew


def load_predictions_from_db() -> list[tuple[str, float, float, float, float, float]]:
    """Return list of (stem, x, y, w, h, confidence) from stamp_predictions."""
    with get_db() as conn:
        return conn.execute(
            """
            SELECT stem, x, y, w, h, confidence
            FROM stamp_predictions
            WHERE w > 0 AND h > 0
            """
        ).fetchall()


def crop_stamp(img: Image.Image, bbox: dict) -> Image.Image:
    """Pad + crop the stamp region using normalized (cx, cy, w, h)."""
    w_img, h_img = img.size
    cx, cy = bbox["x"] * w_img, bbox["y"] * h_img
    bw, bh = bbox["w"] * w_img, bbox["h"] * h_img
    pad_x = bw * PAD_FACTOR
    pad_y = bh * PAD_FACTOR
    x1 = max(0, int(cx - bw / 2 - pad_x))
    y1 = max(0, int(cy - bh / 2 - pad_y))
    x2 = min(w_img, int(cx + bw / 2 + pad_x))
    y2 = min(h_img, int(cy + bh / 2 + pad_y))
    return img.crop((x1, y1, x2, y2))


def build_corpus(per_bucket: int, seed: int) -> dict:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    preds = load_predictions_from_db()
    pred_rows = [(r[0], float(r[5])) for r in preds]
    bbox_map = {r[0]: {"x": float(r[1]), "y": float(r[2]), "w": float(r[3]), "h": float(r[4])} for r in preds}

    sampled, skew = stratified_sample(pred_rows, per_bucket=per_bucket, seed=seed)
    print(f"Sampled {len(sampled)} stems across {len(CONFIDENCE_BUCKETS)} buckets.")
    if skew:
        print(f"Underfilled buckets: {skew}")

    manifest_stems = []
    confidence_lookup = {stem: conf for stem, conf in sampled}

    for stem, conf in sampled:
        src = SCANMYPHOTOS_DIR / f"{stem}.jpg"
        if not src.exists():
            print(f"  MISSING: {src}")
            continue

        dst_source = CORPUS_DIR / f"{stem}.jpg"
        if not dst_source.exists():
            shutil.copy2(src, dst_source)

        bbox = bbox_map[stem]
        img = Image.open(src).convert("RGB")
        crop = crop_stamp(img, bbox)
        if max(crop.size) > CROP_MAX_SIDE:
            crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE), Image.LANCZOS)
        dst_crop = CROPS_DIR / f"{stem}.jpg"
        crop.save(dst_crop, format="JPEG", quality=90)

        manifest_stems.append(
            {
                "stem": stem,
                "bbox": bbox,
                "bbox_source": "yolo",
                "yolo_confidence": conf,
                "confidence_bucket": assign_bucket(conf),
                "crop_path": str(dst_crop.relative_to(BASE_DIR)),
                "source_path": str(dst_source.relative_to(BASE_DIR)),
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "per_bucket_target": per_bucket,
        "stratification": {"confidence_buckets": CONFIDENCE_BUCKETS},
        "skew": skew,
        "ground_truth_frozen": False,
        "stems": manifest_stems,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {MANIFEST_PATH}")
    print(f"Corpus:   {CORPUS_DIR}")
    print(f"Crops:    {CROPS_DIR}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build VLM bench corpus")
    parser.add_argument("--per-bucket", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_corpus(per_bucket=args.per_bucket, seed=args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit tests**

```bash
source .venv/bin/activate && pytest tests/test_build_bench_corpus.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run against real data**

```bash
cd /home/will/photo_project
uv run scripts/ocr/build_bench_corpus.py
```

Expected output ends with something like:
```
Sampled 200 stems across 4 buckets.
Wrote manifest: .../state/bench/manifest.json
Corpus:   .../state/bench/corpus
Crops:    .../state/bench/crops
```

- [ ] **Step 6: Spot-check the outputs**

```bash
source .venv/bin/activate && python -c "
import json
from pathlib import Path
m = json.loads(Path('state/bench/manifest.json').read_text())
print(f\"stems: {len(m['stems'])}\")
print(f\"skew: {m['skew']}\")
from collections import Counter
c = Counter(s['confidence_bucket'] for s in m['stems'])
for b,n in c.items(): print(f'  {b}: {n}')
print()
print('corpus files:', len(list(Path('state/bench/corpus').glob('*.jpg'))))
print('crop files:  ', len(list(Path('state/bench/crops').glob('*.jpg'))))
"
```

Expected: `stems: 200`; bucket counts sum to 200; both directories have 200 JPGs (minus any missing source files, logged above).

Open one crop visually to confirm it contains a legible date stamp:

```bash
xdg-open state/bench/crops/$(ls state/bench/crops | head -1) 2>/dev/null || echo "open manually to verify"
```

- [ ] **Step 7: Commit**

```bash
git add scripts/ocr/build_bench_corpus.py tests/test_build_bench_corpus.py
git commit -m "bench: add build_bench_corpus.py — stratified 200-stem sampler + pre-cropper

Reads stamp_predictions, stratifies by YOLO confidence bucket
(4 buckets x 50 stems), copies source JPGs to state/bench/corpus/ and
writes padded stamp crops to state/bench/crops/. Deterministic with --seed."
```

---

## Task 5: `seed_bench_ground_truth.py` IO half (no LLM calls)

**Context:** The LLM half runs as Claude Code Task-tool subagents invoked by a human operator or orchestrator session (Task 6). This task only covers the deterministic IO: dispatch manifest creation, shard-result merging, idempotency, and the review HTML.

**Files:**
- Create: `scripts/ocr/seed_bench_ground_truth.py`
- Test: `tests/test_seed_bench_ground_truth.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_seed_bench_ground_truth.py`:

```python
"""Tests for the Sonnet ground-truth seeder IO."""

import json
from pathlib import Path

import pytest

from scripts.ocr.seed_bench_ground_truth import (
    build_shard_manifests,
    parse_subagent_output,
    select_pending_stems,
)


def test_select_pending_stems_skips_rows_already_in_db(tmp_path, monkeypatch):
    """If a stem already has a model='sonnet' row, it should be skipped."""

    manifest = {
        "stems": [
            {"stem": "s1", "crop_path": "crops/s1.jpg"},
            {"stem": "s2", "crop_path": "crops/s2.jpg"},
            {"stem": "s3", "crop_path": "crops/s3.jpg"},
        ]
    }

    existing_in_db = {"s2"}

    def fake_load_existing():
        return existing_in_db

    monkeypatch.setattr(
        "scripts.ocr.seed_bench_ground_truth.load_existing_sonnet_stems",
        fake_load_existing,
    )

    pending = select_pending_stems(manifest, force=False)
    assert [s["stem"] for s in pending] == ["s1", "s3"]


def test_select_pending_stems_force_returns_all(monkeypatch):
    manifest = {
        "stems": [{"stem": "s1"}, {"stem": "s2"}, {"stem": "s3"}]
    }
    monkeypatch.setattr(
        "scripts.ocr.seed_bench_ground_truth.load_existing_sonnet_stems",
        lambda: {"s1", "s2", "s3"},
    )
    pending = select_pending_stems(manifest, force=True)
    assert [s["stem"] for s in pending] == ["s1", "s2", "s3"]


def test_build_shard_manifests_splits_into_waves(tmp_path):
    pending = [{"stem": f"s{i}", "crop_path": f"crops/s{i}.jpg"} for i in range(17)]
    shard_dir = tmp_path / "shards"
    shards = build_shard_manifests(pending, wave_size=5, out_dir=shard_dir)
    assert len(shards) == 4  # 5+5+5+2
    assert shards[0]["size"] == 5
    assert shards[-1]["size"] == 2
    # Each manifest file exists and is readable JSON.
    for s in shards:
        p = Path(s["path"])
        assert p.exists()
        data = json.loads(p.read_text())
        assert "stems" in data
        assert len(data["stems"]) == s["size"]


def test_parse_subagent_output_takes_last_nonempty_line():
    raw = "I see the stamp.\nIt reads:\n10 3'99\n"
    assert parse_subagent_output(raw) == "10 3'99"


def test_parse_subagent_output_handles_empty():
    assert parse_subagent_output("") == ""
    assert parse_subagent_output("\n\n  \n") == ""


def test_parse_subagent_output_strips_thinking():
    raw = "<think>some reasoning</think>\n\n10 3'99"
    assert parse_subagent_output(raw) == "10 3'99"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && pytest tests/test_seed_bench_ground_truth.py -v
```

Expected: all 6 tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scripts/ocr/seed_bench_ground_truth.py`**

```python
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pillow>=12.0",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Orchestrator IO for regenerating Sonnet ground truth via Claude Code
subagents.

The LLM half is dispatched by a Claude Code session (see the "Operator
workflow" section in the plan). This script:

  1. Picks pending stems from state/bench/manifest.json (stems without a
     model='sonnet' row in stamp_ocr).
  2. Partitions them into wave-sized shard manifests under
     state/bench/shards_sonnet/.
  3. Merges a shard result file back into stamp_ocr.
  4. Renders state/bench/ground_truth_review.html for human spot-check.
  5. Flips manifest.ground_truth_frozen=true after review.

Subcommands:
    plan [--wave-size 5] [--force]   Produce shard manifests for pending stems
    merge <shard-result.json>        Insert shard result rows into stamp_ocr
    review-html                      Regenerate ground_truth_review.html
    freeze                           Set manifest.ground_truth_frozen=true
    status                           Print counts by state
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402
from scripts.ocr.ocr_util import extract_final_answer, normalize_date  # noqa: E402

BENCH_DIR = BASE_DIR / "state" / "bench"
CROPS_DIR = BENCH_DIR / "crops"
MANIFEST_PATH = BENCH_DIR / "manifest.json"
SHARDS_DIR = BENCH_DIR / "shards_sonnet"
REVIEW_HTML = BENCH_DIR / "ground_truth_review.html"

SONNET_MODEL = "sonnet"
SONNET_HOST_LABEL = "claude-code-subagent"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def load_existing_sonnet_stems() -> set[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT stem FROM stamp_ocr WHERE model = %s", (SONNET_MODEL,)
        ).fetchall()
    return {r[0] for r in rows}


def select_pending_stems(manifest: dict, force: bool) -> list[dict]:
    existing = set() if force else load_existing_sonnet_stems()
    return [s for s in manifest["stems"] if s["stem"] not in existing]


def build_shard_manifests(
    pending: list[dict], wave_size: int, out_dir: Path
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for i in range(0, len(pending), wave_size):
        chunk = pending[i : i + wave_size]
        shard_id = f"sonnet_{i // wave_size:04d}"
        shard = {
            "shard_id": shard_id,
            "size": len(chunk),
            "stems": [
                {"stem": s["stem"], "crop_path": s["crop_path"]} for s in chunk
            ],
        }
        path = out_dir / f"{shard_id}.json"
        path.write_text(json.dumps(shard, indent=2))
        shards.append({"shard_id": shard_id, "size": len(chunk), "path": str(path)})
    return shards


def parse_subagent_output(raw: str) -> str:
    """Trim the subagent's last-line output. Handles <think> blocks."""
    return extract_final_answer(raw)


def merge_shard_result(result_path: Path) -> int:
    """Merge a shard result into stamp_ocr.

    Expected JSON:
      {
        "shard_id": "sonnet_0003",
        "results": [
          {"stem": "d1_00000123", "raw_text": "10 3'99"},
          {"stem": "d1_00000124", "raw_text": "<think>...</think>\n5 17'94"},
          ...
        ]
      }
    """
    payload = json.loads(result_path.read_text())
    items = payload.get("results", [])
    rows = []
    for item in items:
        raw = item["raw_text"]
        cleaned = parse_subagent_output(raw)
        parsed = normalize_date(cleaned)
        rows.append(
            (
                item["stem"],
                cleaned,
                parsed,
                "yolo",
                SONNET_MODEL,
                SONNET_HOST_LABEL,
            )
        )

    if not rows:
        return 0

    with get_db() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO stamp_ocr (stem, raw_text, parsed_date, bbox_source, model, host_label)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (stem, model) DO UPDATE SET
                raw_text    = EXCLUDED.raw_text,
                parsed_date = EXCLUDED.parsed_date,
                bbox_source = EXCLUDED.bbox_source,
                host_label  = EXCLUDED.host_label,
                updated_at  = NOW()
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def render_review_html() -> Path:
    """Produce state/bench/ground_truth_review.html for human spot-check."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT stem, raw_text, parsed_date
            FROM stamp_ocr
            WHERE model = %s
            ORDER BY (parsed_date IS NULL) DESC, stem
            """,
            (SONNET_MODEL,),
        ).fetchall()

    rows_html = []
    for stem, raw, parsed in rows:
        crop_rel = Path("crops") / f"{stem}.jpg"
        abs_crop = BENCH_DIR / crop_rel
        if not abs_crop.exists():
            continue
        with abs_crop.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        status = (
            '<span style="color:green">parsed</span>'
            if parsed is not None
            else '<span style="color:red">UNPARSED</span>'
        )
        rows_html.append(
            f"""
<tr>
  <td><img src="data:image/jpeg;base64,{b64}" style="max-width:300px"/></td>
  <td><code>{stem}</code></td>
  <td><code>{raw!r}</code></td>
  <td><code>{parsed}</code></td>
  <td>{status}</td>
</tr>
"""
        )

    html = f"""<!doctype html>
<html><head><title>VLM bench ground-truth review</title>
<style>
body {{ font-family: sans-serif; margin: 1em; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ccc; padding: 6px; vertical-align: top; }}
</style>
</head><body>
<h1>VLM bench Sonnet ground-truth review</h1>
<p>Rows: {len(rows_html)}. Unparsed shown first. Skim and correct obvious errors
in the DB (or mark genuinely empty stamps) before running
<code>seed_bench_ground_truth.py freeze</code>.</p>
<table>
<tr><th>crop</th><th>stem</th><th>raw_text</th><th>parsed_date</th><th>status</th></tr>
{''.join(rows_html)}
</table>
</body></html>
"""
    REVIEW_HTML.write_text(html)
    return REVIEW_HTML


def cmd_plan(args):
    manifest = load_manifest()
    pending = select_pending_stems(manifest, force=args.force)
    if not pending:
        print("No pending stems. Ground truth is already seeded.")
        return
    shards = build_shard_manifests(pending, wave_size=args.wave_size, out_dir=SHARDS_DIR)
    print(f"Wrote {len(shards)} shard manifests to {SHARDS_DIR}")
    for s in shards:
        print(f"  {s['shard_id']}  size={s['size']}  path={s['path']}")
    print(
        "\nNext: in a Claude Code session, dispatch one subagent per shard "
        "using the prompt template in docs/plans/2026-04-20-vlm-ocr-bench.md "
        "(Task 6). Write each subagent's structured output to "
        f"{SHARDS_DIR}/<shard_id>.result.json, then run "
        "`python scripts/ocr/seed_bench_ground_truth.py merge <path>`."
    )


def cmd_merge(args):
    n = merge_shard_result(Path(args.result))
    print(f"Merged {n} rows into stamp_ocr (model={SONNET_MODEL!r}).")


def cmd_review_html(args):
    p = render_review_html()
    print(f"Wrote {p}")


def cmd_freeze(args):
    manifest = load_manifest()
    manifest["ground_truth_frozen"] = True
    save_manifest(manifest)
    print("Ground truth frozen. Bench reports will trust these Sonnet rows.")


def cmd_status(args):
    manifest = load_manifest()
    total = len(manifest["stems"])
    existing = load_existing_sonnet_stems()
    with_date = 0
    with get_db() as conn:
        with_date = conn.execute(
            "SELECT COUNT(*) FROM stamp_ocr WHERE model=%s AND parsed_date IS NOT NULL",
            (SONNET_MODEL,),
        ).fetchone()[0]
    print(f"manifest stems:          {total}")
    print(f"sonnet rows in stamp_ocr: {len(existing)}")
    print(f"  with parsed_date:       {with_date}")
    print(f"ground_truth_frozen:      {manifest.get('ground_truth_frozen', False)}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--wave-size", type=int, default=5)
    p_plan.add_argument("--force", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("result", help="path to shard result JSON")
    p_merge.set_defaults(func=cmd_merge)

    p_rv = sub.add_parser("review-html")
    p_rv.set_defaults(func=cmd_review_html)

    p_fz = sub.add_parser("freeze")
    p_fz.set_defaults(func=cmd_freeze)

    p_st = sub.add_parser("status")
    p_st.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && pytest tests/test_seed_bench_ground_truth.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Dry-run the plan command**

```bash
cd /home/will/photo_project
python scripts/ocr/seed_bench_ground_truth.py plan --wave-size 5
```

Expected: 40 shard files listed under `state/bench/shards_sonnet/`, each containing 5 stems (the last may contain fewer).

- [ ] **Step 6: Smoke-test `merge` with a hand-crafted shard result**

```bash
# Grab one stem from the first shard
STEM=$(python -c "import json; d = json.load(open('state/bench/shards_sonnet/sonnet_0000.json')); print(d['stems'][0]['stem'])")
cat > /tmp/sonnet_test_result.json <<EOF
{
  "shard_id": "sonnet_test",
  "results": [
    {"stem": "$STEM", "raw_text": "10 3'99"}
  ]
}
EOF
python scripts/ocr/seed_bench_ground_truth.py merge /tmp/sonnet_test_result.json
python scripts/ocr/seed_bench_ground_truth.py status
```

Expected: `Merged 1 rows into stamp_ocr (model='sonnet')`; `status` shows 1 sonnet row.

Then clean up the test row so Task 6 starts from scratch:

```bash
source .venv/bin/activate && python -c "
import psycopg
with psycopg.connect('postgresql://dedup:dedup_local_dev@localhost:5432/dedup', autocommit=True) as conn:
    conn.execute(\"DELETE FROM stamp_ocr WHERE model='sonnet'\")
print('test row cleaned')
"
rm /tmp/sonnet_test_result.json
```

Also regenerate shards (since `select_pending_stems` just returned fewer entries after the merge):

```bash
python scripts/ocr/seed_bench_ground_truth.py plan --wave-size 5
```

- [ ] **Step 7: Commit**

```bash
git add scripts/ocr/seed_bench_ground_truth.py tests/test_seed_bench_ground_truth.py
git commit -m "bench: add seed_bench_ground_truth.py — Sonnet IO orchestrator

Partitions pending bench stems into shard manifests for Claude Code
subagent dispatch, merges shard results into stamp_ocr, renders a
review HTML for spot-check, and freezes the manifest once reviewed."
```

---

## Task 6: Operator runbook — dispatch Sonnet subagents

**Context:** This is an operator-driven step, not a script. The plan lives here so whoever runs the bench has the exact prompt and the exact shape of the JSON file the subagent must produce.

**Files:** Only produces data (shard result JSON files). No code changes.

- [ ] **Step 1: Open a Claude Code orchestrator session**

In the photo_project working directory, open a Claude Code session. This session will dispatch subagents; no code changes happen here.

- [ ] **Step 2: Dispatch the first shard**

Prompt template for each subagent (one subagent per shard):

```
You are an OCR subagent. For each stem listed below, Read the crop file
at `state/bench/crops/<stem>.jpg` and output the date-stamp text you see.

Rules:
- The stamp is orange LED digits overlaid on the photo.
- Transcribe EXACTLY what you see, including spaces and apostrophes.
- Example formats: "10 3'99" or "'94 6 22" or "8 24'95"
- If there is no date stamp, output the single word NONE.
- If the stamp is completely unreadable, output the single word UNREADABLE.

Stems for this shard:
  <stem-1>
  <stem-2>
  ...

After reading all crops, write your output as a JSON file to
`state/bench/shards_sonnet/<shard_id>.result.json` with this schema:

{
  "shard_id": "<shard_id>",
  "results": [
    {"stem": "<stem-1>", "raw_text": "<what you read>"},
    {"stem": "<stem-2>", "raw_text": "<what you read>"},
    ...
  ]
}

Use the Write tool to create the file. Do not print the JSON to chat.
```

Dispatch one Task per shard. With 40 shards of 5 stems, expect ~10-20s per stem wall-clock → ~1-1.5 hours with wave-of-5 parallelism.

- [ ] **Step 3: Merge each shard result as it lands**

For each completed shard result:

```bash
python scripts/ocr/seed_bench_ground_truth.py merge state/bench/shards_sonnet/sonnet_0000.result.json
```

- [ ] **Step 4: Verify 200 rows are seeded**

```bash
python scripts/ocr/seed_bench_ground_truth.py status
```

Expected: `sonnet rows in stamp_ocr: 200`.

- [ ] **Step 5: Render the review HTML and spot-check**

```bash
python scripts/ocr/seed_bench_ground_truth.py review-html
xdg-open state/bench/ground_truth_review.html
```

Manually scan every unparsed row first, then spot-check 10-20 parsed rows by opening the referenced crop files. For any wrong `raw_text`, fix it directly in the DB:

```bash
source .venv/bin/activate && python -c "
import psycopg
from scripts.ocr.ocr_util import normalize_date
STEM='d1_XXXXXXXX'
CORRECTED=\"'94 6 22\"
with psycopg.connect('postgresql://dedup:dedup_local_dev@localhost:5432/dedup', autocommit=True) as conn:
    conn.execute('UPDATE stamp_ocr SET raw_text=%s, parsed_date=%s WHERE stem=%s AND model=%s',
                 (CORRECTED, normalize_date(CORRECTED), STEM, 'sonnet'))
print('corrected')
"
```

- [ ] **Step 6: Freeze ground truth**

Once the review HTML looks correct:

```bash
python scripts/ocr/seed_bench_ground_truth.py freeze
```

Expected: `Ground truth frozen. Bench reports will trust these Sonnet rows.`

- [ ] **Step 7: Commit the shard result files (optional but recommended)**

Shards are small JSON. Checking them in gives a reproducible record of the ground-truth run without the corpus JPGs:

```bash
# shards_sonnet is covered by state/bench/ in .gitignore; override with -f.
git add -f state/bench/shards_sonnet/*.json
git commit -m "bench: Sonnet ground-truth subagent results (200 stems)"
```

(If you prefer to keep these out of git entirely, skip this step.)

---

## Task 7: `bench_vlm_ocr.py` Ollama runner

**Files:**
- Create: `scripts/ocr/bench_vlm_ocr.py`
- Test: `tests/test_bench_vlm_ocr.py`

- [ ] **Step 1: Write failing tests for the non-network logic**

Create `tests/test_bench_vlm_ocr.py`:

```python
"""Tests for bench_vlm_ocr.py helper functions.

Does NOT call Ollama. The smoke test (Step 6) covers the network path.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.ocr.bench_vlm_ocr import (
    encode_crop,
    load_manifest_stems,
    make_model_key,
    write_jsonl_row,
)


def test_make_model_key_joins_with_at_sign():
    assert make_model_key("kimi-vl:latest", "ares-cpu") == "kimi-vl:latest@ares-cpu"


def test_make_model_key_strips_whitespace():
    assert make_model_key("  qwen2.5-vl:7b ", " m2pro ") == "qwen2.5-vl:7b@m2pro"


def test_encode_crop_produces_base64(tmp_path):
    from PIL import Image

    p = tmp_path / "t.jpg"
    Image.new("RGB", (100, 100), "red").save(p, format="JPEG")
    b64 = encode_crop(p, max_side=512)
    # Round-trip decode to confirm valid image bytes.
    import base64
    import io
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.size == (100, 100)


def test_encode_crop_resizes_when_over_max(tmp_path):
    from PIL import Image
    import base64
    import io

    p = tmp_path / "big.jpg"
    Image.new("RGB", (2000, 1000), "red").save(p, format="JPEG")
    b64 = encode_crop(p, max_side=512)
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert max(img.size) == 512


def test_load_manifest_stems_returns_subset_when_limit(tmp_path):
    m = {
        "stems": [
            {"stem": "s1", "crop_path": "crops/s1.jpg"},
            {"stem": "s2", "crop_path": "crops/s2.jpg"},
            {"stem": "s3", "crop_path": "crops/s3.jpg"},
        ]
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    assert len(load_manifest_stems(p, limit=2)) == 2
    assert len(load_manifest_stems(p, limit=None)) == 3


def test_write_jsonl_row_appends_single_line(tmp_path):
    out = tmp_path / "results.jsonl"
    row1 = {"stem": "a", "raw_text": "1 2'99", "parsed_date": "1999-01-02"}
    row2 = {"stem": "b", "raw_text": "TIMEOUT", "parsed_date": None}
    write_jsonl_row(out, row1)
    write_jsonl_row(out, row2)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == row1
    assert json.loads(lines[1]) == row2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && pytest tests/test_bench_vlm_ocr.py -v
```

Expected: all 6 tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scripts/ocr/bench_vlm_ocr.py`**

```python
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pillow>=12.0",
#     "psycopg[binary]>=3.1.0",
#     "requests>=2.31",
# ]
# ///
"""Portable Ollama VLM runner for the date-stamp bench.

Reads pre-cropped images from a manifest, calls Ollama's /api/chat with a
fixed prompt, and writes per-image rows to either Postgres (stamp_ocr)
or a JSONL shard file.

Usage:
    bench_vlm_ocr.py --model <ollama-tag>
                     --host <ollama-url>
                     --manifest state/bench/manifest.json
                     --crops-dir state/bench/crops
                     --output postgres://... | jsonl://state/bench/results/<tag>.jsonl
                     --host-label <ares-cpu|m2pro|cloud-gpu>
                     [--limit N] [--resume]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402
from scripts.ocr.ocr_util import extract_final_answer, normalize_date  # noqa: E402

PROMPT = """This is a cropped photo showing a camera date stamp -- orange LED digits.
Transcribe EXACTLY what you see, preserving spaces and apostrophes.
Example formats: "10 3 '99" or "'94 6 22" or "8 24'95"
Output ONLY the stamp text, nothing else."""

REQUEST_TIMEOUT = 180
NUM_PREDICT = 2048
CROP_MAX_SIDE = 512


# -------------------- small helpers --------------------


def make_model_key(model: str, host_label: str) -> str:
    return f"{model.strip()}@{host_label.strip()}"


def encode_crop(path: Path, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def load_manifest_stems(path: Path, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text())
    stems = data["stems"]
    return stems[:limit] if limit else stems


def write_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


# -------------------- Ollama --------------------


def wait_for_model(host: str, model: str, max_wait: int = 600) -> None:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{host}/api/tags", timeout=5)
            if r.status_code == 200:
                names = [m["name"] for m in r.json().get("models", [])]
                if any(model == n or model.split(":")[0] == n.split(":")[0] for n in names):
                    return
                # Pull and retry.
                print(f"Pulling {model} ...")
                pull = requests.post(
                    f"{host}/api/pull", json={"name": model, "stream": False}, timeout=3600
                )
                if pull.status_code != 200:
                    raise RuntimeError(f"Pull failed: {pull.status_code} {pull.text[:200]}")
                return
        except requests.ConnectionError:
            time.sleep(2)
    raise RuntimeError(f"Ollama not reachable at {host} within {max_wait}s")


def ocr_one(host: str, model: str, b64: str) -> dict:
    t0 = time.time()
    try:
        r = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": NUM_PREDICT},
                "think": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        return {
            "raw_text": "TIMEOUT",
            "elapsed_s": REQUEST_TIMEOUT,
            "eval_count": 0,
            "prompt_eval_count": 0,
            "error": "timeout",
        }

    elapsed = round(time.time() - t0, 2)
    if r.status_code == 500 and "out of memory" in r.text.lower():
        return {
            "raw_text": "OOM_ERROR",
            "elapsed_s": elapsed,
            "eval_count": 0,
            "prompt_eval_count": 0,
            "error": "oom",
        }
    if r.status_code != 200:
        return {
            "raw_text": f"HTTP_{r.status_code}",
            "elapsed_s": elapsed,
            "eval_count": 0,
            "prompt_eval_count": 0,
            "error": f"http_{r.status_code}",
        }
    data = r.json()
    raw = data.get("message", {}).get("content", "")
    return {
        "raw_text": raw,
        "elapsed_s": elapsed,
        "eval_count": int(data.get("eval_count") or 0),
        "prompt_eval_count": int(data.get("prompt_eval_count") or 0),
        "error": None,
    }


# -------------------- output writers --------------------


def upsert_pg_row(conn, row: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stamp_ocr (stem, raw_text, parsed_date, bbox_source, model, host_label)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (stem, model) DO UPDATE SET
                raw_text    = EXCLUDED.raw_text,
                parsed_date = EXCLUDED.parsed_date,
                bbox_source = EXCLUDED.bbox_source,
                host_label  = EXCLUDED.host_label,
                updated_at  = NOW()
            """,
            (
                row["stem"],
                row["raw_text"],
                row["parsed_date"],
                row.get("bbox_source", "yolo"),
                row["model_key"],
                row["host_label"],
            ),
        )


def load_resume_set_pg(conn, model_key: str) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT stem FROM stamp_ocr WHERE model = %s", (model_key,)
        ).fetchall()
    }


def load_resume_set_jsonl(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["stem"])
    return done


# -------------------- main loop --------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Ollama model tag, e.g. kimi-vl:latest")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--manifest", default="state/bench/manifest.json")
    p.add_argument("--crops-dir", default="state/bench/crops")
    p.add_argument(
        "--output",
        required=True,
        help="postgres://... or jsonl://path/to/file.jsonl",
    )
    p.add_argument("--host-label", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    model_key = make_model_key(args.model, args.host_label)
    manifest_path = Path(args.manifest).resolve()
    crops_dir = Path(args.crops_dir).resolve()
    stems = load_manifest_stems(manifest_path, args.limit)
    print(f"Model key: {model_key}")
    print(f"Stems:     {len(stems)}")

    wait_for_model(args.host, args.model)

    # Output setup
    output_scheme, _, output_loc = args.output.partition("://")
    if output_scheme == "postgres":
        conn = get_db()
        done = load_resume_set_pg(conn, model_key) if args.resume else set()
    elif output_scheme == "jsonl":
        jsonl_path = Path(output_loc).resolve()
        conn = None
        done = load_resume_set_jsonl(jsonl_path) if args.resume else set()
    else:
        raise SystemExit(f"Unknown --output scheme: {output_scheme}")

    if args.resume:
        before = len(stems)
        stems = [s for s in stems if s["stem"] not in done]
        print(f"Resume: skipping {before - len(stems)} stems already present.")

    processed = 0
    total_elapsed = 0.0
    t_start = time.time()

    for i, s in enumerate(stems):
        crop_path = crops_dir / f"{s['stem']}.jpg"
        if not crop_path.exists():
            print(f"  MISSING crop: {crop_path}")
            continue
        b64 = encode_crop(crop_path, CROP_MAX_SIDE)
        res = ocr_one(args.host, args.model, b64)
        cleaned_text = extract_final_answer(res["raw_text"])
        parsed = normalize_date(cleaned_text)
        row = {
            "stem": s["stem"],
            "model_key": model_key,
            "host_label": args.host_label,
            "raw_text": cleaned_text,
            "parsed_date": parsed,
            "bbox_source": s.get("bbox_source", "yolo"),
            "elapsed_s": res["elapsed_s"],
            "eval_count": res["eval_count"],
            "prompt_eval_count": res["prompt_eval_count"],
            "error": res["error"],
            "ran_at": datetime.now(timezone.utc).isoformat(),
        }
        if conn is not None:
            upsert_pg_row(conn, row)
            if (i + 1) % 10 == 0:
                conn.commit()
        else:
            write_jsonl_row(jsonl_path, row)
        processed += 1
        total_elapsed += res["elapsed_s"]
        date_str = f" -> {parsed}" if parsed else " [unparsed]"
        print(f"  [{i+1}/{len(stems)}] {s['stem']}: {cleaned_text!r}{date_str}  ({res['elapsed_s']:.1f}s)")

    if conn is not None:
        conn.commit()
        conn.close()

    wall = time.time() - t_start
    print()
    print(f"Processed:  {processed}")
    if processed:
        print(f"avg/img:    {total_elapsed/processed:.1f}s")
    print(f"Wall time:  {wall/3600:.2f}h")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit tests**

```bash
source .venv/bin/activate && pytest tests/test_bench_vlm_ocr.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Smoke test with a real Ollama model (requires Ollama running locally)**

First check whether Ollama is reachable:

```bash
curl -s http://localhost:11434/api/tags | head -c 200
```

If no Ollama is running locally, skip to Step 6. If it is:

```bash
cd /home/will/photo_project
mkdir -p state/bench/results
uv run scripts/ocr/bench_vlm_ocr.py \
    --model gemma3 \
    --host http://localhost:11434 \
    --manifest state/bench/manifest.json \
    --crops-dir state/bench/crops \
    --output jsonl://state/bench/results/smoke.jsonl \
    --host-label smoke \
    --limit 3
```

Expected: 3 rows printed with a date parse result or `[unparsed]`, each taking a few seconds to a minute depending on model and host. No crashes.

Then:

```bash
wc -l state/bench/results/smoke.jsonl
rm state/bench/results/smoke.jsonl
```

Expected: `3 state/bench/results/smoke.jsonl`.

- [ ] **Step 6: Commit**

```bash
git add scripts/ocr/bench_vlm_ocr.py tests/test_bench_vlm_ocr.py
git commit -m "bench: add bench_vlm_ocr.py — portable Ollama runner

Reads pre-cropped bench crops, calls /api/chat, strips <think> blocks,
normalizes to YYYY-MM-DD. Writes to Postgres (stamp_ocr) or JSONL.
Models rows under '<tag>@<host-label>' so multi-host runs don't
collide on the (stem, model) PK. Handles timeouts and OOM without
retries."
```

---

## Task 8: `report_vlm_bench.py` metrics + markdown + Pareto PNG

**Files:**
- Create: `scripts/ocr/report_vlm_bench.py`
- Test: `tests/test_report_vlm_bench.py`

- [ ] **Step 1: Write failing tests for the metric computation**

Create `tests/test_report_vlm_bench.py`:

```python
"""Tests for report_vlm_bench.py metric computation."""

import pytest

from scripts.ocr.report_vlm_bench import (
    compute_metrics,
    parse_model_key,
    pareto_frontier,
)


def test_parse_model_key_splits_host():
    assert parse_model_key("kimi-vl:latest@ares-cpu") == ("kimi-vl:latest", "ares-cpu")


def test_parse_model_key_no_host_returns_none():
    assert parse_model_key("sonnet") == ("sonnet", None)


def test_compute_metrics_basic():
    sonnet = {"s1": "1999-10-03", "s2": "1994-06-22", "s3": "2001-05-17"}
    candidate = [
        {"stem": "s1", "raw_text": "10 3'99", "parsed_date": "1999-10-03", "elapsed_s": 5.0},
        {"stem": "s2", "raw_text": "'94 6 22", "parsed_date": "1994-06-22", "elapsed_s": 7.0},
        {"stem": "s3", "raw_text": "garbled", "parsed_date": None, "elapsed_s": 6.0},
    ]
    m = compute_metrics(sonnet_by_stem=sonnet, rows=candidate)
    assert m["total"] == 3
    assert m["agree_pct"] == pytest.approx(66.67, abs=0.1)
    assert m["unparsed_pct"] == pytest.approx(33.33, abs=0.1)
    assert m["high_conf_wrong_pct"] == 0.0
    assert m["median_s"] == 6.0


def test_compute_metrics_flags_high_conf_wrong():
    sonnet = {"s1": "1999-10-03"}
    candidate = [
        # Model parsed a date but it disagrees with Sonnet.
        {"stem": "s1", "raw_text": "10 3'98", "parsed_date": "1998-10-03", "elapsed_s": 5.0},
    ]
    m = compute_metrics(sonnet_by_stem=sonnet, rows=candidate)
    assert m["agree_pct"] == 0.0
    assert m["high_conf_wrong_pct"] == 100.0


def test_compute_metrics_timeout_and_oom():
    sonnet = {"s1": "1999-10-03", "s2": "1994-06-22"}
    candidate = [
        {"stem": "s1", "raw_text": "TIMEOUT", "parsed_date": None, "elapsed_s": 180.0},
        {"stem": "s2", "raw_text": "OOM_ERROR", "parsed_date": None, "elapsed_s": 2.0},
    ]
    m = compute_metrics(sonnet_by_stem=sonnet, rows=candidate)
    assert m["timeout_pct"] == 50.0
    assert m["oom_pct"] == 50.0


def test_pareto_frontier_picks_dominant_points():
    # (agree, imgs_per_sec) tuples with model labels.
    points = [
        {"model_key": "a", "agree_pct": 90.0, "imgs_per_sec": 1.0},
        {"model_key": "b", "agree_pct": 85.0, "imgs_per_sec": 5.0},
        {"model_key": "c", "agree_pct": 80.0, "imgs_per_sec": 3.0},  # dominated by a and b
        {"model_key": "d", "agree_pct": 92.0, "imgs_per_sec": 0.5},
    ]
    frontier = pareto_frontier(points)
    keys = {p["model_key"] for p in frontier}
    assert keys == {"a", "b", "d"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && pytest tests/test_report_vlm_bench.py -v
```

Expected: all 5 tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scripts/ocr/report_vlm_bench.py`**

```python
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "psycopg[binary]>=3.1.0",
#     "matplotlib>=3.8",
# ]
# ///
"""Aggregate VLM bench results and emit a Pareto-frontier report.

Reads rows from stamp_ocr (model='sonnet' for ground truth, other models
for candidates). Optionally ingests orphan JSONL shards under
state/bench/results/ into stamp_ocr first.

Outputs:
    output/vlm_bench_report.md
    output/vlm_bench_pareto.png

Usage:
    uv run scripts/ocr/report_vlm_bench.py \\
        [--manifest state/bench/manifest.json] \\
        [--ingest-jsonl state/bench/results]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402

OUTPUT_DIR = BASE_DIR / "output"
REPORT_MD = OUTPUT_DIR / "vlm_bench_report.md"
PARETO_PNG = OUTPUT_DIR / "vlm_bench_pareto.png"


# Rough model size (billions of params) lookup for Pareto bubble sizing.
# Keys matched by prefix of the bare model tag (before @host-label).
MODEL_SIZE_B = {
    "kimi-vl": 16,  # MoE total; active ~3
    "qwen2.5-vl:7b": 7,
    "qwen2.5-vl:3b": 3,
    "minicpm-v": 8,
    "gemma3": 12,  # default tag; varies
    "llama3.2-vision:11b": 11,
    "llama3.2-vision:90b": 90,
    "internvl3:8b": 8,
    "sonnet": 0,  # hide from Pareto; ground truth
}


def parse_model_key(key: str) -> tuple[str, str | None]:
    if "@" in key:
        m, h = key.split("@", 1)
        return m, h
    return key, None


def lookup_size(model_tag: str) -> int:
    for k, v in MODEL_SIZE_B.items():
        if model_tag.startswith(k):
            return v
    return 0


def compute_metrics(sonnet_by_stem: dict[str, str], rows: list[dict]) -> dict:
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "agree_pct": 0.0,
            "unparsed_pct": 0.0,
            "timeout_pct": 0.0,
            "oom_pct": 0.0,
            "high_conf_wrong_pct": 0.0,
            "median_s": 0.0,
            "p95_s": 0.0,
            "imgs_per_sec": 0.0,
        }
    agree = unparsed = timeout = oom = wrong = 0
    latencies = []
    for r in rows:
        raw = (r.get("raw_text") or "").strip()
        parsed = r.get("parsed_date")
        truth = sonnet_by_stem.get(r["stem"])
        if raw == "TIMEOUT":
            timeout += 1
        elif raw == "OOM_ERROR":
            oom += 1
        elif parsed is None:
            unparsed += 1
        elif parsed == truth:
            agree += 1
        else:
            wrong += 1
        if isinstance(r.get("elapsed_s"), (int, float)):
            latencies.append(float(r["elapsed_s"]))
    median_s = statistics.median(latencies) if latencies else 0.0
    p95_s = 0.0
    if latencies:
        sorted_l = sorted(latencies)
        p95_s = sorted_l[int(0.95 * (len(sorted_l) - 1))]
    return {
        "total": total,
        "agree_pct": 100.0 * agree / total,
        "unparsed_pct": 100.0 * unparsed / total,
        "timeout_pct": 100.0 * timeout / total,
        "oom_pct": 100.0 * oom / total,
        "high_conf_wrong_pct": 100.0 * wrong / total,
        "median_s": median_s,
        "p95_s": p95_s,
        "imgs_per_sec": (1.0 / median_s) if median_s > 0 else 0.0,
    }


def pareto_frontier(points: list[dict]) -> list[dict]:
    """Keep points where no other point has higher agree_pct AND higher imgs_per_sec."""
    out = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q["agree_pct"] >= p["agree_pct"] and q["imgs_per_sec"] >= p["imgs_per_sec"] and (
                q["agree_pct"] > p["agree_pct"] or q["imgs_per_sec"] > p["imgs_per_sec"]
            ):
                dominated = True
                break
        if not dominated:
            out.append(p)
    return out


def ingest_jsonl_dir(directory: Path) -> int:
    """Import orphan JSONL shards into stamp_ocr. Returns rows inserted."""
    if not directory.exists():
        return 0
    rows = []
    for p in directory.glob("*.jsonl"):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(
                (
                    r["stem"],
                    r["raw_text"],
                    r["parsed_date"],
                    r.get("bbox_source", "yolo"),
                    r["model_key"],
                    r["host_label"],
                )
            )
    if not rows:
        return 0
    with get_db() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO stamp_ocr (stem, raw_text, parsed_date, bbox_source, model, host_label)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (stem, model) DO UPDATE SET
                raw_text    = EXCLUDED.raw_text,
                parsed_date = EXCLUDED.parsed_date,
                bbox_source = EXCLUDED.bbox_source,
                host_label  = EXCLUDED.host_label,
                updated_at  = NOW()
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def load_sonnet_truth(stems: list[str]) -> dict[str, str]:
    if not stems:
        return {}
    with get_db() as conn:
        rows = conn.execute(
            "SELECT stem, parsed_date FROM stamp_ocr WHERE model='sonnet' AND stem = ANY(%s)",
            (stems,),
        ).fetchall()
    return {r[0]: r[1].isoformat() if r[1] else None for r in rows}


def load_candidate_rows(stems: list[str]) -> dict[str, list[dict]]:
    """Return {model_key: [row_dicts]} for all non-sonnet models covering `stems`."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT stem, raw_text, parsed_date, model, host_label, updated_at
            FROM stamp_ocr
            WHERE model <> 'sonnet' AND stem = ANY(%s)
            """,
            (stems,),
        ).fetchall()
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r[3], []).append(
            {
                "stem": r[0],
                "raw_text": r[1],
                "parsed_date": r[2].isoformat() if r[2] else None,
                "model_key": r[3],
                "host_label": r[4],
                "elapsed_s": 0.0,  # elapsed is not persisted in stamp_ocr; JSONL has it
            }
        )
    return grouped


def render_markdown(rows_by_key: dict, metrics_by_key: dict, manifest: dict) -> str:
    frozen = manifest.get("ground_truth_frozen", False)
    header_warn = (
        ""
        if frozen
        else "> **⚠ Ground truth NOT frozen.** Metrics below compare against an "
        "unreviewed Sonnet run.\n\n"
    )
    lines = [
        "# VLM OCR Bench Report",
        "",
        f"Generated: {manifest.get('generated_at', '?')}",
        f"Corpus: {len(manifest['stems'])} stems, seed={manifest.get('seed')}",
        "",
        header_warn,
        "| Model (tag@host) | N | Agree% | HiConfWrong% | Unparsed% | Timeout% | OOM% | Median s | P95 s | img/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = sorted(metrics_by_key.items(), key=lambda kv: (-kv[1]["agree_pct"], kv[1]["median_s"]))
    for key, m in rows:
        lines.append(
            "| `{k}` | {n} | {a:.1f} | {w:.1f} | {u:.1f} | {t:.1f} | {o:.1f} | {med:.1f} | {p95:.1f} | {ips:.2f} |".format(
                k=key,
                n=m["total"],
                a=m["agree_pct"],
                w=m["high_conf_wrong_pct"],
                u=m["unparsed_pct"],
                t=m["timeout_pct"],
                o=m["oom_pct"],
                med=m["median_s"],
                p95=m["p95_s"],
                ips=m["imgs_per_sec"],
            )
        )
    return "\n".join(lines) + "\n"


def render_pareto_png(metrics_by_key: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = []
    for key, m in metrics_by_key.items():
        model_tag, host = parse_model_key(key)
        points.append(
            {
                "model_key": key,
                "model_tag": model_tag,
                "host": host or "unknown",
                "agree_pct": m["agree_pct"],
                "imgs_per_sec": m["imgs_per_sec"],
                "size_b": lookup_size(model_tag),
            }
        )
    frontier = pareto_frontier(points)

    hosts = sorted({p["host"] for p in points})
    colors = plt.cm.tab10.colors
    host_color = {h: colors[i % 10] for i, h in enumerate(hosts)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for p in points:
        is_frontier = any(fp["model_key"] == p["model_key"] for fp in frontier)
        ax.scatter(
            p["imgs_per_sec"],
            p["agree_pct"],
            s=60 + 30 * p["size_b"],
            color=host_color[p["host"]],
            edgecolors="black" if is_frontier else "none",
            linewidths=2 if is_frontier else 0,
            alpha=0.8,
            label=f"{p['model_tag']} @ {p['host']}",
        )
        ax.annotate(p["model_tag"], (p["imgs_per_sec"], p["agree_pct"]), fontsize=7)

    ax.set_xlabel("images / second (1 / median latency)")
    ax.set_ylabel("agreement with Sonnet ground truth (%)")
    ax.set_title("VLM bench Pareto frontier")
    ax.grid(alpha=0.3)

    # Deduplicate the legend.
    seen = set()
    handles = []
    labels = []
    for h, lab in zip(*ax.get_legend_handles_labels()):
        if lab in seen:
            continue
        seen.add(lab)
        handles.append(h)
        labels.append(lab)
    ax.legend(handles, labels, fontsize=7, loc="lower right")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PARETO_PNG, dpi=120)
    plt.close(fig)
    return PARETO_PNG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="state/bench/manifest.json")
    p.add_argument("--ingest-jsonl", default=None, help="Import orphan JSONL shards first")
    args = p.parse_args()

    if args.ingest_jsonl:
        n = ingest_jsonl_dir(Path(args.ingest_jsonl))
        print(f"Ingested {n} rows from {args.ingest_jsonl}")

    manifest = json.loads(Path(args.manifest).read_text())
    stems = [s["stem"] for s in manifest["stems"]]
    sonnet = load_sonnet_truth(stems)
    cand_by_key = load_candidate_rows(stems)

    if not cand_by_key:
        print("No non-sonnet rows in stamp_ocr yet. Run a bench first.")
        return

    metrics = {k: compute_metrics(sonnet, v) for k, v in cand_by_key.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render_markdown(cand_by_key, metrics, manifest))
    print(f"Wrote {REPORT_MD}")
    render_pareto_png(metrics)
    print(f"Wrote {PARETO_PNG}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests**

```bash
source .venv/bin/activate && pytest tests/test_report_vlm_bench.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Dry-run (requires at least one model benched end-to-end; safe to run even with empty results)**

```bash
cd /home/will/photo_project
uv run scripts/ocr/report_vlm_bench.py
```

Expected (if no bench runs yet): `No non-sonnet rows in stamp_ocr yet. Run a bench first.`

- [ ] **Step 6: Commit**

```bash
git add scripts/ocr/report_vlm_bench.py tests/test_report_vlm_bench.py
git commit -m "bench: add report_vlm_bench.py — markdown table + Pareto PNG

Joins candidate rows to Sonnet ground truth, computes agree/unparsed/
timeout/OOM/high-confidence-wrong percentages and median/p95 latency,
emits output/vlm_bench_report.md and output/vlm_bench_pareto.png."
```

---

## Task 9: Profile wrappers (`ares-cpu.sh`, `m2pro.sh`, `cloud-gpu.sh`)

**Files:**
- Create: `scripts/ocr/bench_profiles/ares-cpu.sh`
- Create: `scripts/ocr/bench_profiles/m2pro.sh`
- Create: `scripts/ocr/bench_profiles/cloud-gpu.sh`

- [ ] **Step 1: Create `ares-cpu.sh`**

```bash
#!/usr/bin/env bash
# Bench profile: run bench_vlm_ocr.py on ares's Ollama container.
#
# Usage:
#   bench_profiles/ares-cpu.sh <ollama-tag>
#
# Example:
#   bench_profiles/ares-cpu.sh kimi-vl:latest

set -euo pipefail

MODEL="${1:?usage: ares-cpu.sh <ollama-tag>}"
HOST="${OLLAMA_HOST:-http://ares.savannah-mimosa.ts.net:11434}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Force sequential requests so llama3.2-vision:11b + internvl3:8b don't
# collide in RAM.
export OLLAMA_NUM_PARALLEL=1

cd "$REPO_ROOT"
source .venv/bin/activate

uv run scripts/ocr/bench_vlm_ocr.py \
    --model "$MODEL" \
    --host "$HOST" \
    --manifest state/bench/manifest.json \
    --crops-dir state/bench/crops \
    --output "postgres://${DATABASE_URL:-dedup:dedup_local_dev@localhost:5432/dedup}" \
    --host-label ares-cpu \
    --resume
```

Make executable:

```bash
chmod +x scripts/ocr/bench_profiles/ares-cpu.sh
```

- [ ] **Step 2: Create `m2pro.sh`**

```bash
#!/usr/bin/env bash
# Bench profile: run on the user's M2 Pro MacBook with native Ollama.
#
# Usage:
#   bench_profiles/m2pro.sh <ollama-tag>
#
# Results go to local JSONL (no direct tailnet Postgres write); after
# completion, sync the JSONL back to ares and run report_vlm_bench.py
# with --ingest-jsonl to import.

set -euo pipefail

MODEL="${1:?usage: m2pro.sh <ollama-tag>}"
HOST="${OLLAMA_HOST:-http://localhost:11434}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$REPO_ROOT"
source .venv/bin/activate

SAFE_TAG=$(echo "$MODEL" | tr '/:' '__')
OUT="state/bench/results/${SAFE_TAG}_m2pro.jsonl"

uv run scripts/ocr/bench_vlm_ocr.py \
    --model "$MODEL" \
    --host "$HOST" \
    --manifest state/bench/manifest.json \
    --crops-dir state/bench/crops \
    --output "jsonl://$OUT" \
    --host-label m2pro \
    --resume

echo
echo "Done. JSONL at: $OUT"
echo "Sync back to ares with: rsync $OUT ares:photo_project/state/bench/results/"
```

Make executable:

```bash
chmod +x scripts/ocr/bench_profiles/m2pro.sh
```

- [ ] **Step 3: Create `cloud-gpu.sh`**

```bash
#!/usr/bin/env bash
# Bench profile: run on an ephemeral AWS GPU spot joined to the tailnet.
#
# Usage (on the spot instance, after tailscale up + ollama installed):
#   bench_profiles/cloud-gpu.sh <ollama-tag>
#
# The spot instance writes directly to ares Postgres via tailnet.

set -euo pipefail

MODEL="${1:?usage: cloud-gpu.sh <ollama-tag>}"
HOST="${OLLAMA_HOST:-http://localhost:11434}"
PG_DSN="${DATABASE_URL:-postgresql://dedup:dedup_local_dev@ares.savannah-mimosa.ts.net:5432/dedup}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$REPO_ROOT"
source .venv/bin/activate

HOST_LABEL="${HOST_LABEL:-cloud-gpu-$(uname -m)}"

uv run scripts/ocr/bench_vlm_ocr.py \
    --model "$MODEL" \
    --host "$HOST" \
    --manifest state/bench/manifest.json \
    --crops-dir state/bench/crops \
    --output "postgres://${PG_DSN#postgresql://}" \
    --host-label "$HOST_LABEL" \
    --resume
```

Make executable:

```bash
chmod +x scripts/ocr/bench_profiles/cloud-gpu.sh
```

- [ ] **Step 4: Sanity-check the wrappers**

```bash
cd /home/will/photo_project
bash -n scripts/ocr/bench_profiles/ares-cpu.sh
bash -n scripts/ocr/bench_profiles/m2pro.sh
bash -n scripts/ocr/bench_profiles/cloud-gpu.sh
echo "OK"
```

Expected: `OK` (no syntax errors).

- [ ] **Step 5: Commit**

```bash
git add scripts/ocr/bench_profiles/
git commit -m "bench: add per-host profile wrappers (ares-cpu, m2pro, cloud-gpu)"
```

---

## Task 10: Justfile recipes

**Files:**
- Modify: `justfile`

- [ ] **Step 1: Check existing recipes**

```bash
grep -n "^[a-z]" justfile | head
```

- [ ] **Step 2: Append bench recipes to `justfile`**

Append:

```just
# ---------- VLM OCR bench ----------

# Build the 200-stem bench corpus (run once)
bench-build:
    uv run scripts/ocr/build_bench_corpus.py

# Partition pending Sonnet ground-truth stems into shard manifests
bench-seed-plan:
    uv run scripts/ocr/seed_bench_ground_truth.py plan

# Render the ground-truth review HTML
bench-seed-review:
    uv run scripts/ocr/seed_bench_ground_truth.py review-html
    xdg-open state/bench/ground_truth_review.html 2>/dev/null || true

# Freeze ground truth once review is complete
bench-seed-freeze:
    uv run scripts/ocr/seed_bench_ground_truth.py freeze

# Run a single Ollama model via the given profile, e.g. just bench-run ares-cpu kimi-vl:latest
bench-run profile model:
    scripts/ocr/bench_profiles/{{profile}}.sh {{model}}

# Build the markdown + PNG report
bench-report:
    uv run scripts/ocr/report_vlm_bench.py --ingest-jsonl state/bench/results
```

- [ ] **Step 3: Verify `just --list` picks them up**

```bash
cd /home/will/photo_project
just --list | grep bench
```

Expected: all six `bench-*` recipes listed.

- [ ] **Step 4: Commit**

```bash
git add justfile
git commit -m "bench: justfile recipes for corpus build, ground-truth, runs, and report"
```

---

## Task 11: End-to-end dry-run on willbook

**Context:** Not a code change — a verification pass that catches integration bugs before committing to a 10h bench.

**Files:** None. Produces data only.

- [ ] **Step 1: Confirm Ollama is reachable and pull gemma3**

```bash
curl -s http://localhost:11434/api/tags | head -c 400
```

If no Ollama is running locally, start one (`ollama serve` in a separate terminal) or point at ares via `export OLLAMA_HOST=http://ares.savannah-mimosa.ts.net:11434`.

Pull gemma3 (smallest model in the slate — good for the dry-run):

```bash
curl -s -X POST $OLLAMA_HOST/api/pull -d '{"name":"gemma3","stream":false}' | head -c 200
```

- [ ] **Step 2: Run gemma3 on 10 stems**

```bash
cd /home/will/photo_project
uv run scripts/ocr/bench_vlm_ocr.py \
    --model gemma3 \
    --host "${OLLAMA_HOST:-http://localhost:11434}" \
    --manifest state/bench/manifest.json \
    --crops-dir state/bench/crops \
    --output "jsonl://state/bench/results/gemma3_dryrun.jsonl" \
    --host-label dryrun \
    --limit 10
```

Expected: 10 rows printed with date parse results, latencies sane (1-40s depending on host).

- [ ] **Step 3: Ingest JSONL and generate a report**

```bash
uv run scripts/ocr/report_vlm_bench.py --ingest-jsonl state/bench/results
```

Expected: `Ingested 10 rows from state/bench/results`; `Wrote output/vlm_bench_report.md`; `Wrote output/vlm_bench_pareto.png`. If ground truth is not frozen, report includes the warning.

- [ ] **Step 4: Open the report**

```bash
cat output/vlm_bench_report.md
xdg-open output/vlm_bench_pareto.png 2>/dev/null || true
```

Expected: markdown table with one row for `gemma3@dryrun`; PNG with a single data point.

- [ ] **Step 5: Clean up the dryrun rows before the real bench**

```bash
source .venv/bin/activate && python -c "
import psycopg
with psycopg.connect('postgresql://dedup:dedup_local_dev@localhost:5432/dedup', autocommit=True) as conn:
    conn.execute(\"DELETE FROM stamp_ocr WHERE host_label='dryrun'\")
print('dryrun rows removed')
"
rm -f state/bench/results/gemma3_dryrun.jsonl
```

- [ ] **Step 6: (No commit — this task only verifies.)**

If anything failed above, loop back to the relevant task (4, 7, or 8) and fix.

---

## Task 12: Write the final session handoff

**Files:**
- Create: `docs/handoff/2026-04-20 Handoff - VLM OCR Bench Scaffolding.md`

- [ ] **Step 1: Use the handoff-create skill or write manually**

The handoff records:

1. Spec + plan paths.
2. Migrations applied (Task 0, Task 1).
3. Corpus built (Task 4) — 200 stems at `state/bench/`.
4. Ground-truth generation status (Task 6) — probably still in progress at the end of most implementation sessions; record which shard_id you're on.
5. Pointer to which benches (Task 11+) have run.
6. Open questions from the spec still unresolved.

Run:

```bash
# Via skill if available:
/handoff-create VLM OCR Bench Scaffolding

# OR write manually to docs/handoff/2026-04-20 Handoff - VLM OCR Bench Scaffolding.md
```

- [ ] **Step 2: Commit the handoff**

```bash
git add "docs/handoff/2026-04-20 Handoff - VLM OCR Bench Scaffolding.md" docs/handoff/.handoffs.md
git commit -m "handoff: VLM OCR bench scaffolding"
```

---

## Post-plan: running real benches

Once all 12 tasks are complete and ground truth is frozen, the bench is now ready to run. This is a reference, not part of the implementation plan:

```bash
# On ares (10h — kick off overnight):
for m in kimi-vl:latest qwen2.5-vl:7b minicpm-v gemma3 llama3.2-vision:11b internvl3:8b; do
  just bench-run ares-cpu "$m"
done

# On M2 Pro (1-2h total):
for m in kimi-vl:latest qwen2.5-vl:7b minicpm-v gemma3 llama3.2-vision:11b internvl3:8b; do
  bash scripts/ocr/bench_profiles/m2pro.sh "$m"
done
rsync state/bench/results/*_m2pro.jsonl ares:photo_project/state/bench/results/

# On ares, generate the final report:
just bench-report
```

Check `output/vlm_bench_report.md` + `output/vlm_bench_pareto.png`, and write the follow-up decision memo (outside this plan's scope).
