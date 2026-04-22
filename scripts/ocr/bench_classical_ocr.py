# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pillow>=12.0",
#     "opencv-python>=4.10",
#     "pytesseract>=0.3.10",
#     "numpy>=1.26",
#     "requests>=2.31",
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Non-LLM OCR bench over the 200 date-stamp crops in state/bench/crops/.

Engines:
    tesseract -- local, multi-variant (preprocess x psm), pick best per crop
    easyocr   -- imported lazily; CPU; downloads model on first run
    paddleocr -- imported lazily; CPU; downloads model on first run
    docling   -- docling-serve at http://localhost:5001 (Docker)

Output: state/bench/results/<engine>.jsonl  (one JSON row per stem)
Score:  --score prints accuracy table vs Sonnet ground truth in stamp_ocr.

Usage:
    uv run scripts/ocr/bench_classical_ocr.py --engine tesseract
    uv run scripts/ocr/bench_classical_ocr.py --engine easyocr
    uv run scripts/ocr/bench_classical_ocr.py --engine paddleocr
    uv run scripts/ocr/bench_classical_ocr.py --engine docling
    uv run scripts/ocr/bench_classical_ocr.py --score
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402
from ocr.ocr_util import normalize_date  # noqa: E402

BENCH_DIR = BASE_DIR / "state" / "bench"
CROPS_DIR = BENCH_DIR / "crops"
MANIFEST_PATH = BENCH_DIR / "manifest.json"
RESULTS_DIR = BENCH_DIR / "results"


# -------------------- shared helpers --------------------


def load_stems() -> list[dict]:
    data = json.loads(MANIFEST_PATH.read_text())
    return data["stems"]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# -------------------- tesseract --------------------


def _prep_orange_otsu(im: np.ndarray, invert: bool) -> np.ndarray:
    b, g, r = cv2.split(im)
    gray = cv2.subtract(r, g)
    h, w = gray.shape
    gray = cv2.resize(gray, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(bw) if invert else bw


def _prep_gray_otsu(im: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _prep_raw_up(im: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    return cv2.resize(gray, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)


TESS_PREPS: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("orange_otsu_inv", lambda im: _prep_orange_otsu(im, invert=True)),
    ("orange_otsu", lambda im: _prep_orange_otsu(im, invert=False)),
    ("gray_otsu", _prep_gray_otsu),
    ("raw_up", _prep_raw_up),
]

TESS_WHITELIST = "0123456789 "
TESS_PSMS = [7, 8, 11, 13]


def run_tesseract(stem: str, crop_path: Path) -> dict:
    import pytesseract

    im = cv2.imread(str(crop_path))
    if im is None:
        return {"stem": stem, "raw_text": "", "parsed_date": None, "variant": None, "elapsed_s": 0.0}
    t0 = time.time()

    best_parsed = None
    best_raw = ""
    best_variant = None

    all_candidates: list[tuple[str, str, str | None]] = []

    for pname, pfn in TESS_PREPS:
        try:
            img = pfn(im)
        except Exception:
            continue
        for psm in TESS_PSMS:
            cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist={TESS_WHITELIST}"
            try:
                txt = pytesseract.image_to_string(img, config=cfg).strip()
            except Exception:
                continue
            if not txt:
                continue
            flat = " ".join(txt.split())
            parsed = normalize_date(flat)
            all_candidates.append((f"{pname}_psm{psm}", flat, parsed))
            if parsed and best_parsed is None:
                best_parsed = parsed
                best_raw = flat
                best_variant = f"{pname}_psm{psm}"

    if best_parsed is None and all_candidates:
        # no full-date parse; pick the longest digit-containing candidate
        variant, flat, parsed = max(all_candidates, key=lambda c: len(c[1]))
        best_raw = flat
        best_parsed = parsed
        best_variant = variant

    return {
        "stem": stem,
        "raw_text": best_raw,
        "parsed_date": best_parsed,
        "variant": best_variant,
        "elapsed_s": round(time.time() - t0, 3),
    }


# -------------------- easyocr --------------------


_EASY_READER = None


def run_easyocr(stem: str, crop_path: Path) -> dict:
    global _EASY_READER
    if _EASY_READER is None:
        import easyocr  # type: ignore

        _EASY_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    t0 = time.time()
    # Pre-upscale helps a lot for tiny crops.
    im = cv2.imread(str(crop_path))
    h, w = im.shape[:2]
    up = cv2.resize(im, (w * 3, h * 3), interpolation=cv2.INTER_LANCZOS4)
    res = _EASY_READER.readtext(up, allowlist="0123456789' ", detail=0, paragraph=False)
    raw = " ".join(x.strip() for x in res if x.strip())
    parsed = normalize_date(raw)
    return {
        "stem": stem,
        "raw_text": raw,
        "parsed_date": parsed,
        "variant": "easyocr_x3",
        "elapsed_s": round(time.time() - t0, 3),
    }


# -------------------- paddleocr --------------------


_PADDLE_OCR = None


def run_paddleocr(stem: str, crop_path: Path) -> dict:
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        from paddleocr import PaddleOCR  # type: ignore

        _PADDLE_OCR = PaddleOCR(use_angle_cls=False, lang="en", enable_mkldnn=False, show_log=False)
    t0 = time.time()
    im = cv2.imread(str(crop_path))
    h, w = im.shape[:2]
    up = cv2.resize(im, (w * 3, h * 3), interpolation=cv2.INTER_LANCZOS4)
    try:
        result = _PADDLE_OCR.ocr(up, cls=False)
    except TypeError:
        result = _PADDLE_OCR.ocr(up)
    texts: list[str] = []
    if result and result[0]:
        for line in result[0]:
            try:
                texts.append(line[1][0].strip())
            except (IndexError, TypeError):
                pass
    raw = " ".join(t for t in texts if t)
    parsed = normalize_date(raw)
    return {
        "stem": stem,
        "raw_text": raw,
        "parsed_date": parsed,
        "variant": "paddle_en_x3",
        "elapsed_s": round(time.time() - t0, 3),
    }


# -------------------- docling --------------------


def run_docling(stem: str, crop_path: Path, url: str) -> dict:
    t0 = time.time()
    with crop_path.open("rb") as f:
        files = {"files": (crop_path.name, f, "image/jpeg")}
        data = {
            "to_formats": "md",
            "do_ocr": "true",
            "image_export_mode": "placeholder",
        }
        try:
            r = requests.post(f"{url}/v1/convert/file", files=files, data=data, timeout=120)
        except requests.RequestException as e:
            return {
                "stem": stem,
                "raw_text": f"ERROR_{type(e).__name__}",
                "parsed_date": None,
                "variant": "docling",
                "elapsed_s": round(time.time() - t0, 3),
            }
    if r.status_code != 200:
        return {
            "stem": stem,
            "raw_text": f"HTTP_{r.status_code}",
            "parsed_date": None,
            "variant": "docling",
            "elapsed_s": round(time.time() - t0, 3),
        }
    try:
        payload = r.json()
        md = payload.get("document", {}).get("md_content", "") or ""
    except Exception:
        md = r.text
    flat = " ".join(md.split())
    parsed = normalize_date(flat)
    return {
        "stem": stem,
        "raw_text": flat,
        "parsed_date": parsed,
        "variant": "docling_md",
        "elapsed_s": round(time.time() - t0, 3),
    }


# -------------------- scoring --------------------


def score_all() -> None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT stem, raw_text, parsed_date FROM stamp_ocr WHERE model = 'sonnet'")
    gt_rows = cur.fetchall()
    gt_parsed = {r[0]: r[2] for r in gt_rows}
    gt_raw = {r[0]: (r[1] or "").strip() for r in gt_rows}

    total = len(gt_parsed)
    gt_with_date = sum(1 for v in gt_parsed.values() if v is not None)
    gt_none = total - gt_with_date

    print(f"Ground truth: {total} stems ({gt_with_date} dated, {gt_none} NONE)")
    print()
    header = f"{'engine':<32}{'rows':>6}{'nonempty':>10}{'full_date':>11}{'md_only':>9}{'year_only':>11}{'none_hit':>10}{'time_avg':>10}"
    print(header)
    print("-" * len(header))

    for jsonl in sorted(RESULTS_DIR.glob("*.jsonl")):
        rows = [json.loads(ln) for ln in jsonl.read_text().splitlines() if ln.strip()]
        nonempty = 0
        full_date = 0
        md_only = 0
        year_only = 0
        none_hit = 0
        n_dated = 0
        n_none = 0
        t_total = 0.0
        t_count = 0
        for row in rows:
            t_total += float(row.get("elapsed_s") or 0.0)
            t_count += 1
            rt = row.get("raw_text") or ""
            if rt and not rt.startswith(("ERROR_", "HTTP_", "TIMEOUT", "RATE_LIMIT", "OOM_")):
                nonempty += 1
            stem = row["stem"]
            gp = gt_parsed.get(stem)
            pp = row.get("parsed_date")
            if gp is None:
                n_none += 1
                if pp is None:
                    none_hit += 1
                continue
            n_dated += 1
            if pp is None:
                continue
            if isinstance(gp, str):
                gy, gm, gd = gp.split("-")
            else:
                gy, gm, gd = str(gp.year), f"{gp.month:02d}", f"{gp.day:02d}"
            py, pm, pd = pp.split("-")
            ymatch = gy == py
            mdmatch = gm == pm and gd == pd
            if ymatch and mdmatch:
                full_date += 1
            elif mdmatch:
                md_only += 1
            elif ymatch:
                year_only += 1
        avg = (t_total / t_count) if t_count else 0.0
        print(
            f"{jsonl.stem:<32}"
            f"{len(rows):>6}"
            f"{nonempty:>10}"
            f"{full_date:>4}/{n_dated:<6}"
            f"{md_only:>4}/{n_dated:<4}"
            f"{year_only:>4}/{n_dated:<6}"
            f"{none_hit:>4}/{n_none:<5}"
            f"{avg:>9.2f}s"
        )


# -------------------- main --------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["tesseract", "easyocr", "paddleocr", "docling"])
    ap.add_argument("--docling-url", default="http://localhost:5001")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    if args.score:
        score_all()
        return

    if not args.engine:
        ap.error("--engine required unless --score")

    stems = load_stems()
    if args.limit:
        stems = stems[: args.limit]
    print(f"Engine: {args.engine}  |  Stems: {len(stems)}")

    rows: list[dict] = []
    t_start = time.time()
    for i, s in enumerate(stems):
        crop_path = CROPS_DIR / f"{s['stem']}.jpg"
        if not crop_path.exists():
            print(f"  MISSING: {crop_path}")
            continue
        if args.engine == "tesseract":
            row = run_tesseract(s["stem"], crop_path)
        elif args.engine == "easyocr":
            row = run_easyocr(s["stem"], crop_path)
        elif args.engine == "paddleocr":
            row = run_paddleocr(s["stem"], crop_path)
        elif args.engine == "docling":
            row = run_docling(s["stem"], crop_path, args.docling_url)
        else:
            raise SystemExit(f"unknown engine: {args.engine}")
        rows.append(row)
        tag = row.get("parsed_date") or "--"
        print(f"  [{i+1}/{len(stems)}] {s['stem']}: {row['raw_text']!r:30s} -> {tag}  ({row['elapsed_s']:.2f}s)")

    out = RESULTS_DIR / f"{args.engine}.jsonl"
    write_jsonl(out, rows)
    wall = time.time() - t_start
    print()
    print(f"Wrote {len(rows)} rows -> {out}")
    print(f"Wall: {wall:.1f}s  |  avg: {wall/max(1,len(rows)):.2f}s/img")


if __name__ == "__main__":
    main()
