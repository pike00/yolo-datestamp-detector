# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pillow>=10.0",
#     "psycopg[binary]>=3.1.0",
#     "requests>=2.31",
# ]
# ///
"""Quick size-comparison bench for local VLM OCR.

Runs one or more Ollama vision models on the first N crops from
state/bench/crops/, compares normalized-date output to the Sonnet ground
truth (stamp_ocr rows with model='sonnet'), and prints per-image + summary
stats (agreement rate, median/p95 latency).

No DB writes. Stdout-only report plus a JSONL side file per model.

Usage:
    uv run scripts/ocr/bench_compare_sizes.py \
        --models gemma4:e4b gemma4:26b \
        --n 20
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402
from ocr.ocr_util import extract_final_answer, normalize_date  # noqa: E402

BENCH_DIR = BASE_DIR / "state" / "bench"
CROPS_DIR = BENCH_DIR / "crops"
MANIFEST_PATH = BENCH_DIR / "manifest.json"
RESULTS_DIR = BENCH_DIR / "size_compare"

OLLAMA_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 300

PROMPT = """This is a cropped photo showing a camera date stamp -- orange LED digits.
Transcribe EXACTLY what you see, preserving spaces and apostrophes.
Example formats: "10 3 '99" or "'94 6 22" or "8 24'95"
Output ONLY the stamp text, nothing else."""


def load_ground_truth(stems: list[str]) -> dict[str, tuple[str, str | None]]:
    """Return {stem: (raw_text, parsed_date_iso)} for the Sonnet rows."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT stem, raw_text, parsed_date
               FROM stamp_ocr
               WHERE model = 'sonnet' AND stem = ANY(%s)""",
            (stems,),
        ).fetchall()
    return {
        r[0]: (r[1] or "", r[2].isoformat() if r[2] else None)
        for r in rows
    }


def encode_crop(crop_path: Path) -> str:
    return base64.b64encode(crop_path.read_bytes()).decode()


def run_model(model: str, stems: list[str], crops: dict[str, Path]) -> list[dict]:
    url = f"{OLLAMA_URL}/api/chat"
    results = []
    for i, stem in enumerate(stems):
        b64 = encode_crop(crops[stem])
        t0 = time.time()
        try:
            resp = requests.post(
                url,
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": PROMPT, "images": [b64]}
                    ],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 2048},
                    "think": False,
                },
                timeout=REQUEST_TIMEOUT,
            )
            elapsed = time.time() - t0
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("message", {}).get("content", "")
            final = extract_final_answer(raw)
            parsed = normalize_date(final)
            if parsed and parsed.endswith("-00"):
                parsed = None
            eval_count = data.get("eval_count", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            err = None
        except requests.Timeout:
            elapsed = time.time() - t0
            raw = ""
            final = ""
            parsed = None
            eval_count = 0
            prompt_eval_count = 0
            err = "TIMEOUT"
        except Exception as e:
            elapsed = time.time() - t0
            raw = ""
            final = ""
            parsed = None
            eval_count = 0
            prompt_eval_count = 0
            err = f"ERROR: {e}"

        results.append(
            {
                "stem": stem,
                "raw": raw,
                "final": final,
                "parsed": parsed,
                "elapsed_s": round(elapsed, 2),
                "eval_count": eval_count,
                "prompt_eval_count": prompt_eval_count,
                "error": err,
            }
        )
        note = err or ("" if parsed else " [unparsed]")
        print(
            f"  [{i + 1}/{len(stems)}] {stem}: {final!r:<30} -> {parsed}"
            f"  ({elapsed:.1f}s){note}"
        )
    return results


def summarize(
    model: str,
    results: list[dict],
    gt: dict[str, tuple[str, str | None]],
) -> dict:
    parsed_count = sum(1 for r in results if r["parsed"] is not None)
    timeout_count = sum(1 for r in results if r["error"] == "TIMEOUT")
    error_count = sum(1 for r in results if r["error"] and r["error"] != "TIMEOUT")

    date_agree = 0
    date_compared = 0
    for r in results:
        gt_raw, gt_parsed = gt.get(r["stem"], ("", None))
        if gt_parsed is None:
            continue  # Sonnet also couldn't parse; skip
        date_compared += 1
        if r["parsed"] == gt_parsed:
            date_agree += 1

    latencies = [r["elapsed_s"] for r in results if r["error"] is None]
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = (
        statistics.quantiles(latencies, n=20)[-1]
        if len(latencies) >= 20
        else max(latencies)
        if latencies
        else 0.0
    )

    return {
        "model": model,
        "n": len(results),
        "parsed": parsed_count,
        "date_compared": date_compared,
        "date_agree": date_agree,
        "agreement_pct": (100.0 * date_agree / date_compared) if date_compared else 0.0,
        "p50_s": round(p50, 1),
        "p95_s": round(p95, 1),
        "total_s": round(sum(latencies), 1),
        "timeouts": timeout_count,
        "errors": error_count,
    }


def print_summary_table(summaries: list[dict]) -> None:
    print()
    print("=" * 92)
    print(
        f"{'model':<24}  {'n':>3}  {'parsed':>6}  {'agree/cmp':>10}  "
        f"{'agree%':>7}  {'p50s':>6}  {'p95s':>6}  {'wall':>6}  {'T/E':>5}"
    )
    print("-" * 92)
    for s in summaries:
        agree_frac = f"{s['date_agree']}/{s['date_compared']}"
        print(
            f"{s['model']:<24}  {s['n']:>3}  {s['parsed']:>6}  {agree_frac:>10}  "
            f"{s['agreement_pct']:>6.1f}%  {s['p50_s']:>6.1f}  {s['p95_s']:>6.1f}  "
            f"{s['total_s']:>6.1f}  {s['timeouts']}/{s['errors']}"
        )
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Skip first N stems (lets you pick a different slice)")
    ap.add_argument("--crops-dir", default=str(CROPS_DIR),
                    help="Directory containing <stem>.jpg crops")
    ap.add_argument("--label-suffix", default="",
                    help="Suffix to add to output JSONL names, e.g. '_hf'")
    args = ap.parse_args()

    crops_dir = Path(args.crops_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST_PATH.read_text())
    # Prefer stems where Sonnet gave a parseable date so we can measure agreement.
    all_stems = [s["stem"] for s in manifest["stems"]]
    gt_all = load_ground_truth(all_stems)
    usable = [s for s in all_stems if gt_all.get(s, ("", None))[1] is not None]
    picked = usable[args.seed_offset : args.seed_offset + args.n]
    if len(picked) < args.n:
        print(f"WARNING: only {len(picked)} stems with Sonnet ground truth available")
    crops = {s: crops_dir / f"{s}.jpg" for s in picked}
    missing = [s for s, p in crops.items() if not p.exists()]
    if missing:
        print(f"WARNING: missing crops in {crops_dir} for {len(missing)} stems; skipping them")
        picked = [s for s in picked if s not in set(missing)]
        crops = {s: crops[s] for s in picked}
    if not picked:
        print("ERROR: no crops to run")
        sys.exit(1)

    print(f"Bench set: {len(picked)} crops (all with Sonnet-parsed ground truth)\n")

    summaries = []
    for model in args.models:
        label = f"{model}{args.label_suffix}"
        print(f"=== {label} ===")
        results = run_model(model, picked, crops)
        out_path = RESULTS_DIR / f"{label.replace(':', '_').replace('/', '_')}.jsonl"
        with out_path.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {out_path}")
        summaries.append(summarize(label, results, gt_all))
        print()

    print_summary_table(summaries)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
