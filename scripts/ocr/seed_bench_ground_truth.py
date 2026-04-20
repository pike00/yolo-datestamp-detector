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
from ocr.ocr_util import extract_final_answer, normalize_date  # noqa: E402

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
