# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Render a side-by-side HTML comparing VLM bench runs for the size-compare bench.

Reads all JSONL files from state/bench/size_compare/, joins on Sonnet ground
truth from stamp_ocr, and emits state/bench/size_compare/compare.html with
each crop inlined as base64 alongside every model's raw + parsed output.
"""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402

BENCH_DIR = BASE_DIR / "state" / "bench"
CROPS_DIR = BENCH_DIR / "crops"
RESULTS_DIR = BENCH_DIR / "size_compare"
OUT_HTML = RESULTS_DIR / "compare.html"


def load_model_runs() -> dict[str, dict[str, dict]]:
    """Return {model: {stem: result_dict}} from every JSONL in results dir."""
    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        model = path.stem.replace("_", ":", 1)
        runs[model] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            runs[model][r["stem"]] = r
    return runs


def load_ground_truth(stems: list[str]) -> dict[str, tuple[str, str | None]]:
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


def crop_b64(stem: str) -> str | None:
    p = CROPS_DIR / f"{stem}.jpg"
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()


def summarize(
    model: str,
    rows: dict[str, dict],
    gt: dict[str, tuple[str, str | None]],
) -> dict:
    latencies = [r["elapsed_s"] for r in rows.values() if not r.get("error")]
    agree = sum(
        1
        for stem, r in rows.items()
        if gt.get(stem, (None, None))[1] is not None
        and r["parsed"] == gt[stem][1]
    )
    compared = sum(1 for stem in rows if gt.get(stem, (None, None))[1] is not None)
    parsed = sum(1 for r in rows.values() if r["parsed"] is not None)
    return {
        "model": model,
        "n": len(rows),
        "parsed": parsed,
        "agree": agree,
        "compared": compared,
        "agree_pct": (100 * agree / compared) if compared else 0.0,
        "p50": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
        "total": round(sum(latencies), 1),
    }


def render() -> Path:
    runs = load_model_runs()
    if not runs:
        print(f"No JSONL files in {RESULTS_DIR}")
        sys.exit(1)

    all_stems: list[str] = []
    for rows in runs.values():
        for stem in rows:
            if stem not in all_stems:
                all_stems.append(stem)
    gt = load_ground_truth(all_stems)
    models = list(runs.keys())
    summaries = [summarize(m, runs[m], gt) for m in models]

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>VLM size-compare bench</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:1em;background:#fafafa;color:#222}",
        "h1{margin-top:0}",
        "table.summary{border-collapse:collapse;margin:0 0 1.5em 0}",
        "table.summary th,table.summary td{border:1px solid #bbb;padding:6px 10px;text-align:left}",
        "table.summary th{background:#eee}",
        "table.grid{border-collapse:collapse;width:100%;background:#fff}",
        "table.grid th,table.grid td{border:1px solid #ccc;padding:8px;vertical-align:top}",
        "table.grid th{background:#eee;position:sticky;top:0}",
        "img.crop{max-height:120px;display:block}",
        ".stem{font-family:monospace;font-size:0.85em;color:#555}",
        ".ok{background:#d7f0d7}",
        ".miss{background:#f7dcdc}",
        ".unparsed{background:#f5efd7;color:#7a5c00}",
        ".raw{font-family:monospace;font-size:0.9em}",
        ".parsed{font-family:monospace;color:#333}",
        ".meta{color:#666;font-size:0.8em}",
        "</style></head><body>",
        "<h1>gemma4 size comparison on photo date stamps</h1>",
        f"<p>Corpus: first {len(all_stems)} crops from <code>state/bench/crops/</code> "
        "with Sonnet-parsed ground truth. Green = matches Sonnet; red = disagrees; "
        "yellow = unparsed (day missing or garbled).</p>",
        "<table class='summary'>",
        "<tr><th>model</th><th>parsed</th><th>agree / compared</th><th>agree %</th>"
        "<th>p50 latency (s)</th><th>wall (s)</th></tr>",
    ]
    for s in summaries:
        parts.append(
            f"<tr><td><code>{s['model']}</code></td><td>{s['parsed']}/{s['n']}</td>"
            f"<td>{s['agree']}/{s['compared']}</td>"
            f"<td>{s['agree_pct']:.1f}%</td>"
            f"<td>{s['p50']:.1f}</td><td>{s['total']:.1f}</td></tr>"
        )
    parts.append("</table>")

    parts.append("<table class='grid'>")
    header = ["<th>crop</th>", "<th>stem</th>", "<th>sonnet (truth)</th>"]
    for m in models:
        header.append(f"<th>{html.escape(m)}</th>")
    parts.append("<tr>" + "".join(header) + "</tr>")

    for stem in all_stems:
        gt_raw, gt_parsed = gt.get(stem, ("", None))
        b64 = crop_b64(stem)
        img_html = (
            f"<img class='crop' src='data:image/jpeg;base64,{b64}'/>"
            if b64
            else "<em>missing</em>"
        )
        row = [f"<td>{img_html}</td>"]
        row.append(f"<td class='stem'>{stem}</td>")
        row.append(
            f"<td><div class='raw'>{html.escape(gt_raw)!r}</div>"
            f"<div class='parsed'>{gt_parsed}</div></td>"
        )
        for m in models:
            r = runs[m].get(stem)
            if not r:
                row.append("<td><em>missing</em></td>")
                continue
            parsed = r["parsed"]
            if parsed is None:
                cls = "unparsed"
            elif gt_parsed is not None and parsed == gt_parsed:
                cls = "ok"
            else:
                cls = "miss"
            row.append(
                f"<td class='{cls}'>"
                f"<div class='raw'>{html.escape(r['final'])!r}</div>"
                f"<div class='parsed'>{parsed}</div>"
                f"<div class='meta'>{r['elapsed_s']:.1f}s</div>"
                f"</td>"
            )
        parts.append("<tr>" + "".join(row) + "</tr>")

    parts.append("</table></body></html>")
    OUT_HTML.write_text("".join(parts))
    return OUT_HTML


if __name__ == "__main__":
    out = render()
    print(f"Wrote {out}")
