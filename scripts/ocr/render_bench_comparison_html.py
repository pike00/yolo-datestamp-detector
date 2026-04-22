# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "psycopg[binary]>=3.1.0",
# ]
# ///
"""Render a side-by-side HTML comparing VLM bench model reads against the
Sonnet ground truth, one row per stem with the crop inlined as base64.

Reads:
- state/bench/manifest.json (stem list + crop paths)
- state/bench/results/*.jsonl (per-model raw_text + parsed_date + elapsed)
- stamp_ocr where model='sonnet' (ground truth)

Writes:
- output/vlm_bench_comparison.html
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from _db import get_db  # noqa: E402

BENCH_DIR = BASE_DIR / "state" / "bench"
CROPS_DIR = BENCH_DIR / "crops"
MANIFEST_PATH = BENCH_DIR / "manifest.json"
RESULTS_DIR = BENCH_DIR / "results"
OUTPUT_HTML = BASE_DIR / "output" / "vlm_bench_comparison.html"


def load_manifest_stems() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text())["stems"]


def load_sonnet_truth(stems: list[str]) -> dict[str, dict]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT stem, raw_text, parsed_date
            FROM stamp_ocr
            WHERE model='sonnet' AND stem = ANY(%s)
            """,
            (stems,),
        ).fetchall()
    return {
        r[0]: {
            "raw_text": r[1],
            "parsed_date": r[2].isoformat() if r[2] else None,
        }
        for r in rows
    }


def load_jsonl_reads() -> dict[str, dict[str, dict]]:
    """Return {model_key: {stem: {raw_text, parsed_date, elapsed_s}}}."""
    out: dict[str, dict[str, dict]] = {}
    for p in sorted(RESULTS_DIR.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if "model_key" not in d:
                continue
            key = d["model_key"]
            out.setdefault(key, {})[d["stem"]] = {
                "raw_text": d.get("raw_text", ""),
                "parsed_date": d.get("parsed_date"),
                "elapsed_s": d.get("elapsed_s"),
            }
    return out


def cell_html(read: dict | None, truth_parsed: str | None) -> str:
    if not read:
        return '<td class="missing">—</td>'
    raw = read.get("raw_text") or ""
    parsed = read.get("parsed_date")
    elapsed = read.get("elapsed_s")
    err = raw in ("RATE_LIMIT", "TIMEOUT", "OOM_ERROR") or raw.startswith("HTTP_")

    if err:
        css = "err"
        status = raw
    elif parsed is None:
        css = "unparsed"
        status = "unparsed"
    elif parsed == truth_parsed:
        css = "agree"
        status = "agree"
    else:
        css = "wrong"
        status = f"wrong (got {parsed})"

    elapsed_str = f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
    return (
        f'<td class="{css}">'
        f'<div class="raw">{raw!s}</div>'
        f'<div class="parsed">{parsed or "—"}</div>'
        f'<div class="meta">{status} · {elapsed_str}</div>'
        "</td>"
    )


def truth_cell(truth: dict | None) -> str:
    if not truth:
        return '<td class="missing">—</td>'
    raw = truth.get("raw_text") or ""
    parsed = truth.get("parsed_date") or "—"
    return (
        '<td class="truth">'
        f'<div class="raw">{raw!s}</div>'
        f'<div class="parsed">{parsed}</div>'
        "</td>"
    )


def b64_crop(stem: str) -> str:
    p = CROPS_DIR / f"{stem}.jpg"
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode()


def render() -> Path:
    manifest = load_manifest_stems()
    stems = [s["stem"] for s in manifest]
    truth = load_sonnet_truth(stems)
    by_model = load_jsonl_reads()
    model_keys = sorted(by_model.keys())

    rows = []
    for m in manifest:
        stem = m["stem"]
        conf = m.get("yolo_confidence", 0.0)
        bucket = m.get("confidence_bucket", "")
        t = truth.get(stem)
        b64 = b64_crop(stem)
        img = (
            f'<img src="data:image/jpeg;base64,{b64}" alt="{stem}" />'
            if b64
            else "(missing crop)"
        )
        cells = "".join(
            cell_html(by_model.get(k, {}).get(stem), t["parsed_date"] if t else None)
            for k in model_keys
        )
        rows.append(
            "<tr>"
            f'<td class="crop">{img}</td>'
            f'<td class="stem"><code>{stem}</code><br><span class="meta">yolo={conf:.2f}<br>{bucket}</span></td>'
            f"{truth_cell(t)}"
            f"{cells}"
            "</tr>"
        )

    model_headers = "".join(
        f'<th class="model">{k}</th>' for k in model_keys
    )
    legend = (
        '<span class="agree">agree</span> · '
        '<span class="wrong">wrong</span> · '
        '<span class="unparsed">unparsed</span> · '
        '<span class="err">error</span>'
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>VLM bench comparison</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1em; background: #fafafa; }}
h1 {{ margin: 0 0 0.3em; }}
.legend {{ margin: 0 0 1em; color: #555; font-size: 0.9em; }}
.legend span {{ padding: 2px 8px; border-radius: 3px; margin-right: 4px; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; text-align: left; }}
th {{ background: #eee; font-size: 0.85em; position: sticky; top: 0; z-index: 1; }}
th.model {{ max-width: 180px; word-break: break-all; }}
td.crop img {{ max-width: 220px; display: block; }}
td.stem {{ font-size: 0.85em; white-space: nowrap; }}
td.stem .meta {{ color: #888; font-size: 0.9em; }}
td.truth {{ background: #eef2ff; }}
td.agree {{ background: #e9f7ea; }}
td.wrong {{ background: #fdecea; }}
td.unparsed {{ background: #fff4e5; }}
td.err {{ background: #f4f4f4; color: #888; }}
td.missing {{ color: #bbb; text-align: center; }}
.raw {{ font-family: ui-monospace, monospace; font-size: 0.9em; }}
.parsed {{ font-family: ui-monospace, monospace; color: #444; font-size: 0.85em; }}
.meta {{ color: #777; font-size: 0.75em; margin-top: 2px; }}
.legend .agree {{ background: #e9f7ea; color: #1a6b28; }}
.legend .wrong {{ background: #fdecea; color: #9b2c2c; }}
.legend .unparsed {{ background: #fff4e5; color: #9c6b1f; }}
.legend .err {{ background: #f4f4f4; color: #777; }}
</style>
</head><body>
<h1>VLM bench comparison ({len(manifest)} stems)</h1>
<div class="legend">Legend: {legend}. "agree" = model's normalized_date matches Sonnet's.</div>
<table>
<thead><tr>
<th>crop</th><th>stem</th><th class="model">sonnet (truth)</th>
{model_headers}
</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
</body></html>
"""

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html)
    return OUTPUT_HTML


if __name__ == "__main__":
    p = render()
    size_mb = p.stat().st_size / 1024 / 1024
    print(f"Wrote {p} ({size_mb:.1f} MB)")
