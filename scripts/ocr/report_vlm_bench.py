# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "psycopg[binary]>=3.1.0",
#     "matplotlib>=3.8",
# ]
# ///
"""Aggregate VLM bench results and emit a Pareto-frontier report.

Reads candidate rows from stamp_ocr. Ground-truth model is taken from the
manifest's ``ground_truth_model`` field (falls back to ``sonnet`` for
backward compatibility) or overridden via ``--ground-truth``. The GT
model is excluded from candidate metrics. Latency (elapsed_s) is hydrated
from JSONL files under ``state/bench/**`` since stamp_ocr doesn't store it.

Outputs:
    output/vlm_bench_report.md
    output/vlm_bench_pareto.png

Usage:
    uv run scripts/ocr/report_vlm_bench.py \\
        [--manifest state/bench/manifest.json] \\
        [--ingest-jsonl state/bench/results_litellm] \\
        [--ground-truth 'gemma4:31b-cloud@ollama-cloud']
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
REPORT_HTML = OUTPUT_DIR / "vlm_bench_report.html"
PARETO_PNG = OUTPUT_DIR / "vlm_bench_pareto.png"


# Rough model size (billions of params) lookup for Pareto bubble sizing.
MODEL_SIZE_B = {
    "kimi-vl": 16,
    "qwen2.5-vl:7b": 7,
    "qwen2.5-vl:3b": 3,
    "minicpm-v": 8,
    "gemma3": 12,
    "llama3.2-vision:11b": 11,
    "llama3.2-vision:90b": 90,
    "internvl3:8b": 8,
    "gemini-3-pro": 100,  # opaque; use large bubble
    "gpt-oss": 20,
    "sonnet": 0,
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


def compute_metrics(truth_by_stem: dict[str, str], rows: list[dict]) -> dict:
    empty = {
        "total": 0,
        "agree": 0,
        "wrong": 0,
        "unparsed": 0,
        "timeout": 0,
        "oom": 0,
        "rate_limit": 0,
        "agree_pct": 0.0,
        "unparsed_pct": 0.0,
        "timeout_pct": 0.0,
        "oom_pct": 0.0,
        "rate_limit_pct": 0.0,
        "high_conf_wrong_pct": 0.0,
        "min_s": 0.0,
        "median_s": 0.0,
        "p95_s": 0.0,
        "max_s": 0.0,
        "imgs_per_sec": 0.0,
        "avg_input_tokens": 0.0,
        "avg_output_tokens": 0.0,
        "first_run": None,
        "last_run": None,
        "host_label": None,
    }
    total = len(rows)
    if total == 0:
        return empty
    agree = unparsed = timeout = oom = rate_limit = wrong = 0
    latencies: list[float] = []
    in_tokens: list[int] = []
    out_tokens: list[int] = []
    run_timestamps: list[str] = []
    host_label = None
    for r in rows:
        raw = (r.get("raw_text") or "").strip()
        parsed = r.get("parsed_date")
        truth = truth_by_stem.get(r["stem"])
        if raw == "TIMEOUT":
            timeout += 1
        elif raw == "OOM_ERROR":
            oom += 1
        elif raw == "RATE_LIMIT":
            rate_limit += 1
        elif parsed is None:
            unparsed += 1
        elif parsed == truth:
            agree += 1
        else:
            wrong += 1
        if isinstance(r.get("elapsed_s"), (int, float)) and r["elapsed_s"] > 0:
            latencies.append(float(r["elapsed_s"]))
        if isinstance(r.get("prompt_eval_count"), (int, float)):
            in_tokens.append(int(r["prompt_eval_count"]))
        if isinstance(r.get("eval_count"), (int, float)):
            out_tokens.append(int(r["eval_count"]))
        if r.get("ran_at"):
            run_timestamps.append(r["ran_at"])
        if host_label is None and r.get("host_label"):
            host_label = r["host_label"]
    if latencies:
        sorted_l = sorted(latencies)
        min_s = sorted_l[0]
        median_s = statistics.median(sorted_l)
        p95_s = sorted_l[int(0.95 * (len(sorted_l) - 1))]
        max_s = sorted_l[-1]
    else:
        min_s = median_s = p95_s = max_s = 0.0
    return {
        "total": total,
        "agree": agree,
        "wrong": wrong,
        "unparsed": unparsed,
        "timeout": timeout,
        "oom": oom,
        "rate_limit": rate_limit,
        "agree_pct": 100.0 * agree / total,
        "unparsed_pct": 100.0 * unparsed / total,
        "timeout_pct": 100.0 * timeout / total,
        "oom_pct": 100.0 * oom / total,
        "rate_limit_pct": 100.0 * rate_limit / total,
        "high_conf_wrong_pct": 100.0 * wrong / total,
        "min_s": min_s,
        "median_s": median_s,
        "p95_s": p95_s,
        "max_s": max_s,
        "imgs_per_sec": (1.0 / median_s) if median_s > 0 else 0.0,
        "avg_input_tokens": (sum(in_tokens) / len(in_tokens)) if in_tokens else 0.0,
        "avg_output_tokens": (sum(out_tokens) / len(out_tokens)) if out_tokens else 0.0,
        "first_run": min(run_timestamps) if run_timestamps else None,
        "last_run": max(run_timestamps) if run_timestamps else None,
        "host_label": host_label,
    }


def pareto_frontier(points: list[dict]) -> list[dict]:
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


def load_ground_truth(stems: list[str], truth_model: str) -> dict[str, str]:
    if not stems:
        return {}
    with get_db() as conn:
        rows = conn.execute(
            "SELECT stem, parsed_date FROM stamp_ocr WHERE model=%s AND stem = ANY(%s)",
            (truth_model, stems),
        ).fetchall()
    return {r[0]: r[1].isoformat() if r[1] else None for r in rows}


def load_details_map(
    jsonl_roots: list[Path],
) -> dict[tuple[str, str], dict]:
    """Walk every *.jsonl under the given roots, collect per-(stem, model) extras.

    stamp_ocr persists only the parsed row; latency and token counts live in
    the JSONL shards. If the same (stem, model) appears in multiple files,
    the last write wins.
    """
    m: dict[tuple[str, str], dict] = {}
    for root in jsonl_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.jsonl"):
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stem = r.get("stem")
                key = r.get("model_key")
                if not (stem and key):
                    continue
                m[(stem, key)] = {
                    "elapsed_s": r.get("elapsed_s"),
                    "eval_count": r.get("eval_count"),
                    "prompt_eval_count": r.get("prompt_eval_count"),
                    "ran_at": r.get("ran_at"),
                    "error": r.get("error"),
                }
    return m


def load_candidate_rows(
    stems: list[str],
    truth_model: str,
    details_map: dict[tuple[str, str], dict],
) -> dict[str, list[dict]]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT stem, raw_text, parsed_date, model, host_label, updated_at
            FROM stamp_ocr
            WHERE model <> %s AND stem = ANY(%s)
            """,
            (truth_model, stems),
        ).fetchall()
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        stem, model_key = r[0], r[3]
        extras = details_map.get((stem, model_key), {})
        grouped.setdefault(model_key, []).append(
            {
                "stem": stem,
                "raw_text": r[1],
                "parsed_date": r[2].isoformat() if r[2] else None,
                "model_key": model_key,
                "host_label": r[4],
                "elapsed_s": extras.get("elapsed_s") or 0.0,
                "eval_count": extras.get("eval_count"),
                "prompt_eval_count": extras.get("prompt_eval_count"),
                "ran_at": extras.get("ran_at"),
                "error": extras.get("error"),
            }
        )
    return grouped


def render_markdown(
    rows_by_key: dict, metrics_by_key: dict, manifest: dict, truth_model: str
) -> str:
    frozen = manifest.get("ground_truth_frozen", False)
    frozen_at = manifest.get("ground_truth_frozen_at")
    if frozen:
        header_warn = (
            f"> Ground truth: `{truth_model}` "
            f"(frozen{' at ' + frozen_at if frozen_at else ''}).\n\n"
        )
    else:
        header_warn = (
            f"> **Ground truth NOT frozen.** Metrics below compare against "
            f"an unreviewed `{truth_model}` run.\n\n"
        )
    lines = [
        "# VLM OCR Bench Report",
        "",
        f"Generated: {manifest.get('generated_at', '?')}",
        f"Corpus: {len(manifest['stems'])} stems, seed={manifest.get('seed')}",
        f"Ground truth: `{truth_model}`",
        "",
        header_warn,
        f"| Model (tag@host) | N | Agree% vs `{truth_model}` | HiConfWrong% | Unparsed% | Timeout% | OOM% | RateLimit% | Median s | P95 s | img/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = sorted(metrics_by_key.items(), key=lambda kv: (-kv[1]["agree_pct"], kv[1]["median_s"]))
    for key, m in rows:
        lines.append(
            "| `{k}` | {n} | {a:.1f} | {w:.1f} | {u:.1f} | {t:.1f} | {o:.1f} | {rl:.1f} | {med:.1f} | {p95:.1f} | {ips:.2f} |".format(
                k=key,
                n=m["total"],
                a=m["agree_pct"],
                w=m["high_conf_wrong_pct"],
                u=m["unparsed_pct"],
                t=m["timeout_pct"],
                o=m["oom_pct"],
                rl=m["rate_limit_pct"],
                med=m["median_s"],
                p95=m["p95_s"],
                ips=m["imgs_per_sec"],
            )
        )
    return "\n".join(lines) + "\n"


def render_pareto_png(metrics_by_key: dict, truth_model: str) -> Path:
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
    ax.set_ylabel(f"agreement with {truth_model} ground truth (%)")
    ax.set_title(f"VLM bench Pareto frontier (GT: {truth_model})")
    ax.grid(alpha=0.3)

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


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "—"
    return ts.replace("T", " ").split(".")[0].rstrip("Z") + " UTC"


def _manifest_bucket_summary(manifest: dict) -> list[tuple[str, int]]:
    buckets: dict[str, int] = {}
    for s in manifest.get("stems", []):
        b = s.get("confidence_bucket", "(unknown)")
        buckets[b] = buckets.get(b, 0) + 1
    order = manifest.get("stratification", {}).get("confidence_buckets") or sorted(buckets)
    return [(b, buckets.get(b, 0)) for b in order]


def render_html(
    metrics_by_key: dict,
    manifest: dict,
    truth_model: str,
    frontier_keys: set[str],
) -> str:
    import html

    frozen = manifest.get("ground_truth_frozen", False)
    frozen_at = manifest.get("ground_truth_frozen_at")
    frozen_note = manifest.get("ground_truth_frozen_note")
    generated_at = manifest.get("generated_at", "?")
    seed = manifest.get("seed", "?")
    stems_n = len(manifest.get("stems", []))
    buckets = _manifest_bucket_summary(manifest)

    rows = sorted(
        metrics_by_key.items(),
        key=lambda kv: (-kv[1]["agree_pct"], kv[1]["median_s"] or 9e9),
    )
    total_runs = sum(m["total"] for _, m in rows)
    all_first = min((m["first_run"] for _, m in rows if m["first_run"]), default=None)
    all_last = max((m["last_run"] for _, m in rows if m["last_run"]), default=None)

    pareto_rel = PARETO_PNG.name

    gt_banner = (
        f'<div class="banner ok">Ground truth: <code>{html.escape(truth_model)}</code> '
        f'(frozen{" at " + html.escape(frozen_at) if frozen_at else ""})'
        + (f' — {html.escape(frozen_note)}' if frozen_note else "")
        + "</div>"
        if frozen
        else
        f'<div class="banner warn">Ground truth NOT frozen. Comparing to '
        f'unreviewed <code>{html.escape(truth_model)}</code> run.</div>'
    )

    bucket_rows = "\n".join(
        f"<tr><td><code>{html.escape(b)}</code></td><td class=\"num\">{n}</td></tr>"
        for b, n in buckets
    )

    table_rows_parts: list[str] = []
    for i, (key, m) in enumerate(rows, start=1):
        model_tag, host = parse_model_key(key)
        size_b = lookup_size(model_tag)
        size_txt = f"{size_b}B" if size_b else "—"
        front_badge = (
            '<span class="badge frontier" title="On Pareto frontier">★</span> '
            if key in frontier_keys
            else ""
        )
        acc = m["agree_pct"]
        acc_cls = "good" if acc >= 60 else "mid" if acc >= 50 else "bad"
        wrong = m["high_conf_wrong_pct"]
        wrong_cls = "good" if wrong <= 15 else "mid" if wrong <= 30 else "bad"
        in_tok = m["avg_input_tokens"]
        out_tok = m["avg_output_tokens"]
        err_total = m["timeout"] + m["oom"] + m["rate_limit"]
        err_cell = (
            f'{m["timeout"]}/{m["oom"]}/{m["rate_limit"]}'
            if err_total
            else '<span class="muted">0</span>'
        )
        table_rows_parts.append(
            f"""<tr>
  <td class="num">{i}</td>
  <td>{front_badge}<code>{html.escape(key)}</code></td>
  <td>{html.escape(host or '—')}</td>
  <td class="num">{size_txt}</td>
  <td class="num">{m['total']}</td>
  <td class="num {acc_cls}">{acc:.1f}%</td>
  <td class="num {wrong_cls}">{wrong:.1f}%</td>
  <td class="num">{m['unparsed_pct']:.1f}%</td>
  <td class="num err">{err_cell}</td>
  <td class="num">{m['min_s']:.2f}</td>
  <td class="num">{m['median_s']:.2f}</td>
  <td class="num">{m['p95_s']:.2f}</td>
  <td class="num">{m['max_s']:.2f}</td>
  <td class="num">{m['imgs_per_sec']:.2f}</td>
  <td class="num">{in_tok:.0f}</td>
  <td class="num">{out_tok:.0f}</td>
  <td class="muted">{_fmt_ts(m['first_run'])}</td>
  <td class="muted">{_fmt_ts(m['last_run'])}</td>
</tr>"""
        )
    table_rows_html = "\n".join(table_rows_parts)

    style = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: 2rem auto; padding: 0 1rem; color: #222; }
h1 { margin-bottom: 0.2rem; }
.muted { color: #888; font-size: 0.9em; }
.banner { padding: 0.6rem 1rem; border-radius: 6px; margin: 1rem 0; }
.banner.ok { background: #e6f4ea; border: 1px solid #8bc09a; }
.banner.warn { background: #fff3cd; border: 1px solid #e0b84a; }
.meta-grid { display: grid; grid-template-columns: auto 1fr; gap: 0.3rem 1.2rem;
             margin: 1rem 0; font-size: 0.92em; }
.meta-grid dt { font-weight: 600; color: #555; }
.meta-grid dd { margin: 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.88em; margin: 1rem 0; }
th, td { padding: 5px 8px; border-bottom: 1px solid #eee; text-align: left; }
th { background: #f5f5f5; position: sticky; top: 0; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.good { color: #1a7c31; font-weight: 600; }
td.mid  { color: #8a6d00; }
td.bad  { color: #a23131; }
td.err  { font-variant-numeric: tabular-nums; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
.badge.frontier { color: #b58500; font-weight: bold; }
.sub { display: grid; grid-template-columns: 2fr 1fr; gap: 2rem; }
.bucket-table td, .bucket-table th { padding: 3px 8px; }
img.pareto { max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
             margin: 1rem 0; display: block; }
.caption { font-size: 0.85em; color: #666; margin-top: -0.3rem; }
"""

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VLM OCR Bench Report</title>
<style>{style}</style>
</head>
<body>
<h1>VLM OCR Bench Report</h1>
<p class="muted">Rendered {html.escape(_fmt_ts(None) if not generated_at else _fmt_ts(generated_at))}</p>

{gt_banner}

<h2>Run metadata</h2>
<div class="sub">
<dl class="meta-grid">
  <dt>Generated</dt>                 <dd>{html.escape(generated_at)}</dd>
  <dt>Corpus size</dt>               <dd>{stems_n} stems</dd>
  <dt>Seed</dt>                      <dd>{html.escape(str(seed))}</dd>
  <dt>Ground truth</dt>              <dd><code>{html.escape(truth_model)}</code></dd>
  <dt>Frozen</dt>                    <dd>{'yes' if frozen else 'no'}{' — ' + html.escape(frozen_at) if frozen and frozen_at else ''}</dd>
  <dt>Candidate models</dt>          <dd>{len(rows)}</dd>
  <dt>Total rows compared</dt>       <dd>{total_runs}</dd>
  <dt>First run observed</dt>        <dd>{_fmt_ts(all_first)}</dd>
  <dt>Last run observed</dt>         <dd>{_fmt_ts(all_last)}</dd>
  <dt>Pareto-frontier models</dt>    <dd>{len(frontier_keys)}</dd>
</dl>
<div>
  <h3>Corpus stratification</h3>
  <table class="bucket-table">
    <thead><tr><th>Confidence bucket</th><th class="num">Stems</th></tr></thead>
    <tbody>{bucket_rows}</tbody>
  </table>
</div>
</div>

<h2>Pareto frontier</h2>
<img class="pareto" src="{pareto_rel}" alt="Pareto chart of accuracy vs throughput">
<p class="caption">Models with bordered bubbles are on the frontier (no other model dominates them on both axes). Bubble size ≈ model parameter count.</p>

<h2>Per-model results</h2>
<p class="caption">Sorted by agreement descending. ★ = on Pareto frontier.
Error column format: <code>timeout / OOM / rate_limit</code>.
Latency is based on JSONL <code>elapsed_s</code> (may be missing for older sonnet-only rows).</p>
<table>
  <thead>
    <tr>
      <th class="num">#</th>
      <th>Model</th>
      <th>Host</th>
      <th class="num">Size</th>
      <th class="num">N</th>
      <th class="num">Agree%</th>
      <th class="num">HiConfWrong%</th>
      <th class="num">Unparsed%</th>
      <th class="num" title="timeout / OOM / rate_limit">Errors</th>
      <th class="num">Min s</th>
      <th class="num">Median s</th>
      <th class="num">P95 s</th>
      <th class="num">Max s</th>
      <th class="num">img/s</th>
      <th class="num">Avg in-tok</th>
      <th class="num">Avg out-tok</th>
      <th>First run</th>
      <th>Last run</th>
    </tr>
  </thead>
  <tbody>
{table_rows_html}
  </tbody>
</table>

<h2>Interpretation notes</h2>
<ul>
  <li><strong>Agreement &ne; correctness.</strong> The ground-truth model is itself a VLM; "Agree%" measures alignment with its parses, not against a human-verified reference.</li>
  <li><strong>Unparsed vs HiConfWrong.</strong> Unparsed rows emitted text we couldn't coerce into a date; HiConfWrong emitted a parseable date that disagrees with GT.</li>
  <li><strong>Latency may undercount.</strong> Only JSONL-sourced runs have <code>elapsed_s</code>. Sonnet (original GT) has none and shows median 0 s; treat its img/s as unknown.</li>
  <li><strong>GT model excluded.</strong> <code>{html.escape(truth_model)}</code> is not in the candidate table because comparing it to itself is trivially 100%.</li>
</ul>

</body>
</html>
"""
    return body


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="state/bench/manifest.json")
    p.add_argument("--ingest-jsonl", default=None, help="Import orphan JSONL shards first")
    p.add_argument(
        "--ground-truth",
        default=None,
        help=(
            "Model key to use as ground truth. Defaults to manifest's "
            "ground_truth_model, then 'sonnet' as last fallback."
        ),
    )
    p.add_argument(
        "--latency-root",
        default="state/bench",
        help="Directory (walked recursively) containing JSONL with elapsed_s.",
    )
    args = p.parse_args()

    if args.ingest_jsonl:
        n = ingest_jsonl_dir(Path(args.ingest_jsonl))
        print(f"Ingested {n} rows from {args.ingest_jsonl}")

    manifest = json.loads(Path(args.manifest).read_text())
    stems = [s["stem"] for s in manifest["stems"]]

    truth_model = args.ground_truth or manifest.get("ground_truth_model") or "sonnet"
    print(f"Ground truth model: {truth_model}")

    details_map = load_details_map([Path(args.latency_root)])
    print(f"Loaded JSONL details for {len(details_map)} (stem, model) pairs")

    truth = load_ground_truth(stems, truth_model)
    if not truth:
        print(f"WARN: no rows found for ground-truth model '{truth_model}'. Agreement will be 0%.")
    cand_by_key = load_candidate_rows(stems, truth_model, details_map)

    if not cand_by_key:
        print(f"No non-'{truth_model}' rows in stamp_ocr yet. Run a bench first.")
        return

    metrics = {k: compute_metrics(truth, v) for k, v in cand_by_key.items()}
    frontier_keys = {
        fp["model_key"]
        for fp in pareto_frontier(
            [
                {
                    "model_key": k,
                    "agree_pct": m["agree_pct"],
                    "imgs_per_sec": m["imgs_per_sec"],
                }
                for k, m in metrics.items()
            ]
        )
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render_markdown(cand_by_key, metrics, manifest, truth_model))
    print(f"Wrote {REPORT_MD}")
    render_pareto_png(metrics, truth_model)
    print(f"Wrote {PARETO_PNG}")
    REPORT_HTML.write_text(
        render_html(metrics, manifest, truth_model, frontier_keys)
    )
    print(f"Wrote {REPORT_HTML}")


if __name__ == "__main__":
    main()
