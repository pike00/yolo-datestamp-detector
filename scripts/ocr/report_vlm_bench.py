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


def compute_metrics(sonnet_by_stem: dict[str, str], rows: list[dict]) -> dict:
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "agree_pct": 0.0,
            "unparsed_pct": 0.0,
            "timeout_pct": 0.0,
            "oom_pct": 0.0,
            "rate_limit_pct": 0.0,
            "high_conf_wrong_pct": 0.0,
            "median_s": 0.0,
            "p95_s": 0.0,
            "imgs_per_sec": 0.0,
        }
    agree = unparsed = timeout = oom = rate_limit = wrong = 0
    latencies = []
    for r in rows:
        raw = (r.get("raw_text") or "").strip()
        parsed = r.get("parsed_date")
        truth = sonnet_by_stem.get(r["stem"])
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
        "rate_limit_pct": 100.0 * rate_limit / total,
        "high_conf_wrong_pct": 100.0 * wrong / total,
        "median_s": median_s,
        "p95_s": p95_s,
        "imgs_per_sec": (1.0 / median_s) if median_s > 0 else 0.0,
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
                "elapsed_s": 0.0,
            }
        )
    return grouped


def render_markdown(rows_by_key: dict, metrics_by_key: dict, manifest: dict) -> str:
    frozen = manifest.get("ground_truth_frozen", False)
    header_warn = (
        ""
        if frozen
        else "> **Ground truth NOT frozen.** Metrics below compare against an "
        "unreviewed Sonnet run.\n\n"
    )
    lines = [
        "# VLM OCR Bench Report",
        "",
        f"Generated: {manifest.get('generated_at', '?')}",
        f"Corpus: {len(manifest['stems'])} stems, seed={manifest.get('seed')}",
        "",
        header_warn,
        "| Model (tag@host) | N | Agree% | HiConfWrong% | Unparsed% | Timeout% | OOM% | RateLimit% | Median s | P95 s | img/s |",
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
