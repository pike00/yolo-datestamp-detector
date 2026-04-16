#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""
Rank images from a benchmark_iqa.py CSV by a composite score.

Composite method: per-metric rank (1 = best), averaged across all valid
metrics for each image. This is robust against metrics that live on
different scales and handles missing / errored values gracefully.

Usage:
    ./rank_composite.py                                # top 50 from data/quality_results.csv
    ./rank_composite.py --top 20
    ./rank_composite.py --csv data/quality_results.csv --top 50 --out data/top50.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Direction per metric: True = lower is better.
LOWER_BETTER = {
    "brisque": True,
    "niqe": True,
    "laplacian": False,
    "musiq": False,
    "nima": False,
    "clipiqa": False,
    "topiq_nr": False,
}


def parse_score(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("data/quality_results.csv"))
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None,
                    help="Write ranked CSV here (default: print only)")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} does not exist", file=sys.stderr)
        return 2

    with args.csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        metrics = [c for c in reader.fieldnames or [] if c != "filename"]

    if not rows:
        print("ERROR: empty CSV", file=sys.stderr)
        return 2

    # For each metric, build a list of (filename, score) with valid scores,
    # sort by direction, assign ranks. Missing/errored values get no rank.
    ranks: dict[str, dict[str, int]] = {m: {} for m in metrics}
    for m in metrics:
        scored = [(r["filename"], parse_score(r[m])) for r in rows]
        scored = [(fn, s) for fn, s in scored if s is not None]
        lower_better = LOWER_BETTER.get(m, False)
        scored.sort(key=lambda x: x[1], reverse=not lower_better)
        for rank, (fn, _) in enumerate(scored, start=1):
            ranks[m][fn] = rank

    # Composite = mean of per-metric ranks across metrics that scored this file.
    composite: list[dict] = []
    for r in rows:
        fn = r["filename"]
        per_metric_ranks = [ranks[m][fn] for m in metrics if fn in ranks[m]]
        if not per_metric_ranks:
            continue
        avg_rank = sum(per_metric_ranks) / len(per_metric_ranks)
        row_out = {
            "filename": fn,
            "composite_rank": round(avg_rank, 2),
            "n_metrics": len(per_metric_ranks),
        }
        for m in metrics:
            row_out[m] = r[m]
            row_out[f"{m}_rank"] = ranks[m].get(fn, "")
        composite.append(row_out)

    composite.sort(key=lambda x: x["composite_rank"])
    top = composite[: args.top]

    # Pretty print
    print(f"Top {len(top)} by composite rank (lower = better)")
    print(f"Metrics used: {metrics}")
    print()
    header = f"{'rank':>4}  {'composite':>9}  {'filename':<30}  " + "  ".join(
        f"{m:>9}" for m in metrics)
    print(header)
    print("-" * len(header))
    for i, r in enumerate(top, start=1):
        vals = "  ".join(f"{str(r[m]):>9}" for m in metrics)
        print(f"{i:>4}  {r['composite_rank']:>9.2f}  {r['filename']:<30}  {vals}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["filename", "composite_rank", "n_metrics"] + [
            x for m in metrics for x in (m, f"{m}_rank")
        ]
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(composite)
        print(f"\nWrote full ranked CSV to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
