#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyiqa>=0.1.13",
#     "torch>=2.2",
#     "torchvision>=0.17",
#     "opencv-python>=4.9",
#     "pillow>=10",
#     "numpy>=1.26",
#     "tqdm>=4.66",
# ]
# ///
"""
Benchmark several no-reference image quality / aesthetic models on a folder of
images and write a CSV with per-image scores. Designed for triaging scanned
photos — quickly see which images each metric ranks best / worst, then decide
which metric(s) to actually use for filtering.

Usage:
    ./benchmark_iqa.py                              # defaults to data/samples
    ./benchmark_iqa.py --src data/samples --out data/quality_results.csv
    ./benchmark_iqa.py --metrics brisque,niqe,musiq
    ./benchmark_iqa.py --device cuda                # if GPU available

Metric reference (all no-reference, no ground truth needed):
    laplacian   custom    blur proxy (variance of Laplacian); higher = sharper
    brisque     classical fast, no weights, lower = better
    niqe        classical fast, no weights, lower = better
    musiq       deep      multi-scale transformer, higher = better
    nima        deep      aesthetic+technical, higher = better
    clipiqa     deep      CLIP-based, higher = better
    topiq_nr    deep      modern SOTA-ish, higher = better

Adding a deep metric the first time downloads weights (~100MB-600MB each)
into ~/.cache/torch/hub. Run once with --metrics brisque,niqe,laplacian if
you just want a fast sanity pass.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# pyiqa metric name -> "higher is better" flag. pyiqa exposes this via
# .lower_better but we hardcode for the curated set so the CSV is annotated
# consistently even if a model fails to load.
DEEP_METRICS = {
    "brisque":  {"lower_better": True},
    "niqe":     {"lower_better": True},
    "musiq":    {"lower_better": False},
    "nima":     {"lower_better": False},
    "clipiqa":  {"lower_better": False},
    "topiq_nr": {"lower_better": False},
}

DEFAULT_METRICS = ["laplacian", "brisque", "niqe", "musiq", "clipiqa"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def laplacian_variance(path: Path) -> float:
    """Cheap blur proxy. Higher = sharper. Works on grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return float("nan")
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def load_pyiqa_metric(name: str, device: str):
    import pyiqa
    return pyiqa.create_metric(name, device=device)


def score_with_pyiqa(metric, path: Path) -> float:
    # pyiqa metrics accept either a path string or a torch tensor.
    # Path form handles preprocessing (resize/normalize) internally.
    with torch.no_grad():
        out = metric(str(path))
    if isinstance(out, torch.Tensor):
        return float(out.detach().cpu().item())
    return float(out)


def collect_images(src: Path, skip_pattern: str | None) -> list[Path]:
    files = [p for p in sorted(src.iterdir())
             if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if skip_pattern:
        files = [p for p in files if skip_pattern not in p.name]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/samples"),
                    help="Directory of images to score (default: data/samples)")
    ap.add_argument("--out", type=Path, default=Path("data/quality_results.csv"),
                    help="Output CSV path")
    ap.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS),
                    help=f"Comma-separated metrics. Available: laplacian,"
                         f"{','.join(DEEP_METRICS)}")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-pattern", type=str, default="_boxes",
                    help="Skip files containing this substring (default: _boxes)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only score first N images (0 = all)")
    args = ap.parse_args()

    src = args.src.resolve()
    if not src.is_dir():
        print(f"ERROR: src not a directory: {src}", file=sys.stderr)
        return 2

    requested = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in requested if m != "laplacian" and m not in DEEP_METRICS]
    if unknown:
        print(f"ERROR: unknown metrics: {unknown}", file=sys.stderr)
        return 2

    images = collect_images(src, args.skip_pattern)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"ERROR: no images found in {src}", file=sys.stderr)
        return 2

    print(f"Scoring {len(images)} images from {src}")
    print(f"Device: {args.device}")
    print(f"Metrics: {requested}")

    # Load deep metrics up front so any download/init cost is paid once.
    loaded: dict[str, object] = {}
    for name in requested:
        if name == "laplacian":
            continue
        t0 = time.time()
        print(f"  loading {name} ...", end="", flush=True)
        try:
            loaded[name] = load_pyiqa_metric(name, args.device)
            print(f" ok ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f" FAILED: {e}")
            requested.remove(name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename"] + requested
    rows: list[dict[str, object]] = []

    t_start = time.time()
    for path in tqdm(images, desc="scoring"):
        row: dict[str, object] = {"filename": path.name}
        for name in requested:
            try:
                if name == "laplacian":
                    row[name] = round(laplacian_variance(path), 4)
                else:
                    row[name] = round(score_with_pyiqa(loaded[name], path), 4)
            except Exception as e:
                row[name] = f"ERR:{type(e).__name__}"
        rows.append(row)

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    elapsed = time.time() - t_start
    per = elapsed / max(len(images), 1)
    print(f"\nWrote {args.out} ({len(rows)} rows) in {elapsed:.1f}s "
          f"({per * 1000:.0f} ms/image)")

    # Quick top/bottom preview per metric, so a single run is informative
    # without needing to open the CSV.
    print("\n=== Top 3 / Bottom 3 per metric ===")
    for name in requested:
        meta = {"lower_better": False} if name == "laplacian" else DEEP_METRICS[name]
        valid = [(r["filename"], r[name]) for r in rows
                 if isinstance(r[name], (int, float)) and not (isinstance(r[name], float) and np.isnan(r[name]))]
        if not valid:
            print(f"\n{name}: no valid scores")
            continue
        valid.sort(key=lambda x: x[1], reverse=not meta["lower_better"])
        direction = "lower=better" if meta["lower_better"] else "higher=better"
        print(f"\n{name} ({direction})")
        print("  best:")
        for fn, s in valid[:3]:
            print(f"    {s:>10.3f}  {fn}")
        print("  worst:")
        for fn, s in valid[-3:]:
            print(f"    {s:>10.3f}  {fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
