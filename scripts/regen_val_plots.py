#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["ultralytics>=8.3", "matplotlib>=3.8"]
# ///
"""Regenerate validation plots and confidence distribution for the new GPU weights.

Runs `model.val()` with the new best.pt to produce fresh PR / F1 / confusion matrix
curves, then copies them into examples/ replacing the old CPU-run plots. Also
regenerates examples/confidence_distribution.png from state/prediction_drift.json.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "detect" / "gpu-40ep" / "weights" / "best.pt"
EXAMPLES = ROOT / "examples"
DRIFT_PATH = ROOT / "state" / "prediction_drift.json"

DATA_YAML_STAGED = ROOT / "dataset" / "data.local.yaml"


def stage_dataset() -> Path:
    """Write a sibling data.yaml pointing at the host-side dataset/ path."""
    content = (
        "names:\n  0: target\n"
        f"path: {ROOT / 'dataset'}\n"
        "train: images/train\n"
        "val: images/val\n"
    )
    DATA_YAML_STAGED.write_text(content)
    return DATA_YAML_STAGED


def resolve_val_symlinks() -> None:
    """val symlinks point at /app/scanmyphotos — rewrite them to host paths."""
    val_dir = ROOT / "dataset" / "images" / "val"
    fixed = 0
    for link in val_dir.iterdir():
        if not link.is_symlink():
            continue
        target = Path(str(link.readlink()))
        if str(target).startswith("/app/"):
            new = ROOT / str(target).removeprefix("/app/")
            link.unlink()
            link.symlink_to(new)
            fixed += 1
    if fixed:
        print(f"[val] rewrote {fixed} symlinks to host paths")


def run_val() -> Path:
    data = stage_dataset()
    resolve_val_symlinks()
    model = YOLO(str(WEIGHTS))
    results = model.val(
        data=str(data),
        imgsz=416,
        batch=16,
        device="cpu",
        plots=True,
        project=str(ROOT / "runs" / "detect"),
        name="gpu-40ep-val",
        exist_ok=True,
    )
    print("metrics:", results.results_dict)
    return ROOT / "runs" / "detect" / "gpu-40ep-val"


def copy_plots(val_dir: Path) -> None:
    mapping = {
        "PR_curve.png": "precision_recall_curve.png",
        "F1_curve.png": "f1_confidence_curve.png",
        "confusion_matrix.png": "confusion_matrix.png",
    }
    for src_name, dst_name in mapping.items():
        src = val_dir / src_name
        if src.exists():
            shutil.copy(src, EXAMPLES / dst_name)
            print(f"copied {src.name} -> examples/{dst_name}")
        else:
            print(f"MISSING: {src}")


def regen_confidence_distribution() -> None:
    drift = json.loads(DRIFT_PATH.read_text())
    new_conf = [e["new"]["confidence"] for e in drift.values() if e.get("new")]
    old_conf = [e["old"]["confidence"] for e in drift.values() if e.get("old")]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.hist(old_conf, bins=40, alpha=0.55, label=f"CPU run (n={len(old_conf)})", color="#d03c3c")
    ax.hist(new_conf, bins=40, alpha=0.75, label=f"GPU run (n={len(new_conf)})", color="#3cb878")
    ax.set_xlabel("Detection confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence distribution: CPU vs GPU model on {len(drift)} scans")
    ax.legend(loc="upper center")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    out = EXAMPLES / "confidence_distribution.png"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    val_dir = run_val()
    copy_plots(val_dir)
    regen_confidence_distribution()


if __name__ == "__main__":
    main()
