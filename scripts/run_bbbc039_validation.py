#!/usr/bin/env python3
"""Subset validation on BBBC039 (U2OS nuclei) using OptiCell metrics.

This script:
  1. Downloads BBBC039 images + masks if missing (~80 MB).
  2. Builds instance label maps from per-nucleus PNG masks.
  3. Runs OptiCell threshold segmentation (and optional Cellpose).
  4. Scores with validation.benchmark_segmentation / paired metrics.
  5. Writes only measured numbers to JSON + CSV.

No invented metrics. Scale with --max-images after the pipeline works.

Usage (from repo root, venv active):

  pip install -e .
  python scripts/run_bbbc039_validation.py --max-images 50
  python scripts/run_bbbc039_validation.py --max-images 200 --backend cellpose

Data source: https://bbbc.broadinstitute.org/BBBC039
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Allow running from repo root without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation import benchmark_segmentation, paired_segmentation_metrics  # noqa: E402
from qc_pipeline import to_grayscale_uint8, segment_threshold  # noqa: E402

BBBC039_IMAGES = "https://data.broadinstitute.org/bbbc/BBBC039/images.zip"
BBBC039_MASKS = "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size} bytes)")
        return
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"[ok] {dest.name} sha256={sha256_file(dest)}")


def unzip(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"[skip] already extracted under {out_dir}")
        return
    print(f"[extract] {archive.name} -> {out_dir}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(out_dir)
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")


def masks_to_instance_labels(mask_dir: Path) -> np.ndarray:
    """Merge per-nucleus binary PNGs into a single integer label image."""
    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No PNG masks in {mask_dir}")
    first = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise IOError(f"Could not read {mask_files[0]}")
    labels = np.zeros(first.shape, dtype=np.int32)
    next_id = 1
    for mf in mask_files:
        m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != labels.shape:
            raise ValueError(f"Mask shape mismatch: {mf}")
        binary = m > 0
        if not binary.any():
            continue
        # Avoid overwriting existing labels if masks accidentally overlap
        free = binary & (labels == 0)
        if free.any():
            labels[free] = next_id
            next_id += 1
    return labels


def load_image_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read image {path}")
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return to_grayscale_uint8(arr)


def find_pairs(images_root: Path, masks_root: Path) -> list[tuple[Path, Path]]:
    """Pair image folders with corresponding mask folders by stem name."""
    # BBBC039 layout after extract is typically flat or one level deep;
    # support both: images/*.tif and images/<id>/*.tif style.
    image_files: list[Path] = []
    for ext in ("*.tif", "*.tiff", "*.png"):
        image_files.extend(images_root.rglob(ext))
    image_files = sorted({p.resolve() for p in image_files if p.is_file()})

    pairs: list[tuple[Path, Path]] = []
    for img in image_files:
        stem = img.stem
        # Common patterns: masks/<stem>/ or masks/<stem>_masks/ or same stem dir
        candidates = [
            masks_root / stem,
            masks_root / f"{stem}_masks",
            masks_root / img.parent.name,
        ]
        # Also: masks next to image if nested
        candidates.append(img.parent / "masks")
        mask_dir = next((c for c in candidates if c.is_dir() and any(c.glob("*.png"))), None)
        if mask_dir is None:
            # Try any subdir under masks_root whose name contains stem
            for d in masks_root.rglob("*"):
                if d.is_dir() and stem in d.name and any(d.glob("*.png")):
                    mask_dir = d
                    break
        if mask_dir is None:
            continue
        pairs.append((img, mask_dir))
    return pairs


def segment(image_gray: np.ndarray, backend: str):
    if backend == "threshold":
        return segment_threshold(image_gray)
    if backend == "adaptive":
        return segment_threshold(image_gray, adaptive=True)
    if backend == "cellpose":
        from qc_pipeline import CellposeSegmenter

        seg = CellposeSegmenter(model_type="nuclei", gpu=False)
        return seg.segment(image_gray)
    raise ValueError(f"Unknown backend: {backend}")


def main() -> int:
    parser = argparse.ArgumentParser(description="BBBC039 OptiCell validation (real metrics only)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/bbbc039"))
    parser.add_argument("--max-images", type=int, default=50, help="Cap for first honest subset run")
    parser.add_argument("--backend", choices=("threshold", "adaptive", "cellpose"), default="threshold")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/bbbc039_validation"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    raw_dir = data_dir / "raw"
    images_zip = raw_dir / "images.zip"
    masks_zip = raw_dir / "masks.zip"
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not args.skip_download:
        download(BBBC039_IMAGES, images_zip)
        download(BBBC039_MASKS, masks_zip)
        unzip(images_zip, images_dir)
        unzip(masks_zip, masks_dir)
    else:
        if not images_dir.exists() or not masks_dir.exists():
            print("ERROR: --skip-download set but images/masks folders missing", file=sys.stderr)
            return 2

    pairs = find_pairs(images_dir, masks_dir)
    if not pairs:
        # Fallback: list structure for debugging
        print("ERROR: could not pair any images with mask folders.", file=sys.stderr)
        print(f"  images under {images_dir}: {len(list(images_dir.rglob('*')))} entries")
        print(f"  masks under {masks_dir}: {len(list(masks_dir.rglob('*')))} entries")
        print("  Top-level images:", [p.name for p in list(images_dir.iterdir())[:20]])
        print("  Top-level masks:", [p.name for p in list(masks_dir.iterdir())[:20]])
        return 3

    pairs = pairs[: max(1, args.max_images)]
    print(f"[run] backend={args.backend} n_images={len(pairs)}")

    pred_labels: list[np.ndarray] = []
    truth_labels: list[np.ndarray] = []
    per_image: list[dict] = []

    for i, (img_path, mask_dir) in enumerate(pairs, 1):
        try:
            gray = load_image_gray(img_path)
            truth = masks_to_instance_labels(mask_dir)
            if truth.shape != gray.shape:
                # Resize labels to image if needed (rare)
                truth = cv2.resize(truth.astype(np.float32), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            seg = segment(gray, args.backend)
            pred = seg.labels
            metrics = paired_segmentation_metrics(pred, truth)
            row = {
                "index": i,
                "image": str(img_path.relative_to(data_dir)),
                "mask_dir": str(mask_dir.relative_to(data_dir)),
                "pred_count": int(seg.count),
                "truth_count": int(truth.max()),
                "method": seg.method,
                **{k: float(v) for k, v in metrics.items()},
            }
            per_image.append(row)
            pred_labels.append(pred)
            truth_labels.append(truth)
            print(
                f"  [{i}/{len(pairs)}] {img_path.name}: "
                f"iou={metrics['iou']:.3f} dice={metrics['dice']:.3f} "
                f"f1={metrics['f1']:.3f} count_err={metrics['absolute_count_error']:.0f}"
            )
        except Exception as exc:
            print(f"  [{i}/{len(pairs)}] FAIL {img_path.name}: {exc}")

    if not pred_labels:
        print("ERROR: no successful segmentations", file=sys.stderr)
        return 4

    summary = benchmark_segmentation(pred_labels, truth_labels)
    payload = {
        "dataset": "BBBC039",
        "source": "https://bbbc.broadinstitute.org/BBBC039",
        "backend": args.backend,
        "n_requested": len(pairs),
        "n_scored": len(pred_labels),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "script": "scripts/run_bbbc039_validation.py",
            "repo": "Virelion-Biotech/Virelion-OptiCell",
        },
        "summary": {k: float(v) for k, v in summary.items()},
        "per_image": per_image,
        "note": "All numbers measured on the listed images only. Not a full-corpus claim.",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"bbbc039_{args.backend}_n{len(pred_labels)}.json"
    out_csv = args.out_dir / f"bbbc039_{args.backend}_n{len(pred_labels)}.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Minimal CSV without pandas dependency requirement beyond what OptiCell already uses
    if per_image:
        keys = list(per_image[0].keys())
        lines = [",".join(keys)]
        for row in per_image:
            lines.append(",".join(str(row[k]) for k in keys))
        out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== SUMMARY (measured only) ===")
    for k, v in sorted(summary.items()):
        print(f"  {k}: {v}")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
