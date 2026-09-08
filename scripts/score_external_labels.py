#!/usr/bin/env python3
"""Score external segmentation labels (e.g. CellProfiler exports) against BBBC039 GT.

Stage-1 comparator path: run CellProfiler on the same FOVs, export label images,
then score with the same OptiCell metrics used for threshold/hybrid/cellpose.

Usage:
  python scripts/score_external_labels.py \
    --pred-dir path/to/cp_labels \
    --data-dir data/bbbc039 \
    --max-images 50 \
    --name cellprofiler \
    --out-dir outputs/bbbc039_validation

Prediction files must share the image stem with BBBC039 TIFFs
(e.g. IXMtest_A02_s1_....png or .tif label maps).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation import benchmark_segmentation, paired_segmentation_metrics  # noqa: E402

# Reuse BBBC039 pairing/decode from the main runner
from scripts.run_bbbc039_validation import (  # noqa: E402
    decode_bbbc039_mask,
    find_pairs,
    load_image_gray,
    relpath,
)


def load_label_map(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read prediction labels {path}")
    if arr.ndim == 3:
        # color-encoded → connected components on nonzero
        if arr.shape[2] >= 3:
            rgb = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
            binary = (rgb.max(axis=2) > 0).astype(np.uint8)
        else:
            binary = (arr[:, :, 0] > 0).astype(np.uint8)
        _n, labels = cv2.connectedComponents(binary, connectivity=8)
        return labels.astype(np.int32)
    # already integer labels or binary
    if arr.dtype == np.uint8 or arr.dtype == np.uint16:
        if int(arr.max()) <= 1:
            _n, labels = cv2.connectedComponents((arr > 0).astype(np.uint8), connectivity=8)
            return labels.astype(np.int32)
        return arr.astype(np.int32)
    return arr.astype(np.int32)


def index_predictions(pred_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in pred_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".tif", ".tiff", ".npy"}:
            continue
        if "__MACOSX" in p.parts:
            continue
        out[p.stem] = p.resolve()
        # also strip common prefixes like label_
        if p.stem.startswith("label_"):
            out[p.stem[len("label_") :]] = p.resolve()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Score external labels vs BBBC039 GT")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/bbbc039"))
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--name", default="external")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/bbbc039_validation"))
    parser.add_argument(
        "--include-empty-gt",
        action="store_true",
        help="Include FOVs with empty decoded GT (default: skip)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    pred_dir = args.pred_dir.expanduser().resolve()
    if not pred_dir.is_dir():
        print(f"ERROR: pred-dir not found: {pred_dir}", file=sys.stderr)
        return 2

    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    pairs = find_pairs(images_dir, masks_dir)
    if not pairs:
        print("ERROR: no BBBC039 pairs", file=sys.stderr)
        return 3
    pairs = pairs[: max(1, args.max_images)]
    pred_index = index_predictions(pred_dir)
    print(f"[pred] indexed {len(pred_index)} label files under {pred_dir}")

    pred_labels: list[np.ndarray] = []
    truth_labels: list[np.ndarray] = []
    per_image: list[dict] = []
    missing_pred = 0
    skipped_empty = 0

    for i, (img_path, mask_path) in enumerate(pairs, 1):
        stem = img_path.stem
        pred_path = pred_index.get(stem)
        if pred_path is None:
            missing_pred += 1
            print(f"  [{i}] no prediction for {stem}")
            continue
        try:
            gray = load_image_gray(img_path)
            truth = decode_bbbc039_mask(mask_path)
            if truth.shape != gray.shape:
                truth = cv2.resize(
                    truth.astype(np.float32),
                    (gray.shape[1], gray.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
            truth_count = int(truth.max())
            if truth_count == 0 and not args.include_empty_gt:
                skipped_empty += 1
                continue
            pred = load_label_map(pred_path)
            if pred.shape != gray.shape:
                pred = cv2.resize(
                    pred.astype(np.float32),
                    (gray.shape[1], gray.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
            metrics = paired_segmentation_metrics(pred, truth)
            row = {
                "index": i,
                "image": relpath(img_path, data_dir),
                "mask": relpath(mask_path, data_dir),
                "pred": relpath(pred_path, pred_dir),
                "pred_count": int(pred.max()),
                "truth_count": truth_count,
                "method": args.name,
                **{k: float(v) for k, v in metrics.items()},
            }
            per_image.append(row)
            pred_labels.append(pred)
            truth_labels.append(truth)
            print(
                f"  [{i}] {stem}: truth={truth_count} pred={int(pred.max())} "
                f"iou={metrics['iou']:.3f} dice={metrics['dice']:.3f} f1={metrics['f1']:.3f}"
            )
        except Exception as exc:
            print(f"  [{i}] FAIL {stem}: {exc}")

    if not pred_labels:
        print("ERROR: nothing scored", file=sys.stderr)
        return 4

    summary = benchmark_segmentation(pred_labels, truth_labels)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(pred_labels)
    payload = {
        "dataset": "BBBC039",
        "backend": args.name,
        "n_requested": len(pairs),
        "n_scored": n,
        "missing_predictions": missing_pred,
        "skipped_empty_gt": skipped_empty,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {k: float(v) for k, v in summary.items()},
        "per_image": per_image,
        "note": "Measured only. External labels scored with OptiCell validation.py.",
    }
    out_json = out_dir / f"bbbc039_{args.name}_n{n}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== SUMMARY (measured only) ===")
    for k, v in sorted(summary.items()):
        print(f"  {k}: {v}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
