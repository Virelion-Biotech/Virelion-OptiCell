#!/usr/bin/env python3
"""BBBC038 (Kaggle 2018 Data Science Bowl nuclei) validation using OptiCell metrics
(measured numbers only).

Dataset: https://bbbc.broadinstitute.org/BBBC038
License: CC0.

Only stage1_train.zip ships ground-truth masks (stage1_test / stage2_test are the
unlabeled Kaggle competition test sets), so this script only targets stage1_train.

Layout (per the BBBC038 page): each image has an ImageId folder containing two
subfolders:
  images/  -> exactly one PNG, the raw image
  masks/   -> one PNG per nucleus, each a single non-overlapping binary mask

Usage:
  python scripts/run_bbbc038_validation.py --max-images 50 --skip-download
  python scripts/run_bbbc038_validation.py --max-images 200 --backend hybrid --gpu --skip-download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation import benchmark_segmentation, paired_segmentation_metrics  # noqa: E402
from qc_pipeline import (  # noqa: E402
    to_grayscale_uint8,
    segment_threshold,
    CellposeSegmenter,
    _HAS_CELLPOSE,
    _CELLPOSE_IMPORT_ERROR,
)
from ensemble import hybrid_threshold_cellpose, fov_confidence  # noqa: E402

BBBC038_STAGE1_TRAIN = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"


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


def _is_record_dir(p: Path) -> bool:
    return p.is_dir() and (p / "images").is_dir()


def resolve_stage1_train_root(extracted_root: Path) -> Path:
    """Find the directory that directly contains per-ImageId record folders.

    Handles both a flat `stage1_train/<id>/...` layout and an extra nesting level
    some zip tools introduce, by picking whichever parent directory contains the
    most `<id>/images/` record folders.
    """
    if not extracted_root.exists():
        raise FileNotFoundError(extracted_root)
    parents: list[Path] = []
    for p in extracted_root.rglob("*"):
        if "__MACOSX" in p.parts:
            continue
        if _is_record_dir(p):
            parents.append(p.parent)
    if not parents:
        raise FileNotFoundError(
            f"No BBBC038 ImageId folders (with an images/ subfolder) found under {extracted_root}"
        )
    best, _count = Counter(parents).most_common(1)[0]
    return best.resolve()


def find_records(stage1_root: Path) -> list[Path]:
    """Sorted per-ImageId record folders (basename sort, matches the BBBC039 script convention)."""
    records = [p for p in stage1_root.iterdir() if _is_record_dir(p)]
    records.sort(key=lambda p: p.name)
    return records


def load_record_image_gray(record: Path) -> np.ndarray:
    img_files = sorted((record / "images").glob("*.png"))
    if not img_files:
        raise IOError(f"No image PNG under {record / 'images'}")
    arr = cv2.imread(str(img_files[0]), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read image {img_files[0]}")
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    return to_grayscale_uint8(arr)


def decode_bbbc038_instance_masks(record: Path, shape: tuple[int, int]) -> np.ndarray:
    """Stack per-nucleus binary PNGs (masks/ folder) into one int32 instance-label array.

    BBBC038 guarantees masks do not overlap (no pixel belongs to two masks), so painting
    them in filename order is safe and never needs to resolve conflicting claims.
    """
    labels = np.zeros(shape, dtype=np.int32)
    mask_dir = record / "masks"
    if not mask_dir.is_dir():
        return labels  # stage1_train records always have masks/; guard anyway
    next_id = 1
    for mf in sorted(mask_dir.glob("*.png")):
        m = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.shape != shape:
            m = cv2.resize(
                m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
            )
        fg = m > 0
        if not fg.any():
            continue
        labels[fg & (labels == 0)] = next_id
        next_id += 1
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BBBC038 (2018 Data Science Bowl nuclei) OptiCell validation (real metrics only)"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/bbbc038"))
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument(
        "--backend", choices=("threshold", "adaptive", "cellpose", "hybrid"), default="threshold"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/bbbc038_validation"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--cellpose-model", default="cpsam")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--include-empty-gt",
        action="store_true",
        help="Score records whose decoded ground truth has zero objects (default: skip)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    raw_dir = data_dir / "raw"
    archive = raw_dir / "stage1_train.zip"
    extract_dir = data_dir / "stage1_train"

    if not args.skip_download:
        download(BBBC038_STAGE1_TRAIN, archive)
        unzip(archive, extract_dir)
    else:
        if not extract_dir.exists():
            print(
                "ERROR: --skip-download set but extracted stage1_train folder missing",
                file=sys.stderr,
            )
            return 2

    stage1_root = resolve_stage1_train_root(extract_dir)
    print(f"[paths] stage1_train_root={stage1_root}")

    records = find_records(stage1_root)
    if not records:
        print("ERROR: no BBBC038 ImageId records found.", file=sys.stderr)
        return 3
    print(f"[pair] records={len(records)}")

    records = records[: max(1, args.max_images)]
    print(f"[run] backend={args.backend} n_images={len(records)} gpu={args.gpu}")
    print(f"[env] cellpose_available={_HAS_CELLPOSE} import_error={_CELLPOSE_IMPORT_ERROR!r}")

    needs_cellpose = args.backend in ("cellpose", "hybrid")
    cellpose_seg: CellposeSegmenter | None = None
    if needs_cellpose:
        if not _HAS_CELLPOSE:
            print(
                f"ERROR: Cellpose not importable. Detail: {_CELLPOSE_IMPORT_ERROR}", file=sys.stderr
            )
            return 5
        print(f"[cellpose] loading model={args.cellpose_model!r} gpu={args.gpu} (once)...")
        cellpose_seg = CellposeSegmenter(model_type=args.cellpose_model, gpu=bool(args.gpu))
        _ = cellpose_seg.model
        print("[cellpose] model ready")

    sample_gray = load_record_image_gray(records[0])
    sample_truth = decode_bbbc038_instance_masks(records[0], sample_gray.shape)
    print(
        f"[mask-check] {records[0].name}: shape={sample_truth.shape} "
        f"max_label={int(sample_truth.max())} fg_pixels={int((sample_truth > 0).sum())}"
    )

    pred_labels: list[np.ndarray] = []
    truth_labels: list[np.ndarray] = []
    per_image: list[dict] = []
    skipped_empty_gt: list[str] = []

    for i, record in enumerate(records, 1):
        try:
            gray = load_record_image_gray(record)
            truth = decode_bbbc038_instance_masks(record, gray.shape)

            truth_count = int(truth.max())
            if truth_count == 0 and not args.include_empty_gt:
                skipped_empty_gt.append(record.name)
                print(f"  [{i}/{len(records)}] SKIP empty-GT {record.name}")
                continue

            if args.backend == "threshold":
                seg = segment_threshold(gray)
            elif args.backend == "adaptive":
                seg = segment_threshold(gray, adaptive=True)
            elif args.backend == "cellpose":
                assert cellpose_seg is not None
                seg = cellpose_seg.segment(gray)
            else:
                seg = hybrid_threshold_cellpose(gray, cellpose_segmenter=cellpose_seg)

            pred = seg.labels
            metrics = paired_segmentation_metrics(pred, truth)
            conf = fov_confidence(gray, pred)
            row = {
                "index": i,
                "image_id": record.name,
                "pred_count": int(seg.count),
                "truth_count": truth_count,
                "method": seg.method,
                "confidence_score": float(conf["confidence_score"]),
                "confidence_flags": str(conf["flags"]),
                **{k: float(v) for k, v in metrics.items()},
            }
            per_image.append(row)
            pred_labels.append(pred)
            truth_labels.append(truth)
            print(
                f"  [{i}/{len(records)}] {record.name}: "
                f"truth={truth_count} pred={int(seg.count)} "
                f"iou={metrics['iou']:.3f} dice={metrics['dice']:.3f} "
                f"f1={metrics['f1']:.3f} count_err={metrics['absolute_count_error']:.0f} "
                f"conf={conf['confidence_score']:.0f}"
            )
        except Exception as exc:
            print(f"  [{i}/{len(records)}] FAIL {record.name}: {exc}")

    if not pred_labels:
        print("ERROR: no successful segmentations", file=sys.stderr)
        return 4

    if skipped_empty_gt:
        print(f"[note] skipped empty-GT records: {len(skipped_empty_gt)} -> {skipped_empty_gt}")

    summary = benchmark_segmentation(pred_labels, truth_labels)
    payload = {
        "dataset": "BBBC038",
        "source": "https://bbbc.broadinstitute.org/BBBC038",
        "archive": "stage1_train.zip",
        "backend": args.backend,
        "cellpose_model": args.cellpose_model if needs_cellpose else None,
        "gpu": bool(args.gpu),
        "n_requested": len(records),
        "n_scored": len(pred_labels),
        "n_skipped_empty_gt": len(skipped_empty_gt),
        "skipped_empty_gt": skipped_empty_gt,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "script": "scripts/run_bbbc038_validation.py",
            "repo": "Virelion-Biotech/Virelion-OptiCell",
        },
        "summary": {k: float(v) for k, v in summary.items()},
        "per_image": per_image,
        "note": (
            "Measured only. Empty-GT records skipped unless --include-empty-gt. "
            "Only stage1_train ships ground-truth masks."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"bbbc038_{args.backend}_n{len(pred_labels)}.json"
    out_csv = out_dir / f"bbbc038_{args.backend}_n{len(pred_labels)}.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
