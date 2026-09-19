#!/usr/bin/env python3
"""LIVECell (Sartorius phase-contrast, 1.6M+ cells) validation using OptiCell metrics
(measured numbers only).

Dataset: https://github.com/sartorius-research/LIVECell
License: CC BY-NC 4.0 (non-commercial) for images/annotations/models.

This is real, dense, label-free phase-contrast microscopy -- the "ugly real data"
case from docs/PRODUCT_ROADMAP.md Stage 5, unlike the curated single-modality
fluorescence sets already validated (BBBC039/BBBC038). Images can carry up to
roughly 2,000 annotated cells each, low contrast, touching/overlapping boundaries,
and eight morphologically distinct cell types. `images.zip` alone is ~1.3GB and
annotations are COCO-format JSON (~hundreds of MB per split), so this download is
much larger than the BBBC039/BBBC038 adapters -- budget time and disk accordingly.

Layout after extracting images.zip (per the LIVECell README):
  images/
      livecell_test_images/<Cell Type>/<Cell Type>_Phase_<Well>_<Loc>_<Time>_<Crop>.tif
      livecell_train_val_images/<Cell Type>/...

Annotations are Microsoft COCO Object Detection-format. `segmentation` is usually a
list of polygons (decoded here with only numpy+cv2, no extra dependency); if an
annotation instead uses compressed RLE (a `counts` string), this script defers to
`pycocotools` for that one case rather than reimplementing its exact codec from
scratch -- a from-scratch RLE decoder that's subtly wrong would produce plausible
but incorrect masks, which is unacceptable for a "measured only" validation script.

Usage:
  python scripts/run_livecell_validation.py --max-images 20 --backend threshold
  python scripts/run_livecell_validation.py --max-images 20 --backend hybrid --gpu --skip-download
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
from ensemble import hybrid_threshold_cellpose, fov_confidence, suggest_backend  # noqa: E402

LIVECELL_IMAGES_URL = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip"
LIVECELL_ANNOTATIONS = {
    "train": "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json",
    "val": "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json",
    "test": "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json",
}


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
    print(f"[download] {url} -> {dest} (this one is large, be patient)")
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


def build_image_index(images_root: Path) -> dict[str, Path]:
    """filename -> full path, built once by walking the extracted images tree."""
    index: dict[str, Path] = {}
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        for p in images_root.rglob(ext):
            index.setdefault(p.name, p)
    return index


def _polygon_to_mask(polygon: list, height: int, width: int) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    pts = np.round(pts).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _rle_to_mask(rle: dict, height: int, width: int) -> np.ndarray:
    counts = rle.get("counts")
    size = rle.get("size", [height, width])
    if isinstance(counts, (list, tuple)):
        h, w = size
        flat = np.zeros(h * w, dtype=np.uint8)
        idx = 0
        val = 0
        for c in counts:
            flat[idx : idx + c] = val
            idx += c
            val = 1 - val
        return flat.reshape((w, h)).T.astype(np.uint8)  # COCO uncompressed RLE is column-major
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "This annotation uses compressed COCO RLE segmentation, which needs "
            "pycocotools to decode correctly (pip install pycocotools). Not "
            "reimplemented here from scratch to avoid a subtly-wrong codec."
        ) from exc
    return np.asarray(mask_utils.decode(rle), dtype=np.uint8)


def decode_coco_instance_masks(anns: list, height: int, width: int) -> np.ndarray:
    """Paint each annotation's segmentation into one int32 instance-label array.

    Painted in listed order (a later instance overwrites an earlier one at any
    overlap). LIVECell polygons are per-cell boundaries and rarely overlap in
    practice, but -- unlike BBBC038 -- that is not a guarantee of the format, so
    treat this as a documented approximation, not an exact reconstruction.
    """
    labels = np.zeros((height, width), dtype=np.int32)
    next_id = 1
    for ann in anns:
        seg = ann.get("segmentation")
        if not seg:
            continue
        if isinstance(seg, list):
            mask = np.zeros((height, width), dtype=np.uint8)
            for poly in seg:
                if len(poly) >= 6:
                    mask |= _polygon_to_mask(poly, height, width)
        elif isinstance(seg, dict):
            mask = _rle_to_mask(seg, height, width)
        else:
            continue
        fg = mask > 0
        if not fg.any():
            continue
        labels[fg & (labels == 0)] = next_id
        next_id += 1
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LIVECell (real, dense phase-contrast) OptiCell validation (real metrics only)"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/livecell"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument(
        "--backend", choices=("auto", "threshold", "adaptive", "cellpose", "hybrid"), default="threshold"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/livecell_validation"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--cellpose-model", default="cpsam")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    raw_dir = data_dir / "raw"
    images_archive = raw_dir / "images.zip"
    images_root = data_dir / "images"
    ann_path = raw_dir / f"livecell_coco_{args.split}.json"

    if not args.skip_download:
        download(LIVECELL_IMAGES_URL, images_archive)
        unzip(images_archive, images_root)
        download(LIVECELL_ANNOTATIONS[args.split], ann_path)
    else:
        if not images_root.exists() or not ann_path.exists():
            print(
                "ERROR: --skip-download set but images/ or the annotation JSON is missing",
                file=sys.stderr,
            )
            return 2

    print(f"[paths] images_root={images_root} annotations={ann_path}")
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    ordered_image_ids = sorted(images_by_id, key=lambda i: images_by_id[i]["file_name"])
    print(f"[pair] images_in_split={len(ordered_image_ids)} total_annotations={len(coco['annotations'])}")

    index = build_image_index(images_root)
    print(f"[index] indexed {len(index)} image files under {images_root}")

    ordered_image_ids = ordered_image_ids[: max(1, args.max_images)]
    print(f"[run] backend={args.backend} n_images={len(ordered_image_ids)} split={args.split} gpu={args.gpu}")
    print(f"[env] cellpose_available={_HAS_CELLPOSE} import_error={_CELLPOSE_IMPORT_ERROR!r}")

    needs_cellpose = args.backend in ("cellpose", "hybrid", "auto")
    cellpose_seg: CellposeSegmenter | None = None
    if needs_cellpose:
        if not _HAS_CELLPOSE:
            print(f"ERROR: Cellpose not importable. Detail: {_CELLPOSE_IMPORT_ERROR}", file=sys.stderr)
            return 5
        print(f"[cellpose] loading model={args.cellpose_model!r} gpu={args.gpu} (once)...")
        cellpose_seg = CellposeSegmenter(model_type=args.cellpose_model, gpu=bool(args.gpu))
        _ = cellpose_seg.model
        print("[cellpose] model ready")

    pred_labels: list = []
    truth_labels: list = []
    per_image: list = []
    missing_files: list = []
    truth_instance_counts: list = []

    for i, image_id in enumerate(ordered_image_ids, 1):
        meta = images_by_id[image_id]
        file_name = meta["file_name"]
        path = index.get(file_name)
        if path is None:
            missing_files.append(file_name)
            print(f"  [{i}/{len(ordered_image_ids)}] SKIP missing-file {file_name}")
            continue
        try:
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise IOError(f"could not read {path}")
            if raw.ndim == 3 and raw.shape[2] >= 3:
                raw = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
            gray = to_grayscale_uint8(raw)
            height, width = int(meta.get("height", gray.shape[0])), int(meta.get("width", gray.shape[1]))
            if (height, width) != gray.shape:
                height, width = gray.shape  # trust the actual decoded image over stale JSON metadata

            truth = decode_coco_instance_masks(anns_by_image.get(image_id, []), height, width)
            truth_count = int(truth.max())
            truth_instance_counts.append(truth_count)

            backend_for_image = args.backend
            backend_reason = f"user requested {args.backend}"
            if args.backend == "auto":
                backend_for_image, backend_reason = suggest_backend(
                    gray, cellpose_available=cellpose_seg is not None
                )
            if backend_for_image == "threshold":
                seg = segment_threshold(gray)
            elif backend_for_image == "adaptive":
                seg = segment_threshold(gray, adaptive=True)
            elif backend_for_image == "cellpose":
                if cellpose_seg is None:
                    seg = segment_threshold(gray)
                    backend_reason = "auto: Cellpose unavailable → threshold"
                else:
                    seg = cellpose_seg.segment(gray)
            else:
                seg = hybrid_threshold_cellpose(gray, cellpose_segmenter=cellpose_seg)

            pred = seg.labels
            metrics = paired_segmentation_metrics(pred, truth)
            conf = fov_confidence(gray, pred)
            row = {
                "index": i,
                "file_name": file_name,
                "cell_type": file_name.split("_")[0] if "_" in file_name else "",
                "pred_count": int(seg.count),
                "truth_count": truth_count,
                "backend_requested": args.backend,
                "backend_resolved": backend_for_image,
                "backend_reason": backend_reason,
                "method": seg.method,
                "confidence_score": float(conf["confidence_score"]),
                "confidence_flags": str(conf["flags"]),
                **{k: float(v) for k, v in metrics.items()},
            }
            per_image.append(row)
            pred_labels.append(pred)
            truth_labels.append(truth)
            print(
                f"  [{i}/{len(ordered_image_ids)}] {file_name}: "
                f"truth={truth_count} pred={int(seg.count)} "
                f"iou={metrics['iou']:.3f} dice={metrics['dice']:.3f} "
                f"f1={metrics['f1']:.3f} count_err={metrics['absolute_count_error']:.0f} "
                f"conf={conf['confidence_score']:.0f}"
            )
        except Exception as exc:
            print(f"  [{i}/{len(ordered_image_ids)}] FAIL {file_name}: {exc}")

    if not pred_labels:
        print("ERROR: no successful segmentations", file=sys.stderr)
        return 4

    if missing_files:
        print(f"[note] {len(missing_files)} annotated file(s) not found under images_root -> {missing_files[:10]}")

    summary = benchmark_segmentation(pred_labels, truth_labels)
    payload = {
        "dataset": "LIVECell",
        "source": "https://github.com/sartorius-research/LIVECell",
        "license": "CC BY-NC 4.0 (non-commercial)",
        "split": args.split,
        "backend": args.backend,
        "cellpose_model": args.cellpose_model if needs_cellpose else None,
        "gpu": bool(args.gpu),
        "n_requested": len(ordered_image_ids),
        "n_scored": len(pred_labels),
        "n_missing_files": len(missing_files),
        "mean_truth_instances_per_image": (
            float(np.mean(truth_instance_counts)) if truth_instance_counts else 0.0
        ),
        "max_truth_instances_per_image": (
            int(np.max(truth_instance_counts)) if truth_instance_counts else 0
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "script": "scripts/run_livecell_validation.py",
            "repo": "Virelion-Biotech/Virelion-OptiCell",
        },
        "summary": {k: float(v) for k, v in summary.items()},
        "per_image": per_image,
        "note": (
            "Measured only. Dense, real phase-contrast data -- ground truth decoded "
            "from COCO polygons (or RLE via pycocotools if present), painted in "
            "listed order at any overlap, unlike BBBC038's guaranteed non-overlapping "
            "masks. CC BY-NC 4.0: non-commercial use only."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"livecell_{args.split}_{args.backend}_n{len(pred_labels)}.json"
    out_csv = out_dir / f"livecell_{args.split}_{args.backend}_n{len(pred_labels)}.csv"
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
    print(
        f"\nMean/max truth instances per image: "
        f"{payload['mean_truth_instances_per_image']:.1f} / {payload['max_truth_instances_per_image']}"
    )
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
