#!/usr/bin/env python3
"""Subset validation on BBBC039 (U2OS nuclei) using OptiCell metrics.

BBBC039 layout after unzip (nested):
  data/bbbc039/images/images/*.tif   (16-bit Hoechst FOVs)
  data/bbbc039/masks/masks/*.png     (color-encoded instance masks)

Mask decode (official gist pattern):
  take channel 0, then connected-component label non-zero pixels.
  https://gist.github.com/jccaicedo/15e811722fca51e3ae90e8b43057f075

Usage (from repo root, venv active):

  pip install -e .
  python scripts/run_bbbc039_validation.py --max-images 50
  python scripts/run_bbbc039_validation.py --max-images 200 --backend cellpose

If you already extracted data, re-run with the same --data-dir (re-extract
is skipped via .extracted marker; delete that marker only if you need a
force re-extract).
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


def resolve_content_root(root: Path, kind: str) -> Path:
    """Handle nested zip folders: images/images/, masks/masks/, ignore __MACOSX."""
    if not root.exists():
        raise FileNotFoundError(root)

    # Prefer explicit nested folder named like the kind
    nested = root / kind
    if nested.is_dir():
        return nested

    # Otherwise pick the directory that actually holds the files
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if "__MACOSX" in p.parts:
            continue
        has_tif = any(p.glob("*.tif")) or any(p.glob("*.tiff"))
        has_png = any(p.glob("*.png"))
        if kind == "images" and has_tif:
            candidates.append(p)
        if kind == "masks" and has_png:
            candidates.append(p)
    if not candidates:
        return root
    # Prefer shallower paths
    candidates.sort(key=lambda x: (len(x.parts), str(x)))
    return candidates[0]


def decode_bbbc039_mask(path: Path) -> np.ndarray:
    """Decode color-encoded BBBC039 PNG into integer instance labels.

    Official pattern (Caicedo gist):
      gt = imread(png); gt = gt[:,:,0]; gt = label(gt)
    Implemented with OpenCV only (no skimage dependency).
    """
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read mask {path}")
    if arr.ndim == 3:
        # Keep first channel (BGR or RGB — same index 0 in OpenCV BGR load)
        channel = arr[:, :, 0]
    else:
        channel = arr
    # Non-zero pixels form instance seeds; connected components give instance IDs
    binary = (channel > 0).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    # labels already 0=bg, 1..n-1 = instances
    return labels.astype(np.int32)


def load_image_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read image {path}")
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    return to_grayscale_uint8(arr)


def find_pairs(images_root: Path, masks_root: Path) -> list[tuple[Path, Path]]:
    """Pair TIFF images to PNG masks by matching basename (stem)."""
    img_root = resolve_content_root(images_root, "images")
    msk_root = resolve_content_root(masks_root, "masks")
    print(f"[paths] images_root={img_root}")
    print(f"[paths] masks_root={msk_root}")

    image_files: list[Path] = []
    for ext in ("*.tif", "*.tiff"):
        image_files.extend(p for p in img_root.glob(ext) if p.is_file())
        image_files.extend(p for p in img_root.rglob(ext) if p.is_file())
    # de-dupe
    image_files = sorted({p.resolve() for p in image_files})

    mask_by_stem: dict[str, Path] = {}
    for p in list(msk_root.glob("*.png")) + list(msk_root.rglob("*.png")):
        if not p.is_file() or "__MACOSX" in p.parts:
            continue
        mask_by_stem[p.stem] = p.resolve()

    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for img in image_files:
        if "__MACOSX" in img.parts:
            continue
        m = mask_by_stem.get(img.stem)
        if m is None:
            missing += 1
            continue
        pairs.append((img, m))

    print(f"[pair] images={len(image_files)} masks={len(mask_by_stem)} paired={len(pairs)} missing_mask={missing}")
    if pairs:
        print(f"[pair] example: {pairs[0][0].name} <-> {pairs[0][1].name}")
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
        print("ERROR: could not pair any images with masks.", file=sys.stderr)
        print(f"  images under {images_dir}: {len(list(images_dir.rglob('*')))} entries")
        print(f"  masks under {masks_dir}: {len(list(masks_dir.rglob('*')))} entries")
        print("  Top-level images:", [p.name for p in list(images_dir.iterdir())[:20]])
        print("  Top-level masks:", [p.name for p in list(masks_dir.iterdir())[:20]])
        # deeper peek
        for sub in ("images", "masks"):
            d = images_dir / sub if sub == "images" else masks_dir / sub
            if d.is_dir():
                print(f"  Sample under {d}:", [p.name for p in list(d.iterdir())[:8]])
        return 3

    pairs = pairs[: max(1, args.max_images)]
    print(f"[run] backend={args.backend} n_images={len(pairs)}")

    pred_labels: list[np.ndarray] = []
    truth_labels: list[np.ndarray] = []
    per_image: list[dict] = []

    for i, (img_path, mask_path) in enumerate(pairs, 1):
        try:
            gray = load_image_gray(img_path)
            truth = decode_bbbc039_mask(mask_path)
            if truth.shape != gray.shape:
                truth = cv2.resize(
                    truth.astype(np.float32),
                    (gray.shape[1], gray.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
            seg = segment(gray, args.backend)
            pred = seg.labels
            metrics = paired_segmentation_metrics(pred, truth)
            row = {
                "index": i,
                "image": str(img_path.relative_to(data_dir)),
                "mask": str(mask_path.relative_to(data_dir)),
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
