#!/usr/bin/env python3
"""Subset validation on BBBC039 (U2OS nuclei) using OptiCell metrics.

BBBC039 layout after unzip (nested):
  data/bbbc039/images/images/*.tif   (16-bit Hoechst FOVs)
  data/bbbc039/masks/masks/*.png     (color-encoded instance masks)

Mask decode (official gist pattern, RGB-aware):
  https://gist.github.com/jccaicedo/15e811722fca51e3ae90e8b43057f075
  skimage loads RGB → channel 0 = red. OpenCV loads BGR, so we convert first.
  Fallback: any non-zero across channels if channel-0 is empty.

Usage:
  python scripts/run_bbbc039_validation.py --max-images 50 --skip-download
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
    if not root.exists():
        raise FileNotFoundError(root)

    nested = root / kind
    if nested.is_dir():
        return nested.resolve()

    candidates: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir() or "__MACOSX" in p.parts:
            continue
        has_tif = any(p.glob("*.tif")) or any(p.glob("*.tiff"))
        has_png = any(p.glob("*.png"))
        if kind == "images" and has_tif:
            candidates.append(p)
        if kind == "masks" and has_png:
            candidates.append(p)
    if not candidates:
        return root.resolve()
    candidates.sort(key=lambda x: (len(x.parts), str(x)))
    return candidates[0].resolve()


def decode_bbbc039_mask(path: Path) -> np.ndarray:
    """Decode color-encoded BBBC039 PNG into integer instance labels.

    Official gist (skimage RGB):
        gt = imread(png); gt = gt[:,:,0]; gt = label(gt)

    OpenCV loads BGR, so convert to RGB before taking channel 0.
    If channel 0 is empty, fall back to any-channel non-zero, then to
    unique-color labeling (handles multi-channel color encodings).
    """
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read mask {path}")

    if arr.ndim == 2:
        channel = arr
        binary = (channel > 0).astype(np.uint8)
        _n, labels = cv2.connectedComponents(binary, connectivity=8)
        return labels.astype(np.int32)

    # BGR → RGB to match skimage channel-0 semantics
    rgb = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    channel0 = rgb[:, :, 0]
    binary = (channel0 > 0).astype(np.uint8)

    if int(binary.sum()) == 0:
        # Channel 0 empty: any non-zero across RGB
        binary = (rgb.max(axis=2) > 0).astype(np.uint8)

    if int(binary.sum()) == 0:
        return np.zeros(arr.shape[:2], dtype=np.int32)

    # Prefer connected components on the binary map (gist behavior)
    _n, labels = cv2.connectedComponents(binary, connectivity=8)

    # If almost everything collapsed to 1 blob but many colors exist,
    # re-label by unique RGB triplets (touching nuclei painted different colors).
    n_cc = int(labels.max())
    flat = rgb.reshape(-1, 3)
    # sample unique colors excluding pure black
    nonzero = flat[np.any(flat > 0, axis=1)]
    if nonzero.size:
        # approx unique count via view
        view = nonzero.view([("r", nonzero.dtype), ("g", nonzero.dtype), ("b", nonzero.dtype)])
        n_colors = int(np.unique(view).size)
    else:
        n_colors = 0

    if n_colors > max(n_cc, 1) * 2 and n_colors > 5:
        # Unique-color instance map
        h, w = rgb.shape[:2]
        packed = (
            rgb[:, :, 0].astype(np.int32) * 256 * 256
            + rgb[:, :, 1].astype(np.int32) * 256
            + rgb[:, :, 2].astype(np.int32)
        )
        unique_vals = np.unique(packed)
        labels = np.zeros((h, w), dtype=np.int32)
        next_id = 1
        for v in unique_vals:
            if v == 0:
                continue
            labels[packed == v] = next_id
            next_id += 1

    return labels.astype(np.int32)


def load_image_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read image {path}")
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    return to_grayscale_uint8(arr)


def relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def find_pairs(images_root: Path, masks_root: Path) -> list[tuple[Path, Path]]:
    img_root = resolve_content_root(images_root, "images")
    msk_root = resolve_content_root(masks_root, "masks")
    print(f"[paths] images_root={img_root}")
    print(f"[paths] masks_root={msk_root}")

    image_files: list[Path] = []
    for ext in ("*.tif", "*.tiff"):
        image_files.extend(p for p in img_root.glob(ext) if p.is_file())
        image_files.extend(p for p in img_root.rglob(ext) if p.is_file())
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

    print(
        f"[pair] images={len(image_files)} masks={len(mask_by_stem)} "
        f"paired={len(pairs)} missing_mask={missing}"
    )
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
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--backend", choices=("threshold", "adaptive", "cellpose"), default="threshold")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/bbbc039_validation"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

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
        return 3

    pairs = pairs[: max(1, args.max_images)]
    print(f"[run] backend={args.backend} n_images={len(pairs)}")

    # Sanity-check first mask decode
    sample_truth = decode_bbbc039_mask(pairs[0][1])
    print(
        f"[mask-check] {pairs[0][1].name}: shape={sample_truth.shape} "
        f"max_label={int(sample_truth.max())} fg_pixels={int((sample_truth > 0).sum())}"
    )
    if int(sample_truth.max()) == 0:
        print("WARNING: first mask decoded empty — check PNG encoding", file=sys.stderr)

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
                "image": relpath(img_path, data_dir),
                "mask": relpath(mask_path, data_dir),
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
                f"truth={int(truth.max())} pred={int(seg.count)} "
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

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"bbbc039_{args.backend}_n{len(pred_labels)}.json"
    out_csv = out_dir / f"bbbc039_{args.backend}_n{len(pred_labels)}.csv"
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
