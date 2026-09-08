#!/usr/bin/env python3
"""Export BBBC039 prediction label images (matching TIFF stems) for score_external_labels.

Nothing like this existed in the repo yet. This writes real label PNGs so
`--pred-dir` is a real folder — not a placeholder.

Note: this is **OptiCell** segmentation export, not CellProfiler.
For a true CellProfiler baseline, run CP GUI/CLI and export labels the same way;
then point score_external_labels at that folder instead.

Usage (Colab or laptop):
  cd /content/Virelion-OptiCell   # or your clone
  python scripts/export_bbbc039_labels.py --max-images 50 --backend threshold --skip-download
  python scripts/score_external_labels.py --pred-dir outputs/bbbc039_pred_labels/threshold --max-images 50 --name threshold_export
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qc_pipeline import (  # noqa: E402
    segment_threshold,
    CellposeSegmenter,
    _HAS_CELLPOSE,
    _CELLPOSE_IMPORT_ERROR,
)
from ensemble import hybrid_threshold_cellpose  # noqa: E402


def _load_bbbc039_helpers():
    path = ROOT / "scripts" / "run_bbbc039_validation.py"
    spec = importlib.util.spec_from_file_location("run_bbbc039_validation", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_bbbc = _load_bbbc039_helpers()
find_pairs = _bbbc.find_pairs
load_image_gray = _bbbc.load_image_gray
download = _bbbc.download
unzip = _bbbc.unzip
BBBC039_IMAGES = _bbbc.BBBC039_IMAGES
BBBC039_MASKS = _bbbc.BBBC039_MASKS


def save_labels(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # uint16 label map; score_external_labels reads this fine
    out = np.clip(labels, 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), out):
        raise IOError(f"Failed to write {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BBBC039 pred labels for scorer")
    parser.add_argument("--data-dir", type=Path, default=Path("data/bbbc039"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: outputs/bbbc039_pred_labels/<backend>",
    )
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument(
        "--backend",
        choices=("threshold", "adaptive", "cellpose", "hybrid"),
        default="threshold",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cellpose-model", default="cpsam")
    parser.add_argument(
        "--also-score",
        action="store_true",
        help="After export, run score_external_labels on the same folder",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else (ROOT / "outputs" / "bbbc039_pred_labels" / args.backend).resolve()
    )

    raw_dir = data_dir / "raw"
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not args.skip_download:
        download(BBBC039_IMAGES, raw_dir / "images.zip")
        download(BBBC039_MASKS, raw_dir / "masks.zip")
        unzip(raw_dir / "images.zip", images_dir)
        unzip(raw_dir / "masks.zip", masks_dir)
    elif not images_dir.exists():
        print("ERROR: images missing; drop --skip-download or download first", file=sys.stderr)
        return 2

    pairs = find_pairs(images_dir, masks_dir)
    if not pairs:
        print("ERROR: no pairs", file=sys.stderr)
        return 3
    pairs = pairs[: max(1, args.max_images)]

    needs_cp = args.backend in ("cellpose", "hybrid")
    cellpose_seg = None
    if needs_cp:
        if not _HAS_CELLPOSE:
            print(f"ERROR: Cellpose not available: {_CELLPOSE_IMPORT_ERROR}", file=sys.stderr)
            return 5
        print(f"[cellpose] loading {args.cellpose_model!r} gpu={args.gpu}")
        cellpose_seg = CellposeSegmenter(model_type=args.cellpose_model, gpu=bool(args.gpu))
        _ = cellpose_seg.model

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] backend={args.backend} n={len(pairs)} -> {out_dir}")

    n_ok = 0
    for i, (img_path, _mask_path) in enumerate(pairs, 1):
        try:
            gray = load_image_gray(img_path)
            if args.backend == "threshold":
                seg = segment_threshold(gray)
            elif args.backend == "adaptive":
                seg = segment_threshold(gray, adaptive=True)
            elif args.backend == "cellpose":
                seg = cellpose_seg.segment(gray)
            else:
                seg = hybrid_threshold_cellpose(gray, cellpose_segmenter=cellpose_seg)

            # Same stem as TIFF so score_external_labels can pair
            dest = out_dir / f"{img_path.stem}.png"
            save_labels(dest, seg.labels)
            n_ok += 1
            print(f"  [{i}/{len(pairs)}] wrote {dest.name} count={seg.count}")
        except Exception as exc:
            print(f"  [{i}/{len(pairs)}] FAIL {img_path.name}: {exc}")

    print(f"\n[done] wrote {n_ok} label images under:\n  {out_dir}")
    print("\nScore them with:")
    print(
        f"  python scripts/score_external_labels.py --pred-dir {out_dir} "
        f"--data-dir {data_dir} --max-images {args.max_images} --name {args.backend}_export"
    )

    if args.also_score and n_ok:
        import subprocess

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "score_external_labels.py"),
            "--pred-dir",
            str(out_dir),
            "--data-dir",
            str(data_dir),
            "--max-images",
            str(args.max_images),
            "--name",
            f"{args.backend}_export",
        ]
        print("\n[also-score]", " ".join(cmd))
        return subprocess.call(cmd)

    return 0 if n_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
