#!/usr/bin/env python3
"""Stage-2 entry: QC → segmentation → confidence for a folder of images.

Killer use case (from product plan):
  Automated QC → segmentation → (later: tracking → phenotype) for cell assays.

This script is the **first slice**: acquisition QC flags + chosen segmentation
backend + per-FOV confidence. No fabricated metrics.

Usage:
  python scripts/run_killer_workflow.py /path/to/images -o outputs/workflow_run
  python scripts/run_killer_workflow.py /path/to/images --backend hybrid --gpu
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

from qc_pipeline import (  # noqa: E402
    to_grayscale_uint8,
    segment_threshold,
    CellposeSegmenter,
    compute_focus_score,
    _HAS_CELLPOSE,
    _CELLPOSE_IMPORT_ERROR,
)
from ensemble import hybrid_threshold_cellpose, fov_confidence  # noqa: E402


def iter_images(folder: Path):
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        yield from sorted(folder.glob(ext))
        yield from sorted(folder.rglob(ext))


def load_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(path)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    return to_grayscale_uint8(arr)


def main() -> int:
    parser = argparse.ArgumentParser(description="OptiCell QC→segment→confidence workflow")
    parser.add_argument("images", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("outputs/workflow_run"))
    parser.add_argument(
        "--backend",
        choices=("threshold", "adaptive", "cellpose", "hybrid"),
        default="threshold",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cellpose-model", default="cpsam")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    folder = args.images.expanduser().resolve()
    if not folder.is_dir():
        print(f"ERROR: not a directory: {folder}", file=sys.stderr)
        return 2

    paths = []
    seen = set()
    for p in iter_images(folder):
        rp = p.resolve()
        if rp in seen or "__MACOSX" in p.parts:
            continue
        seen.add(rp)
        paths.append(rp)
    if args.max_images > 0:
        paths = paths[: args.max_images]
    if not paths:
        print("ERROR: no images found", file=sys.stderr)
        return 3

    needs_cp = args.backend in ("cellpose", "hybrid")
    cellpose_seg = None
    if needs_cp:
        if not _HAS_CELLPOSE:
            print(f"ERROR: Cellpose unavailable: {_CELLPOSE_IMPORT_ERROR}", file=sys.stderr)
            return 5
        cellpose_seg = CellposeSegmenter(model_type=args.cellpose_model, gpu=bool(args.gpu))
        _ = cellpose_seg.model

    rows = []
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    for i, path in enumerate(paths, 1):
        try:
            gray = load_gray(path)
            focus = compute_focus_score(gray)
            if args.backend == "threshold":
                seg = segment_threshold(gray)
            elif args.backend == "adaptive":
                seg = segment_threshold(gray, adaptive=True)
            elif args.backend == "cellpose":
                seg = cellpose_seg.segment(gray)
            else:
                seg = hybrid_threshold_cellpose(gray, cellpose_segmenter=cellpose_seg)

            conf = fov_confidence(gray, seg.labels, focus_score=focus)
            mask_path = masks_dir / f"{path.stem}_labels.png"
            # save labels as uint16 PNG
            labels_u16 = np.clip(seg.labels, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(mask_path), labels_u16)

            row = {
                "index": i,
                "image": str(path),
                "backend": args.backend,
                "method": seg.method,
                "object_count": int(seg.count),
                "focus_score": float(focus),
                "confidence_score": float(conf["confidence_score"]),
                "confidence_flags": conf["flags"],
                "foreground_fraction": float(seg.foreground_fraction),
                "quality_score": float(seg.quality_score) if seg.quality_score is not None else None,
                "mask": str(mask_path),
            }
            rows.append(row)
            print(
                f"  [{i}/{len(paths)}] {path.name}: count={seg.count} "
                f"focus={focus:.1f} conf={conf['confidence_score']:.0f} flags={conf['flags'] or '-'}"
            )
        except Exception as exc:
            print(f"  [{i}/{len(paths)}] FAIL {path.name}: {exc}")

    if not rows:
        print("ERROR: no successful images", file=sys.stderr)
        return 4

    confs = [r["confidence_score"] for r in rows]
    payload = {
        "workflow": "qc_segment_confidence",
        "backend": args.backend,
        "n_images": len(rows),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "mean_confidence": float(np.mean(confs)),
            "mean_object_count": float(np.mean([r["object_count"] for r in rows])),
            "mean_focus": float(np.mean([r["focus_score"] for r in rows])),
            "n_low_confidence_lt_50": int(sum(c < 50 for c in confs)),
        },
        "per_image": rows,
        "note": "Workflow slice only — not a phenotype or tracking claim.",
    }
    out_json = out_dir / "workflow_summary.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # CSV
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in keys))
    (out_dir / "workflow_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== WORKFLOW SUMMARY ===")
    for k, v in payload["summary"].items():
        print(f"  {k}: {v}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
