#!/usr/bin/env python3
"""Stage-2 killer workflow: QC → segment → features → phenotype [+ optional tracking].

Product use case:
  Automated acquisition QC, segmentation, per-cell phenotype features, and
  (when images are ordered frames of the same field) assignment tracking.

Honest limits:
  - Tracking is only meaningful for true time-lapse sequences. Enabling it on
    unrelated FOVs (e.g. BBBC039 plate wells) produces link tables, not biology.
  - Phenotype rules are explicit morphology thresholds — not a trained classifier.
  - Intensity measurements in cell_features_phenotype.csv preserve the native
    grayscale pixel values; 8-bit normalization is used only for QC/segmentation.
  - No fabricated metrics.

Usage:
  # Single-plane assay / multi-FOV phenotype (no tracking claim)
  python scripts/run_killer_workflow.py data/bbbc039/images/images \
      -o outputs/stage2_bbbc039 --backend threshold --max-images 30

  # True time-lapse folder (frames sorted by filename)
  python scripts/run_killer_workflow.py /path/to/timelapse \
      -o outputs/stage2_tl --backend threshold --enable-tracking
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qc_pipeline import (  # noqa: E402
    load_image,
    to_grayscale_uint8,
    segment_threshold,
    CellposeSegmenter,
    compute_focus_score,
    compute_brightness,
    compute_saturation_fraction,
    extract_object_features,
    _HAS_CELLPOSE,
    _CELLPOSE_IMPORT_ERROR,
)
from ensemble import hybrid_threshold_cellpose, fov_confidence  # noqa: E402
from tracking import TrackingConfig, link_frames, summarize_tracks  # noqa: E402
from phenotype import Rule, score_cells, group_phenotype_summary  # noqa: E402


def iter_images(folder: Path):
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        yield from sorted(folder.glob(ext))
        yield from sorted(folder.rglob(ext))


def load_gray(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return native grayscale pixels plus an 8-bit QC/segmentation copy."""
    arr = load_image(str(path))
    if arr.ndim == 2:
        native_gray = arr
    else:
        native_gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    gray8 = to_grayscale_uint8(native_gray)
    return native_gray, gray8


def default_phenotype_rules() -> list[Rule]:
    """Explicit morphology rules with no bit-depth-dependent intensity cutoff."""
    return [
        Rule(feature="area_px", threshold=50.0, direction=">=", weight=1.0, label="area_ok"),
        Rule(feature="circularity", threshold=0.4, direction=">=", weight=1.0, label="roundish"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="OptiCell Stage-2 killer workflow")
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
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        help="Link frames as a time-lapse (filename sort order = time). "
        "Do NOT use on unordered multi-well FOVs.",
    )
    parser.add_argument("--track-max-distance", type=float, default=30.0)
    parser.add_argument("--track-max-gap", type=int, default=1)
    args = parser.parse_args()

    folder = args.images.expanduser().resolve()
    if not folder.is_dir():
        print(f"ERROR: not a directory: {folder}", file=sys.stderr)
        return 2

    paths: list[Path] = []
    seen: set[Path] = set()
    for p in iter_images(folder):
        rp = p.resolve()
        if rp in seen or "__MACOSX" in p.parts or p.name.startswith("._"):
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

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    fov_rows: list[dict] = []
    feature_frames: list[pd.DataFrame] = []
    labels_by_time: list[np.ndarray] = []

    for i, path in enumerate(paths, 1):
        try:
            native_gray, gray = load_gray(path)
            focus = compute_focus_score(gray)
            bright_mean, bright_std = compute_brightness(gray)
            sat = compute_saturation_fraction(gray)

            if args.backend == "threshold":
                seg = segment_threshold(gray)
            elif args.backend == "adaptive":
                seg = segment_threshold(gray, adaptive=True)
            elif args.backend == "cellpose":
                seg = cellpose_seg.segment(gray)
            else:
                seg = hybrid_threshold_cellpose(gray, cellpose_segmenter=cellpose_seg)

            conf = fov_confidence(gray, seg.labels, focus_score=focus)
            feats = extract_object_features(native_gray, seg.labels)
            if not feats.empty:
                feats = feats.copy()
                feats.insert(0, "frame_index", i - 1)
                feats.insert(1, "image", path.name)
                feature_frames.append(feats)

            mask_path = masks_dir / f"{path.stem}_labels.png"
            cv2.imwrite(str(mask_path), np.clip(seg.labels, 0, 65535).astype(np.uint16))

            if args.enable_tracking:
                labels_by_time.append(seg.labels.astype(np.int32))

            row = {
                "index": i,
                "image": str(path),
                "backend": args.backend,
                "method": seg.method,
                "object_count": int(seg.count),
                "focus_score": float(focus),
                "brightness_mean": float(bright_mean),
                "brightness_std": float(bright_std),
                "saturation_fraction": float(sat),
                "confidence_score": float(conf["confidence_score"]),
                "confidence_flags": conf["flags"],
                "foreground_fraction": float(seg.foreground_fraction),
                "quality_score": float(seg.quality_score) if seg.quality_score is not None else None,
                "mask": str(mask_path),
            }
            fov_rows.append(row)
            print(
                f"  [{i}/{len(paths)}] {path.name}: count={seg.count} "
                f"focus={focus:.1f} conf={conf['confidence_score']:.0f} "
                f"flags={conf['flags'] or '-'}"
            )
        except Exception as exc:
            print(f"  [{i}/{len(paths)}] FAIL {path.name}: {exc}")

    if not fov_rows:
        print("ERROR: no successful images", file=sys.stderr)
        return 4

    # --- features + phenotype ---
    cells_df = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    phenotype_summary = None
    if not cells_df.empty:
        rules = default_phenotype_rules()
        cells_df = score_cells(cells_df, rules, positive_label="pass_rules", negative_label="fail_rules")
        phenotype_summary = group_phenotype_summary(cells_df)
        cells_path = out_dir / "cell_features_phenotype.csv"
        cells_df.to_csv(cells_path, index=False)
        print(f"Wrote {cells_path} ({len(cells_df)} objects)")
    else:
        cells_path = None

    # --- optional tracking ---
    tracks_df = None
    track_summary = None
    if args.enable_tracking and len(labels_by_time) >= 2:
        cfg = TrackingConfig(
            max_distance_px=float(args.track_max_distance),
            max_gap=int(args.track_max_gap),
        )
        tracks_df = link_frames(labels_by_time, config=cfg)
        track_summary = summarize_tracks(tracks_df)
        tracks_df.to_csv(out_dir / "tracks.csv", index=False)
        track_summary.to_csv(out_dir / "track_summary.csv", index=False)
        print(
            f"Wrote tracks: n_observations={len(tracks_df)} "
            f"n_tracks={track_summary['track_id'].nunique() if len(track_summary) else 0}"
        )
        print(
            "NOTE: tracking assumes filename order is time on ONE field of view."
        )
    elif args.enable_tracking:
        print("NOTE: tracking skipped (need >= 2 frames)")

    confs = [r["confidence_score"] for r in fov_rows]
    counts = [r["object_count"] for r in fov_rows]
    summary = {
        "n_images": len(fov_rows),
        "mean_confidence": float(np.mean(confs)),
        "mean_object_count": float(np.mean(counts)),
        "mean_focus": float(np.mean([r["focus_score"] for r in fov_rows])),
        "n_low_confidence_lt_50": int(sum(c < 50 for c in confs)),
        "n_objects_total": int(len(cells_df)) if cells_df is not None and not cells_df.empty else 0,
        "tracking_enabled": bool(args.enable_tracking and tracks_df is not None),
        "n_tracks": int(track_summary["track_id"].nunique()) if track_summary is not None and len(track_summary) else 0,
    }
    if phenotype_summary is not None and len(phenotype_summary):
        # single-group summary
        row0 = phenotype_summary.iloc[0].to_dict()
        summary["phenotype_positive_fraction"] = float(row0.get("positive_fraction", float("nan")))
        summary["phenotype_mean_score"] = float(row0.get("mean_score", float("nan")))

    payload = {
        "workflow": "qc_segment_features_phenotype"
        + ("_tracking" if summary["tracking_enabled"] else ""),
        "backend": args.backend,
        "stage": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "phenotype_rules": [
            {
                "feature": r.feature,
                "threshold": r.threshold,
                "direction": r.direction,
                "weight": r.weight,
                "label": r.label,
            }
            for r in default_phenotype_rules()
        ],
        "per_image": fov_rows,
        "note": (
            "Stage-2 measured pipeline outputs. "
            "Phenotype = explicit morphology rules. Native grayscale intensity "
            "features are preserved in the per-cell table. 8-bit normalization "
            "is used only for QC and segmentation. "
            "Tracking only if --enable-tracking and ordered time-lapse frames."
        ),
    }
    out_json = out_dir / "workflow_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    fov_df = pd.DataFrame(fov_rows)
    fov_df.to_csv(out_dir / "workflow_summary.csv", index=False)
    if phenotype_summary is not None:
        phenotype_summary.to_csv(out_dir / "phenotype_summary.csv", index=False)

    print("\n=== STAGE-2 WORKFLOW SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
