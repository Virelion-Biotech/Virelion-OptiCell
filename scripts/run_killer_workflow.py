#!/usr/bin/env python3
"""Stage-2 killer workflow: QC → segment → features → phenotype [+ optional tracking].

Product default:
  --backend auto  →  cellpose if installed (always on low-contrast / phase-like), else threshold.

Usage:
  python scripts/run_killer_workflow.py /path/to/frames -o outputs/run --enable-tracking --gpu
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
from ensemble import hybrid_threshold_cellpose, fov_confidence, suggest_backend  # noqa: E402
from tracking import TrackingConfig, link_frames, summarize_tracks  # noqa: E402
from phenotype import Rule, score_cells, group_phenotype_summary  # noqa: E402
from acceptance import segmentation_acceptance  # noqa: E402


def resolve_backend(requested: str, *, gpu: bool, gray: np.ndarray | None = None) -> tuple[str, str]:
    """Map 'auto' to a concrete backend. Returns (backend, reason)."""
    req = (requested or "auto").strip().lower()
    if req != "auto":
        return req, f"user requested {req}"
    backend, reason = suggest_backend(gray, cellpose_available=_HAS_CELLPOSE)
    return backend, f"{reason} (gpu={bool(gpu)})"


def iter_images(folder: Path):
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        yield from sorted(folder.glob(ext))
        yield from sorted(folder.rglob(ext))


def load_gray(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return native grayscale pixels plus an 8-bit QC/segmentation copy."""
    raw = load_image(str(path))
    if raw.ndim == 2:
        native_gray = raw
    else:
        native_gray = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_RGB2GRAY)
    gray8 = to_grayscale_uint8(native_gray)
    return native_gray, gray8


def default_phenotype_rules() -> list[Rule]:
    """Explicit morphology rules with no bit-depth-dependent intensity cutoff."""
    return [
        Rule(feature="area_px", threshold=50.0, direction=">=", weight=1.0, label="area_ok"),
        Rule(feature="circularity", threshold=0.4, direction=">=", weight=1.0, label="roundish"),
    ]


def build_review_queue(rows: list[dict], decisions_path: Path | None = None) -> pd.DataFrame:
    """Build an auditable HITL queue from REVIEW/FAIL or flagged FOVs."""
    review_columns = [
        "index", "image", "status", "backend", "method", "object_count",
        "acceptance_status", "acceptance_reason", "confidence_score",
        "confidence_flags", "error", "review_status", "decision", "reviewer", "notes",
    ]
    queued = []
    for row in rows:
        if row.get("review_required"):
            queued.append({**row, "review_status": "PENDING", "decision": "", "reviewer": "", "notes": ""})
    queue = pd.DataFrame(queued, columns=review_columns)

    if decisions_path is not None:
        decisions = pd.read_csv(decisions_path)
        required = {"image", "decision"}
        missing = required - set(decisions.columns)
        if missing:
            raise ValueError(f"review decisions missing required columns: {sorted(missing)}")
        if decisions["image"].duplicated().any():
            raise ValueError("review decisions contain duplicate image entries")
        keep = [c for c in ("image", "decision", "reviewer", "notes") if c in decisions.columns]
        decisions = decisions[keep].copy()
        decisions["decision"] = decisions["decision"].fillna("").astype(str).str.strip().str.lower()
        allowed = {"", "accept", "reject", "rerun"}
        invalid = sorted(set(decisions.loc[~decisions["decision"].isin(allowed), "decision"]))
        if invalid:
            raise ValueError(f"unsupported review decisions: {invalid}; use accept/reject/rerun")
        queue = queue.drop(columns=["decision", "reviewer", "notes"], errors="ignore")
        queue = queue.merge(decisions, on="image", how="left")
        queue["decision"] = queue["decision"].fillna("")
        if "reviewer" not in queue:
            queue["reviewer"] = ""
        if "notes" not in queue:
            queue["notes"] = ""
        queue["reviewer"] = queue["reviewer"].fillna("")
        queue["notes"] = queue["notes"].fillna("")
        queue["review_status"] = np.where(queue["decision"].eq(""), "PENDING", "COMPLETED")
        queue = queue[review_columns]
    return queue
def main() -> int:
    parser = argparse.ArgumentParser(description="OptiCell killer workflow (default backend=auto)")
    parser.add_argument("images", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("outputs/workflow_run"))
    parser.add_argument(
        "--backend",
        choices=("auto", "threshold", "adaptive", "cellpose", "hybrid"),
        default="auto",
        help="auto = cellpose if installed (phase-like always cellpose) else threshold",
    )
    parser.add_argument("--gpu", action="store_true", help="Cellpose GPU when backend is cellpose/hybrid/auto")
    parser.add_argument("--cellpose-model", default="cpsam")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        help="Link frames as a time-lapse (filename sort order = time). "
        "Do NOT use on unordered multi-well FOVs.",
    )
    parser.add_argument("--track-max-distance", type=float, default=50.0)
    parser.add_argument("--track-max-gap", type=int, default=1)
    parser.add_argument("--write-review-queue", action="store_true",
                        help="Write an auditable CSV containing FOVs needing human review.")
    parser.add_argument("--review-decisions", type=Path, default=None,
                        help="CSV with image, decision[, reviewer, notes] to complete a prior review queue.")
    args = parser.parse_args()

    if args.max_images < 0:
        parser.error("--max-images must be >= 0")
    if args.track_max_gap < 0:
        parser.error("--track-max-gap must be >= 0")
    if args.track_max_distance <= 0 or not np.isfinite(args.track_max_distance):
        parser.error("--track-max-distance must be finite and > 0")
    if args.review_decisions is not None and not args.review_decisions.expanduser().is_file():
        parser.error(f"--review-decisions file not found: {args.review_decisions}")

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

    # Resolve auto using first frame contrast when possible
    probe_gray = None
    try:
        _, probe_gray = load_gray(paths[0])
    except Exception:
        probe_gray = None

    backend, backend_reason = resolve_backend(args.backend, gpu=bool(args.gpu), gray=probe_gray)
    print(f"[backend] {backend} ({backend_reason})")

    needs_cp = backend in ("cellpose", "hybrid")
    cellpose_seg = None
    if needs_cp:
        if not _HAS_CELLPOSE:
            if args.backend == "auto":
                backend = "threshold"
                backend_reason = (
                    "auto: Cellpose unavailable at runtime; "
                    f"falling back to threshold ({_CELLPOSE_IMPORT_ERROR or 'initialization failed'})"
                )
                print(f"WARNING: {backend_reason}", file=sys.stderr)
            else:
                print(f"ERROR: Cellpose unavailable: {_CELLPOSE_IMPORT_ERROR}", file=sys.stderr)
                print("Install: pip install -e '.[cellpose]'  or use --backend threshold", file=sys.stderr)
                return 5
        else:
            try:
                cellpose_seg = CellposeSegmenter(model_type=args.cellpose_model, gpu=bool(args.gpu))
                _ = cellpose_seg.model
            except Exception as exc:
                if args.backend == "auto":
                    backend = "threshold"
                    backend_reason = f"auto: Cellpose initialization failed; falling back to threshold ({exc})"
                    print(f"WARNING: {backend_reason}", file=sys.stderr)
                    cellpose_seg = None
                else:
                    print(f"ERROR: Cellpose initialization failed: {exc}", file=sys.stderr)
                    return 5

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    fov_rows: list[dict] = []
    failed_rows: list[dict] = []
    feature_frames: list[pd.DataFrame] = []
    labels_by_time: list[np.ndarray] = []
    tracking_complete = True

    for i, path in enumerate(paths, 1):
        try:
            native_gray, gray = load_gray(path)
            focus = compute_focus_score(gray)
            bright_mean, bright_std = compute_brightness(gray)
            sat = compute_saturation_fraction(gray)

            if backend == "threshold":
                seg = segment_threshold(gray)
            elif backend == "adaptive":
                seg = segment_threshold(gray, adaptive=True)
            elif backend == "cellpose":
                seg = cellpose_seg.segment(gray)
            else:
                seg = hybrid_threshold_cellpose(gray, cellpose_segmenter=cellpose_seg)

            conf = fov_confidence(gray, seg.labels, focus_score=focus)
            acceptance = (
                segmentation_acceptance(
                    quality_score=seg.quality_score,
                    border_fraction=seg.border_fraction,
                    tiny_object_fraction=seg.tiny_object_fraction,
                    merged_object_fraction=seg.merged_object_fraction,
                )
                if seg.error is None
                else None
            )
            review_required = (
                acceptance is None
                or acceptance.status != "PASS"
                or bool(conf["flags"])
            )
            feats = extract_object_features(native_gray, seg.labels)
            if not feats.empty:
                feats = feats.copy()
                feats.insert(0, "frame_index", i - 1)
                feats.insert(1, "image", path.name)
                feature_frames.append(feats)

            mask_path = masks_dir / f"{path.stem}_labels.png"
            max_label = int(seg.labels.max()) if seg.labels.size else 0
            if max_label > 65535:
                raise ValueError("segmentation produced more than 65535 labels; PNG uint16 export is unsafe")
            if not cv2.imwrite(str(mask_path), np.clip(seg.labels, 0, 65535).astype(np.uint16)):
                raise IOError(f"Could not write mask: {mask_path}")

            if args.enable_tracking:
                labels_by_time.append(seg.labels.astype(np.int32))

            row = {
                "index": i,
                "status": "success",
                "image": str(path),
                "backend": backend,
                "backend_requested": args.backend,
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
                "acceptance_status": acceptance.status if acceptance is not None else None,
                "acceptance_reason": acceptance.reason if acceptance is not None else None,
                "review_required": bool(review_required),
                "error": None,
            }
            fov_rows.append(row)
            print(
                f"  [{i}/{len(paths)}] {path.name}: count={seg.count} "
                f"focus={focus:.1f} conf={conf['confidence_score']:.0f} "
                f"flags={conf['flags'] or '-'}"
            )
        except Exception as exc:
            tracking_complete = False
            failed = {
                "index": i,
                "status": "failed",
                "image": str(path),
                "backend": backend,
                "backend_requested": args.backend,
                "object_count": None,
                "acceptance_status": None,
                "acceptance_reason": None,
                "review_required": True,
                "error": f"{type(exc).__name__}: {exc}",            }
            failed_rows.append(failed)
            print(f"  [{i}/{len(paths)}] FAIL {path.name}: {exc}")

    if not fov_rows:
        print("ERROR: no successful images", file=sys.stderr)
        return 4

    cells_df = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    phenotype_summary = None
    if not cells_df.empty:
        rules = default_phenotype_rules()
        cells_df = score_cells(cells_df, rules, positive_label="pass_rules", negative_label="fail_rules")
        phenotype_summary = group_phenotype_summary(cells_df, positive_label="pass_rules")
        cells_path = out_dir / "cell_features_phenotype.csv"
        cells_df.to_csv(cells_path, index=False)
        print(f"Wrote {cells_path} ({len(cells_df)} objects)")
    else:
        cells_path = None

    tracks_df = None
    track_summary = None
    if args.enable_tracking and len(labels_by_time) >= 2 and tracking_complete and len(labels_by_time) == len(paths):
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
        print("NOTE: tracking assumes filename order is time on ONE field of view.")
    elif args.enable_tracking:
        reason = "a frame failed" if not tracking_complete else "fewer than 2 successful frames"
        print(f"NOTE: tracking skipped ({reason}; temporal integrity preserved)", file=sys.stderr)

    confs = [r["confidence_score"] for r in fov_rows]
    counts = [r["object_count"] for r in fov_rows]
    mean_count = float(np.mean(counts)) if counts else 0.0
    count_cv = float(np.std(counts) / mean_count) if counts and mean_count > 0 else float("nan")
    failed_count = len(failed_rows)
    review_queue = None
    if args.write_review_queue or args.review_decisions is not None:
        review_queue = build_review_queue(
            fov_rows + failed_rows,
            args.review_decisions.expanduser() if args.review_decisions is not None else None,
        )
        review_path = out_dir / "review_queue.csv"
        review_queue.to_csv(review_path, index=False)
        print(f"Wrote {review_path} ({len(review_queue)} FOVs)")
    requested_count = len(paths)
    successful_count = len(fov_rows)
    complete = failed_count == 0 and successful_count == requested_count
    summary = {
        "n_images": successful_count,
        "n_requested_images": requested_count,
        "n_successful_images": successful_count,
        "n_failed_images": failed_count,
        "complete": complete,
        "mean_confidence": float(np.mean(confs)),
        "mean_object_count": mean_count,
        "count_cv": count_cv,
        "mean_focus": float(np.mean([r["focus_score"] for r in fov_rows])),
        "n_low_confidence_lt_50": int(sum(c < 50 for c in confs)),
        "n_review_required": int(sum(bool(r.get("review_required")) for r in fov_rows + failed_rows)),
        "n_objects_total": int(len(cells_df)) if cells_df is not None and not cells_df.empty else 0,
        "tracking_enabled": bool(args.enable_tracking and tracks_df is not None),
        "n_tracks": int(track_summary["track_id"].nunique()) if track_summary is not None and len(track_summary) else 0,
    }
    if phenotype_summary is not None and len(phenotype_summary):
        row0 = phenotype_summary.iloc[0].to_dict()
        summary["phenotype_positive_fraction"] = float(row0.get("positive_fraction", float("nan")))
        summary["phenotype_mean_score"] = float(row0.get("mean_score", float("nan")))

    if counts and mean_count > 0 and count_cv > 1.0:
        print(
            f"WARNING: count_cv={count_cv:.2f} > 1.0 — possible segmentation collapse. "
            "If Cellpose was not used, install it or re-run with --backend cellpose --gpu.",
            file=sys.stderr,
        )

    per_image = fov_rows + failed_rows
    per_image.sort(key=lambda row: int(row["index"]))
    payload = {
        "workflow": "qc_segment_features_phenotype"
        + ("_tracking" if summary["tracking_enabled"] else ""),
        "backend": backend,
        "backend_requested": args.backend,
        "backend_reason": backend_reason,
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
        "per_image": per_image,
        "failed_images": failed_rows,
        "review_queue_written": bool(review_queue is not None),
        "note": (
            "Default backend=auto → cellpose if installed (phase-like always cellpose) else threshold. "
            "Phenotype = explicit morphology rules. Tracking only with --enable-tracking on ordered TL. Review queue includes failed, non-PASS, or flagged FOVs when requested."
        ),
    }
    out_json = out_dir / "workflow_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    fov_df = pd.DataFrame(per_image)
    fov_df.to_csv(out_dir / "workflow_summary.csv", index=False)
    if phenotype_summary is not None:
        phenotype_summary.to_csv(out_dir / "phenotype_summary.csv", index=False)

    print("\n=== WORKFLOW SUMMARY ===")
    print(f"  backend: {backend} ({backend_reason})")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Wrote {out_json}")
    if review_queue is not None:
        print("HITL: fill review_queue.csv and re-run with --review-decisions to record accept/reject/rerun decisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
