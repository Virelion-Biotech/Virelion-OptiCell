#!/usr/bin/env python3
"""Stage-3 orchestration: run multiple backends, score FOV signals, pick winner.

Selection rule v2 (from measured CTC TRA, Stage-3 report):

  1. Discard failed backends and count_cv > CV_REJECT (collapse / FP explosion).
  2. Prefer cellpose, then hybrid, then threshold among survivors
     (Cellpose won TRA on all 6 CTC sequences in measured runs).
  3. Tie-break: lower count_cv, then higher mean_confidence, then more frames.

Does NOT invent TRA. FOV confidence alone is often saturated at 100 on clean
fluorescence — do not rely on it as the primary rank key.

Usage:
  python scripts/run_stage3_orchestrate.py /path/to/frames -o outputs/s3 \\
      --backends threshold,hybrid,cellpose --gpu --enable-tracking
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Measured SIM+02 threshold/hybrid had count_cv ~2.3 and TRA=0.
# Cellpose on same seq had count_cv ~0.58 and TRA~0.50.
CV_REJECT = 1.0

BACKEND_PRIORITY = {"cellpose": 0, "hybrid": 1, "adaptive": 2, "threshold": 3}


def run_backend(
    images: Path,
    out_dir: Path,
    backend: str,
    gpu: bool,
    enable_tracking: bool,
    track_max_distance: float,
    track_max_gap: int,
    max_images: int,
    cellpose_model: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_killer_workflow.py"),
        str(images),
        "-o",
        str(out_dir),
        "--backend",
        backend,
        "--track-max-distance",
        str(track_max_distance),
        "--track-max-gap",
        str(track_max_gap),
        "--cellpose-model",
        cellpose_model,
    ]
    if gpu:
        cmd.append("--gpu")
    if enable_tracking:
        cmd.append("--enable-tracking")
    if max_images > 0:
        cmd.extend(["--max-images", str(max_images)])

    print("CMD:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:] if r.stderr else "")
        raise RuntimeError(f"backend={backend} failed code={r.returncode}")

    summary_path = out_dir / "workflow_summary.json"
    payload = json.loads(summary_path.read_text())
    s = payload.get("summary", {})
    fov_path = out_dir / "workflow_summary.csv"
    fov = pd.read_csv(fov_path) if fov_path.is_file() else pd.DataFrame()
    counts = (
        fov["object_count"].to_numpy(float)
        if "object_count" in fov.columns and len(fov)
        else np.array([])
    )
    confs = (
        fov["confidence_score"].to_numpy(float)
        if "confidence_score" in fov.columns and len(fov)
        else np.array([])
    )
    mean_count = float(np.mean(counts)) if len(counts) else 0.0
    count_cv = (
        float(np.std(counts) / mean_count)
        if len(counts) and mean_count > 0
        else float("inf")
    )
    # Fraction of frames with zero objects — strong failure signal
    zero_frac = float(np.mean(counts == 0)) if len(counts) else 1.0
    return {
        "backend": backend,
        "out_dir": str(out_dir),
        "n_images": int(s.get("n_images", 0)),
        "mean_confidence": float(s.get("mean_confidence", 0)),
        "mean_object_count": mean_count,
        "count_cv": count_cv,
        "zero_object_fraction": zero_frac,
        "n_low_confidence_lt_50": int(s.get("n_low_confidence_lt_50", 0)),
        "n_tracks": int(s.get("n_tracks", 0)),
        "n_objects_total": int(s.get("n_objects_total", 0)),
        "mean_focus": float(s.get("mean_focus", 0)),
    }


def select_backend(rows: list[dict], cv_reject: float = CV_REJECT) -> dict:
    """Explicit, auditable selection v2 — not a black-box model.

    Measured evidence (Stage-3 CTC):
      - Cellpose had best TRA on all 6 sequences.
      - count_cv > 1 tracked threshold/hybrid collapse (SIM+02 TRA=0).
      - mean_confidence was 100 for every backend — useless as primary key.
    """
    if not rows:
        raise ValueError("no backend results")

    viable: list[dict] = []
    rejected: list[dict] = []
    for r in rows:
        reason = None
        if r.get("mean_confidence", -1) < 0 or r.get("error"):
            reason = "run_failed"
        elif r.get("n_images", 0) <= 0:
            reason = "no_images"
        elif r.get("mean_object_count", 0) <= 0:
            reason = "zero_mean_count"
        elif float(r.get("count_cv", float("inf"))) > cv_reject:
            reason = f"count_cv>{cv_reject}"
        elif float(r.get("zero_object_fraction", 0)) > 0.25:
            reason = "zero_object_fraction>0.25"
        if reason:
            rejected.append({**r, "reject_reason": reason})
        else:
            viable.append(r)

    pool = viable if viable else [
        r for r in rows if r.get("mean_confidence", -1) >= 0 and not r.get("error")
    ]
    if not pool:
        raise ValueError("no viable backends after filters")

    def sort_key(r: dict):
        pri = BACKEND_PRIORITY.get(str(r.get("backend", "")), 50)
        return (
            pri,  # cellpose first among survivors
            float(r.get("count_cv", float("inf"))),
            -float(r.get("mean_confidence", 0)),
            int(r.get("n_low_confidence_lt_50", 0)),
            -int(r.get("n_images", 0)),
        )

    ranked = sorted(pool, key=sort_key)
    winner = ranked[0]
    return {
        "selected_backend": winner["backend"],
        "rule": "v2",
        "cv_reject": cv_reject,
        "reason": (
            f"v2: reject count_cv>{cv_reject} / zero-count collapse; "
            f"prefer cellpose>hybrid>threshold; then min count_cv. "
            f"winner={winner['backend']} cv={float(winner.get('count_cv', float('nan'))):.3f} "
            f"conf={float(winner.get('mean_confidence', float('nan'))):.1f}"
        ),
        "ranking": [r["backend"] for r in ranked],
        "rejected": [
            {"backend": r["backend"], "reason": r.get("reject_reason")}
            for r in rejected
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Stage-3 multi-backend orchestration")
    p.add_argument("images", type=Path)
    p.add_argument("-o", "--out-dir", type=Path, required=True)
    p.add_argument(
        "--backends",
        default="threshold,hybrid,cellpose",
        help="comma-separated",
    )
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--enable-tracking", action="store_true")
    p.add_argument("--track-max-distance", type=float, default=50.0)
    p.add_argument("--track-max-gap", type=int, default=1)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--cellpose-model", default="cpsam")
    p.add_argument(
        "--cv-reject",
        type=float,
        default=CV_REJECT,
        help="Reject backends with object-count CV above this (default 1.0)",
    )
    args = p.parse_args()

    images = args.images.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    rows = []
    for backend in backends:
        bdir = out_root / f"backend_{backend}"
        print(f"\n===== BACKEND {backend} =====")
        try:
            row = run_backend(
                images,
                bdir,
                backend=backend,
                gpu=bool(args.gpu),
                enable_tracking=bool(args.enable_tracking),
                track_max_distance=float(args.track_max_distance),
                track_max_gap=int(args.track_max_gap),
                max_images=int(args.max_images),
                cellpose_model=args.cellpose_model,
            )
            rows.append(row)
            print(json.dumps(row, indent=2))
        except Exception as exc:
            print(f"BACKEND FAIL {backend}: {exc}")
            rows.append(
                {
                    "backend": backend,
                    "out_dir": str(bdir),
                    "error": str(exc),
                    "mean_confidence": -1.0,
                    "count_cv": float("inf"),
                    "zero_object_fraction": 1.0,
                    "n_low_confidence_lt_50": 10**9,
                    "n_images": 0,
                    "mean_object_count": 0.0,
                    "n_tracks": 0,
                    "n_objects_total": 0,
                    "mean_focus": 0.0,
                }
            )

    decision = select_backend(
        [r for r in rows if True],
        cv_reject=float(args.cv_reject),
    )
    payload = {
        "stage": 3,
        "workflow": "multi_backend_orchestrate",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "images": str(images),
        "backends_requested": backends,
        "per_backend": rows,
        "decision": decision,
        "selection_rule": (
            "v2: reject count_cv>cv_reject / zero-count; "
            "prefer cellpose>hybrid>threshold; then min count_cv. "
            "Auditable, not learned. Derived from measured Stage-3 CTC TRA."
        ),
        "note": "Measured only. TRA must be scored separately against CTC GT.",
    }
    out_json = out_root / "stage3_decision.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    pd.DataFrame(rows).to_csv(out_root / "stage3_backends.csv", index=False)
    print("\n=== STAGE-3 DECISION ===")
    print(json.dumps(decision, indent=2))
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
