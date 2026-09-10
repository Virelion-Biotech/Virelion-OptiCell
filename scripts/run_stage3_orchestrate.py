#!/usr/bin/env python3
"""Stage-3 orchestration: run multiple backends, score FOV confidence, pick winner.

Does NOT invent metrics. Writes measured per-backend workflow summaries and an
auto-select decision table based on explicit rules:

  prefer higher mean confidence, then lower count CV, then fewer low-conf FOVs.

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
    fov = pd.read_csv(out_dir / "workflow_summary.csv") if (out_dir / "workflow_summary.csv").is_file() else pd.DataFrame()
    counts = fov["object_count"].to_numpy(float) if "object_count" in fov.columns else np.array([])
    confs = fov["confidence_score"].to_numpy(float) if "confidence_score" in fov.columns else np.array([])
    count_cv = float(np.std(counts) / np.mean(counts)) if len(counts) and np.mean(counts) > 0 else float("inf")
    return {
        "backend": backend,
        "out_dir": str(out_dir),
        "n_images": int(s.get("n_images", 0)),
        "mean_confidence": float(s.get("mean_confidence", 0)),
        "mean_object_count": float(s.get("mean_object_count", 0)),
        "count_cv": count_cv,
        "n_low_confidence_lt_50": int(s.get("n_low_confidence_lt_50", 0)),
        "n_tracks": int(s.get("n_tracks", 0)),
        "n_objects_total": int(s.get("n_objects_total", 0)),
        "mean_focus": float(s.get("mean_focus", 0)),
    }


def select_backend(rows: list[dict]) -> dict:
    """Explicit, auditable selection — not a black-box model."""
    if not rows:
        raise ValueError("no backend results")
    # Sort: higher confidence, lower count_cv, fewer low-conf FOVs, higher n_tracks stability proxy
    ranked = sorted(
        rows,
        key=lambda r: (
            -r["mean_confidence"],
            r["count_cv"],
            r["n_low_confidence_lt_50"],
            -r["n_images"],
        ),
    )
    winner = ranked[0]
    return {
        "selected_backend": winner["backend"],
        "reason": (
            f"max mean_confidence then min count_cv then min low_conf FOVs; "
            f"winner={winner['backend']} conf={winner['mean_confidence']:.1f} "
            f"cv={winner['count_cv']:.3f}"
        ),
        "ranking": [r["backend"] for r in ranked],
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
                    "n_low_confidence_lt_50": 10**9,
                    "n_images": 0,
                    "mean_object_count": 0.0,
                    "n_tracks": 0,
                    "n_objects_total": 0,
                    "mean_focus": 0.0,
                }
            )

    decision = select_backend([r for r in rows if r.get("mean_confidence", -1) >= 0])
    payload = {
        "stage": 3,
        "workflow": "multi_backend_orchestrate",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "images": str(images),
        "backends_requested": backends,
        "per_backend": rows,
        "decision": decision,
        "selection_rule": (
            "Sort by (-mean_confidence, count_cv, n_low_confidence_lt_50, -n_images). "
            "Auditable, not learned."
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
