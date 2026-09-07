#!/usr/bin/env python3
"""Fair multi-backend benchmark on BBBC039 (same FOV list for every backend).

Why this exists
---------------
Stage-1 product question: is OptiCell measurably competitive on a fixed public set?
This driver runs chosen backends on the **same** image list and writes one comparison
table. No fabricated numbers — only measured aggregates.

Examples
--------
  # Full corpus classical + hybrid (CPU-friendly)
  python scripts/run_bbbc039_multi_backend.py --max-images 200 --backends threshold,hybrid

  # Pilot with Cellpose GPU
  python scripts/run_bbbc039_multi_backend.py --max-images 50 --backends threshold,cellpose,hybrid --gpu

  # Skip download if data already present
  python scripts/run_bbbc039_multi_backend.py --max-images 200 --backends threshold,adaptive,hybrid --skip-download
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_bbbc039_validation.py"


def run_one(backend: str, max_images: int, skip_download: bool, gpu: bool, cellpose_model: str, out_dir: Path) -> Path:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--max-images",
        str(max_images),
        "--backend",
        backend,
        "--out-dir",
        str(out_dir),
        "--cellpose-model",
        cellpose_model,
    ]
    if skip_download:
        cmd.append("--skip-download")
    if gpu and backend in ("cellpose", "hybrid"):
        cmd.append("--gpu")

    print("\n" + "=" * 72)
    print("RUN", " ".join(cmd))
    print("=" * 72)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"backend={backend} failed with code {proc.returncode}")

    # Find newest matching json for this backend
    matches = sorted(out_dir.glob(f"bbbc039_{backend}_n*.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise SystemExit(f"No JSON written for backend={backend} under {out_dir}")
    return matches[-1]


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "backend": data.get("backend"),
        "n_scored": data.get("n_scored"),
        "gpu": data.get("gpu"),
        "cellpose_model": data.get("cellpose_model"),
        "summary": data.get("summary", {}),
        "timestamp_utc": data.get("timestamp_utc"),
    }


def fmt(v, digits=4):
    try:
        x = float(v)
        if x != x:  # NaN
            return "nan"
        return f"{x:.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def write_report(rows: list[dict], out_md: Path, max_images: int) -> None:
    lines = [
        "# BBBC039 multi-backend comparison (measured only)",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Requested max images: {max_images}",
        "- Same FOV list order as `scripts/run_bbbc039_validation.py` (basename sort).",
        "- Policy: no fabricated metrics.",
        "",
        "| Backend | n | IoU | Dice | Instance F1 | \|count err\| | rel count err | GPU |",
        "|---------|--:|----:|-----:|------------:|-------------:|--------------:|:---:|",
    ]
    for r in rows:
        s = r["summary"]
        lines.append(
            "| {backend} | {n} | {iou} | {dice} | {f1} | {ac} | {rc} | {gpu} |".format(
                backend=r["backend"],
                n=r["n_scored"],
                iou=fmt(s.get("iou", s.get("pixel_iou_mean"))),
                dice=fmt(s.get("dice", s.get("pixel_dice_mean"))),
                f1=fmt(s.get("f1", s.get("instance_f1_mean"))),
                ac=fmt(s.get("absolute_count_error", s.get("absolute_count_error_mean")), 2),
                rc=fmt(s.get("relative_count_error", s.get("relative_count_error_mean")), 4),
                gpu=str(bool(r.get("gpu"))),
            )
        )
    lines += [
        "",
        "## Source JSON",
        "",
    ]
    for r in rows:
        lines.append(f"- `{r['path']}` ({r.get('timestamp_utc')})")
    lines += [
        "",
        "## Notes",
        "",
        "- Hybrid = count-gated switch between threshold and Cellpose (see `ensemble.hybrid_threshold_cellpose`).",
        "- Cellpose rows require optional dependency + weights download on first use.",
        "- This is not a CellProfiler comparison; that baseline is a separate Stage-1 item.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote comparison report: {out_md}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-backend BBBC039 benchmark (measured only)")
    parser.add_argument(
        "--backends",
        default="threshold,hybrid",
        help="Comma-separated: threshold,adaptive,cellpose,hybrid",
    )
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cellpose-model", default="cpsam")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/bbbc039_validation"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/bbbc039_validation/BBBC039_MULTI_BACKEND_COMPARISON.md"),
    )
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    allowed = {"threshold", "adaptive", "cellpose", "hybrid"}
    bad = [b for b in backends if b not in allowed]
    if bad:
        print(f"Unknown backends: {bad}. Allowed: {sorted(allowed)}", file=sys.stderr)
        return 2
    if not backends:
        print("No backends specified", file=sys.stderr)
        return 2
    if not RUNNER.exists():
        print(f"Missing runner: {RUNNER}", file=sys.stderr)
        return 3

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # First backend may download; subsequent force skip if data appeared
    skip = bool(args.skip_download)
    rows = []
    for i, backend in enumerate(backends):
        path = run_one(
            backend=backend,
            max_images=args.max_images,
            skip_download=skip or i > 0,
            gpu=bool(args.gpu),
            cellpose_model=args.cellpose_model,
            out_dir=out_dir,
        )
        # After first successful run, data should exist
        skip = True
        rows.append(load_summary(path))

    report_path = args.report.expanduser().resolve()
    write_report(rows, report_path, args.max_images)

    # Machine-readable sidecar
    sidecar = report_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "dataset": "BBBC039",
                "max_images": args.max_images,
                "backends": backends,
                "gpu": bool(args.gpu),
                "rows": rows,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "note": "Measured only. Same FOV ordering across backends.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {sidecar}")

    print("\n=== COMPARISON (measured only) ===")
    for r in rows:
        s = r["summary"]
        print(
            f"  {r['backend']:10s}  n={r['n_scored']}  "
            f"iou={fmt(s.get('iou', s.get('pixel_iou_mean')))}  "
            f"dice={fmt(s.get('dice', s.get('pixel_dice_mean')))}  "
            f"f1={fmt(s.get('f1', s.get('instance_f1_mean')))}  "
            f"|count|={fmt(s.get('absolute_count_error', s.get('absolute_count_error_mean')), 2)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
