#!/usr/bin/env python3
"""BBBC006 focus-metric validation (Stage 1 — second public dataset).

BBBC006: U2OS Hoechst z-stacks; z≈16 is optimal focus; planes 11–23 labeled
in-focus by experts. Full z zips are ~800 MB each — this script does **not**
auto-download them. Point it at a local extract.

What it measures (no fabricated numbers):
  - OptiCell `compute_focus_score` (Laplacian variance) vs |z - z_focus|
  - Rank correlation: does focus score peak near the true focal plane?

Expected layout (flexible):
  data/bbbc006/z_00/*.tif
  data/bbbc006/z_16/*.tif
  ...
or any folder tree where path or filename encodes z index.

Usage:
  python scripts/run_bbbc006_focus_qc.py --root data/bbbc006 --z-focus 16 --max-sites 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qc_pipeline import compute_focus_score, to_grayscale_uint8  # noqa: E402

Z_RE = re.compile(r"(?:^|[_/\\-])z[_-]?(\d{1,2})(?:[_/\\-]|\.|$)", re.I)


def parse_z(path: Path) -> int | None:
    for part in [path.name, *path.parts[::-1]]:
        m = Z_RE.search(str(part))
        if m:
            return int(m.group(1))
    return None


def load_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(path)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    return to_grayscale_uint8(arr)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="BBBC006 focus QC (measured only)")
    parser.add_argument("--root", type=Path, required=True, help="Root of extracted BBBC006 z folders")
    parser.add_argument("--z-focus", type=int, default=16)
    parser.add_argument("--max-sites", type=int, default=30, help="Max unique site stems to score")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/bbbc006_focus_qc"))
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        print("Download selected z-plane zips from https://bbbc.broadinstitute.org/BBBC006", file=sys.stderr)
        return 2

    # Collect tifs with parseable z
    files: list[tuple[Path, int]] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".tif", ".tiff"}:
            continue
        if "__MACOSX" in p.parts:
            continue
        # prefer Hoechst w1 if present in name
        name_l = p.name.lower()
        if "w2" in name_l and "w1" not in name_l:
            continue
        z = parse_z(p)
        if z is None:
            continue
        files.append((p, z))

    if not files:
        print("ERROR: no TIFFs with z index in path/name under", root, file=sys.stderr)
        return 3

    # Group by site stem (strip z token approximately via parent+stem without z)
    from collections import defaultdict

    sites: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for p, z in files:
        # site key: filename without extension, remove z_XX patterns
        key = Z_RE.sub("_zXX_", p.stem)
        sites[key].append((p, z))

    site_keys = sorted(sites.keys())[: max(1, args.max_sites)]
    rows = []
    rank_hits = 0
    rank_total = 0

    for key in site_keys:
        stack = sorted(sites[key], key=lambda t: t[1])
        scores = []
        for path, z in stack:
            try:
                gray = load_gray(path)
                fs = compute_focus_score(gray)
            except Exception as exc:
                print(f"  FAIL {path.name}: {exc}")
                continue
            scores.append((z, fs, path.name))
            rows.append(
                {
                    "site": key,
                    "z": z,
                    "focus_score": fs,
                    "abs_dz": abs(z - args.z_focus),
                    "file": path.name,
                }
            )
        if len(scores) < 3:
            continue
        zs = np.array([s[0] for s in scores], dtype=float)
        fs = np.array([s[1] for s in scores], dtype=float)
        # best predicted focal plane = argmax focus score
        z_hat = int(zs[int(np.argmax(fs))])
        rank_total += 1
        if abs(z_hat - args.z_focus) <= 2:
            rank_hits += 1
        corr = spearman_corr(fs, -np.abs(zs - args.z_focus))
        print(f"  site {key[:40]}... n_z={len(scores)} z_hat={z_hat} spearman_vs_sharpness_proxy={corr:.3f}")

    if not rows:
        print("ERROR: no scores computed", file=sys.stderr)
        return 4

    abs_dz = np.array([r["abs_dz"] for r in rows], dtype=float)
    focus = np.array([r["focus_score"] for r in rows], dtype=float)
    corr_all = spearman_corr(focus, -abs_dz)

    summary = {
        "n_planes_scored": float(len(rows)),
        "n_sites": float(rank_total),
        "z_focus": float(args.z_focus),
        "frac_argmax_within_2_planes": float(rank_hits / rank_total) if rank_total else float("nan"),
        "spearman_focus_vs_neg_abs_dz": float(corr_all),
    }

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "BBBC006",
        "source": "https://bbbc.broadinstitute.org/BBBC006",
        "metric": "qc_pipeline.compute_focus_score (Laplacian variance)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "per_plane": rows,
        "note": "Measured only on local extract. Positive Spearman means score falls as |z-z_focus| grows.",
    }
    out_json = out_dir / f"bbbc006_focus_sites{int(rank_total)}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== SUMMARY (measured only) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
