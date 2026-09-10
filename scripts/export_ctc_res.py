#!/usr/bin/env python3
"""Export OptiCell Stage-2 outputs to CTC result format (maskT.tif + res_track.txt).

Expected Stage-2 layout:
  out_dir/
    masks/{stem}_labels.png   # per-frame instance labels (from segmentation)
    tracks.csv                # columns: frame, label, track_id, ...
    workflow_summary.csv      # optional; used to map frame index → image stem

Writes:
  res_dir/mask000.tif ...
  res_dir/res_track.txt       # L B E P (parent=0; no division model yet)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import tifffile
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tifffile required: pip install tifffile") from exp


def _frame_from_stem(stem: str) -> int | None:
    # t000, t000_labels, mask000, man_track000
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def export_ctc_res(
    stage2_dir: Path,
    res_dir: Path,
    tracks_csv: Path | None = None,
    masks_dir: Path | None = None,
) -> Path:
    stage2_dir = stage2_dir.resolve()
    tracks_csv = (tracks_csv or stage2_dir / "tracks.csv").resolve()
    masks_dir = (masks_dir or stage2_dir / "masks").resolve()
    if not tracks_csv.is_file():
        raise FileNotFoundError(tracks_csv)
    if not masks_dir.is_dir():
        raise FileNotFoundError(masks_dir)

    tracks = pd.read_csv(tracks_csv)
    for col in ("frame", "label", "track_id"):
        if col not in tracks.columns:
            raise ValueError(f"tracks.csv missing {col}")

    # Map frame index → mask path
    mask_paths: dict[int, Path] = {}
    for p in sorted(masks_dir.glob("*")):
        if p.suffix.lower() not in {".png", ".tif", ".tiff"}:
            continue
        # strip _labels suffix if present
        stem = p.stem.replace("_labels", "")
        fr = _frame_from_stem(stem)
        if fr is None:
            continue
        mask_paths[fr] = p

    if not mask_paths:
        raise RuntimeError(f"No masks under {masks_dir}")

    res_dir = res_dir.resolve()
    res_dir.mkdir(parents=True, exist_ok=True)

    # Per-frame: remap instance label → track_id
    frames_sorted = sorted(mask_paths)
    for fr in frames_sorted:
        lab = cv2.imread(str(mask_paths[fr]), cv2.IMREAD_UNCHANGED)
        if lab is None:
            raise IOError(mask_paths[fr])
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        lab = lab.astype(np.int32)

        sub = tracks[tracks["frame"] == fr]
        out = np.zeros_like(lab, dtype=np.uint16)
        for _, row in sub.iterrows():
            inst = int(row["label"])
            tid = int(row["track_id"])
            if tid <= 0 or inst <= 0:
                continue
            out[lab == inst] = np.uint16(tid)

        out_path = res_dir / f"mask{fr:03d}.tif"
        tifffile.imwrite(str(out_path), out)

    # res_track.txt: L B E P (no parent links from current linker)
    lines = []
    for tid, g in tracks.groupby("track_id"):
        tid = int(tid)
        if tid <= 0:
            continue
        b = int(g["frame"].min())
        e = int(g["frame"].max())
        lines.append(f"{tid} {b} {e} 0")
    lines.sort(key=lambda s: int(s.split()[0]))
    (res_dir / "res_track.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(frames_sorted)} masks + res_track.txt → {res_dir}")
    return res_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Export Stage-2 run to CTC RES format")
    p.add_argument("stage2_dir", type=Path)
    p.add_argument("-o", "--res-dir", type=Path, required=True)
    args = p.parse_args()
    export_ctc_res(args.stage2_dir, args.res_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
