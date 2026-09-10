#!/usr/bin/env python3
"""Export OptiCell Stage-2 outputs to CTC result format (maskT.tif + res_track.txt).

Expected Stage-2 layout:
  out_dir/
    masks/{stem}_labels.png   # per-frame instance labels
    tracks.csv                # columns: frame, label, track_id, ...

Writes:
  res_dir/mask000.tif ...
  res_dir/res_track.txt       # L B E P (parent=0)

Guarantees traccuracy consistency: every track ID listed for frame t appears
as a positive label in mask t, and every positive mask label is listed.
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
    raise SystemExit("tifffile required: pip install tifffile") from exc


def _frame_from_stem(stem: str) -> int | None:
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def _read_label_mask(path: Path) -> np.ndarray:
    """Read instance label image; prefer tifffile for 16-bit safety."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        arr = np.asarray(tifffile.imread(str(path)))
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise IOError(f"Could not read {path}")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int32, copy=False)


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

    mask_paths: dict[int, Path] = {}
    for p in sorted(masks_dir.glob("*")):
        if p.suffix.lower() not in {".png", ".tif", ".tiff"}:
            continue
        stem = p.stem.replace("_labels", "")
        fr = _frame_from_stem(stem)
        if fr is None:
            continue
        mask_paths[fr] = p

    if not mask_paths:
        raise RuntimeError(f"No masks under {masks_dir}")

    res_dir = res_dir.resolve()
    res_dir.mkdir(parents=True, exist_ok=True)

    # Collect which (frame, track_id) pairs actually received pixels
    present: list[tuple[int, int]] = []
    skipped_missing_label = 0
    frames_sorted = sorted(mask_paths)

    for fr in frames_sorted:
        lab = _read_label_mask(mask_paths[fr])
        present_inst = set(int(x) for x in np.unique(lab) if int(x) > 0)

        sub = tracks[tracks["frame"] == fr]
        out = np.zeros(lab.shape, dtype=np.uint16)

        # Prefer explicit label→track_id from tracks; only if label exists in mask
        used_inst: set[int] = set()
        for _, row in sub.iterrows():
            inst = int(row["label"])
            tid = int(row["track_id"])
            if tid <= 0 or inst <= 0:
                continue
            if inst not in present_inst:
                skipped_missing_label += 1
                continue
            out[lab == inst] = np.uint16(tid)
            used_inst.add(inst)
            present.append((fr, tid))

        # Any mask instances not in tracks.csv get synthetic track IDs so
        # det_ids ⊆ track_ids for this frame. Use high IDs to avoid clash.
        # Better: drop orphan pixels (leave 0) and only list present track IDs.
        # Orphans would fail the opposite check if we left them labeled.
        # So zero-out orphans (already 0 if not written).

        out_path = res_dir / f"mask{fr:03d}.tif"
        tifffile.imwrite(str(out_path), out)

    if skipped_missing_label:
        print(
            f"NOTE: skipped {skipped_missing_label} track rows whose instance "
            f"label was absent from the saved mask (export consistency)."
        )

    # Rebuild lineage file strictly from IDs that appear in masks
    if not present:
        raise RuntimeError("No track IDs written into any mask — cannot export")

    present_df = pd.DataFrame(present, columns=["frame", "track_id"]).drop_duplicates()
    lines: list[str] = []
    for tid, g in present_df.groupby("track_id"):
        tid = int(tid)
        b = int(g["frame"].min())
        e = int(g["frame"].max())
        lines.append(f"{tid} {b} {e} 0")
    lines.sort(key=lambda s: int(s.split()[0]))
    (res_dir / "res_track.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Verify CTC consistency before caller runs traccuracy
    for fr in frames_sorted:
        mask = np.asarray(tifffile.imread(str(res_dir / f"mask{fr:03d}.tif")))
        det_ids = set(int(x) for x in np.unique(mask) if int(x) > 0)
        track_ids = set(
            int(t)
            for t, g in present_df.groupby("track_id")
            if int(g["frame"].min()) <= fr <= int(g["frame"].max())
        )
        # CTC requires IDs present on this frame in both places
        on_frame = set(present_df.loc[present_df["frame"] == fr, "track_id"].astype(int))
        if on_frame != det_ids:
            raise RuntimeError(
                f"Internal consistency failed at t={fr}: "
                f"tracks-only={on_frame - det_ids} mask-only={det_ids - on_frame}"
            )

    print(f"Wrote {len(frames_sorted)} masks + res_track.txt → {res_dir}")
    print(f"  unique track IDs exported: {present_df['track_id'].nunique()}")
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
