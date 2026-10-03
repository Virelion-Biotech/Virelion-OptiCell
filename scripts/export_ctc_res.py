#!/usr/bin/env python3
"""Export OptiCell Stage-2 outputs to CTC result format (maskT.tif + res_track.txt).

CTC rule: if res_track.txt says track L spans frames B..E, label L must appear
in *every* maskT for T in [B, E]. OptiCell tracks can have gaps, so we split
each track_id into continuous tracklets and assign each tracklet a unique CTC ID.

Stage-2 layout:
  masks/{stem}_labels.png  +  tracks.csv (frame, label, track_id)
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


def _continuous_runs(frames: list[int]) -> list[tuple[int, int, list[int]]]:
    """Split sorted frame list into contiguous runs. Returns (B, E, frames_in_run)."""
    if not frames:
        return []
    frames = sorted(set(int(f) for f in frames))
    runs: list[tuple[int, int, list[int]]] = []
    start = frames[0]
    prev = frames[0]
    buf = [frames[0]]
    for f in frames[1:]:
        if f == prev + 1:
            buf.append(f)
            prev = f
        else:
            runs.append((start, prev, buf))
            start = f
            prev = f
            buf = [f]
    runs.append((start, prev, buf))
    return runs


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

    # Keep only track rows whose instance label exists on that frame's mask
    valid_rows: list[dict] = []
    skipped = 0
    label_cache: dict[int, np.ndarray] = {}
    present_inst_cache: dict[int, set[int]] = {}

    for fr, path in mask_paths.items():
        lab = _read_label_mask(path)
        label_cache[fr] = lab
        present_inst_cache[fr] = {int(x) for x in np.unique(lab) if int(x) > 0}

    for _, row in tracks.iterrows():
        fr = int(row["frame"])
        inst = int(row["label"])
        tid = int(row["track_id"])
        if fr not in present_inst_cache or tid <= 0 or inst <= 0:
            skipped += 1
            continue
        if inst not in present_inst_cache[fr]:
            skipped += 1
            continue
        valid_rows.append({"frame": fr, "label": inst, "track_id": tid})

    if not valid_rows:
        raise RuntimeError("No valid track rows after mask intersection")

    valid = pd.DataFrame(valid_rows)
    if skipped:
        print(f"NOTE: dropped {skipped} track rows with missing instance pixels")

    # Split each OptiCell track_id into continuous CTC tracklets
    # Map (frame, opticell_track_id) -> ctc_id
    pair_to_ctc: dict[tuple[int, int], int] = {}
    track_lines: list[str] = []  # "L B E 0"
    next_ctc = 1
    n_splits = 0

    for oid, g in valid.groupby("track_id", sort=True):
        frames = sorted(int(x) for x in g["frame"].unique())
        runs = _continuous_runs(frames)
        if len(runs) > 1:
            n_splits += len(runs) - 1
        for b, e, run_frames in runs:
            ctc_id = next_ctc
            next_ctc += 1
            for fr in run_frames:
                pair_to_ctc[(fr, int(oid))] = ctc_id
            track_lines.append(f"{ctc_id} {b} {e} 0")

    res_dir = res_dir.resolve()
    res_dir.mkdir(parents=True, exist_ok=True)

    # Write masks with CTC IDs
    frames_sorted = sorted(mask_paths)
    for fr in frames_sorted:
        lab = label_cache[fr]
        out = np.zeros(lab.shape, dtype=np.uint16)
        sub = valid[valid["frame"] == fr]
        for _, row in sub.iterrows():
            oid = int(row["track_id"])
            inst = int(row["label"])
            ctc_id = pair_to_ctc.get((fr, oid))
            if ctc_id is None:
                continue
            out[lab == inst] = np.uint16(ctc_id)
        tifffile.imwrite(str(res_dir / f"mask{fr:03d}.tif"), out)

    track_lines.sort(key=lambda s: int(s.split()[0]))
    (res_dir / "res_track.txt").write_text("\n".join(track_lines) + "\n", encoding="utf-8")

    # Hard consistency check (CTC semantics)
    for fr in frames_sorted:
        mask = np.asarray(tifffile.imread(str(res_dir / f"mask{fr:03d}.tif")))
        det_ids = {int(x) for x in np.unique(mask) if int(x) > 0}
        track_ids = set()
        for line in track_lines:
            L, B, E, _P = line.split()
            L, B, E = int(L), int(B), int(E)
            if B <= fr <= E:
                track_ids.add(L)
        if track_ids != det_ids:
            raise RuntimeError(
                f"CTC consistency failed at t={fr}: "
                f"in_tracks_not_mask={track_ids - det_ids} "
                f"in_mask_not_tracks={det_ids - track_ids}"
            )

    print(f"Wrote {len(frames_sorted)} masks + res_track.txt → {res_dir}")
    print(f"  CTC tracklets: {len(track_lines)} (gap splits: {n_splits})")
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
