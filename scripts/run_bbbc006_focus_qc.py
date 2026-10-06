#!/usr/bin/env python3
"""BBBC006 focus QC on measured z planes; preserve physical well/site identity.

Example: --root data/bbbc006 --max-sites 0 --expected-z 0 8 16 24 33
The expert in-focus interval is z=11..23. z=16 is the laser autofocus target.
Selected-plane results must not be presented as a complete 34-plane study.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from qc_pipeline import compute_focus_score, to_grayscale_uint8  # noqa: E402

Z_RE = re.compile(r"(?:^|[_/\\-])z[_-]?(\d{1,2})(?:[_/\\-]|\.|$)", re.I)
SITE_RE = re.compile(r"(.+_[a-p]\d{2}_s\d+)_w[12]", re.I)


def parse_z(path: Path) -> int | None:
    for part in [path.name, *path.parts[::-1]]:
        match = Z_RE.search(str(part))
        if match:
            return int(match.group(1))
    return None


def site_key(path: Path) -> str:
    # UUIDs after w1 identify individual images, not a physical z-stack.
    match = SITE_RE.match(path.stem)
    return match.group(1).lower() if match else Z_RE.sub("_zXX_", path.stem)


def load_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(path)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    return to_grayscale_uint8(arr)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--z-focus", type=int, default=16)
    parser.add_argument("--max-sites", type=int, default=30, help="0 scores every physical site")
    parser.add_argument("--expected-z", type=int, nargs="+", help="Required plane set per physical site")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/bbbc006_focus_qc"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2
    sites = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"} or "__MACOSX" in path.parts:
            continue
        if "w2" in path.name.lower() and "w1" not in path.name.lower():
            continue
        z = parse_z(path)
        if z is not None:
            sites[site_key(path)].append((path, z))
    keys = sorted(sites)
    if args.max_sites > 0:
        keys = keys[: args.max_sites]
    expected = set(args.expected_z) if args.expected_z else {z for stack in sites.values() for _, z in stack}
    rows, site_rows, failures = [], [], []
    for key in keys:
        stack = sorted(sites[key], key=lambda item: item[1])
        zs = [z for _, z in stack]
        if set(zs) != expected or len(zs) != len(set(zs)):
            failures.append(dict(site=key, error="Missing or duplicate z planes", observed=zs))
            continue
        scores = []
        for path, z in stack:
            try:
                score = float(compute_focus_score(load_gray(path)))
                if not np.isfinite(score):
                    raise ValueError("Non-finite focus score")
            except Exception as exc:
                failures.append(dict(site=key, file=path.name, error=str(exc)))
                continue
            scores.append((z, score))
            rows.append(
                dict(
                    site=key,
                    z=z,
                    focus_score=score,
                    abs_dz=abs(z - args.z_focus),
                    expert_in_focus=11 <= z <= 23,
                    file=path.name,
                )
            )
        if len(scores) != len(expected):
            continue
        zs, focus = np.array(scores).T
        best = int(zs[np.argmax(focus)])
        site_rows.append(
            dict(
                site=key,
                n_planes=len(scores),
                best_z=best,
                within_2_planes=abs(best - args.z_focus) <= 2,
                spearman=spearman_corr(focus, -np.abs(zs - args.z_focus)),
            )
        )
    summary = dict(
        n_planes_scored=len(rows),
        n_sites=len(site_rows),
        n_requested_sites=len(keys),
        planes=sorted(expected),
        z_focus=args.z_focus,
        frac_argmax_within_2_planes=float(np.mean([r["within_2_planes"] for r in site_rows])) if site_rows else None,
    )
    if rows:
        focus = np.array([r["focus_score"] for r in rows])
        truth = np.array([r["expert_in_focus"] for r in rows])
        npos, nneg = int(truth.sum()), int((~truth).sum())
        summary["expert_in_focus_auroc"] = (
            float((rankdata(focus)[truth].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None
        )
        summary["fixed_focus_threshold"] = 100.0
        summary["sensitivity_at_100"] = float(np.mean(focus[truth] >= 100)) if npos else None
        summary["specificity_at_100"] = float(np.mean(focus[~truth] < 100)) if nneg else None
        summary["spearman_focus_vs_neg_abs_dz"] = spearman_corr(focus, -np.array([r["abs_dz"] for r in rows]))
    payload = dict(
        dataset="BBBC006",
        source="https://bbbc.broadinstitute.org/BBBC006",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        summary=summary,
        complete=bool(site_rows) and not failures,
        failures=failures,
        per_plane=rows,
        per_site=site_rows,
        note=("Measured selected planes. Expert focus labels; automated segmentation labels are not expert "
              "segmentation truth. Sites share one plate and are not independent biological replicates."),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"bbbc006_focus_sites{len(site_rows)}.json"
    out.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    return 0 if payload["complete"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
