"""SEG-style scoring against CTC's sparsely annotated gold segmentation frames.

Only annotated GT objects contribute. This is an independent Python scorer,
not a claim of evaluation by the official CTC challenge executable.
"""

from pathlib import Path
import re

import numpy as np
import tifffile


def seg_object_scores(prediction: np.ndarray, truth: np.ndarray) -> list[float]:
    if prediction.shape != truth.shape:
        raise ValueError("CTC prediction and gold segmentation shapes differ")
    for arr in [prediction, truth]:
        if not np.issubdtype(arr.dtype, np.integer) or np.any(arr < 0):
            raise ValueError("CTC labels require nonnegative integer IDs")
    pred_ids, p = np.unique(np.r_[0, prediction.ravel()], return_inverse=True)
    gt_ids, g = np.unique(np.r_[0, truth.ravel()], return_inverse=True)
    table = np.bincount(g[1:] * len(pred_ids) + p[1:], minlength=len(gt_ids) * len(pred_ids))
    table = table.reshape(len(gt_ids), len(pred_ids))
    ga, pa = table.sum(1), table.sum(0)
    scores = []
    for idx in range(1, len(gt_ids)):
        overlap = table[idx, 1:]
        admissible = overlap > 0.5 * ga[idx]
        iou = overlap / (ga[idx] + pa[1:] - overlap)
        scores.append(float(np.max(np.where(admissible, iou, 0))) if iou.size else 0.0)
    return scores


def score_seg_directory(gold: Path, prediction: Path) -> dict:
    frames, scores = [], []
    paths = sorted(gold.glob("man_seg*.tif"))
    if not paths:
        raise ValueError(f"No gold SEG frames at {gold}")
    for path in paths:
        match = re.fullmatch(r"man_seg(\d+)\.tif", path.name)
        if not match:
            raise ValueError(f"Unsupported slice annotation name {path.name}; use a 2D dataset")
        t = int(match.group(1))
        options = list(prediction.glob(f"mask{t:03d}.tif"))
        if len(options) != 1:
            raise FileNotFoundError(f"Missing predicted SEG frame {t}")
        values = seg_object_scores(tifffile.imread(options[0]), tifffile.imread(path))
        frames.append(dict(frame=t, n_gold_objects=len(values), seg=sum(values) / len(values) if values else None))
        scores.extend(values)
    return dict(
        seg_style_mean_iou=float(np.mean(scores)) if scores else None,
        n_gold_seg_frames=len(frames),
        n_gold_seg_objects=len(scores),
        per_frame=frames,
        scorer="Independent Python SEG-style; overlap > 50% GT; no challenge executable",
    )
