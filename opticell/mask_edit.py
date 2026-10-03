"""Pure instance-mask editing operations used by the review UI."""
from __future__ import annotations

import cv2
import numpy as np


def _validated(labels: np.ndarray, stroke_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(labels)
    stroke = np.asarray(stroke_mask, dtype=bool)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("labels must be a non-empty 2-D array")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must contain integer instance IDs")
    if (values < 0).any():
        raise ValueError("labels cannot contain negative instance IDs")
    if stroke.shape != values.shape:
        raise ValueError("stroke_mask must match labels shape")
    return values.astype(np.int32, copy=True), stroke


def apply_brush_edit(
    labels: np.ndarray,
    stroke_mask: np.ndarray,
    *,
    mode: str,
    label_id: int | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Apply one brush layer to an instance mask.

    Modes:
    - add paints the selected existing instance ID.
    - erase sets painted pixels to background.
    - new_object assigns each disconnected painted component a new ID.

    Returns (corrected_labels, affected_or_created_ids).
    """
    result, stroke = _validated(labels, stroke_mask)
    if mode not in {"add", "erase", "new_object"}:
        raise ValueError("mode must be add, erase, or new_object")
    if not stroke.any():
        return result, ()

    if mode == "erase":
        affected = tuple(int(x) for x in np.unique(result[stroke]) if x > 0)
        result[stroke] = 0
        return result, affected

    if mode == "add":
        if not isinstance(label_id, (int, np.integer)) or isinstance(label_id, bool) or int(label_id) <= 0:
            raise ValueError("label_id must be a positive integer for add mode")
        label_id = int(label_id)
        if not np.any(result == label_id):
            raise ValueError(f"label_id {label_id} does not exist in the current mask")
        result[stroke] = label_id
        return result, (label_id,)

    count, components = cv2.connectedComponents(stroke.astype(np.uint8), connectivity=8)
    next_id = int(result.max()) + 1
    created: list[int] = []
    for component_id in range(1, count):
        component = components == component_id
        if not component.any():
            continue
        result[component] = next_id
        created.append(next_id)
        next_id += 1
    return result, tuple(created)


def instance_count(labels: np.ndarray) -> int:
    """Count positive instance IDs without assuming IDs are contiguous."""
    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a 2-D integer array")
    positive = values[values > 0]
    return int(np.unique(positive).size) if positive.size else 0


__all__ = ["apply_brush_edit", "instance_count"]
