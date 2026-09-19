#!/usr/bin/env python3
"""Virelion-OptiCell — Easy UI (Stage 4).

Upload a microscopy image → acquisition QC → segment → acceptance → CSV/overlay.

Run:
    pip install -e '.[ui,cellpose]'
    streamlit run app_streamlit.py

See docs/DEMO_5_MIN.md for the 5-minute checklist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qc_pipeline import (  # noqa: E402
    to_grayscale_uint8,
    segment_threshold,
    CellposeSegmenter,
    SegmentationResult,
    _HAS_CELLPOSE,
    _CELLPOSE_IMPORT_ERROR,
)
from ensemble import (  # noqa: E402
    hybrid_threshold_cellpose,
    fov_confidence,
    suggest_backend,
    image_looks_low_contrast,
)
from acceptance import segmentation_acceptance, SegmentationAcceptance  # noqa: E402
from artifact_quality import acquisition_artifact_metrics, artifact_burden_score  # noqa: E402
from quantitative import object_channel_intensity  # noqa: E402

BACKENDS = ["auto", "threshold", "adaptive", "cellpose", "hybrid"]


def run_pipeline(
    raw_image: np.ndarray,
    backend: str,
    cellpose_segmenter: "CellposeSegmenter | None" = None,
) -> dict:
    """Run acquisition QC + segmentation + segmentation QC on one image."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")

    gray = to_grayscale_uint8(raw_image)
    backend_reason = f"user requested {backend}"
    resolved = backend
    if backend == "auto":
        resolved, backend_reason = suggest_backend(gray, cellpose_available=_HAS_CELLPOSE)

    acq_metrics = acquisition_artifact_metrics(raw_image)
    acq_score = artifact_burden_score(acq_metrics)

    if resolved == "threshold":
        seg = segment_threshold(gray)
    elif resolved == "adaptive":
        seg = segment_threshold(gray, adaptive=True)
    elif resolved == "cellpose":
        if cellpose_segmenter is None:
            raise RuntimeError("cellpose backend requested but no CellposeSegmenter was provided")
        seg = cellpose_segmenter.segment(gray)
    else:  # hybrid
        seg = hybrid_threshold_cellpose(gray, cellpose_segmenter=cellpose_segmenter)

    conf = fov_confidence(gray, seg.labels)

    accept: SegmentationAcceptance | None = None
    if seg.error is None:
        accept = segmentation_acceptance(
            quality_score=seg.quality_score,
            border_fraction=seg.border_fraction,
            tiny_object_fraction=seg.tiny_object_fraction,
            merged_object_fraction=seg.merged_object_fraction,
        )

    obj_df = object_channel_intensity(gray, seg.labels) if seg.count > 0 else pd.DataFrame()

    return {
        "gray": gray,
        "acq_metrics": acq_metrics,
        "acq_score": acq_score,
        "seg": seg,
        "conf": conf,
        "accept": accept,
        "obj_df": obj_df,
        "backend_resolved": resolved,
        "backend_reason": backend_reason,
        "low_contrast": image_looks_low_contrast(gray),
    }


def overlay_labels(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Draw object boundaries over the grayscale image; returns an RGB array."""
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        mask = (labels == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base, contours, -1, (0, 255, 60), 1)
    return cv2.cvtColor(base, cv2.COLOR_BGR2RGB)


def _main() -> None:
    import streamlit as st

    st.set_page_config(page_title="OptiCell — Easy UI", layout="wide")
    st.title("OptiCell — Easy UI")
    st.caption(
        "Upload an image, run a real backend, see every QC signal OptiCell already computes. "
        "Nothing on this page is invented. Demo checklist: docs/DEMO_5_MIN.md"
    )

    with st.sidebar:
        st.header("Backend")
        backend = st.selectbox(
            "Segmentation backend",
            BACKENDS,
            index=0,
            help=(
                "auto → Cellpose if installed (always on low-contrast / phase-like). "
                "threshold/adaptive need no extra install. cellpose/hybrid need "
                "`pip install -e '.[cellpose]'."
            ),
        )
        gpu = st.checkbox("Use GPU (cellpose/hybrid/auto)", value=_HAS_CELLPOSE)
        needs_cellpose = backend in ("cellpose", "hybrid", "auto")
        if needs_cellpose and not _HAS_CELLPOSE and backend != "auto":
            st.error(f"Cellpose not installed: {_CELLPOSE_IMPORT_ERROR}")
        elif not _HAS_CELLPOSE:
            st.info("Cellpose not installed — auto will use threshold. `pip install -e '.[cellpose]'`")

    uploaded = st.file_uploader(
        "Upload a microscopy image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    )
    if uploaded is None:
        st.info("Upload an image to get started. Prefer backend **auto** for unknown modalities.")
        st.markdown(
            "**Quick checklist:** [docs/DEMO_5_MIN.md](https://github.com/Virelion-Biotech/Virelion-OptiCell/blob/main/docs/DEMO_5_MIN.md)"
        )
        st.stop()

    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if raw is None:
        st.error("Could not decode this file as an image.")
        st.stop()

    # Pre-resolve auto so we know whether to load Cellpose before running
    gray_probe = to_grayscale_uint8(raw)
    if backend == "auto":
        resolved_probe, _ = suggest_backend(gray_probe, cellpose_available=_HAS_CELLPOSE)
    else:
        resolved_probe = backend

    cellpose_segmenter = None
    if resolved_probe in ("cellpose", "hybrid") or backend in ("cellpose", "hybrid"):
        if not _HAS_CELLPOSE:
            if backend != "auto":
                st.error(f"Cellpose required: {_CELLPOSE_IMPORT_ERROR}")
                st.stop()
        else:

            @st.cache_resource
            def _get_segmenter(model_type: str, use_gpu: bool) -> CellposeSegmenter:
                return CellposeSegmenter(model_type=model_type, gpu=use_gpu)

            with st.spinner("Loading Cellpose model (cached after first run)..."):
                cellpose_segmenter = _get_segmenter("cpsam", gpu)

    with st.spinner(f"Running {backend}..."):
        result = run_pipeline(raw, backend, cellpose_segmenter=cellpose_segmenter)

    seg: SegmentationResult = result["seg"]

    st.info(
        f"**Backend:** `{result['backend_resolved']}` — {result['backend_reason']}  \n"
        f"Low-contrast heuristic: **{result['low_contrast']}**"
    )

    st.subheader("1. Acquisition QC (raw image, before any segmentation)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Acquisition quality", f"{result['acq_score']:.0f} / 100")
    with c2:
        st.dataframe(
            pd.DataFrame([result["acq_metrics"]]).T.rename(columns={0: "value"}),
            width="stretch",
        )

    st.subheader("2. Segmentation")
    if seg.error:
        st.error(f"Segmentation failed: {seg.error}")
        st.stop()
    col_img1, col_img2 = st.columns(2)
    overlay = overlay_labels(result["gray"], seg.labels)
    with col_img1:
        st.image(result["gray"], caption="Input (grayscale)", width="stretch", clamp=True)
    with col_img2:
        st.image(
            overlay,
            caption=f"{seg.method}: {seg.count} objects",
            width="stretch",
        )

    # Overlay download (PNG bytes)
    ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if ok:
        st.download_button(
            "Download overlay PNG",
            data=buf.tobytes(),
            file_name="opticell_overlay.png",
            mime="image/png",
        )

    st.subheader("3. Segmentation QC")
    accept = result["accept"]
    conf = result["conf"]
    if accept is not None:
        msg = f"Acceptance gate: {accept.status} — {accept.reason}"
        if accept.status == "PASS":
            st.success(msg)
        elif accept.status == "REVIEW":
            st.warning(msg)
        else:
            st.error(msg)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Count", seg.count)
    m2.metric("Quality score", f"{seg.quality_score:.0f}")
    m3.metric("Confidence", f"{conf['confidence_score']:.0f}")
    m4.metric("Border frac.", f"{seg.border_fraction:.2f}")
    m5.metric("Tiny-object frac.", f"{seg.tiny_object_fraction:.2f}")
    if conf["flags"]:
        st.warning(f"Confidence flags: {conf['flags']}")

    st.subheader("4. Per-object measurements")
    obj_df = result["obj_df"]
    if not obj_df.empty:
        st.dataframe(obj_df, width="stretch", height=300)
        st.download_button(
            "Download per-object CSV",
            obj_df.to_csv(index=False),
            file_name="opticell_objects.csv",
        )
        if "mean_intensity" in obj_df.columns and "label" in obj_df.columns:
            st.bar_chart(obj_df.set_index("label")["mean_intensity"])
    else:
        st.info("No objects detected.")

    with st.expander("Raw SegmentationResult"):
        st.json(
            {k: v for k, v in seg.__dict__.items() if k != "labels"}
            | {"labels_shape": list(seg.labels.shape)}
        )


if __name__ == "__main__":
    _main()
