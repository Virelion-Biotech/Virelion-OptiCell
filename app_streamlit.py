#!/usr/bin/env python3
"""Virelion-OptiCell — Easy UI (Stage 4 of docs/PRODUCT_ROADMAP.md).

Local Streamlit app: upload a microscopy image, pick a segmentation backend, see
live segmentation plus every QC signal the library already computes. Nothing
here is invented — every number shown is the direct output of qc_pipeline.py /
ensemble.py / acceptance.py / artifact_quality.py / quantitative.py.

Run:
    pip install -e '.[dev,ui]'
    streamlit run app_streamlit.py

Optional (Cellpose/hybrid backends, needs a lot more disk + ideally a GPU):
    pip install -e '.[cellpose]'
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
from ensemble import hybrid_threshold_cellpose, fov_confidence  # noqa: E402
from acceptance import segmentation_acceptance, SegmentationAcceptance  # noqa: E402
from artifact_quality import acquisition_artifact_metrics, artifact_burden_score  # noqa: E402
from quantitative import object_channel_intensity  # noqa: E402

BACKENDS = ["threshold", "adaptive", "cellpose", "hybrid"]


# --------------------------------------------------------------------------
# Pure pipeline logic — no Streamlit calls in here, so it can be unit-tested
# and reused (e.g. from a batch script) without a Streamlit runtime.
# --------------------------------------------------------------------------
def run_pipeline(
    raw_image: np.ndarray,
    backend: str,
    cellpose_segmenter: "CellposeSegmenter | None" = None,
) -> dict:
    """Run acquisition QC + segmentation + segmentation QC on one image."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")

    gray = to_grayscale_uint8(raw_image)

    acq_metrics = acquisition_artifact_metrics(raw_image)
    acq_score = artifact_burden_score(acq_metrics)

    if backend == "threshold":
        seg = segment_threshold(gray)
    elif backend == "adaptive":
        seg = segment_threshold(gray, adaptive=True)
    elif backend == "cellpose":
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


# --------------------------------------------------------------------------
# Streamlit UI — thin rendering layer over run_pipeline().
# --------------------------------------------------------------------------
def _main() -> None:
    import streamlit as st

    st.set_page_config(page_title="OptiCell — Easy UI", layout="wide")
    st.title("OptiCell — Easy UI")
    st.caption(
        "Upload an image, run a real backend, see every QC signal OptiCell already "
        "computes. Nothing on this page is invented."
    )

    with st.sidebar:
        st.header("Backend")
        backend = st.selectbox(
            "Segmentation backend",
            BACKENDS,
            help=(
                "threshold/adaptive need no extra install. cellpose/hybrid need "
                "`pip install -e '.[cellpose]'."
            ),
        )
        gpu = st.checkbox("Use GPU (cellpose/hybrid)", value=False)
        needs_cellpose = backend in ("cellpose", "hybrid")
        if needs_cellpose and not _HAS_CELLPOSE:
            st.error(f"Cellpose not installed: {_CELLPOSE_IMPORT_ERROR}")

    uploaded = st.file_uploader(
        "Upload a microscopy image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    )
    if uploaded is None:
        st.info("Upload an image to get started.")
        st.stop()

    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if raw is None:
        st.error("Could not decode this file as an image.")
        st.stop()

    cellpose_segmenter = None
    if needs_cellpose:
        if not _HAS_CELLPOSE:
            st.stop()

        @st.cache_resource
        def _get_segmenter(model_type: str, gpu: bool) -> CellposeSegmenter:
            return CellposeSegmenter(model_type=model_type, gpu=gpu)

        with st.spinner("Loading Cellpose model (cached after first run)..."):
            cellpose_segmenter = _get_segmenter("cpsam", gpu)

    with st.spinner(f"Running {backend}..."):
        result = run_pipeline(raw, backend, cellpose_segmenter=cellpose_segmenter)

    seg: SegmentationResult = result["seg"]

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
    with col_img1:
        st.image(result["gray"], caption="Input (grayscale)", width="stretch", clamp=True)
    with col_img2:
        st.image(
            overlay_labels(result["gray"], seg.labels),
            caption=f"{seg.method}: {seg.count} objects",
            width="stretch",
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
            "Download per-object CSV", obj_df.to_csv(index=False), file_name="opticell_objects.csv"
        )
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
