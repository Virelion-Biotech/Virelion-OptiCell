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

    st.set_page_config(page_title="OptiCell — Easy UI", page_icon="🔬", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background:#f6f8fb; }
        .block-container { max-width:1500px; padding-top:2rem; padding-bottom:3rem; }
        .hero { padding:1.35rem 1.5rem; border-radius:18px;
          background:linear-gradient(135deg,#102a43,#174a5b 58%,#1f6f78);
          color:white; margin-bottom:1.25rem; }
        .hero h1 { margin:0; font-size:2rem; letter-spacing:-.03em; }
        .hero p { margin:.4rem 0 0; opacity:.82; }
        .eyebrow { text-transform:uppercase; font-size:.72rem; letter-spacing:.12em; font-weight:700; opacity:.72; }
        .status { display:inline-flex; gap:.45rem; padding:.35rem .65rem; border-radius:999px;
          background:rgba(255,255,255,.12); font-size:.8rem; }
        .dot { width:8px; height:8px; border-radius:50%; background:#66e3a4; display:inline-block; }
        .pill { display:inline-block; padding:.22rem .55rem; border-radius:999px;
          background:#e8eef5; color:#35546f; font-size:.75rem; font-weight:650; margin-right:.25rem; }
        div[data-testid="stMetric"] { background:white; border:1px solid #e5eaf0; border-radius:14px; padding:.65rem .8rem; }
        div[data-testid="stFileUploader"] { background:white; border:1px dashed #b9c6d4; border-radius:14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Virelion Biotech · Quantitative Microscopy</div>
          <h1>OptiCell Research Workbench</h1>
          <p>From raw microscopy to segmentation, quality gates, and reproducible measurements.</p>
          <div style="margin-top:.8rem"><span class="status"><span class="dot"></span> Analysis engine ready</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Analysis setup")
        backend = st.selectbox(
            "Segmentation backend",
            BACKENDS,
            index=0,
            help="Auto uses OptiCell's existing routing logic.",
        )
        gpu = st.checkbox("Use GPU", value=_HAS_CELLPOSE, disabled=not _HAS_CELLPOSE)
        if not _HAS_CELLPOSE:
            st.info("Cellpose is unavailable. Auto will use a non-Cellpose backend.")
            st.caption("Install with: pip install -e '.[cellpose]'")
        st.divider()
        st.markdown("**Workflow**")
        st.caption("01 · Upload\n\n02 · QC\n\n03 · Segment\n\n04 · Review\n\n05 · Export")

    uploaded = st.file_uploader(
        "Drop a microscopy image here",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="PNG/JPEG/TIFF/BMP are supported.",
    )
    if uploaded is None:
        a, b = st.columns([1.5, 1])
        with a:
            st.subheader("Start with one field of view")
            st.caption(
                "OptiCell checks acquisition quality before segmentation, then exposes "
                "the backend decision, confidence, acceptance gate, and measurements."
            )
        with b:
            st.info("Tip: choose auto for an unknown microscopy modality.")
        st.stop()

    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if raw is None:
        st.error("Could not decode this file as an image.")
        st.stop()

    gray_probe = to_grayscale_uint8(raw)
    if backend == "auto":
        resolved_probe, probe_reason = suggest_backend(gray_probe, cellpose_available=_HAS_CELLPOSE)
    else:
        resolved_probe, probe_reason = backend, "user requested " + backend

    cellpose_segmenter = None
    if resolved_probe in ("cellpose", "hybrid"):
        if not _HAS_CELLPOSE:
            st.error("Cellpose required: " + str(_CELLPOSE_IMPORT_ERROR))
            st.stop()

        @st.cache_resource
        def _get_segmenter(model_type: str, use_gpu: bool) -> CellposeSegmenter:
            return CellposeSegmenter(model_type=model_type, gpu=use_gpu)

        with st.spinner("Preparing Cellpose model…"):
            cellpose_segmenter = _get_segmenter("cpsam", gpu)

    with st.spinner("Running OptiCell analysis…"):
        result = run_pipeline(raw, backend, cellpose_segmenter=cellpose_segmenter)

    seg: SegmentationResult = result["seg"]
    if seg.error:
        st.error("Segmentation failed: " + str(seg.error))
        st.stop()

    accept = result["accept"]
    conf = result["conf"]
    overlay = overlay_labels(result["gray"], seg.labels)

    st.markdown(
        '<span class="pill">File · ' + uploaded.name + '</span>'
        '<span class="pill">Backend · ' + str(result["backend_resolved"]) + '</span>'
        '<span class="pill">Image · ' + str(raw.shape[1]) + ' × ' + str(raw.shape[0]) + '</span>'
        '<span class="pill">Low contrast · ' + str(result["low_contrast"]) + '</span>',
        unsafe_allow_html=True,
    )

    st.subheader("Run summary")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Objects", seg.count)
    q2.metric("Segmentation quality", f"{seg.quality_score:.0f}")
    q3.metric("Confidence", f"{conf['confidence_score']:.0f}")
    q4.metric("Acquisition quality", f"{result['acq_score']:.0f}")
    q5.metric("Gate", accept.status if accept else "N/A")

    if accept:
        gate_text = "**" + accept.status + "** · " + accept.reason
        if accept.status == "PASS":
            st.success(gate_text)
        elif accept.status == "REVIEW":
            st.warning(gate_text)
        else:
            st.error(gate_text)

    tabs = st.tabs(["🔬 Field of view", "📊 QC & measurements", "⚙️ Run details"])

    with tabs[0]:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Input**")
            st.image(result["gray"], width="stretch", clamp=True)
        with c2:
            st.markdown("**Segmentation · " + str(seg.method) + " · " + str(seg.count) + " objects**")
            st.image(overlay, width="stretch")
        ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if ok:
            st.download_button(
                "Download overlay PNG",
                data=png.tobytes(),
                file_name="opticell_overlay.png",
                mime="image/png",
            )

    with tabs[1]:
        c1, c2 = st.columns([1, 1.25], gap="large")
        with c1:
            st.markdown("**Segmentation QC**")
            qc_df = pd.DataFrame({
                "Signal": ["Quality score", "Confidence", "Border fraction",
                           "Tiny-object fraction", "Merged-object fraction"],
                "Value": [
                    f"{seg.quality_score:.2f}",
                    f"{conf['confidence_score']:.2f}",
                    f"{seg.border_fraction:.3f}",
                    f"{seg.tiny_object_fraction:.3f}",
                    f"{seg.merged_object_fraction:.3f}",
                ],
            })
            st.dataframe(qc_df, hide_index=True, width="stretch")
            if conf["flags"]:
                st.warning("Confidence flags: " + ", ".join(conf["flags"]))
        with c2:
            st.markdown("**Acquisition QC**")
            acq_df = pd.DataFrame([
                {"Signal": k.replace("_", " ").title(), "Value": v}
                for k, v in result["acq_metrics"].items()
            ])
            st.dataframe(acq_df, hide_index=True, width="stretch")

        st.markdown("**Per-object measurements**")
        obj_df = result["obj_df"]
        if obj_df.empty:
            st.info("No objects detected.")
        else:
            st.dataframe(obj_df, width="stretch", height=320)
            if "mean_intensity" in obj_df.columns and "label" in obj_df.columns:
                st.line_chart(obj_df.set_index("label")["mean_intensity"])
            st.download_button(
                "Download per-object CSV",
                obj_df.to_csv(index=False),
                file_name="opticell_objects.csv",
                mime="text/csv",
            )

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Backend decision**")
            st.write("Resolved backend: " + str(result["backend_resolved"]))
            st.write(result["backend_reason"])
            st.caption("Routing probe: " + probe_reason)
            st.write("GPU requested: " + str(gpu))
        with c2:
            st.markdown("**Reproducibility**")
            st.write("Input: " + uploaded.name)
            st.write("Shape: " + str(raw.shape))
            st.write("dtype: " + str(raw.dtype))
            st.write("Cellpose available: " + str(_HAS_CELLPOSE))
        with st.expander("Raw segmentation metadata"):
            st.json(
                {k: v for k, v in seg.__dict__.items() if k != "labels"}
                | {"labels_shape": list(seg.labels.shape)}
            )

    st.caption(
        "OptiCell reports computed QC and segmentation outputs; acceptance is a review aid, not a clinical decision."
    )

if __name__ == "__main__":
    _main()
