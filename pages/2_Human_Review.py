"""Human review, pixel-level mask correction, and immediate reruns."""
from __future__ import annotations

import hashlib
import io
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from app_streamlit import overlay_labels, run_pipeline
from provenance import build_immutable_manifest
from qc_pipeline import CellposeSegmenter, _HAS_CELLPOSE
from opticell import __version__
from opticell.mask_edit import apply_brush_edit, instance_count
from opticell.ui_auth import current_user

st.set_page_config(page_title="OptiCell · Review", page_icon="✅", layout="wide")
user, store = current_user(st)
st.title("Human review & correction")
st.caption(
    "Inspect saved masks, paint pixel-level corrections, approve/reject, attach a replacement mask, "
    "or rerun a FOV with an auditable parent-child record."
)

projects = store.list_projects(user)
if not projects:
    st.info("No projects exist for this user.")
    st.stop()

project_names = {p["name"]: p["project_id"] for p in projects}
project_name = st.selectbox("Project", list(project_names))
project_id = project_names[project_name]
runs = store.list_runs(project_id)
if not runs:
    st.info("This project has no runs.")
    st.stop()

run_labels = {f'{r["created_utc"]} · {r["run_id"][:8]} · {r["status"]}': r["run_id"] for r in runs}
run_label = st.selectbox("Run", list(run_labels))
run_id = run_labels[run_label]
run = store.get_run(run_id)
per_image = run["summary"].get("per_image", [])
st.dataframe(pd.DataFrame(per_image), width="stretch", height=260)

image_keys = [str(row.get("filename", row.get("index", i))) for i, row in enumerate(per_image)] or ["run"]
image_key = st.selectbox("Image / FOV", image_keys)
selected = next((row for row in per_image if str(row.get("filename", row.get("index"))) == image_key), None)

raw = None
saved_labels = None
if selected:
    input_path = selected.get("input_path")
    mask_path = selected.get("mask_path")
    if input_path and Path(input_path).is_file():
        data = np.frombuffer(Path(input_path).read_bytes(), dtype=np.uint8)
        raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if mask_path and Path(mask_path).is_file():
        try:
            saved_labels = np.load(mask_path, allow_pickle=False)
        except (OSError, ValueError):
            saved_labels = None

gray = None
if raw is not None and saved_labels is not None and saved_labels.shape == raw.shape[:2]:
    from qc_pipeline import to_grayscale_uint8

    gray = to_grayscale_uint8(raw)
    c1, c2 = st.columns(2)
    c1.image(gray, caption="Input", width="stretch", clamp=True)
    c2.image(overlay_labels(gray, saved_labels), caption="Saved segmentation", width="stretch")
elif selected:
    st.info("This run predates persisted masks or its artifacts are not available on this deployment.")

decision = st.radio("Decision", ["accept", "reject", "corrected", "rerun"], horizontal=True)
notes = st.text_area("Reviewer notes")

editor_token = hashlib.sha256(f"{run_id}|{image_key}".encode("utf-8")).hexdigest()[:16]
correction_key = f"opticell_saved_correction_{editor_token}"
corrected_path = st.session_state.get(correction_key)

if decision == "corrected":
    brush_tab, upload_tab = st.tabs(["Brush editor", "Upload replacement mask"])

    with brush_tab:
        if gray is None or saved_labels is None:
            st.info("Pixel editing requires the persisted input image and saved segmentation for this run.")
        else:
            labels_key = f"opticell_editor_labels_{editor_token}"
            revision_key = f"opticell_editor_revision_{editor_token}"
            if labels_key not in st.session_state:
                st.session_state[labels_key] = saved_labels.astype(np.int32, copy=True)
            if revision_key not in st.session_state:
                st.session_state[revision_key] = 0

            working_labels = np.asarray(st.session_state[labels_key], dtype=np.int32)
            existing_ids = [int(x) for x in np.unique(working_labels) if x > 0]

            m1, m2 = st.columns(2)
            m1.metric("Working instances", instance_count(working_labels))
            m2.metric("Saved instances", instance_count(saved_labels))

            edit_label = st.radio(
                "Brush operation",
                ["Add to existing object", "Erase pixels", "Create new object"],
                horizontal=True,
                key=f"edit_mode_{editor_token}",
            )
            brush_width = st.slider(
                "Brush width (pixels)",
                min_value=1,
                max_value=80,
                value=8,
                key=f"brush_width_{editor_token}",
            )

            label_id = None
            mode = "erase"
            if edit_label == "Add to existing object":
                mode = "add"
                if existing_ids:
                    label_id = st.selectbox(
                        "Target instance ID",
                        existing_ids,
                        key=f"target_label_{editor_token}",
                    )
                else:
                    st.warning("There are no existing instances. Use Create new object instead.")
            elif edit_label == "Create new object":
                mode = "new_object"

            st.caption(
                "Draw one or more strokes, then click Apply brush edit. "
                "New disconnected strokes become separate new instance IDs."
            )
            background = Image.fromarray(overlay_labels(gray, working_labels))
            canvas = st_canvas(
                fill_color="rgba(255, 45, 85, 0.65)",
                stroke_width=int(brush_width),
                stroke_color="#ff2d55",
                background_image=background,
                update_streamlit=True,
                height=int(gray.shape[0]),
                width=int(gray.shape[1]),
                drawing_mode="freedraw",
                return_image_data=True,
                max_display_height=700,
                key=f"mask_canvas_{editor_token}_{st.session_state[revision_key]}",
            )

            c_apply, c_reset = st.columns(2)
            if c_apply.button("Apply brush edit", type="primary", key=f"apply_edit_{editor_token}"):
                if canvas.image_data is None:
                    st.warning("Draw on the image before applying an edit.")
                elif mode == "add" and label_id is None:
                    st.warning("Choose an existing instance ID.")
                else:
                    drawing = np.asarray(canvas.image_data)
                    if drawing.ndim != 3 or drawing.shape[:2] != working_labels.shape or drawing.shape[2] < 4:
                        st.error("Canvas returned unexpected drawing dimensions; no mask change was applied.")
                    else:
                        stroke_mask = drawing[..., 3] > 0
                        try:
                            corrected, affected = apply_brush_edit(
                                working_labels,
                                stroke_mask,
                                mode=mode,
                                label_id=int(label_id) if label_id is not None else None,
                            )
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            if not affected:
                                st.warning("The brush layer contained no editable pixels.")
                            else:
                                st.session_state[labels_key] = corrected
                                st.session_state[revision_key] += 1
                                st.session_state.pop(correction_key, None)
                                st.rerun()

            if c_reset.button("Reset to saved segmentation", key=f"reset_edit_{editor_token}"):
                st.session_state[labels_key] = saved_labels.astype(np.int32, copy=True)
                st.session_state[revision_key] += 1
                st.session_state.pop(correction_key, None)
                st.rerun()

            working_labels = np.asarray(st.session_state[labels_key], dtype=np.int32)
            st.image(
                overlay_labels(gray, working_labels),
                caption="Working corrected mask",
                width="stretch",
            )

            if st.button("Save brush-edited mask as correction", key=f"save_edit_{editor_token}"):
                buffer = io.BytesIO()
                np.save(buffer, working_labels.astype(np.int32, copy=False), allow_pickle=False)
                payload = buffer.getvalue()
                digest = hashlib.sha256(payload).hexdigest()
                target_dir = Path(os.getenv("OPTICELL_WORKSPACE", ".opticell")) / "corrections" / run_id
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / (digest[:16] + "-labels.npy")
                if not target.exists():
                    target.write_bytes(payload)
                corrected_path = str(target)
                st.session_state[correction_key] = corrected_path
                st.success("Brush-edited correction saved.")
                st.caption("Correction SHA-256: " + digest)

            if st.session_state.get(correction_key):
                corrected_path = st.session_state[correction_key]
                st.caption("Current saved correction: " + corrected_path)

    with upload_tab:
        mask_upload = st.file_uploader(
            "Corrected instance-label mask (PNG/TIFF)",
            type=["png", "tif", "tiff"],
            key=f"replacement_mask_{editor_token}",
        )
        if mask_upload is not None:
            payload = mask_upload.getvalue()
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.ndim != 2:
                st.error("Corrected mask must decode to one 2-D label image.")
            elif raw is not None and decoded.shape != raw.shape[:2]:
                st.error(f"Corrected mask shape {decoded.shape} does not match input {raw.shape[:2]}.")
            elif not np.issubdtype(decoded.dtype, np.integer):
                st.error("Corrected mask must contain integer instance IDs.")
            elif (decoded < 0).any():
                st.error("Corrected mask cannot contain negative labels.")
            else:
                buffer = io.BytesIO()
                np.save(buffer, decoded.astype(np.int32, copy=False), allow_pickle=False)
                normalized_payload = buffer.getvalue()
                digest = hashlib.sha256(normalized_payload).hexdigest()
                target_dir = Path(os.getenv("OPTICELL_WORKSPACE", ".opticell")) / "corrections" / run_id
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / (digest[:16] + "-labels.npy")
                if not target.exists():
                    target.write_bytes(normalized_payload)
                corrected_path = str(target)
                st.session_state[correction_key] = corrected_path
                st.caption("Validated correction SHA-256: " + digest)
                if gray is not None:
                    st.image(
                        overlay_labels(gray, decoded.astype(np.int32)),
                        caption="Uploaded corrected-mask preview",
                        width="stretch",
                    )

if decision != "rerun":
    if st.button("Record review", type="primary"):
        if decision == "corrected" and not corrected_path:
            st.error("Save a brush edit or attach a valid corrected mask when using the corrected decision.")
        else:
            review_id = store.add_review(
                run_id,
                image_key,
                decision,
                reviewer=user,
                notes=notes,
                corrected_mask_path=corrected_path if decision == "corrected" else None,
            )
            st.success("Review recorded: " + review_id)
else:
    rerun_backend = st.selectbox("Rerun backend", ["auto", "threshold", "adaptive", "cellpose", "hybrid"], index=0)
    if st.button("Execute rerun", type="primary"):
        if raw is None or not selected or not selected.get("input_path"):
            st.error("The original input artifact is unavailable, so this FOV cannot be rerun here.")
        else:
            original_config = dict(run.get("config", {}))
            segmenter = None
            if rerun_backend in {"cellpose", "hybrid", "auto"} and _HAS_CELLPOSE:
                try:
                    segmenter = CellposeSegmenter(
                        model_type=str(original_config.get("cellpose_model", "cpsam")),
                        gpu=bool(original_config.get("gpu", False)),
                        diameter=original_config.get("diameter"),
                    )
                except (RuntimeError, OSError, ValueError) as exc:
                    st.error("Cellpose initialization failed: " + str(exc))
                    st.stop()
            if rerun_backend in {"cellpose", "hybrid"} and segmenter is None:
                st.error("Cellpose is unavailable for this rerun.")
                st.stop()

            result = run_pipeline(
                raw,
                rerun_backend,
                cellpose_segmenter=segmenter,
                min_area=int(original_config.get("min_area", 15)),
                max_area_frac=float(original_config.get("max_area_frac", 0.25)),
                diameter=original_config.get("diameter"),
                minimum_quality=float(original_config.get("minimum_quality", 70.0)),
                maximum_border_fraction=float(original_config.get("maximum_border_fraction", 0.35)),
                maximum_tiny_fraction=float(original_config.get("maximum_tiny_fraction", 0.50)),
                maximum_merged_fraction=float(original_config.get("maximum_merged_fraction", 0.25)),
            )
            seg, conf, accept = result["seg"], result["conf"], result["accept"]
            artifact_dir = Path(os.getenv("OPTICELL_WORKSPACE", ".opticell")) / "reruns" / str(uuid.uuid4())
            artifact_dir.mkdir(parents=True, exist_ok=True)
            new_mask = artifact_dir / "labels.npy"
            np.save(new_mask, seg.labels)
            manifest = build_immutable_manifest(
                opticell_version=__version__,
                inputs=[str(selected["input_path"])],
                parameters={**original_config, "backend": rerun_backend},
                operation="human_review_rerun",
                extra={"parent_run_id": run_id, "image_key": image_key, "reviewer": user},
            )
            row = {
                "filename": image_key,
                "input_path": selected["input_path"],
                "mask_path": str(new_mask),
                "backend": result["backend_resolved"],
                "objects": int(seg.count),
                "segmentation_quality": float(seg.quality_score),
                "qc_confidence": float(conf["confidence_score"]),
                "acquisition_quality": float(result["acq_score"]),
                "gate": accept.status if accept else "N/A",
                "confidence_flags": conf["flags"],
                "error": seg.error or "",
            }
            new_run = store.save_run(
                project_id,
                owner=user,
                backend=rerun_backend,
                config={**original_config, "backend": rerun_backend},
                manifest=manifest,
                summary={"parent_run_id": run_id, "per_image": [row], "n_images": 1},
                status="complete" if not seg.error else "partial",
                sample_id=run.get("sample_id"),
            )
            review_id = store.add_review(
                run_id,
                image_key,
                "rerun",
                reviewer=user,
                notes=(notes + f" Child run: {new_run}").strip(),
            )
            st.success(f"Rerun saved as {new_run}; review record {review_id}.")

reviews = store.list_reviews(run_id)
st.markdown("### Review history")
if reviews:
    st.dataframe(pd.DataFrame(reviews), width="stretch")
else:
    st.info("No review decisions recorded yet.")
