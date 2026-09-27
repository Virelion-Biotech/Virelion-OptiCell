"""OptiCell batch analysis and persistent project workspace."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app_streamlit import run_pipeline
from experiment import parse_metadata
from provenance import build_immutable_manifest
from qc_pipeline import CellposeSegmenter, PIPELINE_VERSION, _HAS_CELLPOSE
from opticell import __version__
from opticell.dataset_qc import annotate_outliers, dataset_qc_summary, plate_view, replicate_qc
from opticell.model_config import PRESETS, config_for_checkpoint, config_from_preset
from opticell.ui_auth import current_user

st.set_page_config(page_title="OptiCell · Batch & Projects", page_icon="🧫", layout="wide")
user, store = current_user(st)
st.title("Batch analysis & projects")
st.caption("Persistent experiments, batch segmentation, dataset QC, replicates, plate views, and auditable run manifests.")

projects = store.list_projects(user)
with st.sidebar:
    st.caption("Signed in as " + user)
    st.markdown("### Project")
    project_labels = {p["name"]: p["project_id"] for p in projects}
    selected_name = st.selectbox("Open project", ["Create new…"] + list(project_labels))
    if selected_name == "Create new…":
        new_name = st.text_input("Project name")
        new_desc = st.text_area("Description")
        if st.button("Create project", disabled=not new_name.strip()):
            project_id = store.create_project(user, new_name, description=new_desc)
            st.success("Project created.")
            st.session_state["project_id"] = project_id
            st.rerun()
        project_id = st.session_state.get("project_id")
    else:
        project_id = project_labels[selected_name]
        st.session_state["project_id"] = project_id

if not project_id:
    st.info("Create a project to begin.")
    st.stop()

project = store.get_project(project_id, owner=user)
st.markdown("**Project:** " + project["name"])

setup, results_tab, history_tab = st.tabs(["Run batch", "Dataset QC", "History"])

with setup:
    c1, c2, c3 = st.columns(3)
    sample_group = c1.text_input("Sample group", placeholder="e.g. donor-01")
    condition = c2.text_input("Condition", placeholder="e.g. control")
    replicate = c3.text_input("Replicate", placeholder="e.g. R1")

    c1, c2 = st.columns(2)
    preset = c1.selectbox("Segmentation preset", list(PRESETS), index=0)
    base_config = config_from_preset(preset)
    use_gpu = c2.checkbox("Use GPU", value=base_config.gpu and _HAS_CELLPOSE, disabled=not _HAS_CELLPOSE)

    checkpoint = st.file_uploader("Optional custom Cellpose checkpoint", type=None, key="batch_checkpoint")
    config = config_from_preset(preset, {"gpu": use_gpu})
    if checkpoint is not None:
        model_dir = Path(os.getenv("OPTICELL_WORKSPACE", ".opticell")) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = Path(checkpoint.name).name
        model_path = model_dir / (str(uuid.uuid4()) + "-" + safe_model_name)
        model_path.write_bytes(checkpoint.getvalue())
        config = config_for_checkpoint(model_path, backend=config.backend if config.backend in {"cellpose", "hybrid"} else "cellpose", gpu=use_gpu)
        st.caption("Custom model SHA-256: " + str(config.checkpoint_sha256))

    uploads = st.file_uploader(
        "Upload microscopy images",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        key="batch_uploads",
    )
    run = st.button("Run batch", type="primary", disabled=not uploads)

    if run and uploads:
        artifact_root = Path(os.getenv("OPTICELL_WORKSPACE", ".opticell")) / "projects" / project_id / str(uuid.uuid4())
        input_dir = artifact_root / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_paths: list[str] = []
        for index, upload in enumerate(uploads):
            safe_name = Path(upload.name).name
            target = input_dir / f"{index:04d}-{safe_name}"
            target.write_bytes(upload.getvalue())
            input_paths.append(str(target))

        segmenter = None
        if config.backend in {"cellpose", "hybrid", "auto"} and _HAS_CELLPOSE:
            try:
                segmenter = CellposeSegmenter(model_type=config.cellpose_model, gpu=config.gpu, diameter=config.diameter)
            except (RuntimeError, OSError, ValueError) as exc:
                st.error("Cellpose initialization failed: " + str(exc))
                st.stop()
        if config.backend in {"cellpose", "hybrid"} and segmenter is None:
            st.error("This preset requires Cellpose, but Cellpose is unavailable.")
            st.stop()

        progress = st.progress(0.0)
        rows, object_frames = [], []
        for index, (upload, path) in enumerate(zip(uploads, input_paths)):
            data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
            raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            row = {"index": index, "filename": upload.name, "sha256": sha, "condition": condition, "replicate": replicate, "sample_group": sample_group}
            row.update(parse_metadata(upload.name))
            try:
                if raw is None:
                    raise ValueError("OpenCV could not decode the image")
                result = run_pipeline(raw, config.backend, cellpose_segmenter=segmenter)
                seg, conf, accept = result["seg"], result["conf"], result["accept"]
                if seg.error:
                    raise RuntimeError(seg.error)
                row.update({
                    "backend": result["backend_resolved"],
                    "objects": int(seg.count),
                    "segmentation_quality": float(seg.quality_score),
                    "qc_confidence": float(conf["confidence_score"]),
                    "acquisition_quality": float(result["acq_score"]),
                    "gate": accept.status if accept else "N/A",
                    "confidence_flags": conf["flags"],
                    "error": "",
                })
                if not result["obj_df"].empty:
                    obj = result["obj_df"].copy()
                    obj.insert(0, "filename", upload.name)
                    object_frames.append(obj)
            except Exception as exc:
                row.update({"backend": config.backend, "objects": np.nan, "segmentation_quality": np.nan, "qc_confidence": np.nan, "acquisition_quality": np.nan, "gate": "ERROR", "confidence_flags": "", "error": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
            progress.progress((index + 1) / len(uploads))

        frame = pd.DataFrame(rows)
        metric_columns = ["segmentation_quality", "qc_confidence", "acquisition_quality", "objects"]
        successful = frame["error"].eq("")
        annotated = frame.copy()
        if successful.any():
            ok = annotate_outliers(frame.loc[successful].copy(), metric_columns)
            annotated.loc[ok.index, ok.columns] = ok
        if "dataset_qc_flags" not in annotated:
            annotated["dataset_qc_flags"] = ""
            annotated["dataset_qc_outlier"] = False

        objects = pd.concat(object_frames, ignore_index=True) if object_frames else pd.DataFrame()
        summary_table = dataset_qc_summary(frame.loc[successful], metric_columns) if successful.any() else pd.DataFrame()
        rep_table = pd.DataFrame()
        if successful.any() and condition.strip() and replicate.strip():
            rep_table = replicate_qc(frame.loc[successful], group_columns=("condition", "replicate"), metrics=metric_columns)

        manifest = build_immutable_manifest(
            opticell_version=__version__,
            inputs=input_paths,
            parameters=config.to_dict(),
            operation="streamlit_batch_analysis",
            models=([{"name": config.cellpose_model, "path": config.cellpose_model, "sha256": config.checkpoint_sha256}] if config.checkpoint_sha256 else [{"name": config.cellpose_model}]),
            extra={"project_id": project_id, "owner": user, "pipeline_version": PIPELINE_VERSION, "sample_group": sample_group, "condition": condition, "replicate": replicate},
        )
        summary = {
            "n_images": int(len(frame)),
            "n_successful": int(successful.sum()),
            "n_failed": int((~successful).sum()),
            "n_review_required": int(frame["gate"].isin(["REVIEW", "FAIL", "ERROR"]).sum()),
            "dataset_qc_outliers": int(annotated["dataset_qc_outlier"].fillna(False).sum()),
            "per_image": annotated.to_dict(orient="records"),
        }
        sample_id = store.add_sample(project_id, sample_group or "batch", sample_group=sample_group or None, condition=condition or None, replicate=replicate or None, actor=user)
        run_id = store.save_run(project_id, owner=user, backend=config.backend, config=config.to_dict(), manifest=manifest, summary=summary, status="complete" if successful.all() else "partial", sample_id=sample_id)
        st.session_state["last_batch"] = {"rows": annotated, "objects": objects, "summary": summary_table, "replicates": rep_table, "manifest": manifest, "run_id": run_id}
        st.success("Saved run " + run_id)

with results_tab:
    payload = st.session_state.get("last_batch")
    if not payload:
        st.info("Run a batch in this session to inspect dataset-level QC.")
    else:
        rows = payload["rows"]
        st.dataframe(rows, width="stretch", height=360)
        c1, c2, c3 = st.columns(3)
        c1.metric("Images", len(rows))
        c2.metric("Review required", int(rows["gate"].isin(["REVIEW", "FAIL", "ERROR"]).sum()))
        c3.metric("Dataset outliers", int(rows["dataset_qc_outlier"].fillna(False).sum()))
        st.markdown("### Distribution summary")
        st.dataframe(payload["summary"], width="stretch")
        numeric = rows.loc[rows["error"].eq(""), ["segmentation_quality", "qc_confidence", "acquisition_quality", "objects"]]
        if not numeric.empty:
            st.bar_chart(numeric)
        if rows["well"].notna().any():
            metric = st.selectbox("Plate metric", ["objects", "segmentation_quality", "qc_confidence", "acquisition_quality"])
            try:
                st.dataframe(plate_view(rows.loc[rows["error"].eq("")], metric), width="stretch")
            except ValueError as exc:
                st.warning(str(exc))
        if not payload["replicates"].empty:
            st.markdown("### Replicate QC")
            st.dataframe(payload["replicates"], width="stretch")

        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("image_results.csv", rows.to_csv(index=False))
            zf.writestr("dataset_qc.csv", payload["summary"].to_csv(index=False))
            if not payload["objects"].empty:
                zf.writestr("object_measurements.csv", payload["objects"].to_csv(index=False))
            zf.writestr("manifest.json", json.dumps(payload["manifest"], indent=2, sort_keys=True, default=str))
        st.download_button("Download complete run bundle", archive.getvalue(), file_name="opticell_batch_run.zip", mime="application/zip")

with history_tab:
    runs = store.list_runs(project_id)
    if runs:
        st.dataframe(pd.DataFrame(runs), width="stretch")
    else:
        st.info("No saved runs yet.")
