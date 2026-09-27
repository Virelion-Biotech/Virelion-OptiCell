"""Human review and correction records for OptiCell runs."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from opticell.ui_auth import current_user

st.set_page_config(page_title="OptiCell · Review", page_icon="✅", layout="wide")
user, store = current_user(st)
st.title("Human review & correction")
st.caption("Approve, reject, request reruns, or attach corrected masks while preserving an audit trail.")

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
st.dataframe(pd.DataFrame(per_image), width="stretch", height=300)

image_keys = [str(row.get("filename", row.get("index", i))) for i, row in enumerate(per_image)] or ["run"]
image_key = st.selectbox("Image / FOV", image_keys)
decision = st.radio("Decision", ["accept", "reject", "rerun", "corrected"], horizontal=True)
notes = st.text_area("Reviewer notes")
mask_upload = st.file_uploader("Corrected label mask (optional; PNG/TIFF)", type=["png", "tif", "tiff"])
corrected_path = None
if mask_upload is not None:
    target_dir = Path(os.getenv("OPTICELL_WORKSPACE", ".opticell")) / "corrections" / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (str(uuid.uuid4()) + "-" + Path(mask_upload.name).name)
    target.write_bytes(mask_upload.getvalue())
    corrected_path = str(target)
    st.caption("Correction staged at " + corrected_path)

if st.button("Record review", type="primary"):
    if decision == "corrected" and not corrected_path:
        st.error("Attach a corrected mask when using the corrected decision.")
    else:
        review_id = store.add_review(run_id, image_key, decision, reviewer=user, notes=notes, corrected_mask_path=corrected_path)
        st.success("Review recorded: " + review_id)

reviews = store.list_reviews(run_id)
st.markdown("### Review history")
if reviews:
    st.dataframe(pd.DataFrame(reviews), width="stretch")
else:
    st.info("No review decisions recorded yet.")
