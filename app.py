"""
app.py
======
Streamlit GUI for the microscopy image QC tool.

Run with:
    streamlit run app.py

Two ways to load a dataset:
  1. Drag & drop / browse for image files (works everywhere Streamlit runs,
     including in a browser where the app can't see local folder paths).
  2. Point at a folder path directly (fastest for large local datasets —
     only works when running the app on your own machine).
"""

import os
import io
import shutil
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2

from qc_pipeline import (
    QCThresholds,
    analyze_paths,
    analyze_folder,
    load_image,
    to_grayscale_uint8,
    estimate_cell_count_threshold,
    SUPPORTED_EXTENSIONS,
    _HAS_CELLPOSE,
)

st.set_page_config(page_title="Microscopy QC", layout="wide")

# --------------------------------------------------------------------------
# Sidebar — settings
# --------------------------------------------------------------------------

st.sidebar.title("⚙️ QC Settings")

with st.sidebar.expander("Flagging thresholds", expanded=True):
    focus_min = st.number_input("Min focus score (below = blurry)", value=100.0, step=10.0)
    brightness_min = st.number_input("Min brightness (below = too dark)", value=25.0, step=5.0)
    brightness_max = st.number_input("Max brightness (above = too bright)", value=230.0, step=5.0)
    min_cell_area = st.number_input("Min cell area (px)", value=15, step=1)
    max_cell_area_frac = st.slider("Max cell area (fraction of image)", 0.01, 1.0, 0.25)
    cell_count_low = st.number_input("Flag if fewer cells than", value=1, step=1)

cell_method_options = ["threshold (fast, no extra install)"]
if _HAS_CELLPOSE:
    cell_method_options.append("cellpose (slower, more accurate)")
cell_method_choice = st.sidebar.selectbox("Cell counting method", cell_method_options)
cell_method = "cellpose" if cell_method_choice.startswith("cellpose") else "threshold"

if not _HAS_CELLPOSE:
    st.sidebar.caption("Install `cellpose` to enable the deep-learning cell counter.")

thresholds = QCThresholds(
    focus_min=focus_min,
    brightness_min=brightness_min,
    brightness_max=brightness_max,
    min_cell_area=int(min_cell_area),
    max_cell_area_frac=max_cell_area_frac,
    cell_count_low=int(cell_count_low),
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("🔬 Microscopy Image QC")
st.caption(
    "Drop in a batch of microscopy images to automatically check focus, "
    "brightness, and estimated cell counts across the whole dataset."
)

tab_upload, tab_folder = st.tabs(["📤 Upload images", "📁 Local folder path"])

image_paths = None
temp_dir = None

with tab_upload:
    uploaded_files = st.file_uploader(
        "Drag & drop images here (or click to browse)",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if uploaded_files:
        temp_dir = tempfile.mkdtemp(prefix="qc_upload_")
        image_paths = []
        for f in uploaded_files:
            dest = os.path.join(temp_dir, f.name)
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            image_paths.append(dest)
        st.success(f"{len(image_paths)} image(s) ready to analyze.")

with tab_folder:
    st.caption("Only works when running Streamlit locally on your own machine.")
    folder_path = st.text_input("Folder path", placeholder="/path/to/microscopy_images")
    if folder_path:
        if os.path.isdir(folder_path):
            from qc_pipeline import find_images
            found = find_images(folder_path)
            st.success(f"Found {len(found)} image(s) in this folder.")
            if found:
                image_paths = found
        else:
            st.error("That folder doesn't exist.")

run_clicked = st.button("▶️ Run QC analysis", type="primary", disabled=not image_paths)

# --------------------------------------------------------------------------
# Run analysis
# --------------------------------------------------------------------------

if run_clicked and image_paths:
    progress_bar = st.progress(0.0, text="Starting...")

    def _update_progress(done, total, name):
        progress_bar.progress(done / total, text=f"Analyzing {name} ({done}/{total})")

    with st.spinner("Running QC pipeline..."):
        df = analyze_paths(
            image_paths, thresholds=thresholds, cell_method=cell_method,
            progress_callback=_update_progress,
        )
    progress_bar.empty()
    st.session_state["qc_df"] = df
    st.session_state["qc_paths"] = {os.path.basename(p): p for p in image_paths}

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

if "qc_df" in st.session_state:
    df = st.session_state["qc_df"]
    path_lookup = st.session_state.get("qc_paths", {})

    n_total = len(df)
    n_flagged = (df["flags"] != "").sum()
    n_failed = (df["error"].notna()).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images analyzed", n_total)
    col2.metric("Flagged for review", int(n_flagged))
    col3.metric("Failed to load", int(n_failed))
    col4.metric("Median cell count", int(df["estimated_cells"].median()) if n_total else 0)

    st.divider()

    result_tab, hist_tab, preview_tab = st.tabs(["📋 Results table", "📊 Distributions", "🖼️ Image preview"])

    # ---- Results table ----
    with result_tab:
        flag_filter = st.multiselect(
            "Filter by flag",
            options=["BLURRY", "TOO_DARK", "TOO_BRIGHT", "FEW_OR_NO_CELLS", "FAILED_TO_LOAD"],
        )
        view_df = df.copy()
        if flag_filter:
            mask = view_df["flags"].apply(lambda f: any(fl in f for fl in flag_filter))
            view_df = view_df[mask]

        display_cols = [
            "filename", "width", "height", "focus_score", "brightness_mean",
            "estimated_cells", "cell_method", "file_size_kb", "flags",
        ]

        def _highlight_flags(row):
            return ["background-color: #ffe4e1" if row["flags"] else "" for _ in row]

        st.dataframe(
            view_df[display_cols].style.apply(_highlight_flags, axis=1),
            use_container_width=True,
            height=420,
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download full summary CSV",
            data=csv_bytes,
            file_name="qc_summary.csv",
            mime="text/csv",
        )

    # ---- Histograms ----
    with hist_tab:
        metric_cols = st.columns(2)
        metrics = [
            ("focus_score", "Focus score (variance of Laplacian)"),
            ("brightness_mean", "Mean brightness"),
            ("estimated_cells", "Estimated cell count"),
            ("file_size_kb", "File size (KB)"),
        ]
        for i, (col_name, title) in enumerate(metrics):
            with metric_cols[i % 2]:
                fig, ax = plt.subplots(figsize=(4.5, 3))
                ax.hist(df[col_name].dropna(), bins=20, color="#4C78A8", edgecolor="white")
                ax.set_title(title, fontsize=11)
                ax.set_xlabel(col_name)
                ax.set_ylabel("Number of images")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ---- Image preview with cell overlay ----
    with preview_tab:
        selected_name = st.selectbox("Choose an image", options=df["filename"].tolist())
        row = df[df["filename"] == selected_name].iloc[0]
        selected_path = path_lookup.get(selected_name)

        has_error = bool(row.get("error")) and not pd.isna(row.get("error"))
        if selected_path and os.path.exists(selected_path) and not has_error:
            try:
                raw = load_image(selected_path)
                gray = to_grayscale_uint8(raw)
                _, labels = estimate_cell_count_threshold(
                    gray, min_area=thresholds.min_cell_area, max_area_frac=thresholds.max_cell_area_frac
                )

                overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                contours, _ = cv2.findContours(
                    (labels > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, (255, 60, 60), 1)

                c1, c2 = st.columns(2)
                c1.image(gray, caption="Original (grayscale)", use_container_width=True)
                c2.image(overlay, caption=f"Detected cells: {row['estimated_cells']}", use_container_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Focus score", f"{row['focus_score']:.1f}")
                m2.metric("Brightness", f"{row['brightness_mean']:.1f}")
                m3.metric("Cells", int(row["estimated_cells"]))
                if row["flags"]:
                    st.warning(f"Flags: {row['flags']}")
                else:
                    st.success("No QC issues flagged.")
            except Exception as e:
                st.error(f"Could not preview this image: {e}")
        else:
            st.info("Preview unavailable for this image (file not found or failed to load).")

else:
    st.info("Upload images or point to a folder, then click **Run QC analysis**.")
