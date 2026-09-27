"""Scientific calibration and validation-evidence dashboard."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from opticell.calibration import calibration_curve, choose_threshold, threshold_performance
from opticell.validation_registry import coverage_summary, known_validation_evidence

st.set_page_config(page_title="OptiCell · Calibration & Validation", page_icon="📐", layout="wide")
st.title("Calibration & validation")
st.caption("Empirical threshold calibration requires labeled outcomes; repository benchmarks do not establish universal validity.")

entries = known_validation_evidence()
coverage = coverage_summary(entries)
c1, c2, c3 = st.columns(3)
c1.metric("Evidence sets", coverage["n_evidence_sets"])
c2.metric("Modalities represented", len(coverage["modalities"]))
c3.metric("Independent-lab sets", coverage["n_independent_lab_sets"])
st.dataframe(pd.DataFrame([entry.to_dict() for entry in entries]), width="stretch")

if not coverage["has_independent_lab_validation"]:
    st.warning("No independent-lab validation dataset is registered. External validation breadth is therefore not complete.")

st.markdown("### Calibrate a QC score against labeled outcomes")
upload = st.file_uploader("CSV with a numeric score column and a binary outcome column", type=["csv"])
if upload is not None:
    frame = pd.read_csv(upload)
    st.dataframe(frame.head(20), width="stretch")
    if len(frame.columns) >= 2:
        c1, c2 = st.columns(2)
        score_col = c1.selectbox("Score column", list(frame.columns), index=0)
        outcome_col = c2.selectbox("Outcome column", list(frame.columns), index=min(1, len(frame.columns)-1))
        bins = st.slider("Calibration bins", 2, 20, 10)
        min_sens = st.slider("Minimum sensitivity", 0.0, 1.0, 0.90, 0.01)
        min_spec = st.slider("Minimum specificity", 0.0, 1.0, 0.80, 0.01)
        try:
            curve = calibration_curve(frame[score_col], frame[outcome_col], bins=bins)
            table = threshold_performance(frame[score_col], frame[outcome_col])
            st.markdown("#### Reliability table")
            st.dataframe(curve, width="stretch")
            if not curve.empty:
                st.line_chart(curve.set_index("mean_score")["observed_success"])
            st.markdown("#### Threshold performance")
            st.dataframe(table, width="stretch", height=320)
            try:
                selected = choose_threshold(frame[score_col], frame[outcome_col], minimum_sensitivity=min_sens, minimum_specificity=min_spec)
                st.success("A threshold meets the explicit constraints.")
                st.json(selected)
            except ValueError as exc:
                st.warning(str(exc))
            payload = io.StringIO()
            table.to_csv(payload, index=False)
            st.download_button("Download threshold analysis", payload.getvalue(), file_name="opticell_threshold_calibration.csv", mime="text/csv")
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
