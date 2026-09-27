"""Large-data planning, model management, health and backups."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from opticell.large_data import iter_ome_planes, plan_ome_iteration, pyramid_levels
from opticell.model_config import PRESETS, compare_configs, config_from_preset
from opticell.observability import health_snapshot
from opticell.ui_auth import current_user

st.set_page_config(page_title="OptiCell · Large Data & Operations", page_icon="🧰", layout="wide")
user, store = current_user(st)
st.title("Large data & operations")

large_tab, models_tab, ops_tab = st.tabs(["OME-TIFF / large data", "Models & presets", "Operations"])

with large_tab:
    data_root = Path(os.getenv("OPTICELL_DATA_ROOT", ".")).expanduser().resolve()
    st.caption("Mounted data root: " + str(data_root))
    relative = st.text_input("Server-side TIFF path relative to data root", placeholder="dataset/image.ome.tif")
    if relative:
        source = (data_root / relative).resolve()
        try:
            source.relative_to(data_root)
        except ValueError:
            st.error("Path escapes OPTICELL_DATA_ROOT.")
        else:
            if source.is_file():
                try:
                    plan = plan_ome_iteration(str(source))
                    st.json(plan.to_dict())
                    st.dataframe(pd.DataFrame(pyramid_levels(str(source))), width="stretch")
                    if plan.strategy == "unsupported_without_full_load":
                        st.error("This series cannot be safely streamed with the current reader; it will not be silently loaded into RAM.")
                    else:
                        max_planes = st.number_input("Preview at most N planes", 1, 100, 4)
                        if st.button("Preview streaming planes"):
                            for index, (coords, plane) in enumerate(iter_ome_planes(str(source))):
                                st.write(coords)
                                st.image(plane, clamp=True)
                                if index + 1 >= int(max_planes):
                                    break
                except (ValueError, OSError, IndexError) as exc:
                    st.error(str(exc))
            else:
                st.info("Enter an existing TIFF/OME-TIFF path.")

with models_tab:
    names = list(PRESETS)
    left = st.selectbox("Preset A", names, index=0)
    right = st.selectbox("Preset B", names, index=min(1, len(names)-1))
    st.json({"A": config_from_preset(left).to_dict(), "B": config_from_preset(right).to_dict()})
    st.dataframe(pd.DataFrame([
        {"parameter": key, "A": values[0], "B": values[1]}
        for key, values in compare_configs(config_from_preset(left), config_from_preset(right)).items()
    ]), width="stretch")

with ops_tab:
    health = health_snapshot(os.getenv("OPTICELL_WORKSPACE", ".opticell"))
    st.json(health)
    backup_dir = Path(os.getenv("OPTICELL_BACKUP_DIR", ".opticell/backups"))
    if st.button("Create database backup"):
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / ("workbench-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".sqlite3")
        st.success("Backup created: " + store.backup(target))
    st.markdown("### Audit log")
    st.dataframe(pd.DataFrame(store.audit_events(limit=200)), width="stretch")
