"""Shared Streamlit authentication gate for optional local-account deployments."""
from __future__ import annotations

import os

from .workbench_store import WorkbenchStore


def current_user(st, *, store: WorkbenchStore | None = None) -> tuple[str, WorkbenchStore]:
    store = store or WorkbenchStore(os.getenv("OPTICELL_DB", ".opticell/workbench.sqlite3"))
    require = os.getenv("OPTICELL_REQUIRE_AUTH", "0").strip().lower() in {"1", "true", "yes"}
    if not require:
        return os.getenv("OPTICELL_USER", "local"), store

    store.ensure_bootstrap_user()
    if st.session_state.get("opticell_user"):
        return str(st.session_state["opticell_user"]), store

    st.subheader("Sign in")
    username = st.text_input("Username", key="opticell_login_user")
    password = st.text_input("Password", type="password", key="opticell_login_password")
    if st.button("Sign in", key="opticell_login_button"):
        if store.authenticate(username, password):
            st.session_state["opticell_user"] = username
            st.rerun()
        else:
            st.error("Invalid credentials.")
    st.stop()


__all__ = ["current_user"]
