"""Regression tests for the local-account Streamlit authentication gate."""

import pytest

from opticell.ui_auth import current_user
from opticell.workbench_store import WorkbenchStore


class _StopCalled(RuntimeError):
    pass


class _RerunCalled(RuntimeError):
    pass


class _FakeStreamlit:
    def __init__(self, *, username="admin", password="correct-horse-battery", button=False):
        self.session_state = {}
        self.username = username
        self.password = password
        self.button_pressed = button
        self.errors = []

    def subheader(self, _label):
        pass

    def text_input(self, _label, *, key, type=None):
        return self.username if key == "opticell_login_user" else self.password

    def button(self, _label, *, key):
        assert key == "opticell_login_button"
        return self.button_pressed

    def error(self, message):
        self.errors.append(message)

    def stop(self):
        raise _StopCalled

    def rerun(self):
        raise _RerunCalled


def test_auth_required_bootstraps_first_admin_and_stops_before_workbench(monkeypatch, tmp_path):
    monkeypatch.setenv("OPTICELL_REQUIRE_AUTH", "1")
    monkeypatch.setenv("OPTICELL_ADMIN_USER", "admin")
    monkeypatch.setenv("OPTICELL_ADMIN_PASSWORD", "correct-horse-battery")
    store = WorkbenchStore(tmp_path / "workbench.sqlite3")
    st = _FakeStreamlit(button=False)

    with pytest.raises(_StopCalled):
        current_user(st, store=store)

    assert store.authenticate("admin", "correct-horse-battery") is True
    assert "opticell_user" not in st.session_state


def test_successful_login_sets_session_user(monkeypatch, tmp_path):
    monkeypatch.setenv("OPTICELL_REQUIRE_AUTH", "true")
    monkeypatch.setenv("OPTICELL_ADMIN_USER", "admin")
    monkeypatch.setenv("OPTICELL_ADMIN_PASSWORD", "correct-horse-battery")
    store = WorkbenchStore(tmp_path / "workbench.sqlite3")
    store.create_user("admin", "correct-horse-battery")
    st = _FakeStreamlit(button=True)

    with pytest.raises(_RerunCalled):
        current_user(st, store=store)

    assert st.session_state["opticell_user"] == "admin"


def test_authentication_can_be_disabled_only_by_explicit_false(monkeypatch, tmp_path):
    monkeypatch.setenv("OPTICELL_REQUIRE_AUTH", "0")
    store = WorkbenchStore(tmp_path / "workbench.sqlite3")

    user, returned_store = current_user(_FakeStreamlit(), store=store)

    assert user == "local"
    assert returned_store is store
