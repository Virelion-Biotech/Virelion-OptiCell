"""Persistent project, run, review, quota, and audit storage for OptiCell."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _password_hash(password: str, salt: bytes, iterations: int = 310_000) -> str:
    if not isinstance(password, str) or len(password) < 10:
        raise ValueError("password must contain at least 10 characters")
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return digest.hex()


class WorkbenchStore:
    """SQLite-backed workspace suitable for one-node/local deployments.

    SQLite gives OptiCell durable projects, samples, runs, reviews and audit logs.
    Multi-node deployments should put this interface behind a transactional service
    or replace it with a server database rather than sharing the SQLite file.
    """

    def __init__(self, path: str | os.PathLike[str] = ".opticell/workbench.sqlite3") -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            iterations INTEGER NOT NULL,
            quota_runs INTEGER NOT NULL DEFAULT 1000 CHECK (quota_runs > 0),
            active INTEGER NOT NULL DEFAULT 1,
            created_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL,
            UNIQUE(owner, name)
        );
        CREATE TABLE IF NOT EXISTS samples (
            sample_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            sample_group TEXT,
            condition_name TEXT,
            replicate TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            sample_id TEXT REFERENCES samples(sample_id) ON DELETE SET NULL,
            owner TEXT NOT NULL,
            status TEXT NOT NULL,
            backend TEXT NOT NULL,
            config_json TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id, created_utc);
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            image_key TEXT NOT NULL,
            decision TEXT NOT NULL CHECK(decision IN ('accept','reject','rerun','corrected')),
            reviewer TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            corrected_mask_path TEXT,
            created_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_utc TEXT NOT NULL
        );
        """
        with self._connect() as conn:
            conn.executescript(schema)

    def log(self, actor: str, action: str, entity_type: str, entity_id: str, payload: Mapping[str, Any] | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_log(actor, action, entity_type, entity_id, payload_json, created_utc) VALUES(?,?,?,?,?,?)",
                (actor, action, entity_type, entity_id, _json(payload or {}), _utcnow()),
            )

    def create_user(self, username: str, password: str, *, quota_runs: int = 1000) -> None:
        username = username.strip()
        if not username or len(username) > 128:
            raise ValueError("username must contain 1..128 characters")
        if not isinstance(quota_runs, int) or isinstance(quota_runs, bool) or quota_runs < 1:
            raise ValueError("quota_runs must be a positive integer")
        salt = secrets.token_bytes(16)
        iterations = 310_000
        digest = _password_hash(password, salt, iterations)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users(username,password_hash,salt,iterations,quota_runs,active,created_utc) VALUES(?,?,?,?,?,1,?)",
                (username, digest, salt.hex(), iterations, quota_runs, _utcnow()),
            )
        self.log(username, "create_user", "user", username, {"quota_runs": quota_runs})

    def ensure_bootstrap_user(self) -> None:
        username = os.getenv("OPTICELL_ADMIN_USER")
        password = os.getenv("OPTICELL_ADMIN_PASSWORD")
        if not username or not password:
            return
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
        if not exists:
            self.create_user(username, password, quota_runs=int(os.getenv("OPTICELL_ADMIN_QUOTA_RUNS", "10000")))

    def authenticate(self, username: str, password: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash,salt,iterations,active FROM users WHERE username=?", (username.strip(),)
            ).fetchone()
        if row is None or not bool(row["active"]):
            return False
        candidate = _password_hash(password, bytes.fromhex(row["salt"]), int(row["iterations"]))
        ok = hmac.compare_digest(candidate, row["password_hash"])
        if ok:
            self.log(username, "login", "user", username)
        return ok

    def create_project(self, owner: str, name: str, *, description: str = "", metadata: Mapping[str, Any] | None = None) -> str:
        owner, name = owner.strip(), name.strip()
        if not owner or not name:
            raise ValueError("owner and project name are required")
        project_id = str(uuid.uuid4())
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO projects(project_id,owner,name,description,metadata_json,created_utc,updated_utc) VALUES(?,?,?,?,?,?,?)",
                (project_id, owner, name, description, _json(metadata or {}), now, now),
            )
        self.log(owner, "create_project", "project", project_id, {"name": name})
        return project_id

    def list_projects(self, owner: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM projects WHERE owner=? ORDER BY updated_utc DESC", (owner,)
            ).fetchall()
        return [dict(row) | {"metadata": json.loads(row["metadata_json"])} for row in rows]

    def get_project(self, project_id: str, *, owner: str | None = None) -> dict[str, Any]:
        query, args = "SELECT * FROM projects WHERE project_id=?", [project_id]
        if owner is not None:
            query += " AND owner=?"
            args.append(owner)
        with self._connect() as conn:
            row = conn.execute(query, tuple(args)).fetchone()
        if row is None:
            raise KeyError("project not found")
        return dict(row) | {"metadata": json.loads(row["metadata_json"])}

    def add_sample(
        self,
        project_id: str,
        name: str,
        *,
        sample_group: str | None = None,
        condition: str | None = None,
        replicate: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        actor: str = "local",
    ) -> str:
        sample_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO samples(sample_id,project_id,name,sample_group,condition_name,replicate,metadata_json,created_utc) VALUES(?,?,?,?,?,?,?,?)",
                (sample_id, project_id, name.strip(), sample_group, condition, replicate, _json(metadata or {}), _utcnow()),
            )
            conn.execute("UPDATE projects SET updated_utc=? WHERE project_id=?", (_utcnow(), project_id))
        self.log(actor, "add_sample", "sample", sample_id, {"project_id": project_id})
        return sample_id

    def list_samples(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM samples WHERE project_id=? ORDER BY created_utc", (project_id,)).fetchall()
        return [dict(row) | {"metadata": json.loads(row["metadata_json"])} for row in rows]

    def _enforce_quota(self, owner: str) -> None:
        with self._connect() as conn:
            user = conn.execute("SELECT quota_runs FROM users WHERE username=?", (owner,)).fetchone()
            if user is None:
                return
            n = conn.execute("SELECT COUNT(*) AS n FROM runs WHERE owner=?", (owner,)).fetchone()["n"]
        if int(n) >= int(user["quota_runs"]):
            raise RuntimeError(f"run quota exceeded for {owner!r}")

    def save_run(
        self,
        project_id: str,
        *,
        owner: str,
        backend: str,
        config: Mapping[str, Any],
        manifest: Mapping[str, Any],
        summary: Mapping[str, Any],
        status: str = "complete",
        sample_id: str | None = None,
    ) -> str:
        self._enforce_quota(owner)
        run_id, now = str(uuid.uuid4()), _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs(run_id,project_id,sample_id,owner,status,backend,config_json,manifest_json,summary_json,created_utc,updated_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, project_id, sample_id, owner, status, backend, _json(config), _json(manifest), _json(summary), now, now),
            )
            conn.execute("UPDATE projects SET updated_utc=? WHERE project_id=?", (now, project_id))
        self.log(owner, "save_run", "run", run_id, {"project_id": project_id, "backend": backend, "status": status})
        return run_id

    def list_runs(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id,sample_id,owner,status,backend,created_utc,updated_utc,summary_json FROM runs WHERE project_id=? ORDER BY created_utc DESC",
                (project_id,),
            ).fetchall()
        return [dict(row) | {"summary": json.loads(row["summary_json"])} for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("run not found")
        result = dict(row)
        for key in ("config_json", "manifest_json", "summary_json"):
            result[key.removesuffix("_json")] = json.loads(result[key])
        return result

    def add_review(
        self,
        run_id: str,
        image_key: str,
        decision: str,
        *,
        reviewer: str,
        notes: str = "",
        corrected_mask_path: str | None = None,
    ) -> str:
        if decision not in {"accept", "reject", "rerun", "corrected"}:
            raise ValueError("decision must be accept, reject, rerun, or corrected")
        review_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO reviews(review_id,run_id,image_key,decision,reviewer,notes,corrected_mask_path,created_utc) VALUES(?,?,?,?,?,?,?,?)",
                (review_id, run_id, image_key, decision, reviewer, notes, corrected_mask_path, _utcnow()),
            )
        self.log(reviewer, "review", "run", run_id, {"image_key": image_key, "decision": decision})
        return review_id

    def list_reviews(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM reviews WHERE run_id=? ORDER BY created_utc", (run_id,)).fetchall()
        return [dict(row) for row in rows]

    def audit_events(self, *, limit: int = 500) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM audit_log ORDER BY audit_id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload_json"])} for row in rows]

    def backup(self, destination: str | os.PathLike[str]) -> str:
        target = Path(destination).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as source, sqlite3.connect(target) as dest:
            source.backup(dest)
        shutil.copystat(self.path, target, follow_symlinks=True)
        return str(target)


__all__ = ["WorkbenchStore"]
