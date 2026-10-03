"""Operational health and structured-event helpers for deployed OptiCell instances."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime import capabilities_dict


def structured_event(event: str, **fields: Any) -> str:
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    return json.dumps(payload, sort_keys=True, default=str)


def health_snapshot(workspace: str = ".opticell") -> dict[str, Any]:
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".healthcheck"
    writable = True
    error = None
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        writable, error = False, f"{type(exc).__name__}: {exc}"
    usage = shutil.disk_usage(root)
    return {
        "ok": writable and usage.free > 100 * 1024 * 1024,
        "workspace": str(root.resolve()),
        "workspace_writable": writable,
        "workspace_error": error,
        "disk_free_bytes": int(usage.free),
        "runtime": capabilities_dict(),
        "pid": os.getpid(),
    }


__all__ = ["health_snapshot", "structured_event"]
