#!/usr/bin/env python3
"""Create a consistent backup of the OptiCell workbench SQLite database."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from opticell.workbench_store import WorkbenchStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.getenv("OPTICELL_DB", ".opticell/workbench.sqlite3"))
    parser.add_argument("--output-dir", default=os.getenv("OPTICELL_BACKUP_DIR", ".opticell/backups"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / ("workbench-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".sqlite3")
    print(WorkbenchStore(args.db).backup(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
