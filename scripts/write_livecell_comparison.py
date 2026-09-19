#!/usr/bin/env python3
"""Write a measured comparison report for the current LIVECell n=20 runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/livecell_validation"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/livecell_validation/LIVECELL_CURRENT_N20_REPORT.md"),
    )
    args = parser.parse_args()

    rows = []
    for backend in ("cellpose", "hybrid", "auto", "threshold"):
        path = args.input_dir / f"livecell_val_{backend}_n20.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append((backend, payload["summary"]))

    if not rows:
        raise SystemExit("No LIVECell n20 JSON results found.")

    lines = [
        "# LIVECell current implementation validation — n=20",
        "",
        "Measured on the validation split using the same first 20 FOVs selected by the validation script.",
        "LIVECell images/annotations are CC BY-NC 4.0; this repository stores metrics, not the dataset.",
        "",
        "| Backend | Dice | Instance F1 | Mean |count error| |",
        "|---|---:|---:|---:|",
    ]
    for backend, summary in rows:
        dice = float(summary.get("dice", float("nan")))
        f1 = float(summary.get("f1", float("nan")))
        count_error = float(summary.get("absolute_count_error", float("nan")))
        lines.append("| {} | {:.3f} | {:.3f} | {:.1f} |".format(
            backend, dice, f1, count_error
        ))

    lines += [
        "",
        "## Interpretation",
        "",
        "- `auto` records the final phase/low-contrast-aware routing behavior.",
        "- `hybrid` records the current count-gated hybrid behavior after the low-contrast override.",
        "- The historical 0.622 hybrid Dice result was measured before the final low-contrast routing rule and is retained only as a pre-rule reference.",
        "- These are benchmark measurements on one n=20 slice, not claims of universal segmentation performance.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
