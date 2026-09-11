#!/usr/bin/env python3
"""Offline check: selection rule v2 vs measured Stage-3 TRA (no GPU re-run).

Reads outputs/stage3/stage3_tra_by_backend.csv and reports whether v2 would have
picked the backend with the highest measured TRA on each sequence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_stage3_orchestrate import select_backend  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "outputs" / "stage3" / "stage3_tra_by_backend.csv",
    )
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    required = {"dataset", "sequence", "backend", "count_cv", "mean_confidence", "TRA"}
    if not required.issubset(df.columns):
        print(f"ERROR: csv missing {required - set(df.columns)}", file=sys.stderr)
        return 2

    rows_out = []
    n_match = 0
    n_total = 0
    for (ds, seq), g in df.groupby(["dataset", "sequence"]):
        candidates = []
        for _, r in g.iterrows():
            candidates.append(
                {
                    "backend": str(r["backend"]),
                    "mean_confidence": float(r["mean_confidence"]),
                    "count_cv": float(r["count_cv"]),
                    "n_low_confidence_lt_50": 0,
                    "n_images": int(r["n_tracks"]) if "n_tracks" in r else 1,
                    "mean_object_count": float(r.get("n_tracks", 1) or 1),
                    "zero_object_fraction": 0.0,
                    "TRA": float(r["TRA"]),
                }
            )
        # n_images proxy: use 1 so it does not dominate; mean_object_count>0 keeps viable
        for c in candidates:
            c["n_images"] = 92
            c["mean_object_count"] = max(float(c.get("mean_object_count", 1)), 1.0)

        decision = select_backend(candidates)
        selected = decision["selected_backend"]
        best = max(candidates, key=lambda x: x["TRA"])
        match = selected == best["backend"]
        n_total += 1
        n_match += int(match)
        rows_out.append(
            {
                "dataset": ds,
                "sequence": seq,
                "v2_selected": selected,
                "v2_TRA": next(c["TRA"] for c in candidates if c["backend"] == selected),
                "best_backend": best["backend"],
                "best_TRA": best["TRA"],
                "match": match,
                "rejected": decision.get("rejected"),
            }
        )
        print(
            f"{ds}/{seq}: v2={selected} (TRA={rows_out[-1]['v2_TRA']:.3f}) "
            f"best={best['backend']} (TRA={best['TRA']:.3f}) "
            f"{'MATCH' if match else 'MISS'}"
        )

    print(f"\nMatch rate: {n_match}/{n_total}")
    out = args.csv.parent / "stage3_selection_v2_validation.csv"
    pd.DataFrame(rows_out).to_csv(out, index=False)
    print(f"Wrote {out}")
    return 0 if n_match == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
