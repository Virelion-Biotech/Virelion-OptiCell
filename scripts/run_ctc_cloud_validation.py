"""Full-sequence CPU CTC tracking characterization with independent traccuracy scoring."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_ctc_res import export_ctc_res  # noqa: E402
from score_ctc_seg import score_seg_directory  # noqa: E402
from traccuracy import run_metrics  # noqa: E402
from traccuracy.loaders import load_ctc_data  # noqa: E402
from traccuracy.matchers import CTCMatcher  # noqa: E402
from traccuracy.metrics import CTCMetrics  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--backend", default="threshold", choices=["threshold", "adaptive", "fluorescence", "cellpose", "hybrid"])
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset in args.datasets:
        root = args.data_dir / dataset
        for sequence in ["01", "02"]:
            images = next((x for x in root.rglob(sequence) if list(x.glob("t*.tif"))), None)
            if images is None:
                results.append(dict(dataset=dataset, sequence=sequence, error="Missing raw sequence"))
                continue
            base = images.parent
            gt_tra = base / (sequence + "_GT") / "TRA"
            man = gt_tra / "man_track.txt"
            out = args.out_dir / (dataset + "_" + sequence + "_" + args.backend)
            workflow = out / "workflow"
            workflow.mkdir(parents=True, exist_ok=True)
            row = dict(
                dataset=dataset,
                sequence=sequence,
                backend=args.backend,
                target_kind="simulated" if "SIM+" in dataset else "measured",
                n_requested=len(list(images.glob("t*.tif"))),
                evaluation_scope="public_training_sequences; characterization, not independent qualification",
                confidence_kind="uncalibrated_qc_heuristic",
                tracking_scope="one-to-one centroid linking; no division parent links",
            )
            try:
                with (out / "run.log").open("w") as log:
                    command = [
                        sys.executable,
                        str(ROOT / "scripts/run_killer_workflow.py"),
                        str(images),
                        "-o",
                        str(workflow),
                        "--backend",
                        args.backend,
                        "--enable-tracking",
                        "--track-max-distance",
                        "50",
                        "--track-max-gap",
                        "1",
                    ]
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f"Workflow exited {completed.returncode}; see run.log")
                summary = json.loads((workflow / "workflow_summary.json").read_text())
                row["workflow_summary"] = summary.get("summary", {})
                res = export_ctc_res(workflow, out / "RES")
                gt = load_ctc_data(str(gt_tra), str(man), name="GT")
                pred = load_ctc_data(str(res), str(res / "res_track.txt"), name="RES")
                scores, _ = run_metrics(gt_data=gt, pred_data=pred, matcher=CTCMatcher(), metrics=[CTCMetrics()])
                score = scores[0]
                row["metrics"] = dict(score.results) if hasattr(score, "results") else score.get("results", score)
                row["segmentation"] = score_seg_directory(base / (sequence + "_GT") / "SEG", res)
                row["status"] = "complete"
            except Exception as e:
                row.update(status="failed", error=repr(e))
            results.append(row)
            (args.out_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str))
            print(json.dumps(row, default=str), flush=True)
    return 0 if all(x.get("status") == "complete" for x in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
