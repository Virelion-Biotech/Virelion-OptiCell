"""Cover every official validation/test image with independently checkpointed chunks."""

import argparse
import gc
import hashlib
import json
import pathlib
import subprocess
import sys
import concurrent.futures
from importlib.metadata import version

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=pathlib.Path, required=True)
parser.add_argument("--out-dir", type=pathlib.Path, required=True)
parser.add_argument("--backend", choices=["threshold", "adaptive"], default="threshold")
parser.add_argument("--chunk-size", type=int, default=100)
parser.add_argument("--workers", type=int, choices=[1, 2], default=1)
args = parser.parse_args()
if args.chunk_size < 1:
    parser.error("--chunk-size must be positive")
ROOT = pathlib.Path(__file__).resolve().parents[1]
data_root = args.data_dir.resolve()
output = args.out_dir.resolve()
stage = output / "_chunks"
stage.mkdir(parents=True, exist_ok=True)
runner = ROOT / "scripts/run_livecell_validation.py"
code_hash = hashlib.sha256(
    b"".join(
        (ROOT / p).read_bytes()
        for p in ["validation.py", "qc_pipeline.py", "ensemble.py", "scripts/run_livecell_validation.py"]
    )
).hexdigest()
packages = {p: version(p) for p in ["numpy", "scipy", "pycocotools"]}

expected = {}
jobs = []
image_paths = {}
for path in (data_root / "images").rglob("*"):
    if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        if path.name in image_paths:
            raise ValueError(f"Ambiguous source image basename: {path.name}")
        image_paths[path.name] = path

for split in ["val", "test"]:
    src = data_root / f"raw/livecell_coco_{split}.json"
    source_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    coco = json.loads(src.read_text())
    expected[split] = [i["file_name"] for i in coco["images"]]
    lookup = {}
    for a in coco["annotations"]:
        lookup.setdefault(a["image_id"], []).append(a)
    images = sorted(coco["images"], key=lambda i: i["id"])
    for k in range(0, len(images), args.chunk_size):
        name = f"{split}_{k:04d}"
        data = stage / name / "data"
        data.mkdir(parents=True, exist_ok=True)
        (data / "raw").mkdir(exist_ok=True)
        symlink = data / "images"
        if not symlink.exists():
            symlink.symlink_to(data_root / "images", target_is_directory=True)
        sub = dict(coco)
        sub["images"] = images[k : k + args.chunk_size]
        sub["annotations"] = [a for i in sub["images"] for a in lookup.get(i["id"], [])]
        (data / "raw" / f"livecell_coco_{split}.json").write_text(json.dumps(sub, separators=(",", ":")))
        image_hash = hashlib.sha256()
        for image in sub["images"]:
            image_hash.update(image_paths[image["file_name"]].read_bytes())
        jobs.append(
            (
                split,
                name,
                data,
                dict(
                    source_sha256=source_hash,
                    code_sha256=code_hash,
                    backend=args.backend,
                    image_ids=[i["id"] for i in sub["images"]],
                    image_sha256=image_hash.hexdigest(),
                    packages=packages,
                ),
            )
        )
    del coco, lookup, images, sub
    gc.collect()
(stage / "expected_coverage.json").write_text(json.dumps(expected, indent=2))


def execute(job):
    split, name, data, manifest = job
    out = stage / name / "results"
    expected_name = f"livecell_{split}_{args.backend}_n{len(manifest['image_ids'])}.json"
    existing = list(out.glob(expected_name))
    prior = stage / name / "manifest.json"
    if (
        existing
        and prior.exists()
        and json.loads(prior.read_text()) == manifest
        and json.loads(existing[0].read_text()).get("complete")
    ):
        return name, "reused"
    with (stage / name / "run.log").open("w") as log:
        cmd = [
            sys.executable,
            str(runner),
            "--data-dir",
            str(data),
            "--split",
            split,
            "--max-images",
            "0",
            "--backend",
            args.backend,
            "--skip-download",
            "--out-dir",
            str(out),
        ]
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}; see run.log")
    (stage / name / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(name, "complete", flush=True)
    return name, "complete"


with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    list(pool.map(execute, jobs))
for split in ["val", "test"]:
    chunks = [
        json.loads(
            (
                stage / name / "results" / f"livecell_{split}_{args.backend}_n{len(manifest['image_ids'])}.json"
            ).read_text()
        )
        for s, name, data, manifest in jobs
        if s == split
    ]
    rows = [r for c in chunks for r in c["per_image"]]
    names = [r["file_name"] for r in rows]
    if len(names) != len(set(names)) or len(names) != len(expected[split]) or set(names) != set(expected[split]):
        raise ValueError(f"Incomplete or duplicate full-split coverage for {split}")
    sys.path.insert(0, str(ROOT))
    from validation import aggregate_segmentation_rows

    payload = dict(chunks[0])
    payload.update(
        n_requested=len(expected[split]),
        n_scored=len(rows),
        n_failed=sum(c["n_failed"] for c in chunks),
        n_missing_files=sum(c["n_missing_files"] for c in chunks),
        failures=[f for c in chunks for f in c["failures"]],
        missing_files=[f for c in chunks for f in c["missing_files"]],
        per_image=rows,
        summary=aggregate_segmentation_rows(rows),
        complete=all(c["complete"] for c in chunks),
        coverage_verified_against_full_official_split=True,
        chunks=len(chunks),
        code_sha256=code_hash,
        packages=packages,
    )
    payload["mean_truth_instances_per_image"] = sum(r["truth_count"] for r in rows) / len(rows)
    payload["max_truth_instances_per_image"] = max(r["truth_count"] for r in rows)
    out = output / split
    out.mkdir(exist_ok=True)
    (out / f"livecell_{split}_{args.backend}_n{len(rows)}.json").write_text(json.dumps(payload, indent=2))
    import csv

    with (out / f"livecell_{split}_{args.backend}_n{len(rows)}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(split, "full coverage verified", len(rows), flush=True)
