"""Count-only characterization of every official CellFMCount record.

Checks the supplied dot annotations against the source metadata cell_count,
preserves trainval/test membership, and includes all zero-count references.
Dot labels do not establish whole-cell boundary or phenotype accuracy.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qc_pipeline import segment_threshold, to_grayscale_uint8, sha256_file  # noqa: E402


def aggregate(rows):
    error = np.array([r['absolute_count_error'] for r in rows])
    relative = [r['relative_count_error'] for r in rows if r['relative_count_error'] is not None]
    return dict(n_scored=len(rows), count_mae=float(error.mean()), count_rmse=float(np.sqrt((error**2).mean())),
                mean_relative_count_error=float(np.mean(relative)) if relative else None,
                n_empty_references=sum(r['truth_count'] == 0 for r in rows),
                n_images_with_outside_dots=sum(r['n_dots_outside_image'] > 0 for r in rows),
                n_dots_outside_image=sum(r['n_dots_outside_image'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    rows = []
    with zipfile.ZipFile(args.archive) as bundle:
        metadata = list(csv.DictReader(io.StringIO(bundle.read('dataset/metadata.csv').decode())))
        images = {Path(n).stem: n for n in bundle.namelist() if '/img/' in n and n.endswith('.tiff')}
        annotations = {Path(n).stem: n for n in bundle.namelist() if '/ground_truth/' in n and n.endswith('.csv')}
        ids = [r['id'] for r in metadata]
        if len(ids) != 3023 or len(set(ids)) != 3023 or set(ids) != set(images) or set(ids) != set(annotations):
            raise ValueError('Incomplete or ambiguous official record coverage')
        for index, ref in enumerate(metadata):
            identifier = ref['id']
            truth = int(ref['cell_count'])
            dot_bytes = bundle.read(annotations[identifier])
            dots = list(csv.DictReader(io.StringIO(dot_bytes.decode())))
            if len(dots) != truth:
                raise ValueError(f'Metadata/dot count disagreement on {identifier}: {truth} vs {len(dots)}')
            content = bundle.read(images[identifier])
            raw = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f'Unreadable source {identifier}')
            if raw.ndim == 3:
                raw = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
            gray = to_grayscale_uint8(raw)
            outside = 0
            for dot in dots:
                x, y = float(dot['X']), float(dot['Y'])
                if not np.isfinite([x, y]).all():
                    raise ValueError(f'Non-finite dot coordinates on {identifier}')
                outside += not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0])
            seg = segment_threshold(gray)
            error = abs(seg.count - truth)
            rows.append(dict(ref, pred_count=seg.count, truth_count=truth, absolute_count_error=error,
                             relative_count_error=error/truth if truth else None,
                             n_dots_outside_image=outside,
                             image_sha256=hashlib.sha256(content).hexdigest(),
                             annotation_sha256=hashlib.sha256(dot_bytes).hexdigest()))
            if index % 250 == 0:
                print(index, '/', len(metadata), flush=True)
    source_hash = sha256_file(str(args.archive))
    payload = dict(dataset='CellFMCount', source='https://doi.org/10.5281/zenodo.17088532',
                   backend='threshold', complete=True, summary=aggregate(rows), per_image=rows,
                   by_split={s: aggregate([r for r in rows if r['set'] == s]) for s in sorted({r['set'] for r in rows})},
                   by_test_staining={s: aggregate([r for r in rows if r['set'] == 'test' and r['staining'] == s])
                                     for s in sorted({r['staining'] for r in rows if r['set'] == 'test'})},
                   source_sha256=source_hash,
                   code_sha256=hashlib.sha256((ROOT / 'qc_pipeline.py').read_bytes()).hexdigest(),
                   note=('Count-only; metadata and dot counts must agree. Out-of-bounds dots remain in author counts '
                         'and are flagged, not moved or removed. These coordinates are unsuitable for localization '
                         'scoring without source clarification. No fitting/tuning or untouched-holdout claim.'))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
    with (args.out_dir / 'per_image.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(payload['summary'], payload['by_split'], flush=True)


if __name__ == '__main__':
    main()
