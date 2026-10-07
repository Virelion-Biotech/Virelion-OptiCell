"""Fixed threshold counting and foreground tests on BBBC001/004/005.

BBBC001 uses the mean of two human counters (six measured HT29 fields).
BBBC004/005 are synthetic stress tests, never biological validation.
Binary reference masks support foreground Dice only, not instance F1.
Raw data must be downloaded from the official BBBC pages; no tuning occurs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import zipfile

import cv2
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qc_pipeline import segment_threshold, to_grayscale_uint8, compute_focus_score, sha256_file  # noqa: E402
from validation import binary_dice  # noqa: E402


def decode(content):
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Unreadable TIFF")
    return to_grayscale_uint8(image)


def measure(image, truth_count, mask=None):
    seg = segment_threshold(image)
    error = abs(seg.count - truth_count)
    return dict(pred_count=seg.count, truth_count=truth_count, absolute_count_error=error,
                relative_count_error=error / truth_count,
                dice=binary_dice(seg.labels > 0, mask > 0) if mask is not None else None,
                focus_score=float(compute_focus_score(image)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', choices=['bbbc001', 'bbbc004', 'bbbc005'],
                   default=['bbbc001', 'bbbc004', 'bbbc005'])
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        folder = args.data_dir / dataset
        rows = []
        sources = []
        for source in sorted(folder.glob('*.zip')) + sorted(folder.glob('*.txt')):
            sources.append(dict(file=source.name, sha256=sha256_file(str(source))))
        if dataset == 'bbbc001':
            with (folder / 'counts.txt').open() as stream:
                references = list(csv.DictReader(stream, delimiter='\t'))
            for ref in references:
                name = ref['Image']
                candidates = list((folder / 'images').rglob(name))
                if len(candidates) != 1:
                    raise ValueError(f'Ambiguous or missing source: {name}')
                truth = (float(ref['manual count #1']) + float(ref['manual count #2'])) / 2
                rows.append(dict(image=name, **measure(decode(candidates[0].read_bytes()), truth)))
            expected = 6
        elif dataset == 'bbbc004':
            for level in ['000', '015', '030', '045', '060']:
                images = sorted((folder / level / 'images').rglob('*.tif'))
                masks = {x.stem: x for x in (folder / level / 'foreground').rglob('*.tif')}
                if len(images) != 20 or len(masks) != 20:
                    raise ValueError(f'Incomplete clustering level {level}')
                for image in images:
                    mask = masks[image.stem.removesuffix('GRAY')]
                    rows.append(dict(image=image.name, overlap_probability=int(level)/100,
                                     **measure(decode(image.read_bytes()), 300, decode(mask.read_bytes()))))
            expected = 100
        else:
            with zipfile.ZipFile(folder / 'images.zip') as images, zipfile.ZipFile(folder / 'ground_truth.zip') as masks:
                refs = {Path(x).name: x for x in masks.namelist() if x.lower().endswith('.tif')}
                names = sorted(x for x in images.namelist() if x.lower().endswith('.tif'))
                if len(names) != 19200 or len(refs) != 1200:
                    raise ValueError('Incomplete BBBC005 archives')
                for index, name in enumerate(names):
                    filename = Path(name).name
                    match = re.search(r'_C(\d+)_F(\d+)_s(\d+)_w(\d+)', filename)
                    if match is None:
                        raise ValueError(f'Unrecognized source metadata: {filename}')
                    count, blur, sample, channel = map(int, match.groups())
                    mask = decode(masks.read(refs[filename])) if filename in refs else None
                    rows.append(dict(image=filename, blur=blur, sample=sample, channel=channel,
                                     **measure(decode(images.read(name)), count, mask)))
                    if index % 1000 == 0:
                        print(dataset, index, '/', len(names), flush=True)
            expected = 19200
        if len(rows) != expected:
            raise ValueError(f'Incomplete {dataset}: {len(rows)}/{expected}')
        summary = dict(n_scored=len(rows), count_mae=float(np.mean([r['absolute_count_error'] for r in rows])),
                       mean_relative_count_error=float(np.mean([r['relative_count_error'] for r in rows])),
                       n_foreground_references=sum(r['dice'] is not None for r in rows))
        dice = [r['dice'] for r in rows if r['dice'] is not None]
        summary['foreground_dice'] = float(np.mean(dice)) if dice else None
        if dataset == 'bbbc005':
            focus = np.array([r['focus_score'] for r in rows])
            truth = np.array([r['blur'] == 1 for r in rows])
            npos, nneg = truth.sum(), (~truth).sum()
            summary['fully_in_focus_auroc'] = float((rankdata(focus)[truth].sum()-npos*(npos+1)/2)/(npos*nneg))
        result = dict(dataset=dataset.upper(), backend='threshold', complete=True, expected_records=expected,
                      evidence_kind='measured' if dataset == 'bbbc001' else 'synthetic', summary=summary,
                      source_hashes=sources, per_image=rows)
        (args.out_dir / f'{dataset}.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        with (args.out_dir / f'{dataset}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(dataset, summary, flush=True)


if __name__ == '__main__':
    main()
