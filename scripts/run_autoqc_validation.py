"""Characterize fixed OptiCell QC rules on the complete AutoQC-Bench test pool.

Requires all 100 normal and 48 anomalous source frames from S-BIAD2133.
No diffusion model is trained and artifact masks are not cell references.
The five published folds reuse this pool; they are not five independent studies.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qc_pipeline import analyze_image, QCThresholds  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    files = sorted((args.data_dir / 'test').rglob('*.tiff'))
    normal = [x for x in files if 'good_data' in x.parts]
    anomaly = [x for x in files if 'anomalies' in x.parts and 'raw' in x.parts]
    if len(normal) != 100 or len(anomaly) != 48:
        raise ValueError(f'Incomplete AutoQC test pool: {len(normal)} normal, {len(anomaly)} anomalous')
    rows = []
    thresholds = QCThresholds()
    for path in normal + anomaly:
        result = analyze_image(str(path), thresholds=thresholds)
        if result.error:
            raise ValueError(f'Pipeline failed on {path}: {result.error}')
        rows.append(dict(image=str(path.relative_to(args.data_dir)), anomaly=path in anomaly,
                         kind=path.parts[-4] if path in anomaly else 'normal',
                         flags=';'.join(result.flags), flagged=bool(result.flags),
                         focus_score=result.focus_score, brightness_mean=result.brightness_mean,
                         saturation_fraction=result.saturation_fraction, sha256=result.sha256))
    truth = np.array([r['anomaly'] for r in rows])
    pred = np.array([r['flagged'] for r in rows])
    summary = dict(n_images=len(rows), n_normal=len(normal), n_anomaly=len(anomaly),
                   sensitivity=float(pred[truth].mean()), specificity=float((~pred[~truth]).mean()),
                   accuracy=float((pred == truth).mean()))
    # Negative focus is a predeclared score: low focus should indicate bad frames.
    scores = -np.array([r['focus_score'] for r in rows])
    npos, nneg = truth.sum(), (~truth).sum()
    summary['low_focus_anomaly_auroc'] = float((rankdata(scores)[truth].sum()-npos*(npos+1)/2)/(npos*nneg))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(dataset='AutoQC-Bench', source='https://doi.org/10.6019/S-BIAD2133',
                   complete=True, summary=summary, per_image=rows,
                   code_sha256=hashlib.sha256((ROOT / 'qc_pipeline.py').read_bytes()).hexdigest(),
                   note='Fixed production flags, no tuning. Test frames can share acquisition sequences; no independent-donor claim.')
    (args.out_dir / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
    with (args.out_dir / 'per_image.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(summary)


if __name__ == '__main__':
    main()
