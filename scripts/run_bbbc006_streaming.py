"""Score all 34 official BBBC006 nuclear planes with bounded temporary storage.

Downloads one official archive at a time, scores all 768 w1 images, and discards
the temporary archive. Checkpoints retain source/image hashes and scores.
Native-intensity Laplacian is an exploratory control, not a production repair.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

import cv2
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qc_pipeline import compute_focus_score, to_grayscale_uint8, sha256_file  # noqa: E402
from scripts.run_bbbc006_focus_qc import site_key  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    code_hash = hashlib.sha256((ROOT / 'qc_pipeline.py').read_bytes()).hexdigest()
    all_rows, sources = [], []
    for z in range(34):
        checkpoint = args.out_dir / f'z_{z:02d}.json'
        if checkpoint.exists():
            cached = json.loads(checkpoint.read_text())
            if cached['code_sha256'] != code_hash or len(cached['per_image']) != 768:
                raise ValueError('Incompatible or incomplete checkpoint')
        else:
            url = f'https://data.broadinstitute.org/bbbc/BBBC006/BBBC006_v1_images_z_{z:02d}.zip'
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / 'images.zip'
                with urllib.request.urlopen(url, timeout=120) as src, archive.open('wb') as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                archive_hash = sha256_file(str(archive))
                rows = []
                with zipfile.ZipFile(archive) as bundle:
                    names = sorted(n for n in bundle.namelist() if n.lower().endswith('.tif') and '_w1' in n.lower())
                    if len(names) != 768:
                        raise ValueError(f'Plane {z} has {len(names)} nuclear images; expected 768')
                    for name in names:
                        content = bundle.read(name)
                        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
                        if image is None or image.ndim != 2:
                            raise ValueError(f'Invalid source {name}')
                        rows.append(dict(image=Path(name).name, site=site_key(Path(name)), z=z,
                                         expert_in_focus=11 <= z <= 23,
                                         focus_score=compute_focus_score(to_grayscale_uint8(image)),
                                         native_laplacian_variance=float(cv2.Laplacian(image.astype(np.float64), cv2.CV_64F).var()),
                                         sha256=hashlib.sha256(content).hexdigest()))
                cached = dict(source_url=url, source_sha256=archive_hash, code_sha256=code_hash, per_image=rows)
                checkpoint.write_text(json.dumps(cached, indent=2, allow_nan=False))
        all_rows.extend(cached['per_image'])
        sources.append({k: cached[k] for k in ['source_url', 'source_sha256']})
        print('completed plane', z, 'images', len(all_rows), flush=True)
    sites = {}
    for row in all_rows:
        sites.setdefault(row['site'], []).append(row['z'])
    if len(sites) != 768 or any(sorted(zs) != list(range(34)) for zs in sites.values()):
        raise ValueError('Incomplete physical site/depth coverage')
    truth = np.array([r['expert_in_focus'] for r in all_rows])
    npos, nneg = truth.sum(), (~truth).sum()
    summary = dict(n_images=len(all_rows), n_sites=len(sites), n_planes=34)
    for metric in ['focus_score', 'native_laplacian_variance']:
        scores = np.array([r[metric] for r in all_rows])
        summary[metric + '_auroc'] = float((rankdata(scores)[truth].sum()-npos*(npos+1)/2)/(npos*nneg))
    production = np.array([r['focus_score'] for r in all_rows])
    summary.update(sensitivity_at_100=float(np.mean(production[truth] >= 100)),
                   specificity_at_100=float(np.mean(production[~truth] < 100)))
    payload = dict(dataset='BBBC006', complete=True, summary=summary, per_plane=all_rows,
                   sources=sources, code_sha256=code_hash,
                   note='One plate. Expert z interval 11..23; native control is exploratory. No independent donor claim.')
    (args.out_dir / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
