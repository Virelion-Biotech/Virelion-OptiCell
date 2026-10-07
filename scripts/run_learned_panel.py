"""Checkpointed Cellpose/hybrid characterization with one inference per image.

Default is an operational panel: first/middle/last BBBC039 fields and the
lowest official validation image ID from each of LIVECell's eight cell types.
Use --all for the complete corpora; a panel is never full scientific validation.
Every checkpoint is bound to code, packages, input, reference, and model hashes.
"""
import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qc_pipeline import CellposeSegmenter, sha256_file  # noqa: E402
from ensemble import hybrid_threshold_cellpose, suggest_backend  # noqa: E402
from validation import paired_segmentation_metrics, aggregate_segmentation_rows, json_ready_metrics  # noqa: E402
from scripts.run_bbbc039_validation import find_pairs, load_image_gray, decode_bbbc039_mask  # noqa: E402
from scripts.run_livecell_validation import build_image_index, decode_coco_instance_masks  # noqa: E402


@dataclass
class CachedSegmenter:
    result: object

    def segment(self, gray, **kwargs):
        return self.result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--all', action='store_true')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--model', default='cpsam')
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cp = CellposeSegmenter(args.model, gpu=args.gpu)
    model = cp.model
    model_path = Path(model.pretrained_model)
    signature = dict(model=args.model, model_sha256=sha256_file(str(model_path)), gpu=args.gpu,
                     packages={x: version(x) for x in ['cellpose', 'torch', 'numpy', 'scipy', 'pycocotools']},
                     code_sha256=hashlib.sha256(b''.join((ROOT / x).read_bytes() for x in [
                         'qc_pipeline.py', 'ensemble.py', 'validation.py', 'scripts/run_learned_panel.py',
                         'scripts/run_bbbc039_validation.py', 'scripts/run_livecell_validation.py'])).hexdigest())
    records = []
    pairs = find_pairs(args.data_dir / 'bbbc039/images', args.data_dir / 'bbbc039/masks')
    if len(pairs) != 200:
        raise ValueError('Expected all 200 BBBC039 sources before panel selection')
    selected = pairs if args.all else [pairs[0], pairs[len(pairs)//2], pairs[-1]]
    records.extend(('BBBC039', image.name, image, mask, None) for image, mask in selected)
    src = args.data_dir / 'livecell/raw/livecell_coco_val.json'
    coco = json.loads(src.read_text())
    images = sorted(coco['images'], key=lambda x: x['id'])
    if len(images) != 570:
        raise ValueError('Expected complete official LIVECell validation annotations')
    selected = {}
    for image in images:
        cell_type = image['file_name'].split('_')[0]
        selected.setdefault(cell_type, image)
    if len(selected) != 8:
        raise ValueError('Expected all eight LIVECell cell types')
    lookup = {}
    for annotation in coco['annotations']:
        lookup.setdefault(annotation['image_id'], []).append(annotation)
    index = build_image_index(args.data_dir / 'livecell/images')
    for image in (images if args.all else list(selected.values())):
        records.append(('LIVECell_val', str(image['id']), index[image['file_name']], src,
                        (image, lookup.get(image['id'], []))))
    rows = []
    for dataset, identifier, image_path, mask_path, meta in records:
        gray = load_image_gray(image_path)
        truth = decode_bbbc039_mask(mask_path) if meta is None else decode_coco_instance_masks(
            meta[1], meta[0]['height'], meta[0]['width'])
        if gray.shape != truth.shape:
            raise ValueError('Image and reference dimensions differ')
        manifest = dict(signature, dataset=dataset, identifier=identifier,
                        image_sha256=sha256_file(str(image_path)),
                        reference_sha256=hashlib.sha256(truth.tobytes()).hexdigest())
        checkpoint = args.out_dir / (dataset + '_' + identifier + '.json')
        if checkpoint.exists():
            prior = json.loads(checkpoint.read_text())
            if prior['manifest'] != manifest:
                raise ValueError(f'Incompatible checkpoint: {checkpoint}')
            record_rows = prior['rows']
        else:
            result = cp.segment(gray)
            hybrid = hybrid_threshold_cellpose(gray, cellpose_segmenter=CachedSegmenter(result))
            auto, reason = suggest_backend(gray, cellpose_available=True)
            if auto != 'cellpose':
                raise ValueError('Auto behavior changed; cannot reuse Cellpose endpoint')
            record_rows = []
            for backend, seg in [('cellpose', result), ('hybrid', hybrid), ('auto', result)]:
                record_rows.append(dict(dataset=dataset, identifier=identifier, image=image_path.name,
                                        backend=backend, method=seg.method, auto_reason=reason,
                                        **json_ready_metrics(paired_segmentation_metrics(seg.labels, truth))))
            checkpoint.write_text(json.dumps(dict(manifest=manifest, rows=record_rows), indent=2, allow_nan=False))
        rows.extend(record_rows)
        summaries = {d: {b: json_ready_metrics(aggregate_segmentation_rows([
            r for r in rows if r['dataset'] == d and r['backend'] == b]))
            for b in ['cellpose', 'hybrid', 'auto']} for d in {r['dataset'] for r in rows}}
        payload = dict(complete=len(rows) == len(records)*3, full_corpus=args.all,
                       n_requested_images=len(records), n_scored_images=len(rows)//3,
                       signature=signature, summaries=summaries, per_image=rows,
                       note='Operational panel unless --all. Auto reuses the exact Cellpose result; no independent inference claim.')
        (args.out_dir / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
        print(dataset, identifier, 'complete', len(rows)//3, '/', len(records), flush=True)


if __name__ == '__main__':
    main()
