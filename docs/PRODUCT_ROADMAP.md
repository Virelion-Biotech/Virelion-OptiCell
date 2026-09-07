# OptiCell — product roadmap (measured-first)

Goal: make OptiCell the **most trustworthy** microscopy QC + segmentation validation toolkit — not the loudest.

## What already works (proven)

| Capability | Evidence |
|------------|----------|
| Classical Otsu nuclei baseline | BBBC039 n=50: Dice 0.945, instance F1 0.955, \|count err\| 6.1 |
| Cellpose-SAM optional backend | BBBC039 n=50: Dice 0.970, instance F1 0.908, \|count err\| 15.0 |
| Hybrid threshold↔Cellpose switch | `ensemble.hybrid_threshold_cellpose` — prefers Cellpose when counts agree, else threshold |
| Model-agnostic metrics | `validation.py` (pixel + instance centroid matching) |
| Reproducible runners | laptop script, Ibex SLURM, Colab cells |
| No-hallucination policy | docs + reports only list measured numbers |

## Power upgrades (priority order)

### P0 — ship measured truth at scale
1. **BBBC039 full 200** for threshold, cellpose, hybrid → one comparison table.
2. Commit JSON/CSV artifacts with image lists + software versions.
3. CI smoke test on synthetic fixtures (not full BBBC) so main never breaks silently.

### P1 — hybrid that wins both axes
Current hybrid is a **count-gated switch**. Next measured iterations:
1. Watershed split of Cellpose blobs using threshold local maxima (target: Cellpose Dice + threshold counts).
2. Confidence-weighted fusion when both backends agree on FOV quality flags.
3. Publish only after n≥50 measured comparison vs pure threshold and pure Cellpose.

### P2 — large public corpora (hundreds of thousands of cells)
| Dataset | Role |
|---------|------|
| LIVECell (~1.6M cells) | Phase-contrast instance benchmark |
| TissueNet (>1M cells) | Multiplex nuclear/whole-cell |
| BBBC006 | Focus / blur QC stress test |

Adapters must record SHA-256 of downloads and exact image lists. **No training claims without held-out measured metrics.**

### P3 — product surface
1. One-command `opticell validate --dataset bbbc039 --backend hybrid`.
2. Auto HTML report from JSON (tables + failure FOV list).
3. Optional tracking / lineage QC modules already in tree — wire to the same metric policy.

## Non-goals (until measured)
- Claiming SOTA on LIVECell/TissueNet without runs.
- Shipping a custom trained weights file without train/val hashes and test metrics.
- Marketing numbers disconnected from `outputs/` artifacts.

## How to contribute a “phenomenal” result
1. Run a backend on a named image list.
2. Drop JSON under `outputs/`.
3. Add a short markdown table with **only** those numbers.
4. PR against main.

---

*Stop rather than invent. Scale only after subset pipelines are deterministic.*
