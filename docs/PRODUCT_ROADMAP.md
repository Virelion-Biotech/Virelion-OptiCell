# OptiCell — strategic product roadmap

**Headline question:** *Is OptiCell measurably better, faster, or more reproducible than existing workflows?*

Differentiator is **not** “we wrap Cellpose.” It is:

> Given raw microscopy data, OptiCell automatically produces **trustworthy biological measurements** with less manual work and better QC than the alternatives.

Policy: publish only measured metrics. Stop rather than invent numbers.

---

## Proven baseline (BBBC039 n=50, measured 2026-09)

| Backend | Dice | Instance F1 | \|count err\| |
|---------|-----:|------------:|-------------:|
| Threshold | 0.945 | **0.955** | **6.1** |
| Cellpose-SAM | **0.970** | 0.908 | 15.0 |
| Hybrid (count-gated) | 0.951 | 0.950 | 6.4 |

Artifacts: `outputs/bbbc039_validation/`.

---

## Stage 1 — Lock the scientific benchmark

### Tooling (in repo now)

```bash
# Full BBBC039, fair same-FOV comparison (CPU: threshold + hybrid)
python scripts/run_bbbc039_multi_backend.py --max-images 200 --backends threshold,hybrid --skip-download

# With Cellpose GPU
python scripts/run_bbbc039_multi_backend.py --max-images 200 --backends threshold,cellpose,hybrid --gpu --skip-download
```

Writes:

- per-backend JSON/CSV under `outputs/bbbc039_validation/`
- `BBBC039_MULTI_BACKEND_COMPARISON.md` + `.json`

### Comparators

- Cellpose / Cellpose-SAM / newer generalist models (via `--backend cellpose`)
- CellProfiler baseline (next Stage-1 item: same FOV list, documented pipeline)

### Metrics

- Segmentation: IoU, Dice, instance F1
- Count error (absolute + relative)
- Runtime / GPU flags in JSON
- Reproducibility: fixed basename sort, software tags in payload

### Datasets

1. BBBC039 full **n=200** (primary)
2. +2–4 public sets (BBBC006 focus, LIVECell subset, BBBC038)
3. Document successes **and** failures

---

## Stage 2 — One killer use case

> Automated **QC → segmentation → tracking → phenotype** for time-lapse cell assays.

---

## Stage 3 — AI orchestration layer

```
image → QC → choose backend/strategy → segment → confidence
      → track → phenotype → experiment-level QC → report
```

1. Confidence scoring per FOV / object
2. Automatic backend selection (extend hybrid)
3. Human-in-the-loop correction hooks
4. Fine-tuning only with held-out measured metrics

---

## Stage 4 — Easier to use

Headless Python core + thin UI: drop folder → experiment type → Run → export + config.

---

## Stage 5 — Ugly real data

3–5 real experiments (low contrast, crowding, debris, focus drift, long time-lapse, 3D).

`Dataset → OptiCell → result → comparison → failure analysis`

---

## Immediate next actions

1. **Run** `run_bbbc039_multi_backend.py` at n=200 (threshold,hybrid; add cellpose if GPU)
2. Commit the generated comparison markdown (measured only)
3. CellProfiler baseline on the same FOV list
4. Confidence/disagreement summary in the multi-backend report

*Trustworthy measurements > feature count.*
