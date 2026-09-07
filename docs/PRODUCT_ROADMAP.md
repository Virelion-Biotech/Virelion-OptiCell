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

Make validation the product core.

**Comparators**

- Cellpose / Cellpose-SAM / newer generalist models
- Representative CellProfiler pipeline (same images, documented settings)

**Metrics**

- Segmentation: IoU, Dice, instance F1
- Count error (absolute + relative)
- Phenotype measurement error (when GT exists)
- Tracking accuracy (when GT exists)
- Runtime, GPU/CPU, failure rate
- Reproducibility (seed, versions, image list hashes)

**Datasets**

1. BBBC039 full **n=200** (threshold, cellpose, hybrid)
2. +2–4 public sets (e.g. BBBC006 focus QC, LIVECell subset, BBBC038)
3. Document successes **and** failures

Deliverable: one comparison table + image lists under `outputs/`, cited in README.

---

## Stage 2 — One killer use case

Do **not** market every module at once.

**Primary workflow**

> Automated **QC → segmentation → tracking → phenotype** for time-lapse cell assays.

A complete path beats “another segmentation library.”

Secondary modules (screening stats, 3D, lineage) stay available but behind the main narrative.

---

## Stage 3 — AI orchestration layer

Cellpose (and SAM/DINO variants) are strong generalists. OptiCell’s edge is **decisioning**:

```
image → QC → choose backend/strategy → segment → confidence
      → track → phenotype → experiment-level QC → report
```

Concrete capabilities

1. **Confidence scoring** per FOV / object (disagreement, border fraction, focus)
2. **Automatic backend selection** (extend hybrid beyond count gate)
3. **Human-in-the-loop** correction hooks
4. Domain fine-tuning only with held-out measured metrics

---

## Stage 4 — Dramatically easier to use

Keep the headless Python core. Add a thin interface:

- Drop folder → experiment type → Run
- Inspect QC, masks, tracks
- Export CSV/Parquet + reproducible config
- Docker deployment

Later: web UI, templates, visual overlays, QC dashboard.

---

## Stage 5 — Validate with real (including ugly) data

3–5 real experiments, deliberately hard:

- low contrast, uneven illumination, crowding, debris
- focus drift, long time-lapse, 3D

Format for each:

`Dataset → OptiCell → quantitative result → comparison → failure analysis`

That beats a long feature list.

---

## Phased delivery

| Phase | Focus | Exit criterion |
|-------|--------|----------------|
| **1 Validation** | BBBC039-200 + 2–4 public sets; Cellpose + CellProfiler baselines | Measured tables in `outputs/` |
| **2 AI** | Confidence, auto backend, HITL | Hybrid beats both pure backends on ≥1 public set |
| **3 Product** | UI/templates/reports/Docker | Non-developer completes QC→segment→export |
| **4 Biology** | Dose-response, phenotype class, plate screening | One published workflow on real assay data |
| **5 Platform** | GPU/cloud, DB, provenance, multi-user API | External lab runs without repo surgery |

---

## Non-goals (for now)

- Adding another 20 analysis modules
- Claiming SOTA without image lists and JSON artifacts
- Marketing “we use Cellpose” as the product

---

## Immediate next actions (Stage 1)

1. Score **n=200** threshold / cellpose / hybrid on BBBC039
2. Add a **CellProfiler** baseline script (same 50 FOVs first)
3. Wire **confidence / disagreement** flags into the hybrid report
4. Keep README headline on the measured comparison table

*Trustworthy measurements > feature count.*
