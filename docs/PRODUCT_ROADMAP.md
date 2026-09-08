# OptiCell — strategic product roadmap

**Headline question:** *Is OptiCell measurably better, faster, or more reproducible than existing workflows?*

Differentiator is **not** “we wrap Cellpose.” It is:

> Given raw microscopy data, OptiCell automatically produces **trustworthy biological measurements** with less manual work and better QC than the alternatives.

Policy: publish only measured metrics. Stop rather than invent numbers.

---

## Stage progress

| Stage | Status |
|-------|--------|
| **1 Scientific benchmark** | **In progress** — BBBC039 n=50/200 measured; CellProfiler path ready; BBBC006 focus script ready |
| **2 Killer use case** | **Started** — `scripts/run_killer_workflow.py` (QC→segment→confidence) |
| **3 AI orchestration** | FOV confidence + hybrid backend selection present; expand next |
| **4 Easy UI** | Not started |
| **5 Ugly real data** | Not started |

---

## Proven baseline (BBBC039, measured)

| Backend | n | Dice | Instance F1 | \|count err\| | Rel count |
|---------|--:|-----:|------------:|-------------:|----------:|
| Threshold | 50 | 0.945 | **0.955** | **6.1** | 0.058 |
| Cellpose-SAM | 50 | **0.970** | 0.908 | 15.0 | — |
| Hybrid | 50 | 0.951 | 0.950 | 6.4 | 0.065 |
| Threshold | **200** | 0.925 | **0.929** | **8.7** | **0.077** |
| Hybrid | **200** | **0.929** | 0.924 | 9.0 | 0.083 |

Artifacts: `outputs/bbbc039_validation/`.

---

## Stage 1 — Lock the scientific benchmark

### Done

- Multi-backend runner + hybrid count-gated switch
- Empty-GT FOV handling (3/200 on BBBC039)
- README measured table
- FOV confidence scores

### Tooling

```bash
# OptiCell backends
python scripts/run_bbbc039_multi_backend.py --max-images 200 --backends threshold,hybrid --skip-download

# CellProfiler (or any external labels) on same FOVs
python scripts/score_external_labels.py --pred-dir cp_labels --max-images 50 --name cellprofiler
# Instructions: docs/CELLPROFILER_BBBC039_BASELINE.md

# Second public dataset — focus QC (local BBBC006 extract)
python scripts/run_bbbc006_focus_qc.py --root data/bbbc006 --z-focus 16 --max-sites 20
```

### Still to measure

1. CellProfiler numbers on BBBC039 (same FOVs) via export + `score_external_labels.py`
2. Cellpose n=200 when GPU available
3. BBBC006 focus correlation on a real local extract (z-stacks are large; no fake scores)
4. Optional LIVECell subset later

---

## Stage 2 — One killer use case

> Automated **QC → segmentation → tracking → phenotype** for time-lapse cell assays.

**Now:**

```bash
python scripts/run_killer_workflow.py /path/to/images -o outputs/workflow_run --backend threshold
```

Emits per-FOV focus, object count, confidence flags, label masks, JSON/CSV.

**Next:** wire tracking + simple phenotype features into the same report.

---

## Stage 3 — AI orchestration

```
image → QC → choose backend → segment → confidence → track → phenotype → report
```

Present: hybrid backend switch, `fov_confidence()`. Next: auto backend from confidence, HITL hooks.

---

## Stages 4–5

UI / Docker / ugly multi-lab datasets — after Stage 1 comparator rows are filled and Stage 2 workflow is end-to-end on one assay type.

---

*Trustworthy measurements > feature count.*
