# OptiCell — strategic product roadmap

**Headline question:** *Is OptiCell measurably better, faster, or more reproducible than existing workflows?*

Differentiator is **not** “we wrap Cellpose.” It is:

> Given raw microscopy data, OptiCell automatically produces **trustworthy biological measurements** with less manual work and better QC than the alternatives.

Policy: publish only measured metrics. Stop rather than invent numbers.

---

## Stage progress

| Stage | Status |
|-------|--------|
| **1 Scientific benchmark** | **Done** — BBBC039 n=197 all backends (threshold, hybrid, Cellpose-SAM, CellProfiler) |
| **2 Killer use case** | **Active** — QC→segment→features→phenotype (+ optional tracking) |
| **3 AI orchestration** | FOV confidence + hybrid switch present; auto-backend next |
| **4 Easy UI** | Not started |
| **5 Ugly real data** | Not started |

---

## Stage 1 — BBBC039 (measured)

| Backend | n | Dice | Instance F1 | Mean \|count err\| | Rel count |
|---------|--:|-----:|------------:|------------------:|----------:|
| OptiCell threshold | 197 | 0.925 | **0.929** | **8.7** | **0.077** |
| OptiCell hybrid | 197 | 0.929 | 0.924 | 9.0 | 0.083 |
| Cellpose-SAM | 197 | **0.969** | 0.907 | 16.7 | 0.172 |
| CellProfiler 4.2 | 197 | 0.895 | 0.722 | 28.1 | 0.281 |

Full write-up: `outputs/bbbc039_validation/BBBC039_STAGE1_COMPLETE.md`.

---

## Stage 2 — Killer use case

> Automated **QC → segmentation → features → phenotype** (+ tracking on real time-lapse).

```bash
python scripts/run_killer_workflow.py data/bbbc039/images/images \
  -o outputs/stage2_bbbc039_threshold --backend threshold --max-images 30
```

Docs: `docs/STAGE2_KILLER_WORKFLOW.md`.

**Next measurements (only with real data):**
1. Run Stage-2 phenotype table on BBBC039 n=30/200 (threshold) — counts + feature distributions only
2. When a true time-lapse assay is available, enable `--enable-tracking` and report track continuity metrics if ground truth exists; otherwise report descriptive track stats only

---

## Stage 3 — AI orchestration

```
image → QC → choose backend → segment → confidence → track → phenotype → report
```

Present: hybrid switch, `fov_confidence()`. Next: auto backend from confidence, HITL hooks.

---

## Stages 4–5

UI / Docker / multi-lab ugly data — after Stage 2 is habitually used on one assay type.

---

*Trustworthy measurements > feature count.*
