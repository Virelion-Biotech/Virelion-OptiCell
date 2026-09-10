# OptiCell — strategic product roadmap

**Headline question:** *Is OptiCell measurably better, faster, or more reproducible than existing workflows?*

Differentiator is **not** “we wrap Cellpose.” It is:

> Given raw microscopy data, OptiCell automatically produces **trustworthy biological measurements** with less manual work and better QC than the alternatives.

Policy: publish only measured metrics. Stop rather than invent numbers.

---

## Stage progress

| Stage | Status |
|-------|--------|
| **1 Scientific benchmark** | **Done** — BBBC039 n=197 (threshold, hybrid, Cellpose-SAM, CellProfiler) |
| **2 Killer use case** | **Done (pipeline + CTC runs)** — QC→segment→features→phenotype→tracking on 6 CTC sequences |
| **3 AI orchestration** | FOV confidence + hybrid present; auto-backend next |
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

---

## Stage 2 — Killer workflow + CTC TL

```bash
python scripts/run_killer_workflow.py data/ctc/Fluo-N2DH-GOWT1/01 \
  -o outputs/stage2_gowt1_01 --backend threshold --enable-tracking
```

**Measured Colab runs** (threshold + tracking, 6 sequences):

| Dataset / seq | Frames | Objects | Tracks |
|---------------|-------:|--------:|-------:|
| GOWT1 01/02 | 92+92 | 2.6k+2.9k | 730+869 |
| SIM+ 01/02 | 65+150 | 2.7k+28k | 596+8013 |
| HeLa 01/02 | 92+92 | 6.7k+21k | 1.6k+5.5k |

Report: `outputs/stage2_ctc/STAGE2_CTC_REPORT.md`.

**Not yet:** TRA/SEG vs CTC ground truth (optional next measurement).

---

## Stage 3 — AI orchestration

```
image → QC → choose backend → segment → confidence → track → phenotype → report
```

Present: hybrid switch, `fov_confidence()`. Next: auto backend from confidence, HITL hooks.

---

*Trustworthy measurements > feature count.*
