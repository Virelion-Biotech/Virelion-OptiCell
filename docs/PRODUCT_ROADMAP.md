# OptiCell — strategic product roadmap

**Headline question:** *Is OptiCell measurably better, faster, or more reproducible than existing workflows?*

> Given raw microscopy data, OptiCell automatically produces **trustworthy biological measurements** with less manual work and better QC than the alternatives.

Policy: publish only measured metrics. Stop rather than invent numbers.

---

## Stage progress

| Stage | Status |
|-------|--------|
| **1 Scientific benchmark** | **Done** — BBBC039 n=197 |
| **2 Killer use case + TRA** | **Done (measured)** — CTC TRA/DET/LNK on 6 sequences |
| **3 AI orchestration** | Next |
| **4 Easy UI** | Not started |
| **5 Ugly real data** | Not started |

---

## Stage 1 — BBBC039 segmentation (measured)

| Backend | n | Dice | Instance F1 | Mean \|count err\| | Rel count |
|---------|--:|-----:|------------:|------------------:|----------:|
| OptiCell threshold | 197 | 0.925 | **0.929** | **8.7** | **0.077** |
| OptiCell hybrid | 197 | 0.929 | 0.924 | 9.0 | 0.083 |
| Cellpose-SAM | 197 | **0.969** | 0.907 | 16.7 | 0.172 |
| CellProfiler 4.2 | 197 | 0.895 | 0.722 | 28.1 | 0.281 |

---

## Stage 2 — CTC tracking accuracy (measured)

Threshold + assignment tracking; `traccuracy` CTCMetrics.

| Dataset / seq | TRA | DET | LNK |
|---------------|----:|----:|----:|
| GOWT1 01 | 0.364 | 0.367 | 0.346 |
| GOWT1 02 | 0.310 | 0.309 | 0.314 |
| SIM+ 01 | **0.809** | **0.831** | **0.664** |
| SIM+ 02 | **0.000** | **0.000** | 0.098 |
| HeLa 01 | 0.652 | 0.684 | 0.439 |
| HeLa 02 | 0.715 | 0.754 | 0.453 |

Full report: `outputs/stage2_ctc/STAGE2_CTC_TRA_REPORT.md`.

**Takeaway:** end-to-end TL pipeline works and is auditable; threshold DET limits TRA on hard sequences; SIM+02 is a documented failure mode (massive FP).

---

## Stage 3 — AI orchestration (next)

```
image → QC → choose backend → segment → confidence → track → phenotype → report
```

Use measured confidence / DET proxies to avoid SIM+02-style collapses.

---

*Trustworthy measurements > feature count.*
