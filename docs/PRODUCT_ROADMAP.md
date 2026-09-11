# OptiCell — strategic product roadmap

**Headline:** trustworthy measurements with less manual work than alternatives.

Policy: publish only measured metrics.

---

## Stage progress

| Stage | Status |
|-------|--------|
| **1 Scientific benchmark (BBBC039)** | **Done** |
| **2 Killer workflow + CTC TRA (threshold)** | **Done** |
| **3 Multi-backend orchestration + TRA** | **Done (measured)** |
| **4 Easy UI** | Not started |
| **5 Ugly real data** | Not started |

---

## Stage 1 — BBBC039 (n=197)

| Backend | Dice | Instance F1 | Mean \|count err\| |
|---------|-----:|------------:|------------------:|
| threshold | 0.925 | **0.929** | **8.7** |
| hybrid | 0.929 | 0.924 | 9.0 |
| Cellpose-SAM | **0.969** | 0.907 | 16.7 |
| CellProfiler | 0.895 | 0.722 | 28.1 |

---

## Stage 2 — CTC TRA (threshold only)

Weak on GOWT1 (~0.3); SIM+02 collapsed (TRA=0); HeLa ~0.65–0.72.

---

## Stage 3 — Multi-backend TRA (measured)

| Seq | thr | hybrid | **cellpose** | Auto-select matched best? |
|-----|----:|-------:|-------------:|:-------------------------:|
| GOWT1 01 | 0.36 | 0.60 | **0.97** | Yes |
| GOWT1 02 | 0.31 | 0.78 | **0.92** | Yes |
| SIM+ 01 | 0.81 | 0.95 | **0.96** | No (picked thr) |
| SIM+ 02 | 0.00 | 0.00 | **0.50** | Yes |
| HeLa 01 | 0.65 | 0.65 | **0.91** | Yes |
| HeLa 02 | 0.72 | 0.72 | **0.93** | No (picked thr) |

Report: `outputs/stage3/STAGE3_REPORT.md`.

**Takeaway:** Cellpose is required for strong TRA. Confidence@100 cannot choose backends; fix selection rule next, or default to Cellpose when GPU present.

---

## Stage 4+ 

UI, experiment templates, and ugly real lab data — after selection rule is tightened.
