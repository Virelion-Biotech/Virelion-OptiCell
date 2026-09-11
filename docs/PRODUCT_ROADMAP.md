# OptiCell — strategic product roadmap

**Headline:** trustworthy measurements with less manual work than alternatives.

Policy: publish only measured metrics.

---

## Stage progress

| Stage | Status |
|-------|--------|
| **1 Scientific benchmark (BBBC039)** | **Done** |
| **2 Killer workflow + CTC TRA** | **Done** |
| **3 Multi-backend orchestration + TRA** | **Done** |
| **3b Ship default path** | **Done** — `backend=auto` → cellpose if installed |
| **4 Easy UI** | Not started |
| **5 Ugly real data** | Not started |

---

## Production default (3b)

```
pip install -e '.[cellpose]'
python scripts/run_killer_workflow.py FRAMES -o out --enable-tracking --gpu
```

- `--backend auto` (default): Cellpose when importable, else threshold
- `count_cv > 1.0` → stderr warning (collapse guard)
- Multi-backend + selection v2: `scripts/run_stage3_orchestrate.py`

Measured basis: Cellpose best TRA on 6/6 CTC sequences; threshold best BBBC039 instance F1 / counts.

---

## Stage 1 — BBBC039 (n=197)

| Backend | Dice | Instance F1 | Mean \|count err\| |
|---------|-----:|------------:|------------------:|
| threshold | 0.925 | **0.929** | **8.7** |
| hybrid | 0.929 | 0.924 | 9.0 |
| Cellpose-SAM | **0.969** | 0.907 | 16.7 |
| CellProfiler | 0.895 | 0.722 | 28.1 |

---

## Stage 3 — CTC TRA

See `outputs/stage3/STAGE3_REPORT.md`. Selection v2 offline match **6/6**.

---

## Next

Stage 4 UI or Stage 5 ugly lab data.
