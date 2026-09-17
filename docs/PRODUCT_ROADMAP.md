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
| **3b Ship default path** | **Done** |
| **4 Easy UI** | **Started** — `app_streamlit.py` |
| **5 Ugly real data** | **In progress** — LIVECell cellpose n=20 measured |

---

## Stage 5 — LIVECell (measured, cellpose n=20)

| Metric | Value |
|--------|------:|
| Dice | 0.930 |
| Instance F1 | 0.904 |
| Mean \|count err\| | 25.65 |
| Mean GT cells / image | 197.6 (max 425) |

Report: `outputs/livecell_validation/LIVECELL_CELLPOSE_N20_REPORT.md`.

Still useful: threshold baseline on same 20 FOVs for comparison.

---

## Stage 4 — Easy UI

```bash
pip install -e '.[ui]'
streamlit run app_streamlit.py
```

---

## Next

1. Optional: LIVECell threshold n=20 on same FOVs → side-by-side table.
2. UI polish only after real usage feedback.
