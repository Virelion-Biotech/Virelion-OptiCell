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
| **5 Ugly real data** | **Measured** — LIVECell cellpose vs threshold n=20 |

---

## Stage 5 takeaway (LIVECell, same 20 FOVs)

| Backend | Dice | F1 | \|count err\| |
|---------|-----:|---:|-------------:|
| cellpose | 0.930 | 0.904 | 25.7 |
| threshold | 0.054 | 0.435 | 87.5 |

Threshold is **not** a viable default on dense phase-contrast. Keep it for fluorescent nuclei (BBBC039). Production `auto` → Cellpose when installed is justified by this result as well as CTC TRA.

---

## Next

UI polish from real usage, or more LIVECell cell types / larger n — only if needed for a claim.
