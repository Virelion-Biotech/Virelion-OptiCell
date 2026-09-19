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
| **4 Easy UI** | **Shipped** — `app_streamlit.py` + [`docs/DEMO_5_MIN.md`](DEMO_5_MIN.md) |
| **5 Ugly real data** | **Measured** — LIVECell threshold / cellpose / hybrid n=20 |

---

## Stage 5 takeaway (LIVECell, same 20 FOVs)

| Backend | Dice | F1 | \|count err\| |
|---------|-----:|---:|-------------:|
| cellpose | 0.930 | 0.904 | 25.7 |
| hybrid (collapse + low-contrast aware) | 0.622* | 0.767 | 37.1 |
| threshold | 0.054 | 0.435 | 87.5 |

\*Hybrid n=20 measured **before** low-contrast prefer-Cellpose rule; collapse fix alone already rescued zero-object FOVs. Production default remains **Cellpose** on phase-like data.

---

## Next

Real usage feedback on the UI; optional human-in-the-loop for REVIEW FOVs.
