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
| **4 Easy UI** | **Started** — `app_streamlit.py` |
| **5 Ugly real data** | **Started** — LIVECell adapter + BBBC038 baseline |

---

## Stage 4 — Easy UI

```bash
pip install -e '.[dev,ui]'
streamlit run app_streamlit.py
```

Upload one image → real backend → acquisition QC, segmentation overlay, acceptance gate, confidence, per-object table. Optional: `pip install -e '.[cellpose]'` + GPU checkbox for cellpose/hybrid.

---

## Stage 5 — Ugly / dense real data

```bash
# BBBC038 (already has measured threshold n=200 on main)
python scripts/run_bbbc038_validation.py --max-images 200 --backend threshold

# LIVECell (dense phase-contrast, ~1.3GB download, CC BY-NC 4.0 non-commercial)
python scripts/run_livecell_validation.py --max-images 20 --backend threshold
# Cellpose on LIVECell needs GPU for practical speed:
python scripts/run_livecell_validation.py --max-images 20 --backend cellpose --gpu
```

Publish only measured JSON/CSV from these runs — do not invent baselines.

---

## Production default (3b)

```
pip install -e '.[cellpose]'
python scripts/run_killer_workflow.py FRAMES -o out --enable-tracking --gpu
```

---

## Next

1. Run LIVECell threshold (and optionally cellpose GPU) → commit measured table to README.
2. Polish UI (folder batch / TL tracking view) only after you have real screenshots.
