# OptiCell — 5-minute demo checklist

**Goal:** show trustworthy QC + segmentation without a terminal rabbit hole.

## 0. Once (2–3 min)

```bash
git clone https://github.com/Virelion-Biotech/Virelion-OptiCell.git
cd Virelion-OptiCell
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[ui,cellpose]'
```

GPU optional but recommended for Cellpose.

## 1. Launch UI (30 s)

```bash
streamlit run app_streamlit.py
```

Browser opens to **OptiCell — Easy UI**.

## 2. Run one image (1 min)

1. Sidebar → backend **`auto`** (Cellpose if installed; prefers Cellpose on low-contrast / phase-like images).
2. Enable **GPU** if you have CUDA.
3. Upload a `.tif` / `.png` microscopy frame.
4. Read the four blocks:
   - **Acquisition QC** (focus / artifact burden)
   - **Segmentation** overlay + object count + which backend `auto` picked
   - **Segmentation QC** (PASS / REVIEW / FAIL + confidence flags)
   - **Per-object CSV** + optional overlay PNG download

Nothing on the page is invented — numbers come from `qc_pipeline` / `ensemble` / `acceptance`.

## 3. Batch path (CLI, optional)

```bash
python scripts/run_killer_workflow.py /path/to/frames \
  -o outputs/demo_run --backend auto --gpu --enable-tracking
```

Open `outputs/demo_run/workflow_summary.json` and `cell_features_phenotype.csv`.

## 4. Talking points (measured, not marketing)

| Dataset | Result |
|---------|--------|
| BBBC039 nuclei | Threshold strong on count/F1; Cellpose strongest Dice |
| CTC TRA (6 sequences) | Cellpose best on all six |
| LIVECell phase-contrast | Cellpose Dice **0.93**; threshold **0.05** — do not default threshold here |

**Product line:** *Given raw microscopy data, OptiCell produces measurements with explicit QC gates — and defaults to Cellpose when the data looks low-contrast / phase-like.*

## 5. If something fails

| Symptom | Fix |
|---------|-----|
| Cellpose import error | `pip install -e '.[cellpose]'` (+ GPU torch if needed) |
| Empty segmentation on phase | Use **auto** or **cellpose**, not threshold |
| Tracking nonsense | Only `--enable-tracking` on **ordered time-lapse of one FOV** |

## Done when

- [ ] UI opens and runs one image end-to-end
- [ ] Acceptance + confidence visible
- [ ] CSV / overlay download works
- [ ] You can state the LIVECell / BBBC039 contrast without reading the full README
