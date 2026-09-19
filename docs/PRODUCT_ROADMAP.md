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
| **5 Ugly real data** | **Measured** — LIVECell current threshold / cellpose / hybrid / auto n=20 |

---

## Stage 5 takeaway (LIVECell, same 20 FOVs)

| Backend | Dice | F1 | \|count err\| |
|---------|-----:|---:|-------------:|
Current measured values are generated into [`outputs/livecell_validation/LIVECELL_CURRENT_N20_REPORT.md`](../outputs/livecell_validation/LIVECELL_CURRENT_N20_REPORT.md) by the reproducible benchmark workflow. The historical hybrid 0.622 Dice result is retained there only as a pre-rule reference.

---

## Human-in-the-loop

The killer workflow can emit `review_queue.csv` for failed, non-PASS, or flagged FOVs with `--write-review-queue`. A reviewer can record `accept`, `reject`, or `rerun` with reviewer/notes, then re-run with `--review-decisions` to preserve the decision trail.

## Next

Real usage feedback on the UI; expand benchmark coverage beyond the current n=20 validation slice.
