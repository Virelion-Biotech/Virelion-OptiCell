# CellProfiler baseline on BBBC039 (Stage 1 — real CP labels)

**Goal:** measure CellProfiler on the **same FOVs** as OptiCell, with the **same** `validation.py` metrics.

Policy: no fabricated CP numbers. This doc only describes how to produce them.

## Reality check

| Environment | CellProfiler |
|-------------|--------------|
| Laptop / workstation | Supported (GUI or CLI) |
| Conda + Java | Supported (`conda install -c bioconda cellprofiler`) |
| Google Colab | **Usually not practical** (Java, wx, Python pin) |

On Colab, use OptiCell backends. Run **real CP** on a machine where `cellprofiler` works, then score.

---

## Path A — one script (CLI on PATH)

```bash
cd ~/Virelion-OptiCell   # or your clone
git pull origin main

# BBBC039 data must already exist under data/bbbc039/
MAX_IMAGES=50 bash scripts/run_cellprofiler_bbbc039.sh
```

This will:

1. Stage first N TIFFs  
2. Run `cellprofiler -c -r -p pipelines/bbbc039_nuclei.cppipe`  
3. Collect label PNGs into `outputs/cellprofiler_bbbc039/labels/`  
4. Call `score_external_labels.py --name cellprofiler`  

**Requires:** `cellprofiler` on `PATH`.

If the bundled `.cppipe` fails to load in your CP version, use Path B (GUI) and keep the same score command.

---

## Path B — CellProfiler GUI (most reliable)

1. Install: https://cellprofiler.org/releases  
2. Open CellProfiler → **File → Import → Pipeline from file** → `pipelines/bbbc039_nuclei.cppipe`  
   (or build IdentifyPrimaryObjects manually with settings below)  
3. **Images** module: drag `data/bbbc039/images/images/*.tif` (first 50 for pilot)  
4. **IdentifyPrimaryObjects** (recommended starting point):

   | Setting | Value |
   |---------|--------|
   | Input | DNA / grayscale |
   | Diameter (min, max) | 10, 80 px |
   | Discard outside diameter | Yes |
   | Discard touching border | Yes |
   | Threshold | Global **Otsu**, **two classes** |
   | Declump | Shape |

5. **ConvertObjectsToImage** → uint16 labels  
6. **SaveImages** → PNG, one file per FOV  
   - File name **from image filename**  
   - Prefer exact stem match: `IXMtest_....png` (same as TIFF without `.tif`)  
7. Score:

```bash
python scripts/score_external_labels.py \
  --pred-dir /path/to/your/cp_label_pngs \
  --data-dir data/bbbc039 \
  --max-images 50 \
  --name cellprofiler
```

Paste the SUMMARY block into the repo (measured only).

---

## Path C — you already have label PNGs

```bash
python scripts/score_external_labels.py \
  --pred-dir /absolute/path/to/cp_labels \
  --data-dir data/bbbc039 \
  --max-images 50 \
  --name cellprofiler
```

Stems must match BBBC039 TIFF stems.

---

## OptiCell numbers already measured (same set)

| Backend | n | Dice | F1 | \|count err\| |
|---------|--:|-----:|---:|-------------:|
| Threshold | 50 | 0.945 | 0.955 | 6.1 |
| Cellpose-SAM | 50 | 0.970 | 0.908 | 15.0 |
| Hybrid | 50 | 0.951 | 0.950 | 6.4 |
| Threshold export→score round-trip | 50 | 0.945 | 0.955 | 6.1 |
| **CellProfiler** | 50 | *run Path A/B* | | |

---

## Install hints

```bash
# Conda (often easiest on Linux)
conda create -n cp python=3.9 -y
conda activate cp
conda install -c bioconda -c conda-forge cellprofiler -y
cellprofiler --version
```

Desktop app: note the full path to the binary and either add it to `PATH` or call it explicitly in the shell script.
