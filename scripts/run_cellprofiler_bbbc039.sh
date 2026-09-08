#!/usr/bin/env bash
# Run real CellProfiler on BBBC039, then score with OptiCell metrics.
# Requires CellProfiler CLI on PATH (not typical on Google Colab).
#
# Usage:
#   bash scripts/run_cellprofiler_bbbc039.sh
#   MAX_IMAGES=50 bash scripts/run_cellprofiler_bbbc039.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/bbbc039}"
IMG_DIR="${IMG_DIR:-$DATA_DIR/images/images}"
# nested layout fallback
if [[ ! -d "$IMG_DIR" ]]; then
  IMG_DIR="$DATA_DIR/images"
fi
OUT_DIR="${OUT_DIR:-outputs/cellprofiler_bbbc039}"
PIPE="${PIPE:-pipelines/bbbc039_nuclei.cppipe}"
MAX_IMAGES="${MAX_IMAGES:-50}"

if ! command -v cellprofiler >/dev/null 2>&1; then
  echo "ERROR: 'cellprofiler' not on PATH."
  echo "Install options:"
  echo "  - https://cellprofiler.org/releases (desktop app; use its CLI path)"
  echo "  - conda: conda install -c bioconda cellprofiler"
  echo "  - pip (fragile): needs Java + often Python 3.8/3.9"
  echo ""
  echo "After install, re-run this script or run CP GUI with pipelines/bbbc039_nuclei.cppipe"
  exit 2
fi

if [[ ! -f "$PIPE" ]]; then
  echo "ERROR: missing pipeline $PIPE"
  exit 3
fi

if [[ ! -d "$IMG_DIR" ]]; then
  echo "ERROR: images not found at $IMG_DIR"
  echo "Download BBBC039 first, e.g.:"
  echo "  python scripts/run_bbbc039_validation.py --max-images 1"
  exit 4
fi

mkdir -p "$OUT_DIR/labels" "$OUT_DIR/cp_out"

# Optional: limit to first N tifs via a staging folder
STAGE="$OUT_DIR/input_subset"
rm -rf "$STAGE"
mkdir -p "$STAGE"
mapfile -t TIFS < <(find "$IMG_DIR" -maxdepth 3 -type f \( -iname '*.tif' -o -iname '*.tiff' \) | sort | head -n "$MAX_IMAGES")
if [[ ${#TIFS[@]} -eq 0 ]]; then
  echo "ERROR: no TIFFs under $IMG_DIR"
  exit 5
fi
for f in "${TIFS[@]}"; do
  ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$STAGE/$(basename "$f")" || cp "$f" "$STAGE/"
done
echo "[cp] n_images=${#TIFS[@]} input=$STAGE out=$OUT_DIR/cp_out"

cellprofiler -c -r \
  -p "$PIPE" \
  -i "$STAGE" \
  -o "$OUT_DIR/cp_out"

# Collect label PNGs; strip _cp_labels suffix so stems match TIFF stems
shopt -s nullglob
for f in "$OUT_DIR"/cp_out/*_cp_labels.png "$OUT_DIR"/cp_out/**/*_cp_labels.png; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f" _cp_labels.png)"
  cp "$f" "$OUT_DIR/labels/${base}.png"
done
# also copy any plain pngs CP may have written
for f in "$OUT_DIR"/cp_out/*.png; do
  [[ -f "$f" ]] || continue
  bn="$(basename "$f")"
  if [[ "$bn" != *_cp_labels.png ]]; then
    cp "$f" "$OUT_DIR/labels/$bn" || true
  fi
done

N_LABELS=$(find "$OUT_DIR/labels" -name '*.png' | wc -l | tr -d ' ')
echo "[cp] label pngs in $OUT_DIR/labels: $N_LABELS"
if [[ "$N_LABELS" -eq 0 ]]; then
  echo "ERROR: CellProfiler ran but no label PNGs found under $OUT_DIR/cp_out"
  echo "Open the pipeline in CP GUI, fix SaveImages paths, re-export."
  ls -la "$OUT_DIR/cp_out" || true
  exit 6
fi

echo "[score] OptiCell metrics on CP labels..."
python scripts/score_external_labels.py \
  --pred-dir "$OUT_DIR/labels" \
  --data-dir "$DATA_DIR" \
  --max-images "$MAX_IMAGES" \
  --name cellprofiler \
  --out-dir outputs/bbbc039_validation

echo "Done. See outputs/bbbc039_validation/bbbc039_cellprofiler_n*.json"
