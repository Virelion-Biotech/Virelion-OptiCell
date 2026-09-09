#!/usr/bin/env bash
# Run real CellProfiler on BBBC039, then score with OptiCell metrics.
# Requires CellProfiler CLI on PATH (conda env cp42: python -m cellprofiler).
#
# Usage:
#   conda activate cp42
#   MAX_IMAGES=50 bash scripts/run_cellprofiler_bbbc039.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/bbbc039}"
IMG_DIR="${IMG_DIR:-$DATA_DIR/images/images}"
if [[ ! -d "$IMG_DIR" ]]; then
  IMG_DIR="$DATA_DIR/images"
fi
OUT_DIR="${OUT_DIR:-outputs/cellprofiler_bbbc039}"
PIPE="${PIPE:-pipelines/bbbc039_nuclei.cppipe}"
MAX_IMAGES="${MAX_IMAGES:-50}"

CP_CMD="${CP_CMD:-}"
if [[ -z "$CP_CMD" ]]; then
  if command -v cellprofiler >/dev/null 2>&1; then
    CP_CMD="cellprofiler"
  elif python -c "import cellprofiler" >/dev/null 2>&1; then
    CP_CMD="python -m cellprofiler"
  else
    echo "ERROR: CellProfiler not found. conda activate cp42 first."
    exit 2
  fi
fi

if [[ ! -f "$PIPE" ]]; then
  echo "ERROR: missing pipeline $PIPE"
  exit 3
fi

if [[ ! -d "$IMG_DIR" ]]; then
  echo "ERROR: images not found at $IMG_DIR"
  exit 4
fi

mkdir -p "$OUT_DIR/labels" "$OUT_DIR/cp_out"
STAGE="$OUT_DIR/input_subset"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# Exclude macOS AppleDouble (._*) and __MACOSX junk
mapfile -t TIFS < <(
  find "$IMG_DIR" -maxdepth 4 -type f \( -iname '*.tif' -o -iname '*.tiff' \) \
    ! -name '._*' \
    ! -path '*/__MACOSX/*' \
    | sort | head -n "$MAX_IMAGES"
)

if [[ ${#TIFS[@]} -eq 0 ]]; then
  echo "ERROR: no real TIFFs under $IMG_DIR"
  exit 5
fi

for f in "${TIFS[@]}"; do
  base="$(basename "$f")"
  # skip if basename still starts with ._
  [[ "$base" == ._* ]] && continue
  cp -n "$f" "$STAGE/$base"
done

# purge any ._ that slipped in
find "$STAGE" -name '._*' -delete 2>/dev/null || true

N=$(find "$STAGE" -type f | wc -l | tr -d ' ')
echo "[cp] cmd=$CP_CMD n_images=$N input=$STAGE out=$OUT_DIR/cp_out"

# shellcheck disable=SC2086
$CP_CMD -c -r \
  -p "$PIPE" \
  -i "$STAGE" \
  -o "$OUT_DIR/cp_out"

shopt -s nullglob globstar
for f in "$OUT_DIR"/cp_out/*_cp_labels.png "$OUT_DIR"/cp_out/**/*_cp_labels.png; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f" _cp_labels.png)"
  cp "$f" "$OUT_DIR/labels/${base}.png"
done
for f in "$OUT_DIR"/cp_out/*.png; do
  [[ -f "$f" ]] || continue
  bn="$(basename "$f")"
  if [[ "$bn" != *_cp_labels.png ]]; then
    cp "$f" "$OUT_DIR/labels/$bn" || true
  fi
done

N_LABELS=$(find "$OUT_DIR/labels" -name '*.png' ! -name '._*' | wc -l | tr -d ' ')
echo "[cp] label pngs: $N_LABELS"
if [[ "$N_LABELS" -eq 0 ]]; then
  echo "ERROR: no label PNGs under $OUT_DIR/cp_out"
  ls -la "$OUT_DIR/cp_out" || true
  exit 6
fi

echo "[score] OptiCell metrics..."
python scripts/score_external_labels.py \
  --pred-dir "$OUT_DIR/labels" \
  --data-dir "$DATA_DIR" \
  --max-images "$MAX_IMAGES" \
  --name cellprofiler \
  --out-dir outputs/bbbc039_validation

echo "Done. outputs/bbbc039_validation/bbbc039_cellprofiler_n*.json"
