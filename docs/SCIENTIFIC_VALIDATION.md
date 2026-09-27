# Scientific calibration and external validation

OptiCell's built-in QC confidence and acceptance rules are heuristic unless
calibrated on labeled data from the intended domain.

## Calibration workflow

The **Calibration & Validation** page accepts a CSV containing a 0-100 QC score
and a binary observed outcome. It reports:

- empirical reliability bins,
- sensitivity/specificity/precision for every observed threshold,
- a threshold satisfying explicitly chosen minimum sensitivity and specificity,
  when one exists.

Do not transfer a calibrated threshold to a materially different modality,
lab, microscope, staining protocol or cell type without re-evaluation.

## External-validation registry

The in-product registry records the evidence currently present in this
repository. BBBC039, CTC TRA and LIVECell are marked as repository validation,
not independent-lab evidence. The UI intentionally reports the current
independent-lab count rather than converting these studies into a stronger
claim.

## What would close the external-validation gap

At minimum, add locked evaluations from independent laboratories spanning the
intended modalities, with dataset version/hash, microscope/acquisition
metadata, preprocessing policy, ground-truth protocol, backend/model hash and
predeclared metrics. Store each study as immutable artifacts and register it
only after the outputs are committed or otherwise archived.
