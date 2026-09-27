# Product completeness matrix

This document separates implemented software from evidence or infrastructure
that cannot be truthfully manufactured in code.

| Area | Engineering status | Remaining external dependency |
|---|---|---|
| Batch UI | Implemented | Performance at very large batch scale still deployment-dependent |
| Projects/experiments | Implemented with persistent SQLite store | Multi-node server DB only if deploying horizontally |
| Full provenance | Implemented v2 content-addressed manifests | External artifact archive policy is operator-owned |
| Cellpose testing | Mock/unit contract in normal CI; real Cellpose scheduled/manual | GPU E2E needs a connected GPU runner |
| Scientific calibration | Calibration framework implemented | Labeled domain-specific outcomes required |
| External validation breadth | Evidence registry implemented | Independent-lab datasets/studies required |
| Human review | Decisions, notes, correction-mask upload, pixel-level brush editing, preview, rerun, and audit implemented | Advanced contour/spline editing beyond brush operations is optional future polish |
| Dataset QC | Implemented | Domain-specific QC cutoffs still need calibration |
| Model/config management | Presets, comparisons, custom checkpoint hash/persistence implemented | Model performance claims require validation |
| Deployment | Docker, health, local auth, quotas, audit, backup implemented | Enterprise SSO/multi-node DB/object storage are operator integrations |
| Large data | Safe OME-TIFF planning/streaming and pyramid inspection implemented | Unsupported WSI codecs/formats need a dedicated reader/conversion |
| Release polish | Container config + tag release workflow + docs implemented | Hosted public instance is an operational choice |
