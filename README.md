# WeedDG-3M | Agent-controlled Weed Domain Generalization

**Planning version:** 2026-11-02. Research hypothesis, not verified improvements.

Goal: develop a source-only generalizable weed detector (primary candidate YOLO26n; YOLO11n historical baseline, YOLO26s stronger baseline) with three togglable training modules:
A = object-preserving background domain randomization; B = progressive photometric strength scheduling; C = scale-aware ROI feature consistency.

## Recommended first actions

1. Create a new **public** GitHub repository `weed-domain-generalization` (empty repository recommended).
2. Clone it into a local folder outside any private or unrelated repositories.
3. Extract this starter ZIP, copy the files into the cloned repo, and read `AGENTS.md`.
4. Keep all original data and model checkpoints **outside the Git repository**.
5. Supply only local dataset paths to the Coding Agent. Never publish raw datasets or access tokens.
6. Give the Coding Agent the contents of `tasks/EXP-001.md`. Wait for its report and a PR before authorizing full training.
7. Share the public GitHub repo URL (or the PR URL) in ChatGPT for experiment review.

**Source datasets:**
- CornWeed: https://zenodo.org/records/7961764 ; 3,574 labeled images / 4,981 unlabeled images (no quantitative ground truth for the latter).
- Weed-crop V2: https://data.mendeley.com/datasets/mthv4ppwyw/2 ; 1,120 labeled images with 5 weed and 8 crop categories.
- Optional third held-out domain: CottonWeedDet12, pending annotation/license audit.

**Classification policy:** Primary paper target is `weed-only` detection, to avoid creating invalid two-class comparisons across datasets. All crop boxes are background/hard negatives only if annotation completeness is carefully checked. Optionally add `crop/weed` as secondary, conditioned on label-space audit.

**Selection protocol:** Tune only with source-domain training/validation data and pre-registered source-domain robustness diagnostics. Target-domain box labels may be audited for mapping issues, but external test performance must not drive design/threshold/hyperparameter selection. Report external results only after locking the method or explicitly label exploratory external checks as such.

No fabricated results; no acceptance/indexing guarantees.

## Fixed Windows workspace for this project

Workspace root: `C:\Users\gass\Documents\experiment`

Git checkout: `C:\Users\gass\Documents\experiment\weed-domain-generalization`

Data (outside git): `C:\Users\gass\Documents\experiment\datasets`

Runs (outside git): `C:\Users\gass\Documents\experiment\runs`

Weights (outside git): `C:\Users\gass\Documents\experiment\weights`

If the package cannot be downloaded, the research plan can be reconstructed by a local Coding Agent from the EXP-001 prompt in ChatGPT. No download is required to audit the environment.

See `docs/module_reuse_and_novelty.md` before implementing borrowed modules.
