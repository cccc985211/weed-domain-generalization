# Research protocol (pre-registered proposal; not results)

## Problem
Single-source domain generalization for field weed detection (pretrained YOLO detector, one GPU) without using target-domain data in training or validation. Primary annotation space: `weed` (one class). Crops become context; verify annotations are sufficiently complete.

## Model selection (updated 2026-11-02)
- **Primary research candidate: YOLO26n** (`yolo26n.pt`) — current-generation edge-oriented detector; DFL-free head, dual-head (one-to-many + one-to-one) training, MuSGD optimizer, Progressive Loss, STAL small-target label assignment. All three modules A/B/C must be re-validated for overlap against YOLO26's built-in training recipe before implementation.
- **Historical baseline: YOLO11n** (`yolo11n.pt`) — reference point linking this work to the prior YOLO11 literature; modules were originally scoped against it.
- **Stronger baseline: YOLO26s** (`yolo26s.pt`) — larger-capacity control to separate "more parameters" effects from module effects.
- Selection of the primary model is decided by the EXP-001 model-selection smoke test (identical split, `imgsz=640`, same seed, aligned training config, ≤2 epochs) and is a *trainability/compatibility* decision only — 2-epoch metrics are not an accuracy ranking.

## Three modules
- **A / OPBR:** box-protected background perturbations. Union of GT boxes + margin defines protected region; optionally smooth boundary. Only background gets stronger photometric perturbation; crop targets are not mistakenly treated as weed object masks. Preserve coordinate geometry. Include baseline photometric augmentation comparison; avoid label corruption.
- **B / PSAS:** progressive scheduling of perturbation strength (e.g., moderate → stronger → moderate) with a declared formula and fixed stage boundaries. When A off, B operates on whole-image standard photometric augmentation; otherwise it adjusts A's background augmentation strength. Compare to a fixed-strength control with comparable average magnitude.
- **C / SOFC:** scale-aware instance feature consistency. Two views have identical geometric coordinates (photometric changes only). Extract ROI-aggregated features for GT weed boxes at an appropriate backbone/neck level, L2-normalize vectors and minimize 1-cosine similarity between views. One branch may be detached; define how the views share model weights. Foreground small-weed weights clipped to prevent gradient explosion. Log ROI counts and auxiliary loss scale; control for increased training cost. At inference use only the ordinary single-graph forward (no auxiliary branches). For YOLO26, check overlap with Progressive Loss and STAL (see EXP-001 model-selection analysis) and prefer attaching to backbone/neck features with minimal trainer changes.

## Experiment matrix
| ID | A | B | C |
|---|---:|---:|---:|
| E0 | 0 | 0 | 0 |
| E1 | 1 | 0 | 0 |
| E2 | 0 | 1 | 0 |
| E3 | 0 | 0 | 1 |
| E4 | 1 | 1 | 0 |
| E5 | 1 | 0 | 1 |
| E6 | 0 | 1 | 1 |
| E7 | 1 | 1 | 1 |

Also compare standard augmentations, constant-strength perturbation, and a non-scale-weighted C; keep base training hyperparameters as equal as practical.

## Validation & external evaluation
- First, establish label equivalence and annotation coverage; CornWeed has maize+weed, Weed-crop V2 has eight crops+five weed species.
- Source train/val/test split must be independent of near-duplicate frames and collection groups if metadata available.
- Tune on source val only. Pre-register frozen external test datasets; do not test-target-tune modules.
- If source val cannot represent shifts, consider a declared source-only stress-test validation set, while keeping actual external datasets held out.
- For small-object analysis, define size as area ratio or COCO-convention pixels *after resizing*, and declare this before running tests.
- Report per-domain mAP50, mAP50:95, weed recall, params, inference batch-1 latency/FPS, GPU training hours, mean ± std of ≥3 seeds on selected models, failure cases and robustness against no-extra-module baseline.
- Do not imply domain generalization from a single target site; add a third external data domain if audit and time permit.

## Novelty caution
OA-DG (AAAI 2024) already uses object-aware mixing & loss, DSD-YOLOv11 (2026) uses agricultural physics-based augmentation, and several 2026 weed works use strict cross-dataset evaluation. Cite and compare them. Additionally, YOLO26's built-in training recipe (MuSGD, Progressive Loss, STAL) may partially subsume the motivation of B (scheduling) and C (small-target supervision); the EXP-001 model-selection report must state the residual, non-overlapping contribution of each module. The proposed trio is a *research hypothesis*, not a guaranteed independent novelty claim.
