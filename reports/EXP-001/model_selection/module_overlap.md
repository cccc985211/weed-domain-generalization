# Module A/B/C vs YOLO26 built-in recipe — overlap & feasibility analysis

YOLO26 recipe facts (Ultralytics docs, retrieved 2026-11-02):
dual-head training (one-to-many + one-to-one), MuSGD (SGD+Muon hybrid),
**Progressive Loss** (shifts supervision emphasis toward the inference-time
head over training), **STAL** (Small-Target-Aware Label Assignment — keeps
positive label coverage for small objects), DFL-free regression, standard
mosaic/scale/HSV augmentation with `close_mosaic` cutoff.

## A / OPBR (object-preserving background randomization)

| Question | Answer |
|---|---|
| Overlap with YOLO26 recipe? | **None mechanically.** The stock recipe has no background-targeted perturbation; it uses whole-image photometric/geometric aug + mosaic. |
| Closest prior art | OA-Mix (OA-DG, AAAI'24); DSD-YOLOv11 physics-based aug. A is an *adaptation*, not a fresh concept — must run vs an OA-Mix-style control. |
| Interaction risk | Must be applied *after* labels are finalized per-mosaic (or box-mask recomputed post-mosaic), else protected regions desync from transformed boxes. STAL unaffected (labels unchanged). |
| Verdict | **Orthogonal.** Safe to implement as a dataloader transform; no trainer change needed. |

## B / PSAS (progressive augmentation-strength scheduling)

| Question | Answer |
|---|---|
| Overlap with YOLO26 recipe? | **Partial, at concept level.** Progressive Loss schedules *loss* emphasis between heads — different axis from augmentation strength. BUT `close_mosaic` is already a coarse (binary) augmentation schedule in every Ultralytics run, and "progressive training" is now part of YOLO26's own story. |
| Closest prior art | Curriculum/annealed augmentation literature + `close_mosaic` itself. |
| Control required | Fixed-strength arm with *matched integrated intensity* (area-under-strength-curve), plus ablation with `close_mosaic` held identical. |
| Verdict | **Weakest of the three.** Only defensible as a declared-formula schedule that outperforms the matched-intensity fixed control; otherwise demote to a knob inside A. |

## C / SOFC (scale-aware ROI feature consistency)

| Question | Answer |
|---|---|
| Overlap with YOLO26 recipe? | **Motivation-level overlap with STAL, not mechanism.** STAL fixes *which* positions/queries supervise small objects (label assignment). C constrains *representations* of GT weed regions across photometric views (feature space). Orthogonal levers — but any gain of C must be shown **on top of STAL**, which is on by default. |
| Overlap with Progressive Loss? | None — it reweights head supervision over epochs; C is a per-batch auxiliary feature loss. |
| Minimal-change feasibility on YOLO26 | **Feasible without forking ultralytics — and easier than expected.** Smoke-test evidence (smoke_metrics.json): the standard model forward already returns pyramid neck features as `feats` (e.g. yolo26n eval forward → `{'one2many': {'boxes','scores','feats':[P3,P4,P5]}, 'one2one': {...}}`; yolo11n → `{'boxes','scores','feats':[P3,P4,P5]}`), all grad-connected in `_predict_once`. So C does NOT even need forward hooks: run the second (photometric-only) view through `model._predict_once`, read the feats entry, `torchvision.ops.roi_align` on GT weed boxes (identical coords across views), L2-norm + (1−cos) loss with scale-aware clipped weights, add to total loss — a `DetectionTrainer` subclass of ~150–250 LoC in `tools/`, zero edits to the ultralytics package. The DFL-free dual head is untouched (C attaches pre-head). Inference unchanged. Layer indices differ across models but the `feats` dict key does not — the implementation is model-agnostic. |
| Residual risks | 2× backbone compute per step → needs matched-compute control (e.g., E0 with doubled effective epochs is NOT the control; control = same wall-time budget or same FLOPs baseline); hook layer index differs between YOLO11 (C3k2/C2PSA neck) and YOLO26 — must be taken from the parsed model, not hardcoded; confirm gradient path (views share weights; optional stop-grad on one view). |
| Verdict | **Primary contribution candidate.** Implement behind a flag; prove additive to STAL. |

## Decision summary

- A: orthogonal to YOLO26 → implement, but benchmark against OA-Mix-style control.
- B: partially redundant conceptually → keep only if it beats matched-intensity fixed augmentation.
- C: complementary to STAL, attachable at neck features with a trainer subclass, no package fork → proceed to isolated smoke test in EXP-002 after baseline approval.

*No claim here uses target-domain metrics; all controls are source-side.*
