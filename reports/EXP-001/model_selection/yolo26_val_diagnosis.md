# Task 3 — YOLO26n validation anomaly: diagnosis

**Question:** why did the 2-epoch smoke test show `yolo26n` mAP50 ≈ 0.037 while
`yolo11n` ≈ 0.70 and `yolo26s` ≈ 0.72 on the same synthetic split?

## Checks performed (no new training — only evaluation of saved checkpoints)

- **Pretrained weights**: `weights/yolo26n.pt` SHA-256 `9b09cc8b…f4fef`,
  official ultralytics assets release v8.4.0. Trainer transferred 696/708
  tensors; remainder are nc-dependent head convs (COCO nc=80 → weed nc=1,
  `cls_remap=True` in args).
- **Class mapping**: `smoke_weed_v1/data.yaml` single class `weed`; trainer log
  `Overriding model.yaml nc=80 with nc=1` — mapping is correct.
- **Train config**: identical across the three models (see
  `smoke_metrics.json.common_train_args`); resolved optimizer AdamW(lr=0.002)
  for all.
- **Dual-head validation setting** — source-verified
  (`ultralytics/engine/validator.py:167`): validation uses
  `model.end2end = (args.nms is False)`, i.e. **default validation scores the
  one-to-many head with NMS**; the NMS-free one-to-one head is only used when
  `nms=False`.

## Measured result on `yolo26n/weights/best.pt` (synthetic val, 8 imgs)

| Validation path | P | R | mAP50 | mAP50:95 |
|---|---|---|---|---|
| default — **one2many + NMS** | 0.024 | 1.000 | 0.037 | 0.027 |
| `nms=False` — **one2one head, NMS-free** | 0.500 | 0.513 | **0.432** | 0.337 |
| yolo11n best.pt (single head) | 0.746 | 0.601 | 0.696 | 0.474 |

Also confirmed in `nn/modules/head.py:196`: during training the one-to-one head
consumes **detached** neck features (backbone is supervised by one2many only).

## Explanation

Not a defect — a head-selection artifact:

1. After 2 epochs, the one2many branch's class scores are still poorly
   calibrated for the new 1-class head → near-uniform high scores → NMS keeps
   ~33× too many boxes (P=0.024 at R=1.0, val conf=0.001).
2. The one2one branch emits a bounded top-`max_det` set *without* NMS, so its
   precision is far less sensitive to score miscalibration → mAP50 0.432.
3. The in-training epoch-end validation printed the one2many number, which
   undersold the deployed (NMS-free) path.
4. Everything is still synthetic-placeholder scale — **none of these numbers
   is an accuracy ranking**; they only prove both heads train and validate.

## Consequence for methodology

- EXP-002+ validation protocol must **declare which head is scored**. Report
  both `nms` settings for YOLO26 models, or fix `nms=False` if NMS-free
  inference is the deployment claim — decision belongs to the method section,
  not to a hidden default.
- Machine-readable evidence: `yolo26_val_diagnosis.json`.
