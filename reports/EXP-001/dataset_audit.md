# EXP-001B — CornWeed dataset audit

- Audit timestamp: `2026-11-04T15:16:46.523753+00:00`
- Status: **CONDITIONAL** — archive, structure, labels and official splits audited; visual annotation-completeness review and Weed-crop V2 audit remain outstanding.
- Source: [Zenodo record 7961764](https://zenodo.org/records/7961764)
- Archive MD5: `9b409e835b51c0f7b01524dcb1dfeda6` (expected `9b409e835b51c0f7b01524dcb1dfeda6`; match=True)
- License/citation: CC-BY-4.0; Iqbal et al., CornWeed Dataset: A dataset for training maize and weed object detectors for agricultural machines, Zenodo 2023, DOI 10.5281/zenodo.7961764; companion paper DOI 10.15439/2023f2125.

## Inventory

- Labeled image directory: 3574; unlabeled test images: 4981; total: 8555.
- YOLO label files: 3574; boxes: 281725; classes by zero-indexed ID: `{'1': 23985, '0': 257740}`.
- Dimensions: `{'640x480': 4216, '1280x720': 4339}`.
- Official lists: train=2862, valid=712, test=4981. Test images are explicitly unlabeled; `test.txt` is commented out in the shipped `data.yaml`.
- COCO official JSON: train=2862 images/221627 boxes; valid=712 images/60098 boxes.
- YOLO labels: missing=0; parse errors=0; illegal/out-of-bounds boxes=0; degenerate=0.

## Annotation and format checks

- YOLO normalized box area fraction: `{'min': 1.3261354e-05, 'p01': 0.00010295088188, 'median': 0.000806073836, 'p99': 0.08192105882700006, 'max': 0.35615244089599996, 'frac_lt_0_001': 0.5884532789067353}`.
- COCO vs YOLO split image-set comparison: `{'train': {'yolo_list_images': 2862, 'coco_images': 2862, 'same_image_set': True, 'list_minus_coco': 0, 'coco_minus_list': 0}, 'val': {'yolo_list_images': 712, 'coco_images': 712, 'same_image_set': True, 'list_minus_coco': 0, 'coco_minus_list': 0}}`; class box counts: `{'train': {'yolo_counts': {'1': 19021, '0': 202606}, 'coco_counts': {'Weeds': 202606, 'Maize': 19021}}, 'val': {'yolo_counts': {'0': 55134, '1': 4964}, 'coco_counts': {'Weeds': 55134, 'Maize': 4964}}}`. COCO uses category IDs 1/2; YOLO uses zero-based IDs 0/1.
- `weed-only` mapping: map YOLO class 0 `Weeds` → class 0 `weed`; drop class 1 `Maize` boxes so maize becomes background. Mapping is structurally valid, but annotation coverage must be visually assessed because dropped crop boxes cannot prove all weeds were annotated.

## Duplicate and capture-sequence leakage

- Exact SHA-256 duplicate groups: 1585; cross-split exact pair counts: `{'train_test': 1059, 'val_test': 526}`.
- Cross-split pHash candidates (64-bit, Hamming ≤6, equal image dimensions): `{'val_train': 2, 'train_test': 1618, 'val_test': 526}`. These are candidates and require review; examples are in JSON.
- Filename sequence pattern parsed: `site_YYMMDD_[platform_]color_frame (empty platform=handheld)`; 4 site/date/platform streams; 3 streams span more than one official split.
- Cross-split adjacent-frame candidates (frame number delta ≤1 / ≤5 / ≤10): total 4760 / 17484 / 33424. Train–val only: 6 / 70 / 240; the remainder pairs the labeled splits with the unlabeled `testing/` set.
- Every acquisition stream is a single unbroken frame run (`n_frame_runs_gap_gt_10 == 1` for all 4 streams; per-stream frame ranges in JSON). There are no natural frame-gap breakpoints inside a stream to subdivide on.
- The official train/valid partition is image-level and **not group-safe**: exact duplicate images also occur in the unlabeled `testing/` directory (see exact pair counts); adjacent frames and pHash candidates cross splits. Do not treat its validation score as a leakage-controlled DG estimate.

## Split recommendation

- Preserve the official train/valid/test lists as immutable provenance. The shipped test partition is unlabeled and contains exact copies of labeled frames, so exclude it from any held-out evaluation claim.
- **Deprecated option (do not use):** "group by stream, then merge adjacent frame runs (gap ≤10)". The audit shows each stream is already one unbroken frame run, so this procedure yields exactly one group per stream and cannot subdivide anything.
- **Option A — leave-one-stream-out (LOSO), preferred for labeled data.** Only 3 labeled streams exist (`Langsenkamp_210722_bonirob`, `Langsenkamp_210727_bonirob`, `Langsenkamp_210727_handheld`; the 4th stream exists only in unlabeled test images). Run 3 folds, each holding out one stream as validation and training on the other two; report mean ± std across folds. Limitations: only 3 folds (very coarse, high variance), and date/platform co-vary across the streams, so the resulting "domain" mixes acquisition day, platform and scene rather than isolating one factor.
- **Option B — fixed-window blocks with buffer zones.** Because there are no natural gaps inside a stream, cut each labeled stream into fixed-size frame windows (block size and buffer to be preregistered, e.g. 50-frame blocks with a ≥10-frame buffer dropped between blocks), merge reviewed pHash near-duplicate candidates into the same window, and assign windows to 70/15/15 with the preregistered deterministic hash rule. Limitations: windows are artificial groups, not separate acquisitions — blocks from the same stream still share field, illumination, growth stage and weed density, so independence is much weaker than between-stream; the buffer suppresses adjacent-frame leakage but does not prove block independence.
- Both options are coarse: 3 labeled streams from one site (Langsenkamp) on 3 dates bound what any source-internal split can demonstrate about generalization. A third, external domain (e.g. Weed-crop V2, pending authorization) must carry the DG claim.
- No new split manifests are generated now. The protocol choice and its manifest hashes require reviewer approval before any split list is produced or training starts.
- Keep unlabeled test images out of supervised training/evaluation; use only if a separately approved semi-supervised protocol is later registered.

## Limitations / outstanding checks

- No pixels were published. The machine audit does not establish weed-label completeness or weed omissions where maize boxes are ignored; manually inspect a preregistered, hashed sample before approving `weed-only` conversion.
- Weed-crop V2 is not present/audited; cross-dataset compatibility remains pending.
- See `dataset_audit.json` for full manifests, class counts, exact duplicates and capped pHash/adjacency examples.
