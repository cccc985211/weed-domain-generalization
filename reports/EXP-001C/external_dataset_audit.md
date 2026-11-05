# EXP-001C — Weed-crop V2 external dataset audit

- Source: Mendeley Data [mthv4ppwyw v2](https://data.mendeley.com/datasets/mthv4ppwyw/2) · DOI 10.17632/mthv4ppwyw.2 · CC BY 4.0
- Files verified against publisher per-file SHA-256: 2248/2248 (listing preserved in `datasets/WeedCropV2/file_list_publisher.json`; download manifest in `datasets/WeedCropV2/download_manifest.json`, both outside git).
- The publisher's Download-All zip (sha256 `3f137d4f…2acee`, 7,775,732,980 B) is served from an S3 bucket that denies anonymous GETs (403 AccessDenied); files were fetched individually via the per-file endpoints the dataset page itself exposes.

## Classes

- Documented classes: 13 — `Blackbean, Canola, Corn, Field Pea, Flax, Horseweed, Kochia, Lentil, Ragweed, Redroot Pigweed, Soybean, Sugar beet, Waterhemp`
- classes.txt identical across the 8 scenario folders: True
- YOLO ids: `{'Blackbean': 0, 'Canola': 1, 'Corn': 2, 'Field Pea': 3, 'Flax': 4, 'Horseweed': 5, 'Kochia': 6, 'Lentil': 7, 'Ragweed': 8, 'Redroot Pigweed': 9, 'Soybean': 10, 'Sugar beet': 11, 'Waterhemp': 12}`
- Unclassified names: `none`

## Label integrity

- Labeled images: 1120; missing labels: 0; empty label files: 0
- Parse errors: 0; illegal/out-of-range boxes: 420; degenerate: 0
- Illegal boxes are edge-overflow annotations (center ± half-size crosses the image border by a small amount), not corruption; distribution: `{'Blackbean': 36, 'Ragweed': 10, 'Canola': 60, 'Waterhemp': 4, 'Horseweed': 3, 'Corn': 77, 'Field Pea': 57, 'Kochia': 5, 'Flax': 41, 'Lentil': 32, 'Redroot Pigweed': 3, 'Soybean': 64, 'Sugar beet': 28}`. Any derived copy must CLAMP coordinates to [0,1] (keep box) — never silently drop them.
- Image dimensions: 833 unique sizes (drone imagery); most common: `{'6960x4640': 264, '3991x4640': 3, '4373x4640': 2, '4877x4640': 2, '5474x4640': 2}`; size range `929x1741` … `6960x4640`
- Boxes total: 17693; by class: `{'Ragweed': 1086, 'Blackbean': 1970, 'Kochia': 851, 'Redroot Pigweed': 635, 'Waterhemp': 695, 'Horseweed': 407, 'Lentil': 1706, 'Soybean': 2877, 'Flax': 1745, 'Canola': 1804, 'Field Pea': 1384, 'Corn': 1476, 'Sugar beet': 1057}`

## Weed-only mapping (structure only — NO model runs were performed)

- Weed species → class 0: `{'Horseweed': 5, 'Kochia': 6, 'Ragweed': 8, 'Redroot Pigweed': 9, 'Waterhemp': 12}` (3674 boxes kept)
- Crop species → background (dropped in the derived copy): `{'Blackbean': 0, 'Canola': 1, 'Corn': 2, 'Field Pea': 3, 'Flax': 4, 'Lentil': 7, 'Soybean': 10, 'Sugar beet': 11}` (14019 boxes)
- Original labels are preserved untouched; any conversion must write a separate derived tree outside `datasets/`.
- This dataset is used to FREEZE the external evaluation protocol only. No model accuracy evaluation was run and none may be registered on it before the frozen protocol is approved.

## Outstanding checks

- Machine audit cannot establish annotation completeness of Weed-crop V2; a preregistered manual sample (same style as `annotation_review/`) is required before any weed-only TRAINING use.
- External evaluation protocol must declare which label space (13-class or weed-only) is scored and fix the class mapping in `configs/evaluation_protocol.yaml` before any test is registered.
- Full detail: `external_dataset_audit.json`.