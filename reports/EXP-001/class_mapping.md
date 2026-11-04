# EXP-001 — Class mapping proposal (`weed-only` primary label space)

Status: **PROPOSAL — updated after the CornWeed machine audit (2026-11-04); pending
reviewer approval and manual annotation sampling.** CornWeed is downloaded and audited
(see `dataset_audit.md` / `dataset_audit.json`). Weed-crop V2 is NOT downloaded
(Mendeley `mthv4ppwyw`; download requires separate authorization).

## Verified source schemas (CornWeed, from the audit — not from README claims)

CornWeed (Zenodo 7961764, CC-BY-4.0; companion paper DOI 10.15439/2023f2125):

| Format | Category ID → name | Boxes in labeled set |
|--------|--------------------|----------------------|
| YOLO (`.txt`, zero-based) | `0` → Weeds, `1` → Maize | 257,740 Weeds, 23,985 Maize (total 281,725) |
| COCO (`train_2022.json` / `valid_2022.json`) | `1` → Weeds, `2` → Maize | identical counts per split |

- The YOLO↔COCO ID shift is a systematic **off-by-one**: YOLO `0/1` ↔ COCO `1/2`.
  Never mix the two ID spaces when converting.
- Verified consistency: COCO image sets are identical to the official YOLO lists
  (train 2862, valid 712; `list_minus_coco = coco_minus_list = 0`), and per-class box
  counts match exactly between the two formats.
- `data.yaml` ships `names=['Weeds','Maize']`; no third class appears anywhere in the
  label files (only IDs 0 and 1 occur), so the unused `crop-row` instance type mentioned
  in the record cannot leak into the mapping.
- Label integrity: missing labels, parse errors, out-of-bounds and degenerate boxes all 0.
- Weed-crop V2 (not yet present): 5 weed species + 8 crop types per its README; schema
  must be re-verified by audit before any mapping is trusted.

## Proposed primary mapping (single class)

| Source class | Maps to | Rationale |
|--------------|---------|-----------|
| CornWeed YOLO `0` Weeds | `weed` (id 0) | direct |
| Weed-crop V2 — all 5 weed species (once audited) | `weed` (id 0) | species merged; species-level labels retained in manifest for optional secondary analysis |
| CornWeed YOLO `1` Maize | *excluded from labels* (background) | crop-as-context per plan |
| Weed-crop V2 — all 8 crop classes | *excluded from labels* (background) | crop-as-context per plan |

## Preconditions status (checked against the machine audit)

1. **Annotation completeness — STILL OPEN (blocking).** Dropping maize boxes makes any
   *unannotated* weed a systematic false negative; label files cannot prove coverage.
   Requires a preregistered, hash-anchored manual sample of images (especially
   crop-box-rich scenes) before the `weed-only` conversion is approved. This is the
   main reason the audit status is CONDITIONAL.
2. **Illegal coordinates — RESOLVED.** Out-of-bounds / non-positive / degenerate boxes
   and parse errors are all 0; no clamp-vs-drop rule is needed for CornWeed.
3. **Class-token verification — RESOLVED.** YOLO files use IDs 0=Weeds, 1=Maize,
   consistent with `data.yaml`; COCO uses 1=Weeds, 2=Maize (off-by-one, see above).
4. **Crop-row labels — RESOLVED.** Only class IDs 0/1 occur in the 3,574 label files.
5. **Unlabeled CornWeed partition (4,981 imgs) — REMAINS EXCLUDED.** It also contains
   1,585 exact copies of labeled images (see `dataset_audit.md`), reinforcing exclusion
   from supervised training and any labeled metric; candidate only for a future,
   separately approved semi-supervised protocol.

## Secondary option (deferred)

A `crop`/`weed` 2-class variant may be added only if label completeness is verified in
BOTH datasets; cross-dataset fairness otherwise fails (Weed-crop V2 is not yet audited,
and CornWeed's maize boxes are dense small objects — 58.8% of all boxes have relative
area < 0.001). Default stays `weed-only`.

## Output artifacts

- `dataset_audit.json` — per-split class counts, bbox stats, hash manifest, duplicate
  and sequence-leakage evidence (generated 2026-11-04 by `tools/audit_cornweed.py`).
- Converted YOLO trees under `datasets_derived/` with `class_map` recorded in each
  `data.yaml`; originals untouched. **Not generated yet** — conversion waits for
  reviewer approval of the mapping and of the manual completeness sample.
