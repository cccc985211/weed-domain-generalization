# EXP-001 — Data split proposal (for approval)

Status: **PROPOSAL — updated after the CornWeed machine audit (2026-11-04); awaiting
reviewer approval. No split lists and no manifest hashes have been generated.**
Nothing may be trained on until the approved protocol's manifests are frozen
(`reports/EXP-001/dataset_audit.json` holds the audit evidence, not a manifest).

## What the audit established about the official split

The official CornWeed split **exists** (`train.txt` 2862 / `valid.txt` 712 /
`test.txt` 4981) but is **not usable for evaluation**:

- `test.txt` is commented out in the shipped `data.yaml` and its images carry no labels.
- The official train/valid partition is image-level random, **not group-safe**. Evidence
  (all numbers from `dataset_audit.json`, 2026-11-04):
  - **Exact SHA-256 duplicates across splits:** 1,585 duplicate groups; 1,059 pairs
    train↔test, 526 pairs val↔test. The "official test" set is partly the same pixels
    as the labeled data.
  - **pHash near-duplicate candidates** (64-bit, Hamming ≤6, equal dimensions):
    val↔train 2, train↔test 1,618, val↔test 526 (candidates, require review).
  - **Capture-stream leakage:** filenames parse into 4 `site_date_platform` streams;
    3 of them span both train and valid, i.e. the official partition ignored acquisition
    sequences.
  - **Adjacent frames across splits** (frame-number delta ≤1 / ≤5 / ≤10):
    train↔val only 6 / 70 / 240; train↔unlabeled-test 3,175 / 11,623 / 22,155;
    val↔unlabeled-test 1,579 / 5,791 / 11,029. The dominant adjacency leakage is into
    the unlabeled test directory.
- Per-stream frames are **one unbroken run each** (no gap >10 breakpoint in any of the
  4 streams), so there is no natural intra-stream cut to subdivide on.

**Consequence:** keep the official lists as immutable provenance only; do not report
any DG/accuracy claim on them. The unlabeled test directory is excluded from supervised
training and evaluation unless a separately approved semi-supervised protocol is
registered.

## Grouping metadata that exists now (supersedes the old "no metadata" fallback)

Filenames encode `site_YYMMDD_[platform_]color_frame` (empty platform = handheld).
Labeled data = 3 streams:

| Stream | labeled imgs (train/val) | platform | date |
|---|---|---|---|
| `Langsenkamp_210722_bonirob` | 882 / 172 | bonirob robot | 2021-07-22 |
| `Langsenkamp_210727_bonirob` | 500 / 171 | bonirob robot | 2021-07-27 |
| `Langsenkamp_210727_handheld` | 1,480 / 369 | handheld | 2021-07-27 |

(`Langsenkamp_210802_bonirob` occurs only among unlabeled test images.) The earlier
fallback plan — "if no grouping metadata exists, rely on exact-hash + pHash clustering"
— is no longer the primary route, since real sequence metadata exists; pHash/exact-hash
grouping remains a safety overlay on top of streams.

## Protocol options for the labeled source data (choose ONE, then freeze)

**Option A — leave-one-stream-out (LOSO) — recommended.**
3 folds; each fold holds out one whole stream as validation and trains on the other
two. Report mean ± std over the 3 folds. No 15% "test" split is formed inside the
source domain; the external-domain claim is carried by Weed-crop V2 (pending
authorization and audit) or another third domain.
Limitations (coarse grouping, limited independence): only 3 folds → very high variance;
date and platform co-vary across streams, so a fold measures a compound domain shift
(day + platform + scene), not a controlled factor.

**Option B — fixed-window blocks with buffer zones.**
If an in-source train/val/test ratio (e.g. 70/15/15) is required: cut each labeled
stream into fixed-size frame windows (size preregistered, e.g. 50 frames), drop a
buffer zone between adjacent windows (e.g. ≥10 frames assigned to neither split),
merge any reviewed pHash near-duplicates into the same window, then assign whole
windows to splits with the deterministic hash rule below.
Limitations: windows are artificial groups — same-stream blocks share field, weather,
illumination, growth stage and weed density, so independence is much weaker than
between-stream; the buffer suppresses adjacent-frame leakage but cannot prove block
independence.

Both options inherit the same ceiling: 3 labeled streams from a single site on 3 dates
cannot demonstrate broad single-source generalization. Treat whichever source-internal
number they produce as an optimistic auxiliary estimate, never as the DG result.

## Leakage-control rules (unchanged where still applicable)

1. **Grouping before splitting.** Priority order: (a) acquisition stream from filename
   (`site_date_platform`); (b) exact SHA-256 duplicates — always same group as their
   original; (c) reviewed pHash near-duplicates (64-bit, Hamming ≤6, equal dimensions)
   — same group. All members of a group go to the same split — no exceptions.
2. **Ratios (Option B only):** 70% train / 15% val / 15% test of *groups*, not images,
   via `split = sha256(group_id || "EXP001-v1") mod 100 → [0,70) / [70,85) / [85,100)`.
   Option A has no hash assignment — folds are defined by the held-out stream.
3. **Freezing:** once manifests are generated, `manifest_sha256` per split is recorded
   in the report; any later modification requires a declared new protocol version.
4. **Held-out domains:** Weed-crop V2 is NEVER split for training under the primary
   protocol — external test domain only (full-file manifest, hashes only, for integrity).
   CottonWeedDet12 stays optional pending its own audit and authorization.

## Validation of whichever split is approved

Report per split/fold: image count, group count, class distribution (YOLO ID space:
0=Weeds, 1=Maize; COCO JSONs use the off-by-one 1/2), bbox-count stats, and the count
of exact/pHash duplicate pairs crossing split boundaries (target 0).

## Approval gate

The reviewer must approve (a) the choice between Option A and B, and (b) the frozen
manifest hashes, before any split list is written to `datasets_derived/` or any
training run starts. Until then this proposal generates no files.
