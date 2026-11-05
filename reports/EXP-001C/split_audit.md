# EXP-001C — CornWeed LOSO split audit

- Generated: 2026-11-05 · protocol `EXP001C-LOSO-v1` (approved as **Option A** in the
  EXP-001B review)
- Generator: `tools/make_loso_split.py` (read-only over the dataset; rerunning it must
  reproduce every hash below byte-for-byte)
- Frozen artifacts:
  - `reports/EXP-001C/split_manifest.csv` — **manifest_sha256
    `54dc46a34e44318f99781e51f9bc2950266111b54bf0d31c0fa3e0af9fe22979`**
  - `reports/EXP-001C/duplicates_conflicts.csv` — sha256
    `e074fd40a1ab047bd516fe83f4cfaca4dc252fc9b80c3c41a64662f23f8ad1ad`
  - `reports/EXP-001C/split_stats.json` — full statistics
  - `configs/loso_protocol.yaml` — preregistered rules (mirror of what the tool implements)

## Fold definition

| Fold | Test domain (held out entirely) | Internal streams (train + val) |
|------|--------------------------------|-------------------------------|
| fold1 | `Langsenkamp_210722_bonirob` | `210727_bonirob`, `210727_handheld` |
| fold2 | `Langsenkamp_210727_bonirob` | `210722_bonirob`, `210727_handheld` |
| fold3 | `Langsenkamp_210727_handheld` | `210722_bonirob`, `210727_bonirob` |

The test stream is never used for hyperparameters, epochs, early stopping, or weight
selection in its own fold. The unlabeled `testing/` directory is not part of any split.

## Train/val construction inside the internal streams (time-continuous)

- Frames are grouped into **50-frame blocks taken every 60 frames**; the 10 frames
  between blocks form a **buffer zone** that enters neither train nor val (`role=buffer`).
  Tail frames beyond the last full period are buffer as well.
- Each block is assigned by
  `sha256(f"{stream}|{block_index}|EXP001C-LOSO-v1") mod 100`: `[0,80)` train,
  `[80,100)` val — i.e. 80/20 of *blocks*, deterministic and preregistered.
- Because handheld frames are sparser than bonirob frames, val ends up at ~10–12% of
  active internal images; the buffer accounts for ~1/6 of the frame range.

## Frozen split statistics (active images; buffer/duplicates excluded)

| Fold | train | val | test | buffer | weed boxes train/val/test | maize boxes train/val/test |
|------|-------|-----|------|--------|---------------------------|----------------------------|
| fold1 | 1817 | 250 | 1054 | 453 | 152,015 / 16,409 / 51,579 | 14,508 / 1,796 / 4,106 |
| fold2 | 2153 | 200 | 671 | 550 | 157,208 / 14,659 / 45,573 | 15,644 / 1,528 / 2,868 |
| fold3 | 1250 | 150 | 1849 | 325 | 71,609 / 8,066 / 160,588 | 5,042 / 628 / 17,011 |

Per-fold image-list hashes (sorted, newline-joined, sha256) are recorded in
`split_stats.json` (`train_list_sha256`, `val_list_sha256`, `test_list_sha256`).

## Duplicate checks (within each fold; SHA-256 + pHash)

- Exact SHA-256 duplicate groups and pHash clusters (64-bit, Hamming ≤6, equal
  dimensions, union-find) were recomputed over all 3,574 labeled images.
- **Result: 0 cross-role conflicts.** No exact duplicate and no pHash-cluster pair
  lands in two different active roles (test/train/val) of the same fold. The
  `duplicates_conflicts.csv` is therefore empty by construction — it exists so any
  future re-split with changed roles must re-run and populate it.
- 62 / 67 / 5 buffer-zone images per fold overlap a pHash cluster that also contains
  train images (`buffer_images_with_phash_overlap` in `split_stats.json`). These are
  consecutive video frames that look alike; they are already excluded by the buffer
  zone, so no action is taken, but the count is reported for transparency.
- pHash clustering on near-static video frames is deliberately conservative: it links
  frames that are visually similar without being duplicates. Any future conflict found
  by a changed assignment would be machine-flagged `pending_human_review` before use.

## Provenance & integrity

- Original images and labels were not modified; the tool only reads them and writes
  reports. The official `train.txt`/`valid.txt` membership is recorded per image in the
  manifest (`official_split` column) as immutable provenance only.
- The manifest records, per image and per fold: role, stream, frame, official split,
  SHA-256, 64-bit pHash (hex), and action (`active` / `buffer_zone_excluded` /
  `excluded_duplicate`).

## Residual risks

1. **Three folds only.** Any LOSO summary (mean ± std over 3 numbers) is coarse; treat
   fold-level numbers as descriptive, not inferential.
2. **Date/platform confound.** fold1 vs fold2 vs fold3 differ in acquisition day and
   platform simultaneously; LOSO measures a compound shift, not an isolated factor.
3. **Same field across streams.** 210722 and 210727 bonirob ran in the same experiment
   (possibly the same parcel two days apart and different weather); stream-level
   separation cannot remove within-field similarity.
4. **Manual annotation completeness is still PENDING** (90-image review package in
   `annotation_review/`); until it is confirmed, the `weed-only` conversion must not be
   used to produce final training data.
5. Buffer-zone parameters (50/10) were chosen by us, not by the data; they are
   preregistered in `configs/loso_protocol.yaml` and must not be tuned after seeing
   results.
