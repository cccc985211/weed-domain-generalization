# EXP-001 — Data split proposal (for approval)

Status: **PROPOSAL — blocked on dataset availability**. No dataset files are
present under `C:\Users\gass\Documents\experiment\datasets` as of 2026-11-02.
The rules below are pre-registered so the audit implements them unchanged.

## Leakage-control rules

1. **Grouping before splitting.** Images are grouped by, in priority order:
   (a) acquisition metadata (field/date/sequence/video frame id) when present in
   filenames or EXIF; (b) exact SHA-256 duplicates; (c) perceptual near-duplicates
   via 64-bit pHash (Hamming distance ≤ 6 → same group, reviewed).
   All members of a group go to the same split — no exceptions.
2. **Source split ratios (per dataset):** train 70% / val 15% / test 15% of
   *groups*, not images. Deterministic assignment:
   `split = sha256(group_id || "EXP001-v1") mod 100 → [0,70) train, [70,85) val, [85,100) test`.
3. **Freezing:** once manifests are generated, `manifest_sha256` per split is
   recorded in the report; test manifests are write-locked conceptually — any
   later modification requires a declared new protocol version.
4. **Held-out domains:** Weed-crop V2 is NEVER split for training under the
   primary protocol — it is an external test domain only. A full-file manifest
   (hashes only) is still built for integrity tracking. CottonWeedDet12 is
   optional additional held-out domain pending its own audit.

## If no grouping metadata exists

CornWeed documentation does not promise sequence metadata. Fallback:
- exact-hash dedup → pHash clustering (threshold above, union-find on pairwise
  distances within same-resolution partitions) → treat each cluster as a group
  → same deterministic hash assignment.
- Risk documented: if images are independent shots, pHash groups may be tiny —
  split then behaves like random split; residual leakage risk is reported, not
  hidden.

## Validation of the split

Report per split: image count, group count, class distribution, bbox-count,
mean pHash distance to nearest neighbor *across* splits (should be high),
and the count of near-duplicate pairs crossing split boundaries (target 0).

## Approval gate

Do not train on these splits until the reviewer approves the manifest hashes
recorded in `reports/EXP-001/dataset_audit.json`.
