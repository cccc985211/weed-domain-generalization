# EXP-001C — Manual annotation-completeness review package (CornWeed)

Status: **PENDING HUMAN REVIEW — this audit is NOT a PASS until the review form
below has been filled in and confirmed by a human reviewer.**

## Purpose

The `weed-only` conversion drops all maize (YOLO class 1) boxes. Any *unannotated*
weed inside a dropped-crop scene then becomes a systematic false negative, which
label files alone cannot reveal. This package freezes a preregistered 90-image
sample so a human can estimate annotation completeness before `weed-only` training
data is approved.

## Preregistered sampling protocol

- Population: all 3,574 labeled CornWeed images, grouped into the 3 acquisition
  streams (`Langsenkamp_210722_bonirob`, `Langsenkamp_210727_bonirob`,
  `Langsenkamp_210727_handheld`).
- Per stream: **30 images = 20 random (fixed seed 42) + 10 hard.**
- Hard score: `z(n_weed) + z(n_maize) + z(frac_small)`, where `frac_small` is the
  fraction of boxes with relative area < 0.001; top-10 per stream, ties broken by
  filename. Rationale: maize-box-dense, small-object-dense scenes are where a
  dropped-crop background is most likely to hide unannotated weeds.
- Selection is frozen in `sample_selection.csv` (SHA-256 in `sample_hashes.json`;
  generator: `tools/make_annotation_sample.py`). Re-running the generator must
  reproduce identical hashes.

## Files

| File | What it is |
|------|------------|
| `sample_selection.csv` | frozen 90-image list with per-image stats and SHA-256 |
| `review_template.csv` | **empty review form** — copy to `review_form.csv`, fill in |
| `contact_sheets/<stream>_{random20,hard10}.jpg` | in-git overview grids (thumbnails) |
| `runs/EXP-001C/annotation_panels/<stream>/<image>.jpg` | full-resolution review panels with original label boxes (outside git; generated locally by the tool) |

Each panel shows the original YOLO boxes over a **copy** of the image — original
dataset files are never modified. Legend: RED = `Weeds(0)`, BLUE = `Maize(1)`.

## Review procedure

1. Open `runs/EXP-001C/annotation_panels/<stream>/` (or generate panels first:
   `python tools/make_annotation_sample.py`).
2. For each of the 90 images, compare visible weeds against the red boxes.
3. Record one `issue_category` per image in `review_form.csv`:
   - `missing_weed_label` — a visible weed with no red box
   - `wrong_class` — a weed labeled as maize or vice versa
   - `box_localization` — box exists but is badly placed/sized
   - `duplicate_image` — near-identical image already sampled
   - `none` — labels appear complete
   - `other` — anything else, described in `notes`
4. Fill `notes`, `reviewer`, `review_date` for every row; no empty rows.
5. Ship the filled `review_form.csv` back into this directory for the auditor.

## Decision rule (preregistered, subject to reviewer confirmation)

- Report: count of images with ≥1 `missing_weed_label`, per stream and per
  sample_type, plus total missed-weed count observed.
- Suggested gate (reviewer may tighten): if fewer than 10% of the 90 images show
  any missing weed label, the `weed-only` mapping may proceed; otherwise the
  mapping must be revisited (e.g. restrict to scenes with verified coverage).
- Regardless of the outcome, the audit status stays **PENDING** until a named
  human reviewer has confirmed `review_form.csv`.
