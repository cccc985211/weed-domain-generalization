"""Read-only CornWeed audit for EXP-001B.

Audits the official Zenodo archive extraction without changing source files.
Writes a machine-readable JSON and a concise Markdown report.
"""
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import imagehash

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(REPO)  # experiment root, one level above the repo
ROOT = os.path.join(WORKSPACE, "datasets", "CornWeed", "maize_dataset")
REPORT_DIR = os.path.join(REPO, "reports", "EXP-001")
JSON_OUT = os.path.join(REPORT_DIR, "dataset_audit.json")
MD_OUT = os.path.join(REPORT_DIR, "dataset_audit.md")
EXPECTED_MD5 = "9b409e835b51c0f7b01524dcb1dfeda6"
ARCHIVE = os.path.join(WORKSPACE, "datasets", "maize_dataset.zip")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def read_list(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip().replace("\\", "/").removeprefix("./")
                for line in f if line.strip()]


def parse_yolo(path):
    out = []
    errors = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            p = line.split()
            if not p:
                continue
            if len(p) != 5:
                errors.append({"line": ln, "reason": f"field_count={len(p)}"})
                continue
            try:
                cls = int(p[0])
                x, y, w, h = map(float, p[1:])
            except ValueError:
                errors.append({"line": ln, "reason": "non_numeric"})
                continue
            out.append((cls, x, y, w, h, ln))
    return out, errors


def sequence_info(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(
        r"^(?P<site>.+?)_(?P<date>\d{6})_(?P<platform>.*?)color_(?P<frame>\d+)$",
        stem)
    if not m:
        return None
    d = m.groupdict()
    platform = d["platform"].rstrip("_") or "handheld"
    return {"stream": f"{d['site']}_{d['date']}_{platform}",
            "date": d["date"], "platform": platform,
            "frame": int(d["frame"])}


def main():
    t0 = time.time()
    data_dir = os.path.join(ROOT, "data")
    test_dir = os.path.join(ROOT, "testing")
    labeled_images = sorted(
        [os.path.join(data_dir, x) for x in os.listdir(data_dir)
         if x.lower().endswith((".png", ".jpg", ".jpeg"))])
    test_images = sorted(
        [os.path.join(test_dir, x) for x in os.listdir(test_dir)
         if x.lower().endswith((".png", ".jpg", ".jpeg"))])
    train_list = read_list(os.path.join(ROOT, "train.txt"))
    val_list = read_list(os.path.join(ROOT, "valid.txt"))
    test_list = read_list(os.path.join(ROOT, "test.txt"))
    split_lists = {"train": train_list, "val": val_list, "test": test_list}
    split_by_path = {}
    for split, names in split_lists.items():
        for name in names:
            split_by_path[name.replace("\\", "/")] = split

    labels_by_stem = {
        os.path.splitext(f)[0]: os.path.join(data_dir, f)
        for f in os.listdir(data_dir) if f.lower().endswith(".txt")
    }
    class_counts = Counter()
    label_summary = Counter()
    bbox_areas = []
    invalid_norm = []
    degenerate = []
    parse_errors = []
    image_dims = Counter()
    img_records = []
    missing_label = []
    unlabeled_test = []
    image_hashes = defaultdict(list)
    phashes = []
    split_counts = Counter()

    all_images = [(p, "labeled") for p in labeled_images] + [
        (p, "unlabeled_test") for p in test_images]
    for i, (path, role) in enumerate(all_images, 1):
        name = os.path.basename(path)
        relpath = os.path.relpath(path, ROOT).replace("\\", "/")
        split = split_by_path.get(relpath, role)
        split_counts[split] += 1
        with Image.open(path) as im:
            w, h = im.size
            image_dims[f"{w}x{h}"] += 1
            ph = int(str(imagehash.phash(im, hash_size=8)), 16)
        digest = sha256(path)
        image_hashes[digest].append(relpath)
        phashes.append((relpath, split, w, h, ph))
        rec = {"name": name, "relative_path": relpath, "split": split, "role": role,
               "width": w, "height": h, "sha256": digest,
               "sequence": sequence_info(name)}
        img_records.append(rec)

        if role == "labeled":
            lp = labels_by_stem.get(os.path.splitext(name)[0])
            if not lp:
                missing_label.append(name)
            else:
                rows, errs = parse_yolo(lp)
                parse_errors.extend([{"file": name, **e} for e in errs])
                label_summary["files_with_labels"] += bool(rows)
                label_summary["empty_label_files"] += not bool(rows)
                for cls, x, y, bw, bh, ln in rows:
                    class_counts[cls] += 1
                    bbox_areas.append(bw * bh)
                    if bw <= 0 or bh <= 0:
                        degenerate.append({"file": name, "line": ln,
                                           "xywh": [x, y, bw, bh]})
                    if (x < 0 or x > 1 or y < 0 or y > 1 or
                            bw < 0 or bw > 1 or bh < 0 or bh > 1 or
                            x - bw / 2 < -1e-6 or x + bw / 2 > 1 + 1e-6 or
                            y - bh / 2 < -1e-6 or y + bh / 2 > 1 + 1e-6):
                        invalid_norm.append({"file": name, "line": ln,
                                             "class_id": cls,
                                             "xywh": [x, y, bw, bh]})
        if i % 1000 == 0:
            print(f"hashed/audited {i}/{len(all_images)} images", flush=True)

    exact_dupes = [v for v in image_hashes.values() if len(v) > 1]
    split_by_rel = {r["relative_path"]: r["split"] for r in img_records}
    exact_cross_counts = Counter()
    exact_cross_examples = defaultdict(list)
    for group in exact_dupes:
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                sa, sb = split_by_rel[a], split_by_rel[b]
                if sa != sb:
                    key = f"{sa}_{sb}"
                    exact_cross_counts[key] += 1
                    if len(exact_cross_examples[key]) < 100:
                        exact_cross_examples[key].append([a, b])

    # COCO cross-check: images and boxes from official train/valid JSON.
    coco_summary = {}
    coco_annotations = []
    coco_splits = {}
    for split, fname in (("train", "train_2022.json"),
                         ("val", "valid_2022.json")):
        cp = os.path.join(ROOT, "coco_annotations", fname)
        with open(cp, encoding="utf-8") as f:
            d = json.load(f)
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        anns = d.get("annotations", [])
        cc = Counter(cats.get(a["category_id"], f"UNKNOWN_{a['category_id']}")
                     for a in anns)
        coco_splits[split] = {os.path.basename(x["file_name"])
                              for x in d.get("images", [])}
        coco_summary[split] = {
            "images": len(d.get("images", [])),
            "annotations": len(anns),
            "categories": cats,
            "annotation_counts_by_class": dict(cc),
            "image_dimensions": dict(Counter(
                f"{x['width']}x{x['height']}" for x in d.get("images", []))),
            "licenses": d.get("licenses", []),
        }
        coco_annotations.extend((split, a, cats) for a in anns)
    coco_cross_split = sorted(coco_splits["train"] & coco_splits["val"])
    coco_missing_images = {
        s: sorted(names - {os.path.basename(x) for x in labeled_images})[:30]
        for s, names in coco_splits.items()
    }

    # Verify official lists vs COCO sets and detect capture streams crossing splits.
    list_sets = {s: set(names) for s, names in split_lists.items()}
    list_basenames = {s: {os.path.basename(n) for n in names}
                      for s, names in split_lists.items()}
    list_overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        list_overlaps[f"{a}_{b}"] = sorted(list_sets[a] & list_sets[b])
    basename_overlaps = {}
    for a, b in (("train", "test"), ("val", "test")):
        basename_overlaps[f"{a}_{b}"] = sorted(
            list_basenames[a] & list_basenames[b])
    seq_splits = defaultdict(Counter)
    stream_frames = defaultdict(lambda: defaultdict(list))
    sequence_unparsed = []
    for r in img_records:
        si = r["sequence"]
        if not si:
            sequence_unparsed.append(r["name"])
            continue
        seq_splits[si["stream"]][r["split"]] += 1
        stream_frames[si["stream"]][r["split"]].append((si["frame"], r["name"]))
    crossing_streams = {
        k: dict(v) for k, v in seq_splits.items()
        if len([s for s, n in v.items() if n]) > 1
    }
    adjacent_cross = {1: [], 5: [], 10: []}
    adjacent_by_split_pair = {1: Counter(), 5: Counter(), 10: Counter()}
    for stream, by_split in stream_frames.items():
        events = sorted((frame, split, name)
                        for split, rows in by_split.items()
                        for frame, name in rows)
        for i, (fa, sa, na) in enumerate(events):
            j = i + 1
            while j < len(events) and events[j][0] - fa <= 10:
                fb, sb, nb = events[j]
                if sa != sb:
                    delta = fb - fa
                    pair = {"stream": stream, "split_a": sa, "name_a": na,
                            "split_b": sb, "name_b": nb, "frame_delta": delta}
                    key = "-".join(sorted((sa, sb)))
                    for threshold in adjacent_by_split_pair:
                        if delta <= threshold:
                            adjacent_by_split_pair[threshold][key] += 1
                    for threshold in adjacent_cross:
                        if delta <= threshold and len(adjacent_cross[threshold]) < 50000:
                            adjacent_cross[threshold].append(pair)
                j += 1
    acquisition_runs = {}
    for stream, by_split in stream_frames.items():
        all_frames = sorted(frame for rows in by_split.values()
                            for frame, _ in rows)
        unique_frames = sorted(set(all_frames))
        runs = []
        current = []
        for frame in unique_frames:
            if current and frame - current[-1] > 10:
                runs.append(current)
                current = []
            current.append(frame)
        if current:
            runs.append(current)
        acquisition_runs[stream] = {
            "n_frame_runs_gap_gt_10": len(runs),
            "run_frame_ranges": [[r[0], r[-1]] for r in runs[:200]],
        }

    # pHash near-duplicate candidates, Hamming <=6, only within equal dimensions.
    near_pairs = []
    by_dim = defaultdict(list)
    for rec in phashes:
        by_dim[(rec[2], rec[3])].append(rec)
    bit_lut = np.unpackbits(np.arange(256, dtype=np.uint8)).reshape(256, 8).sum(1)
    pair_count_by_split = Counter()
    for dim, rows in by_dim.items():
        vals = np.array([np.uint64(r[4]) for r in rows])
        bsz = 256
        for i0 in range(0, len(rows), bsz):
            for j0 in range(i0, len(rows), bsz):
                aa, bb = vals[i0:i0+bsz, None], vals[None, j0:j0+bsz]
                xx = (aa ^ bb).astype(np.uint64).view(np.uint8).reshape(
                    len(aa), -1, 8)
                dist = bit_lut[xx].sum(-1)
                ii, jj = np.where(dist <= 6)
                for x, y in zip(ii.tolist(), jj.tolist()):
                    ai, bj = i0 + x, j0 + y
                    if ai >= bj:
                        continue
                    ra, rb = rows[ai], rows[bj]
                    if ra[1] == rb[1]:
                        continue
                    pair_count_by_split[f"{ra[1]}_{rb[1]}"] += 1
                    if len(near_pairs) < 1000:
                        near_pairs.append({"image_a": ra[0], "split_a": ra[1],
                                           "image_b": rb[0], "split_b": rb[1],
                                           "hamming": int(dist[x, y]),
                                           "dimensions": list(dim)})

    areas = np.asarray(bbox_areas, dtype=float)
    # Compare YOLO label distribution with COCO class distribution and boxes.
    coco_yolo_box_delta = {}
    coco_yolo_file_delta = {}
    for s in ("train", "val"):
        subset = list_basenames[s]
        ycount = Counter()
        for name in subset:
            lp = labels_by_stem.get(os.path.splitext(name)[0])
            if lp:
                rows, _ = parse_yolo(lp)
                ycount.update(r[0] for r in rows)
        coco_yolo_box_delta[s] = {
            "yolo_counts": dict(ycount),
            "coco_counts": coco_summary[s]["annotation_counts_by_class"],
        }
        coco_yolo_file_delta[s] = {
            "yolo_list_images": len(subset),
            "coco_images": coco_summary[s]["images"],
            "same_image_set": subset == coco_splits[s],
            "list_minus_coco": len(subset - coco_splits[s]),
            "coco_minus_list": len(coco_splits[s] - subset),
        }

    archive_md5 = md5(ARCHIVE)
    result = {
        "status": "AUDITED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "record": "https://zenodo.org/records/7961764",
            "archive": ARCHIVE.replace(WORKSPACE, "<WORKSPACE>"), "archive_bytes": os.path.getsize(ARCHIVE),
            "expected_md5": EXPECTED_MD5, "actual_md5": archive_md5,
            "md5_match": archive_md5 == EXPECTED_MD5,
            "license": "CC-BY-4.0",
            "citation": "Iqbal et al., CornWeed Dataset: A dataset for training maize and weed object detectors for agricultural machines, Zenodo 2023, DOI 10.5281/zenodo.7961764; companion paper DOI 10.15439/2023f2125.",
        },
        "counts": {
            "labeled_images_data_dir": len(labeled_images),
            "unlabeled_test_images": len(test_images),
            "all_images": len(all_images),
            "yolo_label_files": len(labels_by_stem),
            "official_lists": {s: len(v) for s, v in split_lists.items()},
            "split_image_counts_observed": dict(split_counts),
            "image_dimensions": dict(image_dims),
            "yolo_boxes_by_class_id": dict(class_counts),
            "box_total": sum(class_counts.values()),
            "yolo_label_files_with_boxes": int(label_summary["files_with_labels"]),
            "empty_yolo_label_files": int(label_summary["empty_label_files"]),
            "missing_labels_for_labeled_images": len(missing_label),
            "missing_label_examples": missing_label[:100],
            "label_parse_errors": len(parse_errors),
            "parse_error_examples": parse_errors[:50],
            "out_of_bounds_or_invalid_boxes": len(invalid_norm),
            "invalid_box_examples": invalid_norm[:100],
            "degenerate_boxes": len(degenerate),
            "bbox_area_fraction": {
                "min": float(areas.min()), "p01": float(np.percentile(areas, 1)),
                "median": float(np.median(areas)), "p99": float(np.percentile(areas, 99)),
                "max": float(areas.max()), "frac_lt_0_001": float((areas < .001).mean()),
            } if len(areas) else {},
        },
        "official_coco": {
            "splits": coco_summary,
            "train_val_overlap_images": len(coco_cross_split),
            "train_val_overlap_examples": coco_cross_split[:50],
            "missing_image_examples": coco_missing_images,
            "yolo_list_vs_coco": coco_yolo_file_delta,
            "yolo_vs_coco_class_counts": coco_yolo_box_delta,
        },
        "duplicates": {
            "exact_sha256_duplicate_groups": len(exact_dupes),
            "exact_duplicate_examples": exact_dupes[:100],
            "exact_cross_split_pair_counts": dict(exact_cross_counts),
            "exact_cross_split_examples": dict(exact_cross_examples),
            "phash_threshold_hamming": 6,
            "phash_cross_split_pair_counts": dict(pair_count_by_split),
            "phash_cross_split_examples_capped_1000": near_pairs,
            "phash_method": "64-bit pHash, compare equal-resolution images, Hamming<=6; candidates only, not proof of duplicate",
        },
        "sequence_leakage": {
            "filename_pattern": "site_YYMMDD_[platform_]color_frame (empty platform=handheld)",
            "parsed_images": len(all_images) - len(sequence_unparsed),
            "unparsed_examples": sequence_unparsed[:50],
            "capture_stream_count": len(seq_splits),
            "streams_crossing_splits": crossing_streams,
            "acquisition_runs_gap_gt_10": acquisition_runs,
            "cross_split_pairs_with_frame_delta_le_1": len(adjacent_cross[1]),
            "cross_split_pairs_with_frame_delta_le_5": len(adjacent_cross[5]),
            "cross_split_pairs_with_frame_delta_le_10": len(adjacent_cross[10]),
            "cross_split_pairs_by_split_pair": {
                "le_1": dict(adjacent_by_split_pair[1]),
                "le_5": dict(adjacent_by_split_pair[5]),
                "le_10": dict(adjacent_by_split_pair[10]),
            },
            "adjacent_pair_examples": adjacent_cross[1][:100],
        },
        "official_split_files": {
            "train.txt": len(train_list), "valid.txt": len(val_list),
            "test.txt": len(test_list),
            "test_labels_available": False,
            "test_list_in_data_yaml": False,
            "list_overlaps": {k: len(v) for k, v in list_overlaps.items()},
            "basename_overlaps": {
                k: {"count": len(v), "examples": v[:100]}
                for k, v in basename_overlaps.items()},
        },
        "weed_only_mapping": {
            "proposed": {"source YOLO class_id 0 (Weeds)": 0},
            "exclude_as_background": {"source YOLO class_id 1 (Maize)": True},
            "verified_by": "maize_dataset/data.yaml names=['Weeds','Maize']; README says zero-indexed; COCO categories id=1 Weeds,id=2 Maize.",
            "validity": "Technically valid as a one-class remap for annotated weed boxes. Maize annotations dropped; this treats maize/crop as background. Before using, check unannotated weeds/coverage in crop-box-rich scenes; labels alone cannot establish completeness.",
        },
        "timing_seconds": round(time.time() - t0, 2),
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_markdown(result)
    print(f"Wrote {JSON_OUT} and {MD_OUT}", flush=True)


def write_markdown(d):
    c, coco, dup, seq = (d["counts"], d["official_coco"],
                         d["duplicates"], d["sequence_leakage"])
    lines = [
        "# EXP-001B — CornWeed dataset audit",
        "",
        f"- Audit timestamp: `{d['generated_utc']}`",
        f"- Status: **CONDITIONAL** — archive, structure, labels and official splits audited; visual annotation-completeness review and Weed-crop V2 audit remain outstanding.",
        f"- Source: [Zenodo record 7961764]({d['source']['record']})",
        f"- Archive MD5: `{d['source']['actual_md5']}` (expected `{d['source']['expected_md5']}`; match={d['source']['md5_match']})",
        f"- License/citation: {d['source']['license']}; {d['source']['citation']}",
        "",
        "## Inventory",
        "",
        f"- Labeled image directory: {c['labeled_images_data_dir']}; unlabeled test images: {c['unlabeled_test_images']}; total: {c['all_images']}.",
        f"- YOLO label files: {c['yolo_label_files']}; boxes: {c['box_total']}; classes by zero-indexed ID: `{c['yolo_boxes_by_class_id']}`.",
        f"- Dimensions: `{c['image_dimensions']}`.",
        f"- Official lists: train={d['official_split_files']['train.txt']}, valid={d['official_split_files']['valid.txt']}, test={d['official_split_files']['test.txt']}. Test images are explicitly unlabeled; `test.txt` is commented out in the shipped `data.yaml`.",
        f"- COCO official JSON: train={coco['splits']['train']['images']} images/{coco['splits']['train']['annotations']} boxes; valid={coco['splits']['val']['images']} images/{coco['splits']['val']['annotations']} boxes.",
        f"- YOLO labels: missing={c['missing_labels_for_labeled_images']}; parse errors={c['label_parse_errors']}; illegal/out-of-bounds boxes={c['out_of_bounds_or_invalid_boxes']}; degenerate={c['degenerate_boxes']}.",
        "",
        "## Annotation and format checks",
        "",
        f"- YOLO normalized box area fraction: `{c['bbox_area_fraction']}`.",
        f"- COCO vs YOLO split image-set comparison: `{coco['yolo_list_vs_coco']}`; class box counts: `{coco['yolo_vs_coco_class_counts']}`. COCO uses category IDs 1/2; YOLO uses zero-based IDs 0/1.",
        "- `weed-only` mapping: map YOLO class 0 `Weeds` → class 0 `weed`; drop class 1 `Maize` boxes so maize becomes background. Mapping is structurally valid, but annotation coverage must be visually assessed because dropped crop boxes cannot prove all weeds were annotated.",
        "",
        "## Duplicate and capture-sequence leakage",
        "",
        f"- Exact SHA-256 duplicate groups: {dup['exact_sha256_duplicate_groups']}; cross-split exact pair counts: `{dup['exact_cross_split_pair_counts']}`.",
        f"- Cross-split pHash candidates (64-bit, Hamming ≤6, equal image dimensions): `{dup['phash_cross_split_pair_counts']}`. These are candidates and require review; examples are in JSON.",
        f"- Filename sequence pattern parsed: `{seq['filename_pattern']}`; {seq['capture_stream_count']} site/date/platform streams; {len(seq['streams_crossing_splits'])} streams span more than one official split.",
        f"- Cross-split adjacent-frame candidates (frame number delta ≤1 / ≤5 / ≤10): total {seq['cross_split_pairs_with_frame_delta_le_1']} / {seq['cross_split_pairs_with_frame_delta_le_5']} / {seq['cross_split_pairs_with_frame_delta_le_10']}. Train–val only: {seq['cross_split_pairs_by_split_pair']['le_1'].get('train-val', 0)} / {seq['cross_split_pairs_by_split_pair']['le_5'].get('train-val', 0)} / {seq['cross_split_pairs_by_split_pair']['le_10'].get('train-val', 0)}; the remainder pairs the labeled splits with the unlabeled `testing/` set.",
        f"- Every acquisition stream is a single unbroken frame run (`n_frame_runs_gap_gt_10 == 1` for all {seq['capture_stream_count']} streams; per-stream frame ranges in JSON). There are no natural frame-gap breakpoints inside a stream to subdivide on.",
        "- The official train/valid partition is image-level and **not group-safe**: exact duplicate images also occur in the unlabeled `testing/` directory (see exact pair counts); adjacent frames and pHash candidates cross splits. Do not treat its validation score as a leakage-controlled DG estimate.",
        "",
        "## Split recommendation",
        "",
        "- Preserve the official train/valid/test lists as immutable provenance. The shipped test partition is unlabeled and contains exact copies of labeled frames, so exclude it from any held-out evaluation claim.",
        "- **Deprecated option (do not use):** \"group by stream, then merge adjacent frame runs (gap ≤10)\". The audit shows each stream is already one unbroken frame run, so this procedure yields exactly one group per stream and cannot subdivide anything.",
        "- **Option A — leave-one-stream-out (LOSO), preferred for labeled data.** Only 3 labeled streams exist (`Langsenkamp_210722_bonirob`, `Langsenkamp_210727_bonirob`, `Langsenkamp_210727_handheld`; the 4th stream exists only in unlabeled test images). Run 3 folds, each holding out one stream as validation and training on the other two; report mean ± std across folds. Limitations: only 3 folds (very coarse, high variance), and date/platform co-vary across the streams, so the resulting \"domain\" mixes acquisition day, platform and scene rather than isolating one factor.",
        "- **Option B — fixed-window blocks with buffer zones.** Because there are no natural gaps inside a stream, cut each labeled stream into fixed-size frame windows (block size and buffer to be preregistered, e.g. 50-frame blocks with a ≥10-frame buffer dropped between blocks), merge reviewed pHash near-duplicate candidates into the same window, and assign windows to 70/15/15 with the preregistered deterministic hash rule. Limitations: windows are artificial groups, not separate acquisitions — blocks from the same stream still share field, illumination, growth stage and weed density, so independence is much weaker than between-stream; the buffer suppresses adjacent-frame leakage but does not prove block independence.",
        "- Both options are coarse: 3 labeled streams from one site (Langsenkamp) on 3 dates bound what any source-internal split can demonstrate about generalization. A third, external domain (e.g. Weed-crop V2, pending authorization) must carry the DG claim.",
        "- No new split manifests are generated now. The protocol choice and its manifest hashes require reviewer approval before any split list is produced or training starts.",
        "- Keep unlabeled test images out of supervised training/evaluation; use only if a separately approved semi-supervised protocol is later registered.",
        "",
        "## Limitations / outstanding checks",
        "",
        "- No pixels were published. The machine audit does not establish weed-label completeness or weed omissions where maize boxes are ignored; manually inspect a preregistered, hashed sample before approving `weed-only` conversion.",
        "- Weed-crop V2 is not present/audited; cross-dataset compatibility remains pending.",
        "- See `dataset_audit.json` for full manifests, class counts, exact duplicates and capped pHash/adjacency examples.",
        "",
    ]
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
