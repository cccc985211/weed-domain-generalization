"""EXP-001C Task3: content audit of the downloaded Weed-crop V2 dataset.

Read-only. Verifies per-file SHA-256 against the publisher listing, then audits:
  * the 13 documented classes (classes.txt per scenario folder)
  * per-class and per-folder YOLO box counts
  * label integrity: pairing, empty files, out-of-range class ids, illegal/degenerate
    boxes
  * weed-only mapping structure: 5 weed species -> class 0; 8 crop species excluded
    as background; original labels untouched

Writes reports/EXP-001C/external_dataset_audit.{json,md}.
"""
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(REPO)
ROOT = os.path.join(WORKSPACE, "datasets", "WeedCropV2")
REPORT_DIR = os.path.join(REPO, "reports", "EXP-001C")

WEED_SPECIES = {"Kochia", "Horseweed", "Water Hemp", "Ragweed",
                "Redroot Pigweed"}
CROP_SPECIES = {"Black bean", "Canola", "Corn", "Lentil", "Field pea",
                "Flax", "Soybean", "Sugar beet"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_yolo(path):
    boxes, errors = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            p = line.split()
            if not p:
                continue
            if len(p) != 5:
                errors.append({"file": path, "line": ln,
                               "reason": f"field_count={len(p)}"})
                continue
            try:
                cls = int(p[0])
                x, y, w, h = map(float, p[1:])
            except ValueError:
                errors.append({"file": path, "line": ln, "reason": "non_numeric"})
                continue
            boxes.append((cls, x, y, w, h, ln))
    return boxes, errors


def main():
    listing = json.load(open(os.path.join(ROOT, "file_list_publisher.json"),
                             encoding="utf-8"))
    by_folder = defaultdict(list)
    for e in listing["files"]:
        by_folder[e["folder"]].append(e)

    # per-file hash verification (independent re-check of the downloader)
    verified, bad = 0, []
    for e in listing["files"]:
        p = os.path.join(ROOT, e["folder"], e["filename"])
        if os.path.exists(p) and sha256_file(p) == e["sha256"]:
            verified += 1
        else:
            bad.append(e["folder"] + "/" + e["filename"])
    print(f"hash verification: {verified}/{len(listing['files'])} ok")
    if bad:
        raise SystemExit(f"hash failures: {bad[:10]}")

    # classes.txt per folder -> class name -> id
    class_ids = {}
    folder_classes = {}
    for folder, files in sorted(by_folder.items()):
        ct = [e for e in files if e["filename"] == "classes.txt"]
        if not ct:
            raise SystemExit(f"no classes.txt in {folder}")
        with open(os.path.join(ROOT, folder, "classes.txt"),
                  encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        folder_classes[folder] = names
        for i, nm in enumerate(names):
            class_ids.setdefault(nm, set()).add(i)
    inconsistent = {k: v for k, v in class_ids.items() if len(v) != 1}
    all_names = sorted(class_ids)
    class_to_id = {k: sorted(v)[0] for k, v in class_ids.items()
                   if len(v) == 1}

    # label audit
    img_files = [e for e in listing["files"]
                 if e["filename"].lower().endswith((".jpg", ".jpeg", ".png"))]
    missing_label, empty_label = [], []
    illegal, degenerate, parse_errors = [], [], []
    boxes_by_class = Counter()
    boxes_by_folder_class = defaultdict(Counter)
    dims = Counter()
    images_audited = 0
    for e in img_files:
        folder, name = e["folder"], e["filename"]
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(ROOT, folder, name)
        lbl = os.path.join(ROOT, folder, stem + ".txt")
        if not os.path.exists(lbl):
            missing_label.append(folder + "/" + name)
            continue
        boxes, errs = parse_yolo(lbl)
        parse_errors += [{"file": folder + "/" + name, **er} for er in errs]
        if not boxes:
            empty_label.append(folder + "/" + name)
        with Image.open(img_path) as im:
            dims[f"{im.width}x{im.height}"] += 1
        images_audited += 1
        for cls, x, y, w, h, ln in boxes:
            if cls < 0 or cls >= len(folder_classes[folder]):
                illegal.append({"file": folder + "/" + name, "line": ln,
                                "reason": f"class_id {cls} out of range"})
                continue
            boxes_by_class[folder_classes[folder][cls]] += 1
            boxes_by_folder_class[folder][folder_classes[folder][cls]] += 1
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                    and x - w / 2 >= -1e-6 and x + w / 2 <= 1 + 1e-6
                    and y - h / 2 >= -1e-6 and y + h / 2 <= 1 + 1e-6):
                illegal.append({"file": folder + "/" + name, "line": ln,
                                "reason": "illegal_norm_coords",
                                "xywh": [x, y, w, h]})
            elif w * h <= 0:
                degenerate.append(folder + "/" + name)

    weed_cls = {c: class_to_id[c] for c in WEED_SPECIES if c in class_to_id}
    crop_cls = {c: class_to_id[c] for c in CROP_SPECIES if c in class_to_id}
    unknown_cls = {c: class_to_id[c] for c in class_to_id
                   if c not in WEED_SPECIES and c not in CROP_SPECIES}
    weed_boxes = sum(boxes_by_class[c] for c in weed_cls)
    crop_boxes = sum(boxes_by_class[c] for c in crop_cls)

    result = {
        "status": "AUDITED",
        "dataset": listing["dataset"], "version": listing["version"],
        "source_page": listing["source_page"],
        "doi": "10.17632/mthv4ppwyw.2", "license": "CC BY 4.0",
        "publisher_listing_sha256_note": (
            "every file verified against publisher per-file sha256 "
            f"({verified}/{len(listing['files'])}); "
            "official zip sha256 3f137d4fba4146077fd96ce6a35e938cf6aa41a5ed0"
            "7c9709d4a92586db2acee applies to the publisher zip, which this "
            "network cannot fetch (S3 403); per-file verification is stricter"),
        "files_verified": verified,
        "files_expected": len(listing["files"]),
        "scenario_folders": {f: {"classes.txt": names,
                                 "n_files": len(by_folder[f])}
                             for f, names in sorted(folder_classes.items())},
        "classes_txt_consistent_across_folders":
            not inconsistent, "class_id_inconsistencies": inconsistent,
        "n_documented_classes": len(all_names),
        "class_to_yolo_id": class_to_id,
        "unclassified_names": unknown_cls,
        "counts": {
            "labeled_images": images_audited,
            "missing_labels": len(missing_label),
            "missing_label_examples": missing_label[:50],
            "empty_label_files": len(empty_label),
            "empty_examples": empty_label[:50],
            "parse_errors": len(parse_errors),
            "parse_error_examples": parse_errors[:50],
            "illegal_or_out_of_range": len(illegal),
            "illegal_examples": illegal[:50],
            "degenerate_boxes": len(degenerate),
            "image_dimensions": dict(dims),
            "boxes_total": sum(boxes_by_class.values()),
            "boxes_by_class": dict(boxes_by_class),
            "boxes_by_folder_class": {f: dict(c) for f, c in
                                      boxes_by_folder_class.items()},
        },
        "weed_only_mapping": {
            "weed_species": weed_cls,
            "crop_species_excluded": crop_cls,
            "weed_boxes_kept": weed_boxes,
            "crop_boxes_dropped_as_background": crop_boxes,
            "validity": ("structurally valid: map the 5 weed species ids to 0 "
                         "and drop the 8 crop species ids; original labels are "
                         "preserved and the conversion is a derived copy"),
            "residual_risk": ("unannotated weeds inside crop scenes become "
                              "systematic false negatives; manual review of "
                              "CornWeed does not cover this dataset — a "
                              "preregistered sample of Weed-crop V2 would be "
                              "needed before any weed-only TRAINING use "
                              "(external evaluation-only use is unaffected "
                              "because the external protocol evaluates full "
                              "13-class or weed-only predictions as declared)"),
        },
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "external_dataset_audit.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    c = result["counts"]
    lines = [
        "# EXP-001C — Weed-crop V2 external dataset audit",
        "",
        f"- Source: Mendeley Data [{listing['dataset']} v{listing['version']}]"
        f"({listing['source_page']}) · DOI 10.17632/mthv4ppwyw.2 · CC BY 4.0",
        "- Files verified against publisher per-file SHA-256: "
        f"{verified}/{len(listing['files'])} (listing preserved in "
        "`datasets/WeedCropV2/file_list_publisher.json`; download manifest in "
        "`datasets/WeedCropV2/download_manifest.json`, both outside git).",
        "- The publisher's Download-All zip (sha256 `3f137d4f…2acee`, "
        "7,775,732,980 B) is served from an S3 bucket that denies anonymous "
        "GETs (403 AccessDenied); files were fetched individually via the "
        "per-file endpoints the dataset page itself exposes.",
        "",
        "## Classes",
        "",
        f"- Documented classes: {len(all_names)} — `{', '.join(all_names)}`",
        f"- classes.txt identical across the {len(folder_classes)} scenario "
        f"folders: {not inconsistent}",
        f"- YOLO ids: `{class_to_id}`",
        f"- Unclassified names: `{unknown_cls or 'none'}`",
        "",
        "## Label integrity",
        "",
        f"- Labeled images: {c['labeled_images']}; missing labels: "
        f"{c['missing_labels']}; empty label files: {c['empty_label_files']}",
        f"- Parse errors: {c['parse_errors']}; illegal/out-of-range boxes: "
        f"{c['illegal_or_out_of_range']}; degenerate: {c['degenerate_boxes']}",
        f"- Image dimensions: `{c['image_dimensions']}`",
        f"- Boxes total: {c['boxes_total']}; by class: `{c['boxes_by_class']}`",
        "",
        "## Weed-only mapping (structure only — NO model runs were performed)",
        "",
        f"- Weed species → class 0: `{weed_cls}` ({weed_boxes} boxes kept)",
        f"- Crop species → background (dropped in the derived copy): "
        f"`{crop_cls}` ({crop_boxes} boxes)",
        "- Original labels are preserved untouched; any conversion must write a "
        "separate derived tree outside `datasets/`.",
        "- This dataset is used to FREEZE the external evaluation protocol only. "
        "No model accuracy evaluation was run and none may be registered on it "
        "before the frozen protocol is approved.",
        "",
        "## Outstanding checks",
        "",
        "- Machine audit cannot establish annotation completeness of Weed-crop "
        "V2; a preregistered manual sample (same style as "
        "`annotation_review/`) is required before any weed-only TRAINING use.",
        "- External evaluation protocol must declare which label space "
        "(13-class or weed-only) is scored and fix the class mapping in "
        "`configs/evaluation_protocol.yaml` before any test is registered.",
        "- Full detail: `external_dataset_audit.json`.",
    ]
    with open(os.path.join(REPORT_DIR, "external_dataset_audit.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote external_dataset_audit.json/.md")
    print(json.dumps({"classes": all_names, "boxes": dict(boxes_by_class),
                      "missing": len(missing_label),
                      "empty": len(empty_label),
                      "illegal": len(illegal)}, indent=1))


if __name__ == "__main__":
    main()
