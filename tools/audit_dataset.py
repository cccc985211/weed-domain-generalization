"""EXP-001 Step B: dataset & label audit (YOLO-format assumed; COCO noted if found).

Usage:
    .venv/Scripts/python tools/audit_dataset.py --root C:/.../datasets/CornWeed --name CornWeed
    .venv/Scripts/python tools/audit_dataset.py --root C:/.../datasets/WeedCropV2 --name WeedCropV2

Read-only against the dataset: never modifies original files. Writes
reports/EXP-001/dataset_audit_<name>.json (hashes + statistics only, no pixels).
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if limit and n >= limit:
                break
    return h.hexdigest()


def phash64_fast(path):
    import imagehash
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        bits = np.asarray(imagehash.phash(im, hash_size=8).hash,
                          dtype=np.uint8).flatten()
        v = 0
        for b in bits[:64]:
            v = (v << 1) | int(b)
        return v


def parse_yolo_label(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            p = line.split()
            if not p:
                continue
            if len(p) != 5:
                rows.append({"line": ln, "error": f"expected 5 fields, got {len(p)}"})
                continue
            try:
                c, x, y, w, h = int(p[0]), *map(float, p[1:])
            except ValueError:
                rows.append({"line": ln, "error": "non-numeric"})
                continue
            illegal = not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)
            rows.append({"line": ln, "cls": c, "xywh": [x, y, w, h], "illegal": illegal})
    return rows


def audit(root, name):
    imgs, labels = {}, {}
    coco_jsons = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            ext = os.path.splitext(fn)[1].lower()
            rel = os.path.relpath(p, root)
            if ext in IMG_EXT:
                imgs[rel] = p
            elif ext == ".txt":
                labels[rel] = p
            elif ext == ".json" and os.path.getsize(p) > 1000:
                coco_jsons.append(rel)

    # match images to labels by stem when conventional labels/ tree exists
    stem2label = {}
    for rel in labels:
        stem = os.path.splitext(os.path.basename(rel))[0]
        stem2label.setdefault(stem, rel)

    stats = {
        "root": root, "name": name,
        "n_images": len(imgs), "n_label_files": len(labels),
        "candidate_coco_json": coco_jsons,
        "class_counts": Counter(), "illegal_boxes": 0, "parse_errors": 0,
        "missing_label_for_image": 0, "label_without_image": 0,
        "empty_label_files": 0, "image_dims": Counter(),
        "areas": [], "manifest": {},
    }

    try:
        from PIL import Image
        have_pil = True
    except ImportError:
        have_pil = False

    for rel, p in sorted(imgs.items()):
        stats["manifest"][rel] = sha256(p)
        if have_pil:
            try:
                from PIL import Image
                with Image.open(p) as im:
                    stats["image_dims"][f"{im.width}x{im.height}"] += 1
            except Exception as e:
                stats["image_dims"][f"<unreadable: {e}>"] += 1
        lab = stem2label.get(os.path.splitext(os.path.basename(rel))[0])
        if lab is None:
            stats["missing_label_for_image"] += 1
            continue
        rows = parse_yolo_label(labels[lab])
        if not rows:
            stats["empty_label_files"] += 1
        for r in rows:
            if "error" in r:
                stats["parse_errors"] += 1
                continue
            stats["class_counts"][r["cls"]] += 1
            if r["illegal"]:
                stats["illegal_boxes"] += 1
            stats["areas"].append(r["xywh"][2] * r["xywh"][3])

    labeled_stems = {os.path.splitext(os.path.basename(r))[0] for r in imgs}
    stats["label_without_image"] = sum(
        1 for s in stem2label if s not in labeled_stems)

    # --- acquisition-sequence heuristics for leakage analysis ---
    import re
    grp = Counter()
    for rel in imgs:
        stem = os.path.splitext(os.path.basename(rel))[0]
        # group key = filename minus trailing digit run (e.g. seq42_img0007 -> seq42_img)
        key = re.sub(r"[\d_]+$", "", stem)
        grp[key] += 1
    stats["filename_groups"] = {
        "n_groups": len(grp),
        "largest": grp.most_common(10),
        "note": "stem minus trailing digits; proxies a capture sequence"}
    # detect official split layout
    tops = {p.split(os.sep)[0] for p in
            (os.path.relpath(x, root) for x in imgs.values())}
    stats["top_level_dirs"] = sorted(tops)
    stats["official_split_detected"] = bool(
        {"train", "val", "valid", "test"} & {t.lower() for t in tops})
    # EXIF probe on up to 20 images
    exif_dates = Counter()
    if have_pil:
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            for rel, p in list(sorted(imgs.items()))[:20]:
                try:
                    with Image.open(p) as im:
                        ex = im.getexif()
                        dt = ex.get(36867) or ex.get(306)
                        exif_dates[str(dt)] += 1
                except Exception:
                    exif_dates["<unreadable>"] += 1
        except Exception:
            pass
    stats["exif_datetime_sample"] = dict(exif_dates)

    areas = stats.pop("areas")
    if areas:
        import numpy as np
        a = np.asarray(areas)
        stats["bbox_area_rel"] = {
            "min": float(a.min()), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "max": float(a.max()), "frac_lt_0.001": float((a < 0.001).mean()),
            "frac_lt_0.01": float((a < 0.01).mean()),
        }

    # near-duplicate candidates by pHash (optional; slow on large sets)
    stats["phash_groups"] = "skipped (run with --phash to enable)"
    stats["class_counts"] = dict(stats["class_counts"])
    stats["image_dims"] = dict(stats["image_dims"])
    mh = hashlib.sha256()
    for k in sorted(stats["manifest"]):
        mh.update(f"{k}:{stats['manifest'][k]}\n".encode())
    stats["manifest_sha256"] = mh.hexdigest()
    return stats, imgs


def add_phash_groups(stats, imgs, threshold=6):
    import numpy as np
    hashes = {}
    for rel, p in sorted(imgs.items()):
        try:
            hashes[rel] = phash64_fast(p)
        except Exception:
            pass
    keys = sorted(hashes)
    vals = np.array([np.uint64(hashes[k]) for k in keys])

    parent = {r: r for r in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # chunked pairwise hamming distance on uint64 via byte unpackbits popcount
    lut = np.unpackbits(np.arange(256, dtype=np.uint8)).reshape(256, 8).sum(1)
    B = 512
    n = len(keys)
    for i0 in range(0, n, B):
        for j0 in range(i0, n, B):
            a = vals[i0:i0 + B, None]
            b = vals[j0:j0 + B][None, :]
            xr = (a ^ b).astype(np.uint64).view(np.uint8).reshape(len(a), -1, 8)
            dist = lut[xr].sum(-1)
            ii, jj = np.where(dist <= threshold)
            for x, y in zip(ii.tolist(), jj.tolist()):
                if i0 + x >= j0 + y:
                    continue
                pi, pj = find(keys[i0 + x]), find(keys[j0 + y])
                if pi != pj:
                    parent[pi] = pj
    groups = defaultdict(list)
    for k in keys:
        groups[find(k)].append(k)
    dupes = [sorted(v) for v in groups.values() if len(v) > 1]
    stats["phash_groups"] = {
        "threshold_hamming": threshold, "n_images_hashed": len(keys),
        "n_groups_multi": len(dupes),
        "examples": dupes[:25],
        "note": "manifest IDs only; pixels not exported"}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--phash", action="store_true")
    ap.add_argument("--threshold", type=int, default=6)
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        print(f"BLOCKED: dataset root not found: {a.root}")
        sys.exit(2)

    stats, imgs = audit(a.root, a.name)
    if a.phash:
        stats = add_phash_groups(stats, imgs, a.threshold)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reports", "EXP-001", f"dataset_audit_{a.name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
