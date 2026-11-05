"""EXP-001C Task1: build the frozen leave-one-stream-out (LOSO) split for CornWeed.

Preregistered protocol (mirrors configs/loso_protocol.yaml, approved Option A):
  * 3 labeled acquisition streams -> 3 folds; in fold k the whole stream k is the
    test domain (never used for hyperparameters, epochs or weight selection).
  * The two remaining streams are split into train/val by TIME-CONTINUOUS 50-frame
    blocks taken every 60 frames (10-frame buffer zone between blocks is excluded
    from both train and val).
  * Block -> train/val assignment is deterministic:
        sha256(f"{stream}|{block_index}|EXP001C-LOSO-v1") mod 100
        [0,80) -> train, [80,100) -> val  (80/20 of blocks)
  * Duplicate filtering within each fold (priority test > train > val):
      - exact SHA-256 duplicate groups
      - pHash near-duplicates (64-bit, Hamming <= 6, equal image dimensions,
        union-find clustering) - machine heuristic, flagged for human review
    Any image that duplicates an image with a higher-priority role is marked
    excluded_duplicate; every decision is recorded in duplicates_conflicts.csv.
  * Original images and labels are never modified; the official train/valid lists
    are recorded as provenance only.

Outputs (reports/EXP-001C/):
  split_manifest.csv        fold,role,image,stream,frame,official_split,sha256,phash_hex,action
  duplicates_conflicts.csv  per-fold duplicate decisions
  split_stats.json          per-fold statistics + manifest SHA-256
"""
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
import imagehash

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(REPO)
ROOT = os.path.join(WORKSPACE, "datasets", "CornWeed", "maize_dataset")
REPORT_DIR = os.path.join(REPO, "reports", "EXP-001C")
BLOCK = 50
PERIOD = 60  # 50-frame block + 10-frame buffer
TRAIN_PCT = 80
SALT = "EXP001C-LOSO-v1"
PHASH_HAMMING = 6
ROLE_PRIORITY = {"test": 3, "train": 2, "val": 1, "buffer": 0}
STREAMS = {
    "Langsenkamp_210722_bonirob",
    "Langsenkamp_210727_bonirob",
    "Langsenkamp_210727_handheld",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_yolo(path):
    boxes = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) == 5:
                boxes.append((int(p[0]), float(p[1]), float(p[2]),
                              float(p[3]), float(p[4])))
    return boxes


def block_role(stream, block_idx):
    h = int(hashlib.sha256(f"{stream}|{block_idx}|{SALT}".encode()).hexdigest(), 16)
    return "train" if h % 100 < TRAIN_PCT else "val"


def union_find(n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def main():
    t0 = time.time()
    os.makedirs(REPORT_DIR, exist_ok=True)
    data_dir = os.path.join(ROOT, "data")
    labeled = sorted(f for f in os.listdir(data_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg")))
    official = {}
    for split, fname in (("train", "train.txt"), ("val", "valid.txt")):
        with open(os.path.join(ROOT, fname), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    official[os.path.basename(line.strip())] = split

    # --- per-image records: stream, frame, sha256, pHash, label stats ---
    records = []
    phashes = []
    for i, name in enumerate(labeled, 1):
        stem = os.path.splitext(name)[0]
        m = re.match(r"^(?P<site>.+?)_(?P<date>\d{6})_(?P<platform>.*?)color_(?P<frame>\d+)$",
                     stem)
        if not m:
            raise SystemExit(f"unparsed filename: {name}")
        d = m.groupdict()
        platform = d["platform"].rstrip("_") or "handheld"
        stream = f"{d['site']}_{d['date']}_{platform}"
        path = os.path.join(data_dir, name)
        with Image.open(path) as im:
            ph = int(str(imagehash.phash(im, hash_size=8)), 16)
            w, h = im.size
        lp = os.path.join(data_dir, stem + ".txt")
        boxes = parse_yolo(lp) if os.path.exists(lp) else []
        rec = {
            "name": name,
            "relpath": f"data/{name}",
            "stream": stream,
            "frame": int(d["frame"]),
            "width": w, "height": h,
            "sha256": sha256_file(path),
            "phash": ph,
            "label_file": os.path.basename(lp) if os.path.exists(lp) else "",
            "n_boxes": len(boxes),
            "n_weed": sum(1 for b in boxes if b[0] == 0),
            "n_maize": sum(1 for b in boxes if b[0] == 1),
            "official": official.get(name, ""),
        }
        records.append(rec)
        phashes.append((rec["relpath"], w, h, ph))
        if i % 500 == 0:
            print(f"hashed {i}/{len(labeled)}", flush=True)

    streams_present = {r["stream"] for r in records}
    assert streams_present == STREAMS, streams_present

    # --- block layout per stream (over labeled frame range) ---
    stream_min = {s: min(r["frame"] for r in records if r["stream"] == s)
                  for s in STREAMS}
    stream_max = {s: max(r["frame"] for r in records if r["stream"] == s)
                  for s in STREAMS}

    def base_role(rec):
        s = rec["stream"]
        f = rec["frame"]
        k, off = divmod(f - stream_min[s], PERIOD)
        if off < BLOCK and block_start_valid(s, k, stream_min[s], stream_max[s]):
            return block_role(s, k)
        return "buffer"

    def block_start_valid(s, k, base, mx):
        return base + k * PERIOD + BLOCK - 1 <= mx

    # --- pHash clustering within equal dimensions (union-find) ---
    clusters = []  # list of lists of relpaths (clusters with >=2 members)
    by_dim = defaultdict(list)
    for rel, w, h, ph in phashes:
        by_dim[(w, h)].append((rel, ph))
    bit_lut = np.unpackbits(np.arange(256, dtype=np.uint8)).reshape(256, 8).sum(1)
    pair_dist = {}
    for dim, rows in by_dim.items():
        find, union = union_find(len(rows))
        vals = np.array([np.uint64(r[1]) for r in rows])
        bsz = 256
        for i0 in range(0, len(rows), bsz):
            for j0 in range(i0, len(rows), bsz):
                aa, bb = vals[i0:i0+bsz, None], vals[None, j0:j0+bsz]
                xx = (aa ^ bb).astype(np.uint64).view(np.uint8).reshape(
                    len(aa), -1, 8)
                dist = bit_lut[xx].sum(-1)
                ii, jj = np.where(dist <= PHASH_HAMMING)
                for x, y in zip(ii.tolist(), jj.tolist()):
                    ai, bj = i0 + x, j0 + y
                    if ai != bj:
                        union(ai, bj)
                        key = tuple(sorted((rows[ai][0], rows[bj][0])))
                        pair_dist[key] = min(pair_dist.get(key, 99),
                                             int(dist[x, y]))
        comp = defaultdict(list)
        for idx, r in enumerate(rows):
            comp[find(idx)].append(r[0])
        clusters += [v for v in comp.values() if len(v) > 1]

    # --- duplicate groups: exact sha256 + pHash clusters ---
    by_sha = defaultdict(list)
    for r in records:
        by_sha[r["sha256"]].append(r["relpath"])
    exact_groups = [v for v in by_sha.values() if len(v) > 1]
    rec_by_rel = {r["relpath"]: r for r in records}

    conflicts = []  # dict rows for duplicates_conflicts.csv

    # --- assemble folds and resolve duplicates ---
    manifest_rows = []
    stats = {"protocol_salt": SALT, "block": BLOCK, "period": PERIOD,
             "train_pct": TRAIN_PCT, "phash_hamming": PHASH_HAMMING,
             "folds": {}}
    dup_group_id = 0
    for fold in (1, 2, 3):
        test_stream = sorted(STREAMS)[fold - 1]
        base_roles = {}
        for r in records:
            if r["stream"] == test_stream:
                base_roles[r["relpath"]] = "test"
            else:
                base_roles[r["relpath"]] = base_role(r)

        excluded = set()
        buffer_overlap = set()
        # exact duplicates
        for group in exact_groups:
            roles = {rel: base_roles[rel] for rel in group
                     if rel in base_roles}
            if len(roles) < 2:
                continue
            top = max(roles.values(), key=lambda x: ROLE_PRIORITY[x])
            for rel, role in roles.items():
                if role != top:
                    if role == "buffer":
                        buffer_overlap.add(rel)
                        continue
                    excluded.add(rel)
                    conflicts.append({
                        "fold": fold, "kind": "exact_sha256",
                        "kept_image": group[0], "kept_role": top,
                        "removed_image": rel, "removed_role": role,
                        "action": f"excluded_duplicate_from_{role}",
                        "hamming": ""})
        # pHash clusters (machine heuristic; flagged for human review)
        for cluster in clusters:
            roles = {rel: base_roles[rel] for rel in cluster
                     if rel in base_roles}
            if len(roles) < 2:
                continue
            vals = sorted(set(roles.values()),
                          key=lambda x: -ROLE_PRIORITY[x])
            top = vals[0]
            if len(vals) < 2:
                continue
            dup_group_id += 1
            for rel, role in roles.items():
                if role == top:
                    continue
                if role == "buffer":
                    buffer_overlap.add(rel)
                    continue
                if rel not in excluded:
                    direct = min(
                        [d for (a, b), d in pair_dist.items()
                         if rel in (a, b)] or [99])
                    ham = direct if direct <= PHASH_HAMMING else "indirect"
                    conflicts.append({
                        "fold": fold, "kind": f"phash_le{PHASH_HAMMING}",
                        "kept_image": cluster[0], "kept_role": top,
                        "removed_image": rel, "removed_role": role,
                        "action": f"excluded_duplicate_from_{role}"
                                  "_pending_human_review",
                        "hamming": ham})
                excluded.add(rel)

        for r in records:
            rel = r["relpath"]
            role = base_roles[rel]
            action = "active"
            if role == "buffer":
                action = "buffer_zone_excluded"
            if rel in excluded:
                action = "excluded_duplicate"
            manifest_rows.append({
                "fold": fold, "role": role, "image": rel,
                "stream": r["stream"], "frame": r["frame"],
                "official_split": r["official"], "sha256": r["sha256"],
                "phash_hex": f"{r['phash']:016x}", "action": action,
            })

        # --- per-fold stats ---
        active_by_rel = {row["image"]: row for row in manifest_rows
                         if row["fold"] == fold and row["action"] == "active"}
        box_by_role = defaultdict(Counter)
        count_by_role = Counter()
        for row in active_by_rel.values():
            count_by_role[row["role"]] += 1
        for r in records:
            row = active_by_rel.get(r["relpath"])
            if row is None:
                continue
            box_by_role[row["role"]][0] += r["n_weed"]
            box_by_role[row["role"]][1] += r["n_maize"]
        buf_n = sum(1 for row in manifest_rows
                    if row["fold"] == fold and row["action"] == "buffer_zone_excluded")
        dup_n = sum(1 for row in manifest_rows
                    if row["fold"] == fold and row["action"] == "excluded_duplicate")
        buf_overlap_n = sum(
            1 for r in records
            if base_roles[r["relpath"]] == "buffer" and r["relpath"] in buffer_overlap)
        stream_roles = defaultdict(Counter)
        for row in active_by_rel.values():
            stream_roles[row["stream"]][row["role"]] += 1
        stats["folds"][f"fold{fold}"] = {
            "test_stream": test_stream,
            "images_by_role": {k: count_by_role.get(k, 0)
                               for k in ("train", "val", "test")},
            "boxes_by_role": {k: {"weed(0)": v[0], "maize(1)": v[1]}
                              for k, v in box_by_role.items()},
            "buffer_excluded": buf_n,
            "duplicate_excluded": dup_n,
            "buffer_images_with_phash_overlap": buf_overlap_n,
            "stream_role_composition": {s: dict(c)
                                        for s, c in stream_roles.items()},
        }

    manifest_path = os.path.join(REPORT_DIR, "split_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    manifest_sha = sha256_file(manifest_path)

    conflicts_path = os.path.join(REPORT_DIR, "duplicates_conflicts.csv")
    with open(conflicts_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=[
            "fold", "kind", "kept_image", "kept_role", "removed_image",
            "removed_role", "action", "hamming"])
        wtr.writeheader()
        wtr.writerows(conflicts)

    # per-fold frozen list hashes (sorted image lists)
    for fold in (1, 2, 3):
        for role in ("train", "val", "test"):
            imgs = sorted(row["image"] for row in manifest_rows
                          if row["fold"] == fold and row["role"] == role
                          and row["action"] == "active")
            h = hashlib.sha256("\n".join(imgs).encode()).hexdigest()
            stats["folds"][f"fold{fold}"][f"{role}_list_sha256"] = h
    stats["manifest_sha256"] = manifest_sha
    stats["conflicts_csv_sha256"] = sha256_file(conflicts_path)
    stats["labeled_images"] = len(records)
    stats["timing_seconds"] = round(time.time() - t0, 1)

    with open(os.path.join(REPORT_DIR, "split_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in stats.items() if k != "folds"}, indent=1))
    for fk, fv in stats["folds"].items():
        print(fk, fv["test_stream"], fv["images_by_role"],
              "buffer:", fv["buffer_excluded"], "dup:", fv["duplicate_excluded"])
    print(f"manifest sha256: {manifest_sha}")
    print(f"conflicts rows: {len(conflicts)}")


if __name__ == "__main__":
    main()
