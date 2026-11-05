"""EXP-001C Task2: preregistered manual annotation-completeness sample for CornWeed.

Per acquisition stream: 30 images = 20 fixed-seed random + 10 hard samples.
Hard score (preregistered): z(n_maize) + z(n_weed) + z(frac_small_boxes),
frac_small = fraction of boxes with relative area < 0.001 (ties broken by name).
The crop-box-heavy scenes are exactly where dropping maize boxes could hide
unannotated weeds, so they are the priority for human review.

Outputs:
  runs/EXP-001C/annotation_panels/<stream>/<image>.jpg   full-res review panels
      (outside git; original files are never modified)
  reports/EXP-001C/annotation_review/sample_selection.csv  frozen sample list
  reports/EXP-001C/annotation_review/review_template.csv   empty review form
  reports/EXP-001C/annotation_review/contact_sheets/*.jpg  in-git overviews
  reports/EXP-001C/annotation_review/README.md             review protocol
"""
import csv
import hashlib
import os
import random
import re
import statistics
from collections import defaultdict

from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(REPO)
ROOT = os.path.join(WORKSPACE, "datasets", "CornWeed", "maize_dataset")
PAN_DIR = os.path.join(WORKSPACE, "runs", "EXP-001C", "annotation_panels")
REV_DIR = os.path.join(REPO, "reports", "EXP-001C", "annotation_review")
SHEETS_DIR = os.path.join(REV_DIR, "contact_sheets")
SEED = 42
RANDOM_N = 20
HARD_N = 10
SMALL_AREA = 0.001

COLORS = {0: (255, 32, 32), 1: (32, 105, 255)}
NAMES = {0: "Weeds(0)", 1: "Maize(1)"}


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


def stream_of(stem):
    m = re.match(r"^(?P<site>.+?)_(?P<date>\d{6})_(?P<platform>.*?)color_(?P<frame>\d+)$",
                 stem)
    d = m.groupdict()
    return f"{d['site']}_{d['date']}_{(d['platform'].rstrip('_') or 'handheld')}"


def draw_panel(im, boxes, stem):
    d = ImageDraw.Draw(im)
    W, H = im.size
    d.rectangle([0, 0, W, 18], fill=(0, 0, 0))
    d.text((4, 3), f"{stem}   RED=Weeds(0)  BLUE=Maize(1)", fill=(255, 255, 0))
    for cls, x, y, bw, bh in boxes:
        x1 = (x - bw / 2) * W
        y1 = max(0, (y - bh / 2) * H)
        x2 = (x + bw / 2) * W
        y2 = (y + bh / 2) * H
        d.rectangle([x1, y1, x2, y2], outline=COLORS[cls], width=2)
    return im


def contact_sheet(paths, cols, out_path, title):
    tw, th = 480, 270
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 18)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    d.text((4, 2), title, fill=(255, 255, 0))
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        with Image.open(p) as im:
            thumb = im.copy()
        thumb.thumbnail((tw, th))
        sheet.paste(thumb, (c * tw, r * (th + 18) + 18))
        d.text((c * tw + 4, r * (th + 18) + 20),
               os.path.splitext(os.path.basename(p))[0], fill=(255, 255, 0))
    sheet.save(out_path, quality=82)
    return sha256_file(out_path)


def main():
    os.makedirs(PAN_DIR, exist_ok=True)
    os.makedirs(SHEETS_DIR, exist_ok=True)
    data_dir = os.path.join(ROOT, "data")
    labeled = sorted(f for f in os.listdir(data_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg")))

    by_stream = defaultdict(list)
    for name in labeled:
        stem = os.path.splitext(name)[0]
        lp = os.path.join(data_dir, stem + ".txt")
        boxes = parse_yolo(lp) if os.path.exists(lp) else []
        areas = [bw * bh for _, _, _, bw, bh in boxes]
        frac_small = (sum(1 for a in areas if a < SMALL_AREA) / len(areas)
                      if areas else 0.0)
        by_stream[stream_of(stem)].append({
            "name": name, "stem": stem,
            "n_weed": sum(1 for b in boxes if b[0] == 0),
            "n_maize": sum(1 for b in boxes if b[0] == 1),
            "n_boxes": len(boxes), "frac_small": frac_small,
        })

    def zscore(values):
        mu = statistics.fmean(values)
        sd = statistics.pstdev(values) or 1.0
        return lambda v: (v - mu) / sd

    zs = {k: zscore([rec[k] for recs in by_stream.values() for rec in recs])
          for k in ("n_weed", "n_maize", "frac_small")}
    for recs in by_stream.values():
        for rec in recs:
            rec["hard_score"] = round(zs["n_weed"](rec["n_weed"])
                                      + zs["n_maize"](rec["n_maize"])
                                      + zs["frac_small"](rec["frac_small"]), 4)

    selection = []
    rng = random.Random(SEED)
    for stream in sorted(by_stream):
        recs = sorted(by_stream[stream], key=lambda r: r["name"])
        rand = sorted(rng.sample(recs, RANDOM_N), key=lambda r: r["name"])
        hard = sorted(recs, key=lambda r: (-r["hard_score"], r["name"]))[:HARD_N]
        for typ, group in (("random20", rand), ("hard10", hard)):
            for rec in group:
                rec["sample_type"] = typ
        selection += [{"image": r["name"], "stream": stream,
                       "sample_type": r["sample_type"],
                       "hard_score": r["hard_score"],
                       "n_weed": r["n_weed"], "n_maize": r["n_maize"],
                       "frac_small": r["frac_small"],
                       "sha256_image": sha256_file(
                           os.path.join(data_dir, r["name"]))}
                      for r in rand + hard]

    # review panels (full resolution, outside git)
    for rec in selection:
        src = os.path.join(data_dir, rec["image"])
        out = os.path.join(PAN_DIR, rec["stream"], rec["image"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with Image.open(src) as im:
            stem = os.path.splitext(rec["image"])[0]
            lp = os.path.join(data_dir, stem + ".txt")
            draw_panel(im.copy(), parse_yolo(lp), stem).save(out, quality=88)
    print(f"panels: {len(selection)} -> {PAN_DIR}")

    with open(os.path.join(REV_DIR, "sample_selection.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(selection[0].keys()))
        w.writeheader()
        w.writerows(selection)
    sel_sha = sha256_file(os.path.join(REV_DIR, "sample_selection.csv"))

    with open(os.path.join(REV_DIR, "review_template.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "stream", "sample_type", "n_weed", "n_maize",
                    "issue_category", "notes", "reviewer", "review_date"])
        for rec in selection:
            w.writerow([rec["image"], rec["stream"], rec["sample_type"],
                        rec["n_weed"], rec["n_maize"], "", "", "", ""])

    sheets = {}
    for stream in sorted(by_stream):
        recs = [r for r in selection if r["stream"] == stream]
        for typ, cols in (("random20", 4), ("hard10", 2)):
            paths = [os.path.join(PAN_DIR, stream, r["image"])
                     for r in recs if r["sample_type"] == typ]
            out = os.path.join(SHEETS_DIR, f"{stream}_{typ}.jpg")
            sheets[f"{stream}_{typ}"] = contact_sheet(
                paths, cols, out,
                f"{stream}  {typ}  (RED=Weeds(0) BLUE=Maize(1))")

    with open(os.path.join(REV_DIR, "sample_hashes.json"), "w",
              encoding="utf-8") as f:
        import json
        json.dump({"sample_selection_sha256": sel_sha, "seed": SEED,
                   "contact_sheets": sheets,
                   "panels_dir": "runs/EXP-001C/annotation_panels/<stream>/"},
                 f, indent=2)
    print(f"sample_selection sha256: {sel_sha}")
    print(f"selection rows: {len(selection)}")


if __name__ == "__main__":
    main()
