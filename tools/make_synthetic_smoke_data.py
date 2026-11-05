"""Deterministic synthetic 'weed' dataset for EXP-001 pipeline smoke tests.

PLACEHOLDER ONLY. The real CornWeed/Weed-crop V2 datasets are not yet present
under <WORKSPACE>/datasets, so a tiny synthetic
one-class dataset is generated to validate the train/val/save pipeline.
It is NOT used for any accuracy claim. Same fixed split for every model.

Output: <WORKSPACE>/datasets_derived/smoke_weed_v1
"""
import hashlib
import json
import os

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(REPO), "datasets_derived", "smoke_weed_v1")
SEED = 42
W, H = 640, 640
N_TRAIN, N_VAL = 24, 8


def make_image(rng):
    # soil-like background + crop-row stripes + darker blobs (weeds)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[..., 0] = rng.randint(80, 110, (H, W))   # R
    img[..., 1] = rng.randint(90, 120, (H, W))   # G
    img[..., 2] = rng.randint(50, 70, (H, W))    # B
    for x in range(80, W, 160):                  # crop rows (lighter green bands)
        img[:, x : x + 24, 1] = np.clip(img[:, x : x + 24, 1] + 60, 0, 255)
        img[:, x : x + 24, 0] = np.clip(img[:, x : x + 24, 0] - 20, 0, 255)
    boxes = []
    for _ in range(rng.randint(3, 9)):
        bw, bh = rng.randint(20, 90), rng.randint(20, 90)
        x1, y1 = rng.randint(0, W - bw), rng.randint(0, H - bh)
        # dark irregular weed blob
        img[y1 : y1 + bh, x1 : x1 + bw, 0] = rng.randint(20, 50)
        img[y1 : y1 + bh, x1 : x1 + bw, 1] = rng.randint(60, 110)
        img[y1 : y1 + bh, x1 : x1 + bw, 2] = rng.randint(20, 45)
        boxes.append((x1, y1, x1 + bw, y1 + bh))
    return img, boxes


def yolo_line(b):
    x1, y1, x2, y2 = b
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    bw, bh = (x2 - x1) / W, (y2 - y1) / H
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    rng = np.random.RandomState(SEED)
    manifest = {}
    for split, n in (("train", N_TRAIN), ("val", N_VAL)):
        idir = os.path.join(OUT, "images", split)
        ldir = os.path.join(OUT, "labels", split)
        os.makedirs(idir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        for i in range(n):
            img, boxes = make_image(rng)
            name = f"{split}_{i:04d}"
            ipath = os.path.join(idir, name + ".png")
            Image.fromarray(img).save(ipath)
            with open(os.path.join(ldir, name + ".txt"), "w") as f:
                f.write("\n".join(yolo_line(b) for b in boxes))
            h = hashlib.sha256()
            with open(ipath, "rb") as f:
                h.update(f.read())
            manifest[f"{split}/{name}.png"] = h.hexdigest()

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(
            f"path: {OUT}\ntrain: images/train\nval: images/val\n"
            "names:\n  0: weed\nnc: 1\n"
        )

    mh = hashlib.sha256()
    for k in sorted(manifest):
        mh.update(f"{k}:{manifest[k]}\n".encode())
    meta = {
        "note": "SYNTHETIC placeholder dataset; not real field imagery. "
                "Used only for pipeline smoke tests pending real dataset audit.",
        "seed": SEED, "n_train": N_TRAIN, "n_val": N_VAL,
        "image_size": [W, H], "classes": ["weed"],
        "manifest_sha256": mh.hexdigest(), "files": manifest,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {OUT}; manifest_sha256={mh.hexdigest()}")


if __name__ == "__main__":
    main()
