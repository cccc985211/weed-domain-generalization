"""EXP-001B Task1 (v2): segmented resumable CornWeed download.

Official Zenodo only. 6 parallel Range streams -> per-segment files ->
concatenate -> MD5 verify. Segment state survives restarts (.seg json).
"""
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

URL = "https://zenodo.org/records/7961764/files/maize_dataset.zip?download=1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(REPO), "datasets")
FINAL = os.path.join(OUT_DIR, "maize_dataset.zip")
EXPECTED_MD5 = "9b409e835b51c0f7b01524dcb1dfeda6"
EXPECTED_SIZE = 8693397534
N_SEG = 8
CHUNK = 1 << 20
STATE = os.path.join(OUT_DIR, ".cornweed_segments.json")
LOCK = threading.Lock()
PROGRESS = {"done": 0}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def seg_path(i):
    return os.path.join(OUT_DIR, f"maize_dataset.seg{i:02d}")


def worker(i, start, end):
    target = seg_path(i)
    while True:
        have = os.path.getsize(target) if os.path.exists(target) else 0
        if have >= end - start + 1:
            return
        headers = {"User-Agent": "exp001-audit/1.0",
                   "Range": f"bytes={start + have}-{end}"}
        try:
            r = urllib.request.urlopen(urllib.request.Request(URL, headers=headers),
                                       timeout=90)
            if r.status != 206:
                raise IOError(f"status {r.status}")
            with open(target, "ab") as f:
                while True:
                    c = r.read(CHUNK)
                    if not c:
                        break
                    f.write(c)
                    with LOCK:
                        PROGRESS["done"] += len(c)
        except Exception as e:  # noqa: BLE001
            log(f"seg{i} retry: {type(e).__name__}")
            time.sleep(10)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bounds = []
    per = EXPECTED_SIZE // N_SEG
    for i in range(N_SEG):
        s = i * per
        e = EXPECTED_SIZE - 1 if i == N_SEG - 1 else (i + 1) * per - 1
        bounds.append((i, s, e))
    done0 = sum(os.path.getsize(seg_path(i)) if os.path.exists(seg_path(i)) else 0
                for i, s, e in bounds)
    PROGRESS["done"] = done0
    log(f"start: {done0/1e9:.2f}/{EXPECTED_SIZE/1e9:.2f}GB already present")

    threads = [threading.Thread(target=worker, args=b, daemon=True)
               for b in bounds]
    for t in threads:
        t.start()

    last, t_last = done0, time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        with LOCK:
            d = PROGRESS["done"]
        rate = (d - last) / max(time.time() - t_last, 1)
        log(f"{d/1e9:.2f}GB {rate/1e6:.1f}MB/s")
        last, t_last = d, time.time()

    total = sum(os.path.getsize(seg_path(i)) for i, s, e in bounds)
    if total != EXPECTED_SIZE:
        log(f"FAILED size {total} != {EXPECTED_SIZE}")
        sys.exit(1)
    log("all segments complete; concatenating...")
    with open(FINAL + ".part", "wb") as out:
        for i, s, e in bounds:
            with open(seg_path(i), "rb") as f:
                for c in iter(lambda: f.read(1 << 22), b""):
                    out.write(c)
    os.replace(FINAL + ".part", FINAL)
    h = hashlib.md5()
    with open(FINAL, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    m = h.hexdigest()
    log(f"md5={m} expected={EXPECTED_MD5} match={m == EXPECTED_MD5}")
    if m == EXPECTED_MD5:
        for i, s, e in bounds:
            os.remove(seg_path(i))
        log("segments cleaned; DONE")
        sys.exit(0)
    sys.exit(2)


if __name__ == "__main__":
    main()
