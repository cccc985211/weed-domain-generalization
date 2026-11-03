"""EXP-001B Task1: resumable CornWeed download + MD5 verify.

Source: official Zenodo record 7961764 (CC-BY-4.0). No third-party mirrors.
Part file -> final name only on byte-complete transfer; MD5 checked after.
"""
import hashlib
import os
import sys
import time
import urllib.request

URL = "https://zenodo.org/records/7961764/files/maize_dataset.zip?download=1"
OUT_DIR = r"C:\Users\gass\Documents\experiment\datasets"
FINAL = os.path.join(OUT_DIR, "maize_dataset.zip")
PART = FINAL + ".part"
EXPECTED_MD5 = "9b409e835b51c0f7b01524dcb1dfeda6"
EXPECTED_SIZE = 8693397534
MAX_ATTEMPTS = 40
CHUNK = 1 << 22  # 4 MiB


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def download():
    os.makedirs(OUT_DIR, exist_ok=True)
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        have = os.path.getsize(PART) if os.path.exists(PART) else 0
        if have >= EXPECTED_SIZE:
            break
        headers = {"User-Agent": "exp001-audit/1.0"}
        mode = "ab"
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(URL, headers=headers)
            r = urllib.request.urlopen(req, timeout=120)
            if have:
                cr = r.headers.get("Content-Range", "")
                start = int(cr.split()[1].split("-")[0]) if cr.startswith("bytes") else -1
                if r.status != 206 or start != have:
                    log(f"bad resume (status={r.status} cr='{cr}'); restart from 0")
                    have, mode = 0, "wb"
            log(f"attempt {attempt}: resume at {have/1e9:.2f}GB "
                f"status={r.status}")
            t0, done = time.time(), have
            with open(PART, mode + ("" if "b" in mode else "b")) as f:
                while True:
                    c = r.read(CHUNK)
                    if not c:
                        break
                    f.write(c)
                    done += len(c)
                    if done - have > 0 and (done // (512 << 20)) != ((done - len(c)) // (512 << 20)):
                        el = max(time.time() - t0, 1)
                        log(f"{done/1e9:.2f}/{EXPECTED_SIZE/1e9:.2f}GB "
                            f"{(done-have)/el/1e6:.0f}MB/s")
            if os.path.getsize(PART) == EXPECTED_SIZE:
                break
        except Exception as e:  # noqa: BLE001
            log(f"attempt {attempt} failed: {type(e).__name__}: {e}")
            time.sleep(min(30 * attempt, 120))
    size = os.path.getsize(PART) if os.path.exists(PART) else 0
    if size != EXPECTED_SIZE:
        log(f"FAILED: incomplete {size}/{EXPECTED_SIZE}")
        sys.exit(1)
    os.replace(PART, FINAL)
    log("download complete; verifying md5 ...")
    m = md5_file(FINAL)
    log(f"md5={m} expected={EXPECTED_MD5} match={m == EXPECTED_MD5}")
    sys.exit(0 if m == EXPECTED_MD5 else 2)


if __name__ == "__main__":
    download()
