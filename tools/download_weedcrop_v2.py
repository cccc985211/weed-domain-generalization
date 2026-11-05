"""EXP-001C Task3: Weed-crop V2 (Mendeley mthv4ppwyw v2) file-by-file downloader.

The official "Download All" zip lives on an S3 bucket that denies anonymous
GETs (403 AccessDenied), so this tool uses the per-file public endpoints that
the dataset page itself exposes:
  folder tree : GET https://data.mendeley.com/public-api/datasets/mthv4ppwyw/folders/2
  file listing: GET https://data.mendeley.com/public-api/datasets/mthv4ppwyw/files
                ?folder_id=<id>&version=2&$start=<n>&$limit=1000
  download    : content_details.download_url of each file entry
Every entry ships the publisher's sha256_hash; each downloaded file is verified
against it, and a download_manifest.json records source/version/hash provenance.

curl is used instead of urllib because Windows schannel curl needs
--ssl-no-revoke (revocation servers unreachable on this network) and urllib's
OpenSSL path is repeatedly reset.
"""
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://data.mendeley.com"
DATASET = "mthv4ppwyw"
VERSION = 2
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(REPO), "datasets", "WeedCropV2")
WORKERS = 16
MAX_ATTEMPTS = 8


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def curl_text(url):
    r = subprocess.run(
        ["curl", "-sS", "--ssl-no-revoke", "--fail", "--show-error",
         "--retry", "8", "--retry-all-errors", "--retry-delay", "4",
         "--connect-timeout", "30", "--max-time", "300", url],
        capture_output=True, text=True, check=True)
    return r.stdout


def list_folder_files(fid):
    out, start = [], 0
    while True:
        url = (f"{BASE}/public-api/datasets/{DATASET}/files"
               f"?folder_id={fid}&version={VERSION}&$start={start}&$limit=1000")
        batch = json.loads(curl_text(url))
        if not batch:
            return out
        out += batch
        start += len(batch)
        if start > 20000:
            raise SystemExit(f"folder {fid}: pagination runaway")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def download_one(entry, folder_name):
    name = entry["filename"]
    out_dir = os.path.join(OUT_DIR, folder_name)
    out = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    url = entry["download_url"]
    expect = entry["sha256"]
    size = entry["size"]
    if os.path.exists(out) and os.path.getsize(out) == size \
            and sha256_file(out) == expect:
        return name, "cached"
    for attempt in range(MAX_ATTEMPTS):
        try:
            subprocess.run(
                ["curl", "-sSL", "--ssl-no-revoke", "--fail", "--show-error",
                 "--retry", "4", "--retry-all-errors", "--retry-delay", "4",
                 "--connect-timeout", "30", "--max-time", "900",
                 "-o", out, url],
                check=True, capture_output=True, text=True)
            if os.path.getsize(out) != size:
                raise IOError(f"size {os.path.getsize(out)} != {size}")
            if sha256_file(out) != expect:
                raise IOError("sha256 mismatch")
            return name, "downloaded"
        except Exception as e:  # noqa: BLE001 - retry
            log(f"{folder_name}/{name} attempt {attempt + 1}: {e}")
            if os.path.exists(out):
                os.remove(out)
            time.sleep(min(2 ** attempt * 2, 30))
    raise IOError(f"{folder_name}/{name} failed after {MAX_ATTEMPTS} attempts")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    folders = json.loads(curl_text(
        f"{BASE}/public-api/datasets/{DATASET}/folders/{VERSION}"))
    log(f"folders: {len(folders)}")
    entries = []
    for f in folders:
        files = list_folder_files(f["id"])
        log(f"  {f['name']}: {len(files)} files")
        entries += [(f["name"].replace(" ", "_"), e) for e in files]
    log(f"total entries: {len(entries)}")

    listing = [{"folder": fn, "filename": e["filename"], "id": e["id"],
                "size": e["content_details"]["size"],
                "sha256": e["content_details"]["sha256_hash"],
                "download_url": e["content_details"]["download_url"],
                "status": e.get("status")}
               for fn, e in entries]
    with open(os.path.join(OUT_DIR, "file_list_publisher.json"), "w",
              encoding="utf-8") as f:
        json.dump({"dataset": DATASET, "version": VERSION,
                   "source_page": f"{BASE}/datasets/{DATASET}/{VERSION}",
                   "listing_url": f"{BASE}/public-api/datasets/{DATASET}/folders/{VERSION}",
                   "total_bytes": sum(e["size"] for e in listing),
                   "files": listing}, f, indent=1)
    log("file_list_publisher.json written")

    todo = [e for e in listing if e["status"] == "COMPLETED"]
    done = 0
    errors = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(download_one, e, e["folder"]): e for e in todo}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                _, how = fut.result()
                done += 1
                if done % 100 == 0:
                    log(f"{done}/{len(todo)} files ok ({how})")
            except Exception as exc:  # noqa: BLE001
                errors.append((e["folder"], e["filename"], str(exc)))
    log(f"downloaded/cached {done}/{len(todo)}; errors: {len(errors)}")
    if errors:
        for fn, name, msg in errors[:20]:
            log(f"FAILED {fn}/{name}: {msg}")
        raise SystemExit("download incomplete")

    # full re-verification pass over every file on disk
    bad = []
    for e in todo:
        p = os.path.join(OUT_DIR, e["folder"], e["filename"])
        if not os.path.exists(p) or sha256_file(p) != e["sha256"]:
            bad.append(e["folder"] + "/" + e["filename"])
    log(f"verification: {len(todo) - len(bad)} ok, {len(bad)} bad")
    if bad:
        raise SystemExit(f"verification failed: {bad[:10]}")

    with open(os.path.join(OUT_DIR, "download_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "dataset": DATASET, "version": VERSION,
            "source_page": f"{BASE}/datasets/{DATASET}/{VERSION}",
            "doi": "10.17632/mthv4ppwyw.2",
            "license": "CC BY 4.0",
            "zip_endpoint_note": ("Download-All S3 zip denies anonymous GET "
                                  "(403); files fetched individually and "
                                  "verified against publisher sha256."),
            "n_files": len(todo),
            "total_bytes": sum(e["size"] for e in todo),
            "all_files_verified": True,
            "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=1)
    log("download_manifest.json written; DONE")


if __name__ == "__main__":
    main()
