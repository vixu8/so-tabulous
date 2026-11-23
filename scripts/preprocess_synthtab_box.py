"""Fused SynthTab preprocessing straight from Box -- no raw S3 staging.

Walks the public Box share and, for each synth zip (~30 GB): downloads it, unzips,
runs src/data/preprocess_synthtab.py over just those songs, uploads the HDF5 +
manifest to S3, then deletes the raw. So the 2 TB never lands on disk (one zip at
a time) and never goes to S3 unprocessed. Resumable: a zip whose manifest is
already in S3 is skipped. Run scripts/merge_manifests.py at the end.

Layout is walked, not assumed:
  acoustic/<synth>/part_N_*.zip   -> effect=acoustic, synth=<folder>   (4 parts/synth)
  electric/<effect>/<synth>.zip   -> effect=<effect>,  synth=<zip stem> (1 zip/synth)
  allpaths_*.zip are skipped; the single jams zip at the root is fetched once.

Auth (env): BOX_SHARED_LINK, BOX_ROOT_FOLDER_ID, and a token source:
  BOX_TOKEN         a developer token (60 min). On 401 the run stops with a clear
                    message; regenerate it and re-run -- it resumes.
  auto-refresh (unattended), either:
    CCG:   BOX_CLIENT_ID, BOX_CLIENT_SECRET, BOX_ENTERPRISE_ID
    OAuth: BOX_CLIENT_ID, BOX_CLIENT_SECRET, BOX_REFRESH_TOKEN

Needs: requests, the aws CLI, and `unzip` on PATH.

    BOX_SHARED_LINK=https://rochester.app.box.com/v/SynthTab-Full \
    BOX_ROOT_FOLDER_ID=232326877483 BOX_TOKEN=... \
    python scripts/preprocess_synthtab_box.py \
        --s3 s3://<bucket>/features/synthtab --work /data/work --workers 32
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

API = "https://api.box.com/2.0"
REPO = Path(__file__).resolve().parents[1]
PREPROCESS = REPO / "src" / "data" / "preprocess_synthtab.py"


# ----- Box -----

class Box:
    def __init__(self, shared_link):
        self.shared_link = shared_link
        self._token, self._exp = None, 0.0

    def _refresh(self):
        cid, sec = os.environ.get("BOX_CLIENT_ID"), os.environ.get("BOX_CLIENT_SECRET")
        data = None
        if cid and sec and os.environ.get("BOX_ENTERPRISE_ID"):
            data = {"grant_type": "client_credentials", "client_id": cid, "client_secret": sec,
                    "box_subject_type": "enterprise",
                    "box_subject_id": os.environ["BOX_ENTERPRISE_ID"]}
        elif cid and sec and os.environ.get("BOX_REFRESH_TOKEN"):
            data = {"grant_type": "refresh_token", "client_id": cid, "client_secret": sec,
                    "refresh_token": os.environ["BOX_REFRESH_TOKEN"]}
        if data:
            j = requests.post("https://api.box.com/oauth2/token", data=data, timeout=30)
            j.raise_for_status()
            tok = j.json()
            self._token = tok["access_token"]
            self._exp = time.time() + tok.get("expires_in", 3600) - 300
            if "refresh_token" in tok and os.environ.get("BOX_REFRESH_TOKEN"):
                os.environ["BOX_REFRESH_TOKEN"] = tok["refresh_token"]   # rotated
        elif os.environ.get("BOX_TOKEN"):
            self._token, self._exp = os.environ["BOX_TOKEN"], time.time() + 3300
        else:
            raise SystemExit("set BOX_TOKEN or client creds (see module docstring)")

    def _headers(self):
        if not self._token or time.time() >= self._exp:
            self._refresh()
        return {"Authorization": f"Bearer {self._token}",
                "BoxApi": f"shared_link={self.shared_link}"}

    @staticmethod
    def _check_401(r):
        if r.status_code == 401:
            raise SystemExit("Box 401 (token expired?). Regenerate BOX_TOKEN and re-run; "
                             "it resumes from where it left off.")

    def list_folder(self, folder_id):
        items, offset = [], 0
        while True:
            r = requests.get(f"{API}/folders/{folder_id}/items", headers=self._headers(),
                             params={"fields": "id,name,size,type", "limit": 1000,
                                     "offset": offset}, timeout=60)
            self._check_401(r)
            r.raise_for_status()
            j = r.json()
            items += j["entries"]
            offset += len(j["entries"])
            if offset >= j["total_count"] or not j["entries"]:
                break
        return items

    def download(self, file_id, dest):
        for attempt in range(3):
            try:
                with requests.get(f"{API}/files/{file_id}/content", headers=self._headers(),
                                  stream=True, timeout=120, allow_redirects=True) as r:
                    self._check_401(r)
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8 << 20):
                            f.write(chunk)
                return
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))


# ----- planning: walk the tree into (effect, synth, zip) shards -----

def plan(box, root_id):
    """Return (shards, jams_file_id). shard = dict(effect, synth, name, file_id)."""
    shards, jams_id = [], None
    stack = [(root_id, [])]
    while stack:
        fid, path = stack.pop()
        for it in box.list_folder(fid):
            if it["type"] == "folder":
                stack.append((it["id"], path + [it["name"]]))
                continue
            name = it["name"]
            if not name.endswith(".zip") or name.startswith("allpaths"):
                continue
            if not path and "jams" in name.lower():
                jams_id = it["id"]
                continue
            stem = name[:-len(".zip")]
            if name.startswith("part_") and len(path) >= 2:
                effect, synth = path[-2], path[-1]       # acoustic/<synth>/part_N.zip
            else:
                effect, synth = path[-1], stem           # electric/<effect>/<synth>.zip
            shard_name = re.sub(r"[^0-9A-Za-z]+", "_", f"{effect}__{synth}__{stem}").strip("_")
            shards.append({"effect": effect, "synth": synth, "name": shard_name,
                           "file_id": it["id"], "size": it["size"]})
    if jams_id is None:
        raise SystemExit("no jams zip found at the share root")
    shards.sort(key=lambda s: s["name"])
    return shards, jams_id


# ----- S3 (via aws cli) -----

def s3_exists(uri):
    return subprocess.run(["aws", "s3", "ls", uri], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def s3_cp(src, uri):
    subprocess.run(["aws", "s3", "cp", str(src), uri], check=True)


def s3_sync(src, uri):
    subprocess.run(["aws", "s3", "sync", str(src), uri], check=True)


# ----- run -----

def run_shard(box, shard, jams_root, s3, work, workers):
    manifest_uri = f"{s3}/manifests/{shard['name']}.csv"
    if s3_exists(manifest_uri):
        print(f"skip {shard['name']} (manifest in S3)")
        return
    print(f"== {shard['name']} ({shard['size'] / 1e9:.1f} GB) ==", flush=True)

    # Fresh root holding ONLY this shard's audio (+ jams), so preprocess can't pick
    # up leftovers from a prior/crashed shard. rmtree unlinks the jams symlink, not
    # its target. jams must sit at <root>/jams for preprocess_synthtab to find it.
    root, out = work / "raw", work / "out"
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    dest = root / shard["effect"] / shard["synth"]
    dest.mkdir(parents=True)
    (root / "jams").symlink_to(jams_root, target_is_directory=True)

    zpath = work / "shard.zip"
    box.download(shard["file_id"], zpath)
    subprocess.run(["unzip", "-q", str(zpath), "-d", str(dest)], check=True)
    zpath.unlink()
    # VERIFY on the first shard: dest should now hold <song>/<song>/..._body.flac

    subprocess.run([sys.executable, str(PREPROCESS), "--synthtab_root", str(root),
                    "--out_dir", str(out), "--no_kong_events", "--workers", str(workers)],
                   check=True)

    s3_sync(out / "hdf5s", f"{s3}/hdf5s")
    s3_cp(out / "manifest.csv", manifest_uri)
    if (out / "stats.json").exists():
        s3_cp(out / "stats.json", f"{s3}/stats/{shard['name']}.json")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s3", required=True, help="s3://bucket/prefix for the processed output")
    ap.add_argument("--work", default="/data/work")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--dry-run", action="store_true", help="just print the shard plan")
    args = ap.parse_args()
    s3 = args.s3.rstrip("/")
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    box = Box(os.environ["BOX_SHARED_LINK"])
    shards, jams_id = plan(box, os.environ["BOX_ROOT_FOLDER_ID"])
    total_gb = sum(s["size"] for s in shards) / 1e9
    print(f"{len(shards)} shards, {total_gb:.0f} GB raw total")
    if args.dry_run:
        for s in shards:
            print(f"  {s['name']:60s} {s['size']/1e9:6.1f} GB  ({s['effect']}/{s['synth']})")
        return

    jams_root = (work / "jams_extracted").resolve()
    if not jams_root.exists():
        print("== jams (one-time) ==", flush=True)
        jz = work / "jams.zip"
        box.download(jams_id, jz)
        jams_root.mkdir()
        subprocess.run(["unzip", "-q", str(jz), "-d", str(jams_root)], check=True)
        jz.unlink()
        # jams zip may wrap its songs in a top-level folder; expose it as <root>/jams.
        inner = [p for p in jams_root.iterdir() if p.is_dir() and p.name == "jams"]
        if inner:
            jams_root = inner[0].resolve()

    for shard in shards:
        run_shard(box, shard, jams_root, s3, work, args.workers)

    print(f"\nall {len(shards)} shards done. Merge:")
    print(f"  aws s3 sync {s3}/manifests ./manifests")
    print(f"  python scripts/merge_manifests.py ./manifests --out manifest.csv --stats-dir ./stats")
    print(f"  aws s3 cp manifest.csv {s3}/manifest.csv")


if __name__ == "__main__":
    main()
