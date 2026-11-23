#!/usr/bin/env bash
# Sharded SynthTab preprocessing on EC2 (option B): one synth zip at a time, so
# the raw multi-TB dataset never has to land on disk all at once.
#
# Per shard: download one synth zip -> unzip -> preprocess_synthtab.py over just
# that synth (jams stay resident) -> upload HDF5 + this shard's manifest to S3 ->
# delete the raw + output to free disk. Resumable: a shard whose manifest is
# already in S3 is skipped. After all shards, run scripts/merge_manifests.py.
#
# Raw SynthTab is staged to S3 first (Box -> S3 via rclone), so this pulls each
# zip from S3 with `aws s3 cp` (same-region = free + fast).
#
# Usage:
#   JAMS_SRC=s3://<bucket>/synthtab-raw/jams.zip \
#   ./scripts/preprocess_synthtab_sharded.sh shards.tsv s3://<bucket>/features/synthtab [workdir]
#
#   shards.tsv: one tab-separated line per synth zip:  <s3-uri>\t<effect>\t<synth>
#     e.g.   s3://bucket/synthtab-raw/acoustic/gibson_thumb.zip<TAB>acoustic<TAB>gibson_thumb
#     '#' comment lines and blank lines are ignored. Do NOT list allpaths_*.zip
#     (those are the bundled alternative to the per-synth zips).
#
# Env knobs: WORKERS (default: all cores), PY (python), AWS (aws cli path).
set -euo pipefail

SHARDS="${1:?shards.tsv required}"
S3_DEST="${2:?s3://bucket/prefix required}"; S3_DEST="${S3_DEST%/}"
WORK="${3:-/data/work}"
PY="${PY:-.venv/bin/python}"
AWS="${AWS:-aws}"
WORKERS="${WORKERS:-$(nproc)}"

ROOT="$WORK/raw"
mkdir -p "$ROOT" "$WORK"

# ----- one-time: jams (labels), kept resident for every shard -----
if [ ! -d "$ROOT/jams" ]; then
  echo "== downloading jams =="
  $AWS s3 cp "${JAMS_SRC:?set JAMS_SRC to the jams zip s3 uri}" "$WORK/jams.zip"
  unzip -q "$WORK/jams.zip" -d "$ROOT"
  rm -f "$WORK/jams.zip"
  # VERIFY on first run: this must be a directory of per-song folders.
  [ -d "$ROOT/jams" ] || { echo "ERROR: expected $ROOT/jams/ after unzip; adjust the unzip dest"; exit 1; }
fi

# ----- per shard -----
while IFS=$'\t' read -r url effect synth _rest; do
  [ -z "${url:-}" ] && continue
  case "$url" in \#*) continue;; esac

  if $AWS s3 ls "$S3_DEST/manifests/${effect}_${synth}.csv" >/dev/null 2>&1; then
    echo "== skip $effect/$synth (manifest already in S3) =="
    continue
  fi
  echo "== shard $effect/$synth =="
  dest="$ROOT/$effect/$synth"
  out="$WORK/out"
  rm -rf "$dest" "$out"
  mkdir -p "$dest"

  $AWS s3 cp "$url" "$WORK/shard.zip"
  unzip -q "$WORK/shard.zip" -d "$dest"
  rm -f "$WORK/shard.zip"
  # VERIFY on the FIRST shard: iter_synthtab_tracks expects
  #   $ROOT/<effect>/<synth>/<song>/<song>/<synth>_..._body.flac
  # If the zip has an extra top-level folder, move it so the layout matches.

  "$PY" src/data/preprocess_synthtab.py --synthtab_root "$ROOT" \
        --out_dir "$out" --no_kong_events --workers "$WORKERS"

  $AWS s3 sync "$out/hdf5s" "$S3_DEST/hdf5s"
  $AWS s3 cp   "$out/manifest.csv" "$S3_DEST/manifests/${effect}_${synth}.csv"
  $AWS s3 cp   "$out/stats.json"   "$S3_DEST/stats/${effect}_${synth}.json" || true

  rm -rf "$dest" "$out"          # free disk; jams stays resident
done < "$SHARDS"

echo
echo "all shards processed. Now merge:"
echo "  $AWS s3 sync $S3_DEST/manifests ./manifests"
echo "  $AWS s3 sync $S3_DEST/stats     ./stats   # optional"
echo "  $PY scripts/merge_manifests.py ./manifests --out manifest.csv --stats-dir ./stats"
echo "  $AWS s3 cp manifest.csv $S3_DEST/manifest.csv"
