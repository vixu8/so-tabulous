"""Combine per-shard SynthTab manifests into one, and sum their label stats.

Each shard (one synth) wrote its own manifest.csv + stats.json. The clips have
globally unique track_ids (hash of effect+synth+song), so merging is just a
concat -- no clip is reprocessed. Produces the single manifest.csv the loader
reads (filter by its `split` column for train vs val).

    aws s3 sync s3://<bucket>/features/synthtab/manifests ./manifests
    aws s3 sync s3://<bucket>/features/synthtab/stats     ./stats        # optional
    python scripts/merge_manifests.py ./manifests --out manifest.csv --stats-dir ./stats
    aws s3 cp manifest.csv s3://<bucket>/features/synthtab/manifest.csv
"""

import argparse
import glob
import json
import os
from collections import Counter

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests_dir", help="folder of per-shard manifest CSVs")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--stats-dir", default=None, help="folder of per-shard stats JSONs")
    ap.add_argument("--stats-out", default="stats.json")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.manifests_dir, "*.csv")))
    if not csvs:
        raise SystemExit(f"no CSVs in {args.manifests_dir}")
    frames = []
    for c in csvs:
        df = pd.read_csv(c, dtype={"track_id": str})
        df["_shard"] = os.path.basename(c)          # remember each row's source manifest
        frames.append(df)
    m = pd.concat(frames, ignore_index=True)

    dup_mask = m.track_id.duplicated(keep=False)
    if dup_mask.any():
        dup_ids = m.loc[dup_mask, "track_id"].drop_duplicates()
        print(f"{len(dup_ids)} track_ids are duplicated across shards; first few and the "
              "shards they appear in:")
        for tid in dup_ids.head(5):
            shards = sorted(m.loc[m.track_id == tid, "_shard"])
            print(f"  {tid}: {shards}")
        raise SystemExit(f"{len(dup_ids)} duplicate track_ids across shards (unexpected -- "
                         "track_id should be globally unique; likely two shards' --filter "
                         "regexes overlapped and preprocessed the same clips)")
    m = m.drop(columns="_shard")
    # No Guitar Pro file may span splits (same check the preprocessor runs per shard).
    import re

    def gp(s):
        hit = re.match(r"(.*- gp\d+)__\d+", s)
        return hit.group(1) if hit else s

    span = m.assign(_gp=m.tab_id.map(gp)).groupby("_gp").split.nunique()
    leaked = span[span > 1]
    if len(leaked):
        raise SystemExit(f"{len(leaked)} song file(s) span multiple splits, e.g. "
                         f"{leaked.index[0]!r}")

    m.to_csv(args.out, index=False)
    print(f"merged {len(csvs)} manifests -> {args.out}: {len(m)} clips, "
          f"splits {m.split.value_counts().to_dict()}, "
          f"{m.duration.sum() / 3600:.1f} audio-hours")

    if args.stats_dir:
        total = Counter()
        jsons = glob.glob(os.path.join(args.stats_dir, "*.json"))
        for s in jsons:
            total.update(json.load(open(s)).get("label_stats", {}))
        json.dump({"label_stats": dict(total), "shards": len(csvs), "clips": len(m)},
                  open(args.stats_out, "w"), indent=2)
        print(f"summed {len(jsons)} stats -> {args.stats_out}: {dict(total)}")


if __name__ == "__main__":
    main()
