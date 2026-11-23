"""Generate shards.tsv for preprocess_synthtab_sharded.sh from the staged raw S3.

Lists every per-synth zip under the raw prefix and maps it to (effect, synth):
  <prefix>/acoustic/<synth>.zip               -> acoustic / <synth>
  <prefix>/electric/<effect>/<synth>.zip      -> <effect> / <synth>
i.e. effect = the zip's parent folder, synth = the zip's basename. Skips
jams.zip and the allpaths_*.zip bundles. Uses the instance's AWS creds (boto3).

    python scripts/make_shards.py s3://<bucket>/synthtab-raw > shards.tsv
"""

import argparse
import sys
from urllib.parse import urlparse

import boto3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_uri", help="s3://bucket/prefix of the staged raw zips")
    args = ap.parse_args()

    u = urlparse(args.raw_uri)
    bucket, prefix = u.netloc, u.path.strip("/")
    s3 = boto3.client("s3")

    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket,
                                                             Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".zip"):
                continue
            parts = key[len(prefix):].strip("/").split("/")
            base = parts[-1]
            if base == "jams.zip" or base.startswith("allpaths"):
                continue
            if len(parts) < 2:
                print(f"# skip (no effect folder): {key}", file=sys.stderr)
                continue
            effect, synth = parts[-2], base[:-len(".zip")]
            print(f"s3://{bucket}/{key}\t{effect}\t{synth}")
            n += 1
    print(f"# {n} synth zips", file=sys.stderr)


if __name__ == "__main__":
    main()
