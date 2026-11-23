#!/usr/bin/env bash
# Phase 0: stage the raw SynthTab (public Box share) to S3, so later phases pull
# from S3 (same-region = fast + free) instead of Box.
#
# rclone streams remote-to-remote file by file, so this needs NO large local
# disk for the 2 TB -- each zip passes through in turn. Resumable/idempotent:
# rclone skips files already present with the same size.
#
# Prereqs on the (small) staging instance:
#   - rclone installed (curl https://rclone.org/install.sh | sudo bash)
#   - a `box` remote (public SynthTab folder added to your Box account; headless
#     OAuth via `rclone authorize "box"` on a laptop, token pasted into config)
#   - an S3 remote using the instance IAM role, e.g. in ~/.config/rclone/rclone.conf:
#         [s3]
#         type = s3
#         provider = AWS
#         env_auth = true
#         region = <your-region>
#
# Usage:
#   ./scripts/stage_synthtab_box_to_s3.sh box:SynthTab s3:<bucket>/synthtab-raw
set -euo pipefail

SRC="${1:?rclone source, e.g. box:SynthTab}"
DST="${2:?rclone dest, e.g. s3:<bucket>/synthtab-raw}"

rclone copy "$SRC" "$DST" \
  --transfers "${TRANSFERS:-4}" --checkers 8 --fast-list \
  --progress --log-file "${LOG:-stage.log}" --log-level INFO

echo
echo "staged. verify sizes match:"
echo "  rclone size $SRC"
echo "  rclone size $DST"
