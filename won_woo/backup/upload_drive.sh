#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${1:?Usage: bash backup/upload_drive.sh gdrive:won_woo_backup_20261009/files [--dry-run]}
shift
command -v rclone >/dev/null || { echo 'Install rclone and configure a Google Drive remote first.' >&2; exit 1; }
[[ -f "$ROOT/backup/drive_files.txt" ]] || { echo 'Run backup_workspace.py prepare first.' >&2; exit 1; }
mkdir -p "$ROOT/.backup_work"
rclone copy "$ROOT" "$DEST" --files-from-raw "$ROOT/backup/drive_files.txt" --checksum --transfers 2 --checkers 4 --drive-chunk-size 64M --retries 8 --stats 30s --log-file "$ROOT/.backup_work/drive-upload.log" "$@"
if [[ " $* " != *" --dry-run "* ]]; then
  rclone check "$ROOT" "$DEST" --files-from-raw "$ROOT/backup/drive_files.txt" --one-way --checkers 4 --log-file "$ROOT/.backup_work/drive-check.log" "$@"
fi
