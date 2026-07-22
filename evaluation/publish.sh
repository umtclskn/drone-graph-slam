#!/usr/bin/env bash
# Publish a curated snapshot of the shared eval output into the committed repo.
# Copies report.md, metrics.json and all PNGs from the shared analysis output
# dir (default: <workspace>/analysis/slam_eval) into evaluation/results/.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_dir="${1:-$script_dir/../../../analysis/slam_eval}"
dst_dir="$script_dir/results"

if [[ ! -f "$src_dir/report.md" ]]; then
    echo "publish.sh: no report.md in $src_dir — run slam_eval first" >&2
    exit 1
fi

mkdir -p "$dst_dir"
cp "$src_dir/report.md" "$src_dir/metrics.json" "$dst_dir/"
cp "$src_dir"/*.png "$dst_dir/" 2>/dev/null || true
echo "published $(ls "$dst_dir" | wc -l) files to $dst_dir"
