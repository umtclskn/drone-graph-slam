#!/bin/bash
# Offline unit test for run_comparison.sh RUN-MODE TRANSPARENCY (AC-6 a-d).
# Sources the driver (main is guarded, so nothing launches) and exercises the
# pure transparency functions against synthetic reports — no ROS, no bag.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/run_comparison.sh"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0
check() { if eval "$2"; then echo "  PASS: $1"; else echo "  FAIL: $1"; fail=1; fi; }

echo "AC-6(a) mode detection"
LIO_SAM_WS=$TMP/nope LIO_RUN=$TMP/nope/scripts/run_drone_bag.sh
check "missing workspace => OURS_ONLY" '[ "$(detect_mode)" = OURS_ONLY ]'
# build a fake FULL workspace
mkdir -p "$TMP/lio/scripts" "$TMP/lio/install"
: > "$TMP/lio/scripts/run_drone_bag.sh"; chmod +x "$TMP/lio/scripts/run_drone_bag.sh"
LIO_SAM_WS=$TMP/lio LIO_RUN=$TMP/lio/scripts/run_drone_bag.sh
check "present workspace => FULL" '[ "$(detect_mode)" = FULL ]'

echo "AC-6(a) start banner wording"
LIO_SAM_WS=$TMP/nope BAG=bags/slam_loop_03 OUT=$TMP/o \
  check "OURS_ONLY banner says skipped" \
  'start_banner OURS_ONLY | grep -q "MODE: OURS ONLY .* LIO-SAM skipped"'
LIO_SAM_WS=$TMP/lio \
  check "FULL banner says full comparison" \
  'start_banner FULL | grep -q "MODE: FULL COMPARISON (ours + LIO-SAM)"'

echo "AC-6(b)(c) OURS_ONLY report has banner and NO lio row"
OUT=$TMP/oo; mkdir -p "$OUT"
# synthetic single-row report as slam_eval would leave it (ours only)
{ echo "# SLAM evaluation report"; echo
  echo "| System | ATE trans RMSE [m] |"; echo "|---|---|"
  echo "| ours_graph_slam | 0.169 |"; } > "$OUT/report.md"
finalize_report OURS_ONLY > /dev/null
check "banner prepended to report" 'grep -q "this is NOT a comparison" "$OUT/report.md"'
check "no lio_sam row present"     '! grep -qi "lio_sam" "$OUT/report.md"'
check "ours row still present"     'grep -q "ours_graph_slam" "$OUT/report.md"'

echo "AC-6(b) FULL report gets THIS-run LIO provenance (sha256 + loops + bag)"
OUT=$TMP/ff; mkdir -p "$OUT"
{ echo "# SLAM evaluation report"; echo
  echo "| System | ATE trans RMSE [m] |"; echo "|---|---|"
  echo "| ours_graph_slam | 0.169 |"; echo "| lio_sam | 0.053 |"; } > "$OUT/report.md"
BAG=bags/slam_loop_03 LIO_PARAMS_SHA=efddfe0fdeadbeef LIO_LOOPS=1 \
  LIO_PARAMS=/x/params.yaml finalize_report FULL > /dev/null
check "footer carries params sha256" 'grep -q "efddfe0fdeadbeef" "$OUT/report.md"'
check "footer carries loop count"    'grep -q "loop closures (this run): 1" "$OUT/report.md"'
check "footer carries bag"           'grep -q "bag: .bags/slam_loop_03" "$OUT/report.md"'
check "both rows still present"      'grep -q "ours_graph_slam" "$OUT/report.md" && grep -q "lio_sam" "$OUT/report.md"'

echo "AC-6(c) no stale row survives: a pre-existing lio row must be cleared before scoring"
# Simulate a dir left over from a previous FULL session, then an OURS_ONLY run
# reaching the point where main wipes metrics.json/report.md.
OUT=$TMP/stale; mkdir -p "$OUT"
echo '{"systems":{"lio_sam":{"ate":{"trans_rmse_m":0.053}}}}' > "$OUT/metrics.json"
echo "| lio_sam | 0.053 |" > "$OUT/report.md"
# main() clears these before any scoring (guarded run below stops at missing GT).
GT_TUM=$TMP/does_not_exist.tum LIO_SAM_WS=$TMP/nope OUT=$OUT \
  bash "$HERE/run_comparison.sh" > /dev/null 2>&1
check "stale metrics.json cleared" '[ ! -f "$OUT/metrics.json" ]'
check "stale report.md cleared"    '[ ! -f "$OUT/report.md" ]'

echo "AC-6(d) exit summary states which systems ran"
check "OURS_ONLY summary" 'exit_summary OURS_ONLY | grep -q "ran OURS ONLY .* NOT a comparison"'
LIO_LOOPS=1 check "FULL summary" 'exit_summary FULL | grep -q "ran ours + LIO-SAM"'

echo
[ "$fail" = 0 ] && echo "ALL AC-6 TRANSPARENCY CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
