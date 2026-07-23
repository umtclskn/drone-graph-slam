#!/bin/bash
# EVAL-08 / AC-6 — ours-vs-LIO-SAM comparison driver with RUN-MODE TRANSPARENCY.
#
# One command that (a) detects and announces whether LIO-SAM can run, runs OURS
# always and LIO-SAM when available, scores both with the SAME slam_eval engine
# against the SAME ground truth, and (b)(c)(d) emits a report that states
# unambiguously which systems actually ran THIS invocation — never a copied
# historical row.
#
# It wraps the two already-validated replay scripts (the OURS scratch launch and
# ~/lio_sam_ws/scripts/run_drone_bag.sh); it invents no new estimation. A Release
# build is assumed (colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release).
#
# Transparency is the point of this file, so its logic lives in functions and the
# entrypoint is guarded (`main` runs only when executed, not when sourced) so the
# AC-6 guarantees can be unit-tested offline without launching ROS — see
# test_run_comparison_ac6.sh.
#
# Overridable env: BAG, OUT, LIO_SAM_WS, LIO_PARAMS, GT_TUM, GT_LABEL, FORCE.
set -u

WS=${WS:-/home/umut/portfolio_ws/drone_ws}
BAG=${BAG:-bags/slam_loop_03}
LIO_SAM_WS=${LIO_SAM_WS:-$HOME/lio_sam_ws}
LIO_RUN=$LIO_SAM_WS/scripts/run_drone_bag.sh
LIO_PARAMS=${LIO_PARAMS:-$LIO_SAM_WS/src/LIO-SAM/config/params_drone_x500.yaml}
EVAL_DIR=$WS/src/drone-graph-slam/evaluation
# Fresh, timestamped output dir per invocation: the structural guarantee behind
# AC-6(c) — a new dir cannot inherit a previous session's metrics.json row.
OUT=${OUT:-$WS/analysis/slam_eval_comparison_$(date -u +%Y%m%dT%H%M%SZ)}

# Filled by run_lio(); reported in the LIO row provenance footer (AC-6b).
LIO_PARAMS_SHA=""
LIO_LOOPS=""
LIO_RUN_DIR=""

# ---------------------------------------------------------------------------
# AC-6(a) — mode detection + start banner
# ---------------------------------------------------------------------------
detect_mode() {
  # FULL only if the workspace, its runnable launch script, and a built install
  # all exist. Any missing piece => OURS ONLY, never a half-run masquerading full.
  if [ -d "$LIO_SAM_WS" ] && [ -x "$LIO_RUN" ] && [ -d "$LIO_SAM_WS/install" ]; then
    echo FULL
  else
    echo OURS_ONLY
  fi
}

start_banner() {  # $1 = MODE
  echo "=================================================================="
  if [ "$1" = FULL ]; then
    echo "MODE: FULL COMPARISON (ours + LIO-SAM)"
    echo "  LIO-SAM workspace: $LIO_SAM_WS"
  else
    echo "MODE: OURS ONLY — LIO-SAM skipped ($LIO_SAM_WS not found)"
  fi
  echo "  bag:    $BAG"
  echo "  output: $OUT"
  echo "=================================================================="
}

# ---------------------------------------------------------------------------
# Reset hygiene (COMPARISON.md §2.5) — a stale node silently corrupts a run.
# ---------------------------------------------------------------------------
hygiene_check() {
  local pat="ndt_frontend_node|graph_backend_node|ground_truth_bridge|gazebo_truth_bridge|\
ekf2_odometry_adapter|lio_sam_|imu_bridge_liosam|odom_recorder|static_transform_publisher"
  local hits
  hits=$(pgrep -af "$pat" | grep -v "run_comparison" || true)
  if [ -n "$hits" ]; then
    echo "RESET HYGIENE: stale SLAM/GT processes are alive —"
    echo "$hits"
    if [ "${FORCE:-0}" = 1 ]; then
      echo "  FORCE=1 set — continuing anyway (not recommended)."
    else
      echo "  Kill them (by PID) and re-run, or set FORCE=1. Aborting." >&2
      return 1
    fi
  else
    echo "RESET HYGIENE: clean (no stale SLAM/GT processes)."
  fi
}

# ---------------------------------------------------------------------------
# System runs — thin wrappers over the validated replay scripts. These are the
# only steps that need a live ROS stack; the transparency logic above/below does
# not. Each writes its exports into the fresh $OUT so every scored row is from
# THIS invocation (AC-6c).
# ---------------------------------------------------------------------------
run_ours() {
  echo "--- OURS: replay + score ------------------------------------------"
  OUT="$OUT/ours_launch" BAG="$BAG" bash "$EVAL_DIR/comparison/ours/run_ours.sh" || return 1
  ( cd "$EVAL_DIR" && \
    python3 comparison/ours/export_from_eval05_csv.py \
      --csv "$WS/analysis/eval05_covariance_log.csv" --out "$OUT" && \
    python3 -m slam_eval --est "$OUT/ours_est.tum" --cov "$OUT/ours_est_cov.txt" \
      --gt "$GT_TUM" --out "$OUT" --name ours_graph_slam --align se3 \
      --bag "$BAG" --gt_label "$GT_LABEL" ) || return 1
}

run_lio() {
  echo "--- LIO-SAM: replay + score ---------------------------------------"
  LIO_PARAMS_SHA=$(sha256sum "$LIO_PARAMS" 2>/dev/null | awk '{print $1}')
  LIO_RUN_DIR=$WS/analysis/.lio_run_$(date -u +%Y%m%dT%H%M%SZ)
  BAG="$BAG" OUT="$LIO_RUN_DIR" \
    REC="$LIO_SAM_WS/scripts/odom_recorder_eval.py" bash "$LIO_RUN" || return 1
  LIO_LOOPS=$(grep -oE "loops [0-9]+" "$LIO_RUN_DIR/odom_recorder.log" 2>/dev/null \
              | tail -1 | awk '{print $2}')
  LIO_LOOPS=${LIO_LOOPS:-unknown}
  ( cd "$EVAL_DIR" && \
    python3 comparison/lio_sam/export_from_run_csv.py --run "$LIO_RUN_DIR" \
      --out "$OUT" --tag "$(date -u +%Y%m%d)" && \
    python3 -m slam_eval --est "$OUT"/lio_sam_est_*.tum \
      --gt "$GT_TUM" --out "$OUT" --name lio_sam --align se3 \
      --bag "$BAG" --gt_label "$GT_LABEL" ) || return 1
}

# ---------------------------------------------------------------------------
# AC-6(b)(c) — finalize the emitted report so it can only describe THIS run.
# ---------------------------------------------------------------------------
finalize_report() {  # $1 = MODE ; report.md already written by slam_eval
  local mode=$1 rep="$OUT/report.md"
  [ -f "$rep" ] || { echo "finalize_report: $rep missing" >&2; return 1; }
  if [ "$mode" = OURS_ONLY ]; then
    # No LIO row exists (fresh dir, only ours scored). Prepend the explicit
    # banner so the report itself declares it is not a comparison.
    local banner
    banner="> **LIO-SAM did not run in this session — this is NOT a comparison.**\n\
> See COMPARISON.md for the recorded comparison.\n"
    { printf "%b\n" "$banner"; cat "$rep"; } > "$rep.tmp" && mv "$rep.tmp" "$rep"
    printf "%b\n" "\nLIO-SAM did not run in this session — this is NOT a comparison.\n\
See COMPARISON.md for the recorded comparison."
  else
    # Both rows present. Attach this invocation's LIO provenance so its row is
    # traceable to this run, not a historical copy.
    { echo
      echo "## LIO-SAM run provenance (this invocation)"
      echo "- bag: \`$BAG\`"
      echo "- params: \`$LIO_PARAMS\`"
      echo "- params sha256: \`${LIO_PARAMS_SHA:-unknown}\`"
      echo "- loop closures (this run): ${LIO_LOOPS:-unknown}"
    } >> "$rep"
  fi
}

# ---------------------------------------------------------------------------
# AC-6(d) — one-line exit summary
# ---------------------------------------------------------------------------
exit_summary() {  # $1 = MODE
  echo "=================================================================="
  if [ "$1" = FULL ]; then
    echo "SUMMARY: ran ours + LIO-SAM (loops=${LIO_LOOPS:-?}); artifacts in $OUT"
  else
    echo "SUMMARY: ran OURS ONLY — LIO-SAM skipped; NOT a comparison; artifacts in $OUT"
  fi
  echo "=================================================================="
}

main() {
  # GT defaults are set here (not at top) so a sourced test controls them.
  GT_TUM=${GT_TUM:-$OUT/gt_gazebo.tum}
  GT_LABEL=${GT_LABEL:-"Gazebo model-state (independent), base_link"}

  local mode; mode=$(detect_mode)
  start_banner "$mode"

  mkdir -p "$OUT"
  # Belt-and-braces for AC-6(c): even under an OUT override, no prior row survives.
  rm -f "$OUT/metrics.json" "$OUT/report.md"

  hygiene_check || return 1

  if [ ! -f "$GT_TUM" ]; then
    echo "GT_TUM ($GT_TUM) not present — a live run must export the independent" >&2
    echo "Gazebo ground truth first (see COMPARISON.md §2.2 / §5). Aborting." >&2
    return 2
  fi

  run_ours || { echo "OURS run failed — aborting." >&2; return 1; }
  if [ "$mode" = FULL ]; then
    run_lio || { echo "LIO-SAM run failed — aborting." >&2; return 1; }
  fi

  finalize_report "$mode"
  exit_summary "$mode"
}

# Run main only when executed directly; sourcing exposes the functions for tests.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
