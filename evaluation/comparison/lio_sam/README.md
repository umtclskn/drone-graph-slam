# LIO-SAM comparison glue (LIO-SAM-04 — DONE 2026-07-12)

LIO-SAM is scored as row 2 of the shared report by the SAME engine as our
system. Ground truth is now the **independent Gazebo model-state** truth on
`/ground_truth/odom` (EVAL-06), which `bags/slam_loop_03` onward carries; the
recorder subscribes to it directly (EVAL-07 B1). The 2026-07-12 findings below
predate that and were scored against PX4 EKF2 on `bags/slam_loop_02` — see the
caveat there.

## Reproduce (three steps)

1. **Run** LIO-SAM on the bag (workspace `~/lio_sam_ws`, built per LIO-SAM-01;
   params `src/LIO-SAM/config/params_drone_x500.yaml` per LIO-SAM-03):

   ```bash
   BAG=bags/slam_loop_03 \
   OUT=~/lio_sam_ws/runs/<run_id> \
   REC=~/lio_sam_ws/scripts/odom_recorder_eval.py \
   bash ~/lio_sam_ws/scripts/run_drone_bag.sh
   ```

   `BAG` is relative to `drone_ws` and defaults to `bags/slam_loop_02`, which has
   no `/ground_truth/odom` — use a loop_03-or-later bag for a scorable GT.
   The eval recorder writes `odom_liosam.csv` + `odom_gt.csv` (trajectories),
   `cov_check.txt` (STEP 0 covariance evidence) and `loop_count.txt` (final
   loop-edge count from `/lio_sam/mapping/loop_closure_constraints`).

2. **Export** to the slam_eval file contract (timestamped names — never
   clobbers our system's baseline artifacts):

   ```bash
   ./export_from_run_csv.py --run ~/lio_sam_ws/runs/<run_id> --tag <date>
   ```

3. **Score** with the shared engine, default flags (same as our row). `--bag` /
   `--gt_label` are what the report prints per row; omit `--gt_label` and it
   falls back to the `#` header the exporter wrote:

   ```bash
   cd ../..   # evaluation/
   python3 -m slam_eval \
       --est analysis/slam_eval/lio_sam_est_<date>.tum \
       --gt  analysis/slam_eval/gt_gazebo_<date>.tum \
       --out /home/umut/portfolio_ws/drone_ws/analysis/slam_eval/ --name lio_sam \
       --bag bags/slam_loop_03 --gt_label "Gazebo model-state (independent)"
   ```

   Score it into the same `--out` as our row for that bag, so both land in one
   table; the report states bag + GT per row and refuses to claim they match
   when they do not.

## Findings (2026-07-12, run `liosam04_20260712`)

**Scored on `bags/slam_loop_02` against PX4 EKF2 GT** — superseded as a fair
comparison by EVAL-06/07; re-run on `slam_loop_03` for the independent-GT
numbers. Kept as the record of what the stock stack did.

- **ATE** 0.382 m / 5.30° (se3), RPE@1s 0.543 m, 118/118 matched — vs ours
  0.544 m / 5.57° (see `analysis/slam_eval/report.md` for the full table).
  Both are EKF2-relative, so both are flattered; EVAL-06 measured a ~3x ATE
  inflation from EKF2 GT on our row.
- **No covariance → NEES N/A**: `/lio_sam/mapping/odometry` pose covariance is
  all-zero (max |entry| = 0.0 over 118 msgs; source only ever writes a 0/1
  degeneracy flag on `odometry_incremental` covariance[0], and its internal
  iSAM2 `poseCovariance` is used solely for GPS gating, never published).
  NEES would require patching marginal extraction — deliberately not done.
- **Loop closures: 0 (stock) vs our 19 — structural**: stock
  `historyKeyframeSearchTimeDiff: 30 s` exceeds the ~24 s bag. Diagnostic run
  (`liosam04_loopdiag_20260712`, gate lowered to 2 s ≈ our
  `loop_min_age_difference` gate) accepted **22** ICP closures. The scored row
  is the stock-gate run.
- Honesty caveats (emitted into `report.md` by the engine, only for the rows
  they apply to): EKF2 GT; coupling asymmetry (LIO-SAM's IMU orientation is
  seeded from EKF2 by `imu_bridge_liosam`); ~2.8° yaw-init offset
  (`useImuHeadingInitialization: false`); deskew disabled (no per-point time).
