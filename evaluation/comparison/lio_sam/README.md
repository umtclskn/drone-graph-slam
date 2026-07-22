# LIO-SAM comparison glue (LIO-SAM-04 — DONE 2026-07-12)

LIO-SAM is scored as row 2 of the shared report by the SAME engine, same bag
(`bags/slam_loop_02`), same EKF2 ground truth (EVAL-06 will swap in independent
Gazebo GT by regenerating the `--gt` file — the engine is GT-agnostic).

## Reproduce (three steps)

1. **Run** LIO-SAM on the bag (workspace `~/lio_sam_ws`, built per LIO-SAM-01;
   params `src/LIO-SAM/config/params_drone_x500.yaml` per LIO-SAM-03):

   ```bash
   OUT=~/lio_sam_ws/runs/<run_id> \
   REC=~/lio_sam_ws/scripts/odom_recorder_eval.py \
   bash ~/lio_sam_ws/scripts/run_drone_bag.sh
   ```

   The eval recorder writes `odom_liosam.csv` + `odom_ekf2.csv` (trajectories),
   `cov_check.txt` (STEP 0 covariance evidence) and `loop_count.txt` (final
   loop-edge count from `/lio_sam/mapping/loop_closure_constraints`).

2. **Export** to the slam_eval file contract (timestamped names — never
   clobbers our system's baseline artifacts):

   ```bash
   ./export_from_run_csv.py --run ~/lio_sam_ws/runs/<run_id> --tag <date>
   ```

3. **Score** with the shared engine, default flags (same as our row):

   ```bash
   cd ../..   # evaluation/
   python3 -m slam_eval \
       --est analysis/slam_eval/lio_sam_est_<date>.tum \
       --gt  analysis/slam_eval/gt_ekf2_<date>.tum \
       --out /home/umut/portfolio_ws/drone_ws/analysis/slam_eval/ --name lio_sam
   ```

## Findings (2026-07-12, run `liosam04_20260712`)

- **ATE** 0.382 m / 5.30° (se3), RPE@1s 0.543 m, 118/118 matched — vs ours
  0.544 m / 5.57° (see `analysis/slam_eval/report.md` for the full table).
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
- Honesty caveats (baked into `report.md` by the engine): GT = EKF2; coupling
  asymmetry (LIO-SAM's IMU orientation is seeded from EKF2 by
  `imu_bridge_liosam`); ~2.8° yaw-init offset
  (`useImuHeadingInitialization: false`); deskew disabled (no per-point time).
