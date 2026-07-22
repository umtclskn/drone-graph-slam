# SLAM evaluation report

Same bag, same ground truth, same metric engine (`slam_eval`) for every row.

| System | ATE trans RMSE [m] | ATE rot RMSE [deg] | RPE@1s trans [m] | RPE@1s rot [deg] | drift [m/m] | ANEES | median NEES | 95% coverage [%] | Consistency | matched/dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| ours_graph_slam | 0.513 | 5.27 | 0.621 | 1.43 | 0.1537 | 14.56 | 3.69 | 85.7 | overconfident | 112/0 |

## Run details

- **ours_graph_slam**: align=se3, max_dt=0.02s, cov_scale=1.0, RPE deltas=[1.0, 2.0, 5.0, 10.0]s.
  - RPE sweep: Δ=1s: 0.621 m / 1.43° (0.1537 m/m, n=110); Δ=2s: 0.944 m / 1.54° (0.1234 m/m, n=108); Δ=5s: 1.524 m / 2.27° (0.0743 m/m, n=93); Δ=10s: 0.782 m / 2.79° (0.0299 m/m, n=61)
  - NEES: n=112 (PD-skipped 0), ANEES 95% band [2.56, 3.47].

## Caveats (apply to every row on `bags/slam_loop_02`)

- **Ground truth is PX4 EKF2 output, not independent Gazebo truth** (EVAL-06 open): all accuracy numbers measure consistency with EKF2. Our system seeds NDT with an EKF2-derived initial guess (loosely coupled, one-way); LIO-SAM tightly couples the same IMU that produces this GT — both rows are flattered, differently.
- **`ndt_cov_scale_factor = 435` was NEES-calibrated on this same bag** (`analysis/calibration_log.md`): our consistency row is in-sample until cross-validated on a second bag.
