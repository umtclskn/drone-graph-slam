# slam_eval — shared SLAM metric engine

System-agnostic, ROS-free, pure files-in → numbers-out. Every system in the
comparison (our `graph_slam`, LIO-SAM, …) exports the file contract below and
is scored by the **same command** — one table row per system, zero engine
changes to add a system.

## Entry point

```bash
cd src/drone-graph-slam/evaluation        # so `python -m slam_eval` resolves
python -m slam_eval \
    --est  ours_est.tum \
    --cov  ours_est_cov.txt \
    --gt   gt.tum \
    --out  /home/umut/portfolio_ws/drone_ws/analysis/slam_eval/ \
    --name ours_graph_slam
```

Requires only `numpy` and `matplotlib`. Each run scores one system, upserts it
into `<out>/metrics.json`, and regenerates `<out>/report.md` (one table row per
system) plus three plots: `<name>_xy_trajectory.png`,
`<name>_error_over_time.png`, `<name>_nees_over_time.png`.

## File I/O contract

### `--est` / `--gt` — trajectory (TUM format)

One pose per line, `#` comments allowed:

```
timestamp tx ty tz qx qy qz qw
```

- `timestamp`: seconds (float). Estimate and GT must share one clock domain
  (on our bags: sim time — replay **without** `--clock`, see the launch docs).
- Position in metres, quaternion in `x y z w` order, both in a single fixed
  world frame per file (ENU here). The engine is GT-agnostic: today GT is the
  PX4 EKF2 bridge output (**not** independent Gazebo truth — EVAL-06), and the
  report discloses that.

### `--cov` — pose covariance (optional; enables consistency metrics)

One line per pose:

```
timestamp v0 v1 ... v20
```

`v0..v20` = the **row-major upper triangle** of the 6×6 pose covariance in
axis order **[x y z roll pitch yaw]** (translation first). Explicitly:
`v0..v5` = row x (xx xy xz xr xp xy̑), `v6..v10` = row y from the diagonal
(yy yz yr yp yy̑), …, `v20` = yaw-yaw. Units m², m·rad, rad².

> Our back-end logs Σ_post in GTSAM tangent order `[rx ry rz x y z]`
> (`analysis/eval05_covariance_log.csv`); the converter
> `../comparison/ours/export_from_eval05_csv.py` performs the block swap into
> this contract order. Any exporter for another system must do the same.

### Timestamp association

Nearest-neighbour per estimate pose, rejected when the gap exceeds `--max_dt`
(default **0.02 s**). Matched/dropped counts and the largest accepted gap are
printed and stored in `metrics.json`. The covariance file is associated to the
estimate stamps the same way.

## Metrics computed (the confirmed must-have set)

| Metric | Notes |
|---|---|
| ATE trans (RMSE/mean/median/max) + rot RMSE | `--align se3` (Umeyama, no scale, default) \| `first` (first-pose pinning — reproduces the in-repo C++ `trajectory_metrics.cpp` numbers) \| `sim3` (with scale; flatters metric sensors, flagged in the report) |
| RPE trans + rot over a delta sweep | `--rpe_delta 1,2,5,10` (**seconds**, not frame indices — keyframe rates differ across systems) |
| Drift per metre | mean(‖RPE trans residual‖ / GT segment length) per delta |
| NEES consistency (position, dof=3) | ANEES + median + 95 % χ² coverage + verdict (overconfident / consistent / conservative, from the Wilson–Hilferty ANEES band). Ported from `graph_slam/scripts/eval05_nees.py`. NEES never uses the ATE `--align` (Umeyama would optimize the very errors NEES scores); `--nees_align none` (default, raw errors — the EVAL-05 convention, frames must coincide) or `first` (pin first poses when the system's world frame differs from GT's). |

### `--cov_scale` (default **1.0** — deliberate deviation from the draft spec)

Scalar multiplied into the covariances before NEES. The calibrated
α = 435 (see `NDT_COV_SCALE_ALPHA_CALIBRATED` in `__init__.py`, from the
EVAL-05 NEES calibration, `analysis/calibration_log.md`) is **already baked
into the live pipeline** via `slam_params.yaml: ndt_cov_scale_factor: 435`, so
covariances exported from the current stack are pre-calibrated — scoring them
with another ×435 would double-apply it. Pass `--cov_scale 435` only when
re-scoring a raw, uncalibrated covariance log.

## Output folder & publishing

Default shared output dir: `/home/umut/portfolio_ws/drone_ws/analysis/slam_eval/`
(workspace `analysis/` convention — un-versioned). To commit a curated snapshot
into the repo, run `../publish.sh [<out-dir>]`, which copies `report.md`,
`metrics.json`, and the PNGs into `evaluation/results/`.
