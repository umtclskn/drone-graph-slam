#!/usr/bin/env python3
"""Export a LIO-SAM bag-run (LIO-SAM-03/04 recorder CSVs) into the slam_eval
file contract.

Reads <run_dir>/odom_liosam.csv and <run_dir>/odom_ekf2.csv (columns
stamp,x,y,z,qw,qx,qy,qz — recorded from /lio_sam/mapping/odometry and
/odometry/ekf2 on the same replay, shared sim-time clock) and writes:

  <out>/lio_sam_est_<tag>.tum   LIO-SAM mapping trajectory (TUM)
  <out>/gt_ekf2_<tag>.tum       EKF2 ground truth on the same run (TUM)

Filenames are tagged (default: today's date) so repeated runs never clobber
each other or our system's baseline artifacts (EVAL-05 lesson).

No covariance export: stock LIO-SAM leaves pose.covariance on
/lio_sam/mapping/odometry all-zero (the only covariance use in its source is a
0/1 degenerate flag on covariance[0] of odometry_incremental), so there is
nothing honest to feed NEES — the lio_sam row's consistency columns stay n/a.
Pure file-to-file, no ROS.

Usage: export_from_run_csv.py --run <run_dir> [--out <dir>] [--tag <tag>]
"""
import argparse
import csv
import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from slam_eval.tum_io import write_tum  # noqa: E402


def _load_rows(path: str) -> np.ndarray:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append([float(row["stamp"]), float(row["x"]), float(row["y"]),
                         float(row["z"]), float(row["qx"]), float(row["qy"]),
                         float(row["qz"]), float(row["qw"])])
    if not rows:
        raise ValueError(f"{path}: no poses")
    return np.array(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="recorder output dir (odom_*.csv)")
    p.add_argument("--out", default="analysis/slam_eval")
    p.add_argument("--tag", default=datetime.date.today().isoformat())
    args = p.parse_args()

    est = _load_rows(os.path.join(args.run, "odom_liosam.csv"))
    gt = _load_rows(os.path.join(args.run, "odom_ekf2.csv"))

    os.makedirs(args.out, exist_ok=True)
    est_path = os.path.join(args.out, f"lio_sam_est_{args.tag}.tum")
    gt_path = os.path.join(args.out, f"gt_ekf2_{args.tag}.tum")
    src = f"exported from {args.run}"
    write_tum(est_path, est, f"LIO-SAM /lio_sam/mapping/odometry — {src}")
    write_tum(gt_path, gt,
              f"ground truth (PX4 EKF2 bridge, NOT independent Gazebo truth) — {src}")
    print(f"export_from_run_csv: wrote {len(est)} est / {len(gt)} gt poses -> "
          f"{est_path}, {gt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
