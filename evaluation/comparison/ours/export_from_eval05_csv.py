#!/usr/bin/env python3
"""Export our system's EVAL-05 covariance log into the slam_eval file contract.

Reads analysis/eval05_covariance_log.csv (written live by graph_backend_node,
ARCHITECTURE §9 contract: per-keyframe est pose + GT pose + upper-tri 6x6
Sigma_post in GTSAM tangent order [rx ry rz x y z]) and writes:

  <out>/ours_est.tum      estimated keyframe trajectory (TUM)
  <out>/ours_est_cov.txt  Sigma_post reordered to contract order [x y z r p y]
  <out>/gt.tum            ground-truth poses at the same stamps (TUM)

Only `event == keyframe` rows are used; `loop_closure` rows re-log the same
keyframe after correction and would double-count (same convention as
eval05_nees.py). Pure file-to-file, no ROS.

Usage: export_from_eval05_csv.py [--csv <path>] [--out <dir>]
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from slam_eval.tum_io import write_cov, write_tum  # noqa: E402

# CSV upper-tri column suffixes, row-major over GTSAM order [rx ry rz x y z].
_TRI_COLS = [f"cov_{i}{j}" for i in range(6) for j in range(i, 6)]
# GTSAM [rx ry rz x y z] -> contract [x y z roll pitch yaw]
_PERM = [3, 4, 5, 0, 1, 2]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="analysis/eval05_covariance_log.csv")
    p.add_argument("--out", default="analysis/slam_eval")
    args = p.parse_args()

    est_rows, gt_rows, stamps, covs = [], [], [], []
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["event"] != "keyframe":
                continue
            t = float(row["t"])
            est_rows.append([t] + [float(row[f"est_{k}"])
                                   for k in ("x", "y", "z", "qx", "qy", "qz", "qw")])
            gt_rows.append([t] + [float(row[f"gt_{k}"])
                                  for k in ("x", "y", "z", "qx", "qy", "qz", "qw")])
            m = np.zeros((6, 6))
            it = iter(_TRI_COLS)
            for i in range(6):
                for j in range(i, 6):
                    v = float(row[next(it)])
                    m[i, j] = v
                    m[j, i] = v
            m = m[np.ix_(_PERM, _PERM)]
            stamps.append(t)
            covs.append(m)

    if not est_rows:
        print(f"export_from_eval05_csv: no keyframe rows in {args.csv}", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    src = f"exported from {args.csv} (keyframe rows only)"
    write_tum(os.path.join(args.out, "ours_est.tum"), np.array(est_rows),
              f"graph_slam optimized keyframe trajectory — {src}")
    write_tum(os.path.join(args.out, "gt.tum"), np.array(gt_rows),
              f"ground truth (PX4 EKF2 bridge, NOT independent Gazebo truth) — {src}")
    write_cov(os.path.join(args.out, "ours_est_cov.txt"), np.array(stamps), np.array(covs),
              "Sigma_post, upper-tri 6x6, contract order [x y z roll pitch yaw], "
              f"reordered from GTSAM [rx ry rz x y z] — {src}")
    print(f"export_from_eval05_csv: wrote {len(est_rows)} keyframes to {args.out}/ "
          "(ours_est.tum, ours_est_cov.txt, gt.tum)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
