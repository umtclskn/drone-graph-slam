#!/usr/bin/env python3
"""NDT-04 scan-quality-gate smoke run over a recorded rosbag.

This is an OFFLINE verification harness (not the gate itself): it mirrors the two
C++ ``QualityChecker`` checks -- point count and PCA smallest-eigenvalue spread --
and reports how many of a bag's LiDAR scans the gate would accept vs reject. The
authoritative gate is the C++ ``graph_slam::QualityChecker`` (gtest-covered); it is
wired into the live pipeline at NDT-13. This script only confirms the thresholds
are not over-aggressive on a healthy bag (acceptance criteria #3 / #5).

Each scan is voxel-downsampled (leaf = --voxel-leaf, matching the NDT-03
Preprocessor default) before counting, because the gate runs on the *preprocessed*
cloud. Usage:

    python3 ndt04_quality_smoke.py --bag <bags/slam_loop_02> [--voxel-leaf 0.2]
"""
import argparse
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def voxel_downsample(xyz: np.ndarray, leaf: float) -> np.ndarray:
    """Keep one point per occupied voxel (approximates pcl::VoxelGrid count)."""
    if leaf <= 0.0 or xyz.shape[0] == 0:
        return xyz
    keys = np.floor(xyz / leaf).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx]


def min_spread_eigenvalue(xyz: np.ndarray) -> float:
    """Smallest eigenvalue of the 3x3 point-position covariance (m^2)."""
    cov = np.cov(xyz.T)  # 3x3, population/sample diff is negligible here
    return float(np.linalg.eigvalsh(cov)[0])  # ascending


def read_scans(bag_path: str, topic: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name == topic:
            yield deserialize_message(data, PointCloud2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--topic", default="/x500/lidar_3d/points")
    ap.add_argument("--voxel-leaf", type=float, default=0.2)
    ap.add_argument("--min-points", type=int, default=100)
    ap.add_argument("--min-spread-eigenvalue", type=float, default=0.1)
    args = ap.parse_args()

    accepted = 0
    rejected_count = 0
    rejected_spread = 0
    total = 0
    examples = []
    for msg in read_scans(args.bag, args.topic):
        total += 1
        pts = np.array(
            [[p[0], p[1], p[2]] for p in point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)],
            dtype=np.float64,
        )
        # Preprocessor (NDT-03) also strips non-finite points; drop any inf rows.
        if pts.shape[0]:
            pts = pts[np.isfinite(pts).all(axis=1)]
        pts = voxel_downsample(pts, args.voxel_leaf)
        n = pts.shape[0]
        if n < args.min_points:
            rejected_count += 1
            if len(examples) < 3:
                examples.append(f"  scan {total}: too few points: {n} < {args.min_points}")
            continue
        min_eig = min_spread_eigenvalue(pts)
        if min_eig < args.min_spread_eigenvalue:
            rejected_spread += 1
            if len(examples) < 3:
                examples.append(
                    f"  scan {total}: insufficient spread: {min_eig:.4f} "
                    f"< {args.min_spread_eigenvalue}")
            continue
        accepted += 1

    if total == 0:
        print(f"No messages on topic {args.topic} in {args.bag}", file=sys.stderr)
        return 1

    pct = 100.0 * accepted / total
    print(f"NDT-04 quality-gate smoke run on {args.bag}")
    print(f"  topic={args.topic}  voxel_leaf={args.voxel_leaf}  "
          f"min_points={args.min_points}  min_spread_eigenvalue={args.min_spread_eigenvalue}")
    print(f"  scans total    : {total}")
    print(f"  accepted       : {accepted} ({pct:.1f}%)")
    print(f"  rejected count : {rejected_count}")
    print(f"  rejected spread: {rejected_spread}")
    if examples:
        print("  example rejections:")
        print("\n".join(examples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
