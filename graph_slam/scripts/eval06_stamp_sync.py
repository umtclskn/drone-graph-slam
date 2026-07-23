#!/usr/bin/env python3
"""EVAL-06 AC3: check /ground_truth/pose stamp sync against the LiDAR scans.

Reads a recorded bag offline and, for every scan, finds the nearest
ground-truth pose by header stamp. The EVAL-01 criterion is a max nearest
stamp difference < 50 ms; both streams must also live in the same (sim-time)
clock domain, which this reports as the raw stamp ranges.

Usage:
  python3 eval06_stamp_sync.py bags/slam_loop_03
  python3 eval06_stamp_sync.py bags/slam_loop_03 --gt /ground_truth/pose \\
      --scans /x500/lidar_3d/points --max-ms 50
"""
import argparse
import bisect
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def _stamps(bag, topics):
    """Header stamps in seconds, per topic, from a bag."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag),
        rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in types]
    if missing:
        raise SystemExit(
            f'eval06_stamp_sync: bag {bag} has no topic(s) {missing}.\n'
            f'  present: {sorted(types)}')

    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    out = {t: [] for t in topics}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        s = msg.header.stamp
        out[topic].append(s.sec + s.nanosec * 1e-9)
    for v in out.values():
        v.sort()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bag')
    p.add_argument('--gt', default='/ground_truth/pose')
    p.add_argument('--scans', default='/x500/lidar_3d/points')
    p.add_argument('--max-ms', type=float, default=50.0)
    args = p.parse_args()

    stamps = _stamps(args.bag, [args.gt, args.scans])
    gt, scans = stamps[args.gt], stamps[args.scans]
    if not gt or not scans:
        raise SystemExit(
            f'eval06_stamp_sync: empty stream (gt={len(gt)}, scans={len(scans)})')

    diffs = []
    for t in scans:
        i = bisect.bisect_left(gt, t)
        near = [gt[j] for j in (i - 1, i) if 0 <= j < len(gt)]
        diffs.append(min(abs(t - g) for g in near))

    worst = max(diffs)
    mean = sum(diffs) / len(diffs)
    print(f'bag:    {args.bag}')
    print(f'{args.scans}: {len(scans)} msgs, stamps {scans[0]:.3f} .. {scans[-1]:.3f} s')
    print(f'{args.gt}: {len(gt)} msgs, stamps {gt[0]:.3f} .. {gt[-1]:.3f} s')
    print(f'nearest-stamp diff: max {worst * 1e3:.1f} ms, mean {mean * 1e3:.1f} ms')
    ok = worst * 1e3 < args.max_ms
    print(f'RESULT: {"PASS" if ok else "FAIL"} (criterion < {args.max_ms:.0f} ms)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
