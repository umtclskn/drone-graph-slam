#!/usr/bin/env python3
"""EVAL-02 (Sprint 2 full slice): plot the dead-reckoned NDT-odometry trajectory
vs ground truth, and show the ATE/RPE the node computed.

Reads the trajectory CSV written by eval02_relative_error_node: a small comment
header of `# key=value` summary lines (ate/rpe), then columns
`idx, stamp, est_{x,y,z,qx,qy,qz,qw}, gt_{x,y,z,qx,qy,qz,qw}` (GT columns empty for
any scan with no matched GT pose). Produces one PNG with four panels:
  1. 2D top-down X-Y overlay (estimate vs GT)
  2. 3D X-Y-Z overlay (GT is already ENU from the EVAL-01 bridge -> up=up, NO
     z-negation needed; the negate-NED-z note in ARCHITECTURE applies to raw PX4
     poses, not the bridge output)
  3. per-axis position error vs time (x, y, z)
  4. yaw error vs time
Both trajectories are pinned at node 0 by the compounding, so est-minus-GT is the
honest accumulated front-end drift.

Usage: eval02_trajectory.py [eval02_trajectory.csv]
       (default analysis/eval02_trajectory.csv relative to CWD)
"""
import csv
import math
import sys

import matplotlib

matplotlib.use("Agg")  # headless: always save a PNG, no display needed
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers 3d proj)


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Yaw (rotation about +z) in radians from a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def load(path: str):
    summary: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    with open(path, newline="") as f:
        data_lines = []
        for line in f:
            if line.startswith("#"):
                key, _, val = line[1:].strip().partition("=")
                summary[key.strip()] = val.strip()
            else:
                data_lines.append(line)
        rows = list(csv.DictReader(data_lines))
    return summary, rows


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "analysis/eval02_trajectory.csv"
    try:
        summary, rows = load(path)
    except FileNotFoundError:
        print(f"eval02: trajectory CSV not found: {path}", file=sys.stderr)
        return 1
    if not rows:
        print(f"eval02: no trajectory rows in {path}", file=sys.stderr)
        return 1

    t0 = float(rows[0]["stamp"])
    est = {"t": [], "x": [], "y": [], "z": [], "yaw": []}
    gt = {"t": [], "x": [], "y": [], "z": [], "yaw": []}
    err = {"t": [], "x": [], "y": [], "z": [], "yaw": []}
    for r in rows:
        t = float(r["stamp"]) - t0
        ex, ey, ez = float(r["est_x"]), float(r["est_y"]), float(r["est_z"])
        eyaw = yaw_from_quat(float(r["est_qx"]), float(r["est_qy"]),
                             float(r["est_qz"]), float(r["est_qw"]))
        est["t"].append(t); est["x"].append(ex); est["y"].append(ey)
        est["z"].append(ez); est["yaw"].append(math.degrees(eyaw))
        if r["gt_x"] == "":  # no matched GT for this scan
            continue
        gx, gy, gz = float(r["gt_x"]), float(r["gt_y"]), float(r["gt_z"])
        gyaw = yaw_from_quat(float(r["gt_qx"]), float(r["gt_qy"]),
                             float(r["gt_qz"]), float(r["gt_qw"]))
        gt["t"].append(t); gt["x"].append(gx); gt["y"].append(gy)
        gt["z"].append(gz); gt["yaw"].append(math.degrees(gyaw))
        err["t"].append(t)
        err["x"].append(ex - gx); err["y"].append(ey - gy); err["z"].append(ez - gz)
        err["yaw"].append(wrap_deg(math.degrees(eyaw - gyaw)))

    fig = plt.figure(figsize=(14, 10))
    ate_t = summary.get("ate_trans_m", "?")
    ate_r = summary.get("ate_rot_deg", "?")
    rpe_t = summary.get("rpe_trans_m", "?")
    rpe_r = summary.get("rpe_rot_deg", "?")
    rpe_d = summary.get("rpe_delta", "?")
    fig.suptitle(
        f"EVAL-02 NDT-odometry vs GT  |  ATE: {ate_t} m / {ate_r} deg   "
        f"RPE(\u0394={rpe_d}): {rpe_t} m / {rpe_r} deg",
        fontsize=13)

    # 1. 2D top-down X-Y overlay
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(gt["x"], gt["y"], "-", c="tab:green", lw=2, label="ground truth")
    ax.plot(est["x"], est["y"], "-", c="tab:blue", lw=1.5, label="NDT odometry")
    if gt["x"]:
        ax.plot(gt["x"][0], gt["y"][0], "ko", ms=6, label="start")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("2D top-down (X-Y)"); ax.axis("equal")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")

    # 2. 3D overlay (GT already ENU -> up=up)
    ax3 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3.plot(gt["x"], gt["y"], gt["z"], c="tab:green", lw=2, label="ground truth")
    ax3.plot(est["x"], est["y"], est["z"], c="tab:blue", lw=1.5, label="NDT odometry")
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_zlabel("z [m] (up)")
    ax3.set_title("3D trajectory"); ax3.legend(loc="best")

    # 3. per-axis position error vs time
    ax_e = fig.add_subplot(2, 2, 3)
    for key, c in (("x", "tab:red"), ("y", "tab:green"), ("z", "tab:blue")):
        ax_e.plot(err["t"], err[key], "-", c=c, lw=1.2, label=f"{key} error")
    ax_e.set_xlabel("time [s]"); ax_e.set_ylabel("position error [m]")
    ax_e.set_title("Per-axis position error vs time")
    ax_e.grid(True, alpha=0.3); ax_e.legend(loc="best")

    # 4. yaw error vs time
    ax_y = fig.add_subplot(2, 2, 4)
    ax_y.plot(err["t"], err["yaw"], "-", c="tab:purple", lw=1.2)
    ax_y.set_xlabel("time [s]"); ax_y.set_ylabel("yaw error [deg]")
    ax_y.set_title("Yaw error vs time"); ax_y.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = path.rsplit(".", 1)[0] + ".png"
    if out == path:
        out = path + ".png"
    fig.savefig(out, dpi=120)
    print(f"eval02: saved {out} ({len(est['x'])} nodes, {len(err['t'])} with GT)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
