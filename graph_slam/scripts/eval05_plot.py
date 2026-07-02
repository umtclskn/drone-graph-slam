#!/usr/bin/env python3
"""EVAL-05 offline covariance plots (ARCHITECTURE.md section 9).

Reads the per-keyframe / per-loop-event CSV written by graph_backend_node
(analysis/eval05_covariance_log.csv, the full Sigma_post logging contract) and
produces four publication-quality figures in the CSV's directory:

  1. eval05_2d_ellipses.png        - X-Y path (estimate vs GT) + 2D Sigma_post
                                     ellipses per keyframe.
  2. eval05_3d_projections.png     - XY / XZ / YZ projections with ellipses.
  3. eval05_uncertainty_trace.png  - trace(Sigma_post position block) vs
                                     keyframe_id, loop closures marked.
  4. eval05_error_vs_keyframe.png  - ||est - GT|| position error vs keyframe_id
                                     with the +/-2 sigma consistency envelope.

Poses and covariances are already in ENU (the graph back-end + EVAL-01 work in
ENU), so z needs no NED negation here. The covariance columns are the upper
triangle of the 6x6 Sigma_post in GTSAM tangent order [rx,ry,rz,x,y,z]; the
position block is indices 3..5.

Usage: eval05_plot.py [--csv analysis/eval05_covariance_log.csv]
"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe; PNGs are always written
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

CHI2_2DOF_95 = 5.991  # 95% confidence, 2 DOF


def load_rows(path):
    """Parse the CSV into a list of dict rows (all numeric fields as float)."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def position_cov_3x3(row):
    """Reconstruct the symmetric 3x3 position block from the upper-tri columns."""
    xx = float(row["cov_33"])
    xy = float(row["cov_34"])
    xz = float(row["cov_35"])
    yy = float(row["cov_44"])
    yz = float(row["cov_45"])
    zz = float(row["cov_55"])
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def ellipse_from_2x2(cov2, chi2=CHI2_2DOF_95):
    """Return (width, height, angle_deg) for a confidence ellipse of a 2x2 block."""
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.clip(vals, 0.0, None)
    # eigh returns ascending; largest eigenvalue is the major axis.
    major = vecs[:, 1]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    width = 2.0 * np.sqrt(chi2 * vals[1])   # full axis length
    height = 2.0 * np.sqrt(chi2 * vals[0])
    return width, height, angle


def loop_keyframes(rows):
    """keyframe_id values flagged as loop_closure events."""
    return [float(r["keyframe_id"]) for r in rows if r["event"] == "loop_closure"]


def draw_2d_ellipses(ax, rows, ix, iy, kx, ky):
    """Overlay per-keyframe ellipses onto axis `ax` for component pair (kx, ky)."""
    for r in rows:
        cov3 = position_cov_3x3(r)
        cov2 = cov3[np.ix_([kx, ky], [kx, ky])]
        w, h, ang = ellipse_from_2x2(cov2)
        e = Ellipse(
            (float(r[ix]), float(r[iy])),
            width=w,
            height=h,
            angle=ang,
            facecolor="orange",
            edgecolor="darkorange",
            alpha=0.25,
            lw=0.6,
        )
        ax.add_patch(e)


def plot_2d_ellipses(rows, out):
    est_x = [float(r["est_x"]) for r in rows]
    est_y = [float(r["est_y"]) for r in rows]
    gt_x = [float(r["gt_x"]) for r in rows]
    gt_y = [float(r["gt_y"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_2d_ellipses(ax, rows, "est_x", "est_y", 0, 1)
    ax.plot(est_x, est_y, "b-", lw=1.5, label="estimate")
    ax.plot(gt_x, gt_y, "g--", lw=1.5, label="ground truth")
    ax.scatter(est_x, est_y, s=8, c="blue", zorder=5)
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("EVAL-05 X-Y path + Sigma_post 95% ellipses")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"eval05_plot: saved {out}")


def plot_3d_projections(rows, out):
    pairs = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (a, b, ka, kb) in zip(axes, pairs):
        est_a = [float(r[f"est_{a}"]) for r in rows]
        est_b = [float(r[f"est_{b}"]) for r in rows]
        gt_a = [float(r[f"gt_{a}"]) for r in rows]
        gt_b = [float(r[f"gt_{b}"]) for r in rows]
        draw_2d_ellipses(ax, rows, f"est_{a}", f"est_{b}", ka, kb)
        ax.plot(est_a, est_b, "b-", lw=1.2, label="estimate")
        ax.plot(gt_a, gt_b, "g--", lw=1.2, label="ground truth")
        ax.set_aspect("equal", "datalim")
        ax.set_xlabel(f"{a} [m]")
        ax.set_ylabel(f"{b} [m]")
        ax.set_title(f"{a.upper()}-{b.upper()} projection")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle("EVAL-05 Sigma_post ellipses (3D position projections)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"eval05_plot: saved {out}")


def plot_uncertainty_trace(rows, out):
    kf = [float(r["keyframe_id"]) for r in rows]
    trace = [
        float(r["cov_33"]) + float(r["cov_44"]) + float(r["cov_55"]) for r in rows
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(kf, trace, "b.-", lw=1.2, label="trace(Sigma_post position)")
    for lc in loop_keyframes(rows):
        ax.axvline(lc, color="red", ls="--", alpha=0.7)
    if loop_keyframes(rows):
        ax.axvline(loop_keyframes(rows)[0], color="red", ls="--", alpha=0.7,
                   label="loop closure")
    ax.set_xlabel("keyframe_id")
    ax.set_ylabel("trace(Sigma_post position) [m^2]")
    ax.set_title("EVAL-05 uncertainty growth + loop-closure step-down")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"eval05_plot: saved {out}")


def plot_error_vs_keyframe(rows, out):
    kf, err, two_sigma = [], [], []
    have_gt = False
    for r in rows:
        est = np.array([float(r["est_x"]), float(r["est_y"]), float(r["est_z"])])
        gt = np.array([float(r["gt_x"]), float(r["gt_y"]), float(r["gt_z"])])
        if np.any(gt != 0.0):
            have_gt = True
        kf.append(float(r["keyframe_id"]))
        err.append(float(np.linalg.norm(est - gt)))
        trace = float(r["cov_33"]) + float(r["cov_44"]) + float(r["cov_55"])
        two_sigma.append(2.0 * np.sqrt(max(trace, 0.0)))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(kf, 0.0, two_sigma, color="green", alpha=0.15,
                    label="+/-2 sigma envelope")
    ax.plot(kf, two_sigma, color="green", lw=0.8, alpha=0.6)
    ax.plot(kf, err, "b.-", lw=1.2, label="||est - GT|| [m]")
    for lc in loop_keyframes(rows):
        ax.axvline(lc, color="red", ls="--", alpha=0.7)
    if loop_keyframes(rows):
        ax.axvline(loop_keyframes(rows)[0], color="red", ls="--", alpha=0.7,
                   label="loop closure")
    ax.set_xlabel("keyframe_id")
    ax.set_ylabel("position error [m]")
    title = "EVAL-05 position error vs keyframe_id"
    if not have_gt:
        title += " (no GT in log)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"eval05_plot: saved {out}")


def main():
    parser = argparse.ArgumentParser(description="EVAL-05 covariance plots")
    parser.add_argument("--csv", default="analysis/eval05_covariance_log.csv",
                        help="path to eval05_covariance_log.csv")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"eval05_plot: CSV not found: {args.csv}", file=sys.stderr)
        return 1
    rows = load_rows(args.csv)
    if not rows:
        print(f"eval05_plot: CSV has no data rows: {args.csv}", file=sys.stderr)
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.csv))
    plot_2d_ellipses(rows, os.path.join(out_dir, "eval05_2d_ellipses.png"))
    plot_3d_projections(rows, os.path.join(out_dir, "eval05_3d_projections.png"))
    plot_uncertainty_trace(rows, os.path.join(out_dir, "eval05_uncertainty_trace.png"))
    plot_error_vs_keyframe(rows, os.path.join(out_dir, "eval05_error_vs_keyframe.png"))
    print(f"eval05_plot: {len(rows)} rows, "
          f"{len(loop_keyframes(rows))} loop-closure event(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
