"""CLI: score ONE system's exported files against ground truth.

  python -m slam_eval --est est.tum --cov est_cov.txt --gt gt.tum --out <dir>/

Computes the confirmed metric set (ATE, time-based RPE sweep + drift-per-metre,
position NEES consistency when --cov is given), writes/updates metrics.json and
report.md in --out, and saves the three PNG plots. Nothing here is specific to
our system — LIO-SAM is scored by pointing the same command at its exports.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import NDT_COV_SCALE_ALPHA_CALIBRATED  # noqa: F401  (documented default source)
from .associate import associate
from .consistency import nees
from .metrics import Trajectory, align_trajectory, ate, rpe
from .plots import plot_error_over_time, plot_nees_over_time, plot_xy
from .report import upsert_metrics, write_report
from .tum_io import load_cov, load_tum, quats_to_rots


def _trajectory(tum: np.ndarray) -> Trajectory:
    return Trajectory(stamps=tum[:, 0], pos=tum[:, 1:4], rot=quats_to_rots(tum[:, 4:8]))


def _file_header(path: str) -> str | None:
    """The leading `#` comment of a TUM file — the exporters' provenance note,
    reused as the default ground-truth label so no new plumbing is needed."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                return line.lstrip("#").strip() or None
            if line:
                return None
    return None


def main() -> int:
    p = argparse.ArgumentParser(prog="slam_eval", description=__doc__)
    p.add_argument("--est", required=True, help="estimated trajectory (TUM)")
    p.add_argument("--gt", required=True, help="ground-truth trajectory (TUM)")
    p.add_argument("--cov", default=None,
                   help="estimated pose covariance file (stamp + 21 upper-tri, "
                        "[x y z roll pitch yaw]); omit to skip consistency metrics")
    p.add_argument("--out", required=True, help="output directory (shared analysis folder)")
    p.add_argument("--name", default=None,
                   help="system name = table row label (default: est filename stem)")
    p.add_argument("--gt_label", default=None,
                   help="ground-truth provenance for THIS row, e.g. 'Gazebo model-state "
                        "(independent)' or 'PX4 EKF2'. Default: the '#' header line of "
                        "--gt. Reported per row — rows may legitimately differ")
    p.add_argument("--bag", default=None,
                   help="source bag for THIS row, e.g. bags/slam_loop_03 (default: "
                        "unspecified). Reported per row — rows may legitimately differ")
    p.add_argument("--align", choices=["se3", "first", "sim3"], default="se3",
                   help="trajectory alignment for ATE (default se3 = Umeyama, no scale; "
                        "'first' reproduces the in-repo C++ first-pose ATE; sim3 flatters "
                        "metric sensors and is flagged in the report)")
    p.add_argument("--rpe_delta", default="1,2,5,10",
                   help="comma-separated RPE deltas in SECONDS (default 1,2,5,10)")
    p.add_argument("--max_dt", type=float, default=0.02,
                   help="max est<->gt stamp gap for association [s] (default 0.02)")
    p.add_argument("--cov_scale", type=float, default=1.0,
                   help="scalar applied to input covariances before NEES. Default 1.0: "
                        "covariances exported from the current stack already include the "
                        "NEES-calibrated alpha=435 (slam_params.yaml ndt_cov_scale_factor). "
                        "Use 435 (NDT_COV_SCALE_ALPHA_CALIBRATED) only on raw uncalibrated logs.")
    p.add_argument("--dof", type=int, default=3, help="NEES dof (v1: position-only, 3)")
    p.add_argument("--nees_align", choices=["none", "first"], default="none",
                   help="alignment for NEES errors (default none = raw est-gt, the repo's "
                        "EVAL-05 convention; frames must coincide. Use 'first' only when the "
                        "system's world frame differs from GT's — first-pose pinning imports "
                        "any noise in the first GT sample into every error)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    name = args.name or os.path.splitext(os.path.basename(args.est))[0]

    est_tum, gt_tum = load_tum(args.est), load_tum(args.gt)
    assoc = associate(est_tum[:, 0], gt_tum[:, 0], args.max_dt)
    if assoc.n_matched < 2:
        print(f"slam_eval: only {assoc.n_matched} matched pairs within max_dt="
              f"{args.max_dt}s — nothing to score", file=sys.stderr)
        return 1
    est = _trajectory(est_tum[assoc.est_idx])
    gt = _trajectory(gt_tum[assoc.gt_idx])
    print(f"association: {assoc.n_matched}/{assoc.n_est} est poses matched, "
          f"{assoc.n_dropped} dropped (max accepted gap {assoc.max_gap * 1e3:.1f} ms)")

    est_aligned, scale = align_trajectory(est, gt, args.align)
    ate_res = ate(est_aligned, gt, args.align, scale)
    print(f"ATE ({args.align}): trans RMSE {ate_res.trans_rmse_m:.3f} m "
          f"(mean {ate_res.trans_mean_m:.3f}, median {ate_res.trans_median_m:.3f}, "
          f"max {ate_res.trans_max_m:.3f}), rot RMSE {ate_res.rot_rmse_deg:.2f} deg, "
          f"n={ate_res.count}")

    deltas = [float(d) for d in args.rpe_delta.split(",") if d.strip()]
    rpe_results = {}
    for d in deltas:
        r = rpe(est, gt, d)
        if r.count == 0:
            print(f"RPE Δ={d:g}s: no pairs (trajectory shorter than delta) — skipped")
            continue
        rpe_results[f"{d:g}"] = r
        print(f"RPE Δ={d:g}s: trans RMSE {r.trans_rmse_m:.3f} m, rot RMSE "
              f"{r.rot_rmse_deg:.2f} deg, drift {r.drift_per_m:.4f} m/m, n={r.count}")

    nees_res = None
    if args.cov:
        cov_stamps, covs = load_cov(args.cov)
        # covariance rows must line up with the matched estimate poses (same stamps)
        cov_assoc = associate(est.stamps, cov_stamps, args.max_dt)
        if cov_assoc.n_matched < 2:
            print("slam_eval: covariance stamps do not match the estimate — "
                  "skipping consistency metrics", file=sys.stderr)
        else:
            sel = cov_assoc.est_idx  # matched subset of the already-matched poses
            # NEES never uses the ATE --align mode: Umeyama optimizes the very
            # errors NEES scores (and dumps residual error onto low-covariance
            # poses), biasing the consistency verdict. See --nees_align.
            if args.nees_align == "first":
                est_nees, _ = align_trajectory(est, gt, "first")
            else:
                est_nees = est
            align_rot = est_nees.rot[0] @ est.rot[0].T
            nees_res = nees(
                pos_err=est_nees.pos[sel] - gt.pos[sel],
                covs=covs[cov_assoc.gt_idx],
                stamps=est.stamps[sel],
                align_rot=align_rot,
                cov_scale=args.cov_scale,
                dof=args.dof,
            )
            print(f"NEES (dof={nees_res.dof}, cov_scale={args.cov_scale:g}): "
                  f"ANEES {nees_res.anees:.2f} (band [{nees_res.anees_lo:.2f}, "
                  f"{nees_res.anees_hi:.2f}]), median {nees_res.median:.2f}, "
                  f"95% coverage {nees_res.coverage_95_pct:.1f}%, n={nees_res.count} "
                  f"(PD-skipped {nees_res.n_skipped_pd})")
            print(f"consistency verdict: {nees_res.verdict.upper()}")

    plot_xy(os.path.join(args.out, f"{name}_xy_trajectory.png"), name, est_aligned, gt,
            args.align)
    plot_error_over_time(os.path.join(args.out, f"{name}_error_over_time.png"), name,
                         est.stamps, ate_res)
    if nees_res is not None:
        plot_nees_over_time(os.path.join(args.out, f"{name}_nees_over_time.png"), name,
                            nees_res)

    entry = {
        "run": {
            "est": os.path.abspath(args.est),
            "gt": os.path.abspath(args.gt),
            "cov": os.path.abspath(args.cov) if args.cov else None,
            "bag": args.bag,
            "gt_label": args.gt_label or _file_header(args.gt),
            "align": args.align,
            "max_dt": args.max_dt,
            "cov_scale": args.cov_scale,
            "nees_align": args.nees_align,
            "rpe_deltas": deltas,
        },
        "association": {"matched": assoc.n_matched, "dropped": assoc.n_dropped,
                        "max_gap_s": assoc.max_gap},
        "ate": {k: getattr(ate_res, k) for k in
                ("align", "scale", "trans_rmse_m", "trans_mean_m", "trans_median_m",
                 "trans_max_m", "rot_rmse_deg", "count")},
        "rpe": {k: {f: getattr(r, f) for f in
                    ("delta_s", "trans_rmse_m", "rot_rmse_deg", "drift_per_m", "count")}
                for k, r in rpe_results.items()},
        "nees": ({k: getattr(nees_res, k) for k in
                  ("dof", "cov_scale", "count", "n_skipped_pd", "anees", "median", "std",
                   "coverage_95_pct", "anees_lo", "anees_hi", "verdict")}
                 if nees_res is not None else None),
    }
    data = upsert_metrics(args.out, name, entry)
    report_path = write_report(args.out, data)
    print(f"wrote {os.path.join(args.out, 'metrics.json')} and {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
