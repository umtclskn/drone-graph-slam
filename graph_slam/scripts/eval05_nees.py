#!/usr/bin/env python3
"""EVAL-05 NEES consistency check + Σ_meas scale calibration.

EVAL-05 showed Σ_post is ~10x overconfident (real position error ~0.4 m but
reported σ ≈ 0.03 m). The Hessian-based Σ_meas (NDT-12) has the right SHAPE
(anisotropy) but the wrong SCALE; a single scalar multiplier α applied to
Σ_meas propagates through the whole chain (Σ_meas → Σ_prop → Σ_post).

NEES (Normalized Estimation Error Squared) recovers α from the data we already
log: for each keyframe ε = eᵀ Σ_pos⁻¹ e, where e is the position error and
Σ_pos is the 3x3 position block of Σ_post. A consistent filter has mean NEES ≈
dim (3 for position); mean NEES / dim is exactly the scale the covariance is off
by, i.e. the calibration multiplier α.

Pure analysis: reads only the existing CSV, writes only to analysis/. No bag
replay, no source changes.

Usage:
  eval05_nees.py --csv analysis/eval05_covariance_log.csv [--dim 3]
                 [--output analysis/nees_report.txt]
"""
import argparse
import csv
import os
import sys

import numpy as np

# χ²(0.95) for 3 DoF: a consistent 3-D estimate keeps NEES below this 95% of the
# time (this script is position-only, dim = 3).
CHI2_3DOF_95 = 7.815


def load_keyframes(path):
    """Return (errors Nx3, position covariances Nx3x3, keyframe_ids,
    loop_closure_keyframe_ids, skipped_count).

    Only event == "keyframe" rows feed the NEES stats (loop_closure rows are the
    same keyframe logged again after correction — counting both double-counts).
    Rows whose Σ_pos is not positive-definite (e.g. the bootstrap prior) are
    skipped. loop_closure keyframe_ids are still collected for the plot markers.
    """
    errors = []
    covs = []
    kf_ids = []
    lc_ids = []
    skipped = 0

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["event"] == "loop_closure":
                lc_ids.append(float(row["keyframe_id"]))
                continue
            if row["event"] != "keyframe":
                continue

            e = np.array([
                float(row["est_x"]) - float(row["gt_x"]),
                float(row["est_y"]) - float(row["gt_y"]),
                float(row["est_z"]) - float(row["gt_z"]),
            ])
            sigma = np.array([
                [float(row["cov_33"]), float(row["cov_34"]), float(row["cov_35"])],
                [float(row["cov_34"]), float(row["cov_44"]), float(row["cov_45"])],
                [float(row["cov_35"]), float(row["cov_45"]), float(row["cov_55"])],
            ])

            # Positive-definite gate: Cholesky succeeds iff Σ_pos is PD. Catches
            # the tight-prior bootstrap (det ≈ 0) and any degenerate marginal.
            try:
                np.linalg.cholesky(sigma)
            except np.linalg.LinAlgError:
                skipped += 1
                continue

            errors.append(e)
            covs.append(sigma)
            kf_ids.append(float(row["keyframe_id"]))

    return (np.array(errors), np.array(covs), np.array(kf_ids),
            np.array(lc_ids), skipped)


def compute_nees(errors, covs):
    """ε_i = e_iᵀ Σ_i⁻¹ e_i via solve() (no explicit inverse)."""
    nees = np.empty(len(errors))
    for i, (e, sigma) in enumerate(zip(errors, covs)):
        nees[i] = float(e @ np.linalg.solve(sigma, e))
    return nees


def build_report(args, errors, covs, nees, n_used, n_skipped, dim):
    mean_nees = float(np.mean(nees))
    median_nees = float(np.median(nees))
    std_nees = float(np.std(nees))
    within_95 = 100.0 * float(np.mean(nees <= CHI2_3DOF_95))

    alpha = mean_nees / dim
    alpha_median = median_nees / dim

    nees_corr = nees / alpha
    mean_corr = float(np.mean(nees_corr))
    within_95_corr = 100.0 * float(np.mean(nees_corr <= CHI2_3DOF_95))

    # Per-axis diagonal NEES contribution: mean_i (e_axis^2 / Sigma_axis_axis).
    # Sum ≈ mean NEES when off-diagonal terms are small; reveals WHICH axis drives
    # the overconfidence (a single scalar alpha cannot fix an anisotropic bias).
    diag_var = np.array([np.diag(S) for S in covs])           # N x 3
    axis_contrib = np.mean(errors ** 2 / diag_var, axis=0)    # 3
    axis_ratio = np.median(np.abs(errors) / np.sqrt(diag_var), axis=0)  # median e/sigma

    lines = [
        "=== NEES Calibration Report ===",
        f"CSV: {args.csv}",
        f"Keyframe rows used: {n_used} (skipped: {n_skipped})",
        f"Dimension: {dim} (position only)",
        "",
        "--- Current (uncalibrated) ---",
        f"Mean NEES:     {mean_nees:8.2f}   (expected: {float(dim):.1f})",
        f"Median NEES:   {median_nees:8.2f}",
        f"Std NEES:      {std_nees:8.2f}",
        f"Within 95% CI: {within_95:6.1f}%       (expected: ~95%)",
        "",
        "--- Calibration ---",
        f"alpha = mean_NEES / dim = {alpha:.2f}",
        f"  (median-based alternative: alpha_median = {alpha_median:.2f})",
        f"Interpretation: Sigma_meas should be multiplied by alpha ~ {alpha:.1f}",
        "",
        "--- Simulated (after alpha correction) ---",
        f"Mean NEES:     {mean_corr:8.2f}   (target: {float(dim):.1f})",
        f"Within 95% CI: {within_95_corr:6.1f}%       (target: ~95%)",
        "",
        "--- Anisotropy diagnostic (why a single scalar is only a partial fix) ---",
        "Per-axis diagonal NEES contribution  [x, y, z] (sum ~ mean NEES):",
        f"  {axis_contrib[0]:8.1f}  {axis_contrib[1]:8.1f}  {axis_contrib[2]:8.1f}",
        "Median |error| / sigma  per axis     [x, y, z]:",
        f"  {axis_ratio[0]:8.2f}  {axis_ratio[1]:8.2f}  {axis_ratio[2]:8.2f}",
        "Note: if one axis dominates, the Hessian SHAPE (not just scale) is off;",
        "a scalar alpha restores mean consistency but leaves the per-axis imbalance.",
        "",
        "--- Recommendation ---",
        f"Apply alpha = {alpha:.2f} to NDT-12 Hessian scale:",
        "  Sigma_meas_calibrated = alpha * H^{-1}",
        "File to modify: graph_slam/src/ndt_registrar.cpp",
        "  (where the Hessian inverse is scaled into Sigma_meas)",
    ]
    return "\n".join(lines), alpha, nees_corr


def make_plot(out_png, kf_ids, nees, nees_corr, lc_ids, alpha, dim):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("eval05_nees: matplotlib unavailable, skipping plot", file=sys.stderr)
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.semilogy(kf_ids, nees, "b.-", lw=1.0, label=f"NEES (alpha=1)")
    ax.semilogy(kf_ids, nees_corr, ".-", color="orange", lw=1.0, alpha=0.7,
                label=f"NEES (calibrated, alpha={alpha:.1f})")
    ax.axhline(dim, color="green", ls="--", lw=1.2, label=f"ideal = {dim}")
    ax.axhline(CHI2_3DOF_95, color="red", ls="--", lw=1.2,
               label=f"95% chi2 = {CHI2_3DOF_95}")
    for i, lc in enumerate(lc_ids):
        ax.axvline(lc, color="red", ls=":", alpha=0.4,
                   label="loop closure" if i == 0 else None)
    ax.set_xlabel("keyframe_id")
    ax.set_ylabel("NEES (log scale)")
    ax.set_title(f"NEES per keyframe (current alpha=1 vs calibrated alpha={alpha:.1f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"eval05_nees: saved {out_png}")
    return True


def main():
    parser = argparse.ArgumentParser(description="EVAL-05 NEES calibration")
    parser.add_argument("--csv", default="analysis/eval05_covariance_log.csv")
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--output", default="analysis/nees_report.txt")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"eval05_nees: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    errors, covs, kf_ids, lc_ids, skipped = load_keyframes(args.csv)
    if len(errors) == 0:
        print("eval05_nees: no positive-definite keyframe rows found", file=sys.stderr)
        return 1

    nees = compute_nees(errors, covs)
    report, alpha, nees_corr = build_report(
        args, errors, covs, nees, len(nees), skipped, args.dim)

    print(report)
    with open(args.output, "w") as f:
        f.write(report + "\n")
    print(f"\neval05_nees: report written to {args.output}")

    out_png = os.path.join(os.path.dirname(os.path.abspath(args.output)),
                           "nees_over_keyframe.png")
    make_plot(out_png, kf_ids, nees, nees_corr, lc_ids, alpha, args.dim)

    # Sanity band (AC-4): flag an α that smells like a parsing bug.
    if alpha < 2.0:
        print(f"\neval05_nees: NOTE alpha={alpha:.2f} < 2 — system already near "
              "consistent, little calibration needed.", file=sys.stderr)
    elif alpha > 50.0:
        print(f"\neval05_nees: NOTE alpha={alpha:.2f} > 50 — first rule out a CSV/column "
              "parsing bug; if parsing checks out, this is genuine severe overconfidence "
              "(NEES is quadratic in error/sigma — see the anisotropy diagnostic).",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
