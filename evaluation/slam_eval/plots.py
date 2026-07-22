"""Matplotlib PNG plots for the confirmed metric set (headless Agg, like the
existing eval02/eval05 scripts these are refactored from):
  <name>_xy_trajectory.png  - top-down XY, aligned estimate vs GT
  <name>_error_over_time.png - position / rotation ATE error vs time
  <name>_nees_over_time.png  - per-pose NEES with ideal + 95% chi2 lines
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .consistency import NeesResult, chi2_95  # noqa: E402
from .metrics import AteResult, Trajectory  # noqa: E402


def plot_xy(path: str, name: str, est_aligned: Trajectory, gt: Trajectory, align: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(gt.pos[:, 0], gt.pos[:, 1], "-", c="tab:green", lw=2, label="ground truth")
    ax.plot(est_aligned.pos[:, 0], est_aligned.pos[:, 1], "-", c="tab:blue", lw=1.5,
            label=f"{name} (aligned: {align})")
    ax.plot(gt.pos[0, 0], gt.pos[0, 1], "ko", ms=6, label="start")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{name}: estimate vs ground truth (top-down XY)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_error_over_time(path: str, name: str, stamps, ate_result: AteResult) -> None:
    t = stamps - stamps[0]
    fig, (ax_t, ax_r) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_t.plot(t, ate_result.trans_errors_m, "-", c="tab:blue", lw=1.2)
    ax_t.axhline(ate_result.trans_rmse_m, color="tab:red", ls="--", lw=1,
                 label=f"RMSE = {ate_result.trans_rmse_m:.3f} m")
    ax_t.set_ylabel("position error [m]")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="best")
    ax_r.plot(t, ate_result.rot_errors_deg, "-", c="tab:purple", lw=1.2)
    ax_r.axhline(ate_result.rot_rmse_deg, color="tab:red", ls="--", lw=1,
                 label=f"RMSE = {ate_result.rot_rmse_deg:.2f} deg")
    ax_r.set_xlabel("time [s]")
    ax_r.set_ylabel("rotation error [deg]")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="best")
    fig.suptitle(f"{name}: ATE error over time (align: {ate_result.align})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_nees_over_time(path: str, name: str, res: NeesResult) -> None:
    t = res.stamps - res.stamps[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(t, res.nees, "b.-", lw=1.0, label="NEES")
    ax.axhline(res.dof, color="green", ls="--", lw=1.2, label=f"ideal = {res.dof}")
    ax.axhline(chi2_95(res.dof), color="red", ls="--", lw=1.2,
               label=f"95% chi2 = {chi2_95(res.dof):.2f}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NEES (log scale)")
    ax.set_title(f"{name}: position NEES over time "
                 f"(ANEES = {res.anees:.2f}, median = {res.median:.2f}, "
                 f"verdict: {res.verdict})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
