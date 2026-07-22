"""Trajectory accuracy metrics: alignment, ATE, time-based RPE, drift-per-metre.

ATE/RPE semantics follow the in-repo C++ reference
(graph_slam/src/trajectory_metrics.cpp): translation RMSE + geodesic rotation
RMSE, with RPE residual = (gt_i^-1 gt_j)^-1 (est_i^-1 est_j). Two deliberate
extensions for cross-system comparison:
  - alignment modes: `se3` (Umeyama, no scale — the cross-system standard),
    `first` (first-pose pinning, the C++ reference's mode, kept so historical
    repo numbers stay reproducible), `sim3` (Umeyama with scale; flatters
    metric sensors — flagged in the report when used).
  - RPE deltas are in SECONDS, not frame indices: keyframe rates differ across
    systems, so index gaps are not comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trajectory:
    stamps: np.ndarray  # (N,)
    pos: np.ndarray     # (N, 3)
    rot: np.ndarray     # (N, 3, 3)


def rotation_angles_deg(rots: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) of each (3, 3) rotation, robust near 0 and pi."""
    tr = np.clip((np.trace(rots, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) mapping src points onto dst."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1
    rot = u @ s_mat @ vt
    scale = float(np.trace(np.diag(d) @ s_mat) / xs.var(axis=0).sum()) if with_scale else 1.0
    trans = mu_d - scale * rot @ mu_s
    return scale, rot, trans


def align_trajectory(est: Trajectory, gt: Trajectory, mode: str) -> tuple[Trajectory, float]:
    """Return the estimate mapped into the GT frame, plus the scale used."""
    if mode == "first":
        rot = gt.rot[0] @ est.rot[0].T
        trans = gt.pos[0] - rot @ est.pos[0]
        scale = 1.0
    elif mode in ("se3", "sim3"):
        scale, rot, trans = umeyama(est.pos, gt.pos, with_scale=(mode == "sim3"))
    else:
        raise ValueError(f"unknown alignment mode: {mode}")
    return Trajectory(
        stamps=est.stamps,
        pos=(scale * est.pos @ rot.T) + trans,
        rot=rot @ est.rot,
    ), scale


@dataclass
class AteResult:
    align: str
    scale: float
    trans_rmse_m: float
    trans_mean_m: float
    trans_median_m: float
    trans_max_m: float
    rot_rmse_deg: float
    count: int
    trans_errors_m: np.ndarray = field(repr=False)  # per-pose, for plots
    rot_errors_deg: np.ndarray = field(repr=False)


def ate(est_aligned: Trajectory, gt: Trajectory, align_mode: str, scale: float) -> AteResult:
    e_trans = np.linalg.norm(est_aligned.pos - gt.pos, axis=1)
    e_rot = rotation_angles_deg(np.einsum("nij,nik->njk", gt.rot, est_aligned.rot))
    return AteResult(
        align=align_mode,
        scale=scale,
        trans_rmse_m=float(np.sqrt(np.mean(e_trans**2))),
        trans_mean_m=float(np.mean(e_trans)),
        trans_median_m=float(np.median(e_trans)),
        trans_max_m=float(np.max(e_trans)),
        rot_rmse_deg=float(np.sqrt(np.mean(e_rot**2))),
        count=len(e_trans),
        trans_errors_m=e_trans,
        rot_errors_deg=e_rot,
    )


@dataclass
class RpeResult:
    delta_s: float
    trans_rmse_m: float
    rot_rmse_deg: float
    drift_per_m: float  # mean(||trans residual|| / GT path length of the segment)
    count: int


def _relative(rot: np.ndarray, pos: np.ndarray, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    r_rel = rot[i].T @ rot[j]
    t_rel = rot[i].T @ (pos[j] - pos[i])
    return r_rel, t_rel


def rpe(est: Trajectory, gt: Trajectory, delta_s: float) -> RpeResult:
    """Time-based RPE: pair each pose i with the first pose j where
    t_j - t_i >= delta_s. RPE is frame-independent, so no alignment is applied
    (matches the C++ reference)."""
    stamps = gt.stamps
    gt_step = np.linalg.norm(np.diff(gt.pos, axis=0), axis=1)
    gt_cum = np.concatenate([[0.0], np.cumsum(gt_step)])

    t_sq, r_sq, drifts, count = 0.0, 0.0, [], 0
    j_candidates = np.searchsorted(stamps, stamps + delta_s)
    for i, j in enumerate(j_candidates):
        if j >= len(stamps):
            break
        rg, tg = _relative(gt.rot, gt.pos, i, j)
        re_, te = _relative(est.rot, est.pos, i, j)
        r_res = rg.T @ re_
        t_res = rg.T @ (te - tg)
        # residual translation of (gt_rel^-1 est_rel): R_g^T t_e - R_g^T t_g
        err_t = float(np.linalg.norm(t_res))
        tr = np.clip((np.trace(r_res) - 1.0) / 2.0, -1.0, 1.0)
        err_r = np.degrees(np.arccos(tr))
        t_sq += err_t**2
        r_sq += err_r**2
        seg_len = gt_cum[j] - gt_cum[i]
        if seg_len > 1e-6:
            drifts.append(err_t / seg_len)
        count += 1

    if count == 0:
        return RpeResult(delta_s, 0.0, 0.0, 0.0, 0)
    return RpeResult(
        delta_s=delta_s,
        trans_rmse_m=float(np.sqrt(t_sq / count)),
        rot_rmse_deg=float(np.sqrt(r_sq / count)),
        drift_per_m=float(np.mean(drifts)) if drifts else 0.0,
        count=count,
    )
