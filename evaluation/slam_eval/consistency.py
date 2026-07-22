"""NEES-based covariance consistency (port of graph_slam/scripts/eval05_nees.py).

Position-only (dof = 3), like the EVAL-05 calibration: per matched pose
NEES_i = e_i^T Sigma_pos_i^-1 e_i with e_i the position error in the GT frame
and Sigma_pos the [x y z] 3x3 block of the pose covariance (contract order
[x y z roll pitch yaw] -> indices 0..2). Reports ANEES (mean), median,
95%-chi2 coverage, and a verdict from the Wilson-Hilferty ANEES confidence
band. Mean NEES is outlier-dominated (repo evidence: mean 22.6 vs median 2.19
on the calibrated run), which is why all three numbers ship together.

Errors never use the ATE Umeyama alignment (it optimizes the very errors NEES
scores and dumps residual error onto low-covariance poses, biasing the
verdict). Default is raw est-minus-gt (--nees_align none, the repo's EVAL-05
convention; frames must coincide); --nees_align first pins the first poses for
systems whose world frame differs from GT's. Any alignment rotation is applied
to both the error and the covariance so NEES stays frame-consistent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# chi2 95th percentiles for common dofs (exact); Wilson-Hilferty fallback otherwise.
_CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815, 6: 12.592}


def _chi2_ppf_wh(p: float, k: float) -> float:
    """Wilson-Hilferty chi-square quantile approximation (no scipy dependency)."""
    # normal quantile via Acklam-lite: good enough for the 2.5/95/97.5 points used here
    z = {0.025: -1.959964, 0.95: 1.644854, 0.975: 1.959964}[p]
    return k * (1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))) ** 3


def chi2_95(dof: int) -> float:
    return _CHI2_95.get(dof, _chi2_ppf_wh(0.95, dof))


@dataclass
class NeesResult:
    dof: int
    cov_scale: float
    count: int
    n_skipped_pd: int          # rows dropped by the positive-definite gate
    anees: float               # mean NEES (expected ~= dof when consistent)
    median: float
    std: float
    coverage_95_pct: float     # % of samples below chi2_95(dof); expected ~95
    anees_lo: float            # 95% ANEES consistency band
    anees_hi: float
    verdict: str               # overconfident | consistent | conservative
    nees: np.ndarray = field(repr=False)         # per-sample, for plots
    stamps: np.ndarray = field(repr=False)


def nees(pos_err: np.ndarray, covs: np.ndarray, stamps: np.ndarray,
         align_rot: np.ndarray, cov_scale: float, dof: int = 3) -> NeesResult:
    """pos_err: (N, 3) est-minus-gt position errors already in the GT frame.
    covs: (N, 6, 6) in contract order [x y z roll pitch yaw]. align_rot: the
    3x3 alignment rotation applied to the estimate (rotates the covariance)."""
    if dof != 3:
        raise ValueError("v1 is position-only: --dof must be 3")

    sig = cov_scale * np.einsum("ij,njk,lk->nil", align_rot, covs[:, :3, :3], align_rot)

    vals, used_stamps, skipped = [], [], 0
    for e, s, t in zip(pos_err, sig, stamps):
        try:
            # Cholesky succeeds iff Sigma_pos is PD; drops the tight bootstrap prior.
            np.linalg.cholesky(s)
        except np.linalg.LinAlgError:
            skipped += 1
            continue
        vals.append(float(e @ np.linalg.solve(s, e)))
        used_stamps.append(t)
    arr = np.array(vals)
    if len(arr) == 0:
        raise ValueError("no positive-definite covariance rows to score")

    n = len(arr)
    anees = float(np.mean(arr))
    # ANEES 95% band: sum of N chi2(dof) samples ~ chi2(N*dof); divide by N.
    lo = _chi2_ppf_wh(0.025, n * dof) / n
    hi = _chi2_ppf_wh(0.975, n * dof) / n
    if anees > hi:
        verdict = "overconfident"
    elif anees < lo:
        verdict = "conservative"
    else:
        verdict = "consistent"

    return NeesResult(
        dof=dof,
        cov_scale=cov_scale,
        count=n,
        n_skipped_pd=skipped,
        anees=anees,
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        coverage_95_pct=100.0 * float(np.mean(arr <= chi2_95(dof))),
        anees_lo=float(lo),
        anees_hi=float(hi),
        verdict=verdict,
        nees=arr,
        stamps=np.array(used_stamps),
    )
