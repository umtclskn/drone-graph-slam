"""Nearest-neighbour timestamp association between estimate and ground truth.

Same policy as the existing in-repo matchers (`nearestPose` in
eval02_relative_error_node.cpp, `nearest_gt` in ndt14_replay_harness.py):
for every estimate stamp take the closest GT stamp; drop the pair when the
gap exceeds `max_dt`. GT poses may be matched more than once (GT is usually
much denser than the estimate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Association:
    est_idx: np.ndarray  # indices into the estimate arrays
    gt_idx: np.ndarray   # matched indices into the GT arrays
    n_est: int           # total estimate poses
    n_dropped: int       # estimate poses with no GT within max_dt
    max_gap: float       # largest |dt| among accepted matches (s)

    @property
    def n_matched(self) -> int:
        return len(self.est_idx)


def associate(est_stamps: np.ndarray, gt_stamps: np.ndarray, max_dt: float) -> Association:
    """Match each estimate stamp to its nearest GT stamp within `max_dt` seconds.

    Both stamp arrays must be sorted ascending (tum_io loaders guarantee this).
    """
    pos = np.searchsorted(gt_stamps, est_stamps)
    lo = np.clip(pos - 1, 0, len(gt_stamps) - 1)
    hi = np.clip(pos, 0, len(gt_stamps) - 1)
    pick = np.where(
        np.abs(gt_stamps[hi] - est_stamps) < np.abs(gt_stamps[lo] - est_stamps), hi, lo)
    gaps = np.abs(gt_stamps[pick] - est_stamps)
    ok = gaps <= max_dt
    est_idx = np.nonzero(ok)[0]
    return Association(
        est_idx=est_idx,
        gt_idx=pick[ok],
        n_est=len(est_stamps),
        n_dropped=int(np.sum(~ok)),
        max_gap=float(gaps[ok].max()) if ok.any() else 0.0,
    )
