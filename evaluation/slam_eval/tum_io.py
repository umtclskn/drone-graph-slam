"""File parsers/writers for the slam_eval I/O contract (see README.md).

- Trajectory: TUM format, one pose per line: `timestamp tx ty tz qx qy qz qw`.
- Covariance: one line per pose: `timestamp` + 21 values = row-major upper
  triangle of the 6x6 pose covariance in [x y z roll pitch yaw] order.
Lines starting with `#` are comments in both formats.
"""
from __future__ import annotations

import numpy as np

# Row-major upper-triangle index pairs of a 6x6 symmetric matrix (21 entries).
_UPPER_TRI_6 = [(i, j) for i in range(6) for j in range(i, 6)]


def load_tum(path: str) -> np.ndarray:
    """Load a TUM trajectory as an (N, 8) array [t, tx, ty, tz, qx, qy, qz, qw],
    sorted by timestamp."""
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.replace(",", " ").split()
            if len(vals) != 8:
                raise ValueError(f"{path}:{ln}: expected 8 fields (TUM), got {len(vals)}")
            rows.append([float(v) for v in vals])
    if not rows:
        raise ValueError(f"{path}: no poses found")
    arr = np.array(rows)
    return arr[np.argsort(arr[:, 0])]


def load_cov(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a covariance file as (stamps (N,), covs (N, 6, 6)) sorted by stamp.

    Each line: timestamp + 21 upper-triangle values, row-major,
    axis order [x y z roll pitch yaw].
    """
    stamps, covs = [], []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(v) for v in line.replace(",", " ").split()]
            if len(vals) != 22:
                raise ValueError(
                    f"{path}:{ln}: expected 22 fields (stamp + 21 upper-tri), got {len(vals)}")
            m = np.zeros((6, 6))
            for v, (i, j) in zip(vals[1:], _UPPER_TRI_6):
                m[i, j] = v
                m[j, i] = v
            stamps.append(vals[0])
            covs.append(m)
    if not stamps:
        raise ValueError(f"{path}: no covariance rows found")
    order = np.argsort(stamps)
    return np.array(stamps)[order], np.array(covs)[order]


def write_tum(path: str, rows: np.ndarray, header: str = "") -> None:
    """Write an (N, 8) [t, tx..qw] array in TUM format."""
    with open(path, "w") as f:
        if header:
            f.write(f"# {header}\n")
        for r in rows:
            f.write(f"{r[0]:.6f} " + " ".join(f"{v:.9g}" for v in r[1:]) + "\n")


def write_cov(path: str, stamps: np.ndarray, covs: np.ndarray, header: str = "") -> None:
    """Write (N,), (N, 6, 6) as stamp + 21 upper-tri values per line."""
    with open(path, "w") as f:
        if header:
            f.write(f"# {header}\n")
        for t, m in zip(stamps, covs):
            tri = " ".join(f"{m[i, j]:.9g}" for i, j in _UPPER_TRI_6)
            f.write(f"{t:.6f} {tri}\n")


def quats_to_rots(q: np.ndarray) -> np.ndarray:
    """Convert (N, 4) [qx, qy, qz, qw] quaternions to (N, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r
