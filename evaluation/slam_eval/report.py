"""metrics.json (merged across runs) + report.md (one table row per system).

Each engine run scores ONE system and upserts its entry into
<out>/metrics.json, then regenerates <out>/report.md from every entry — so
LIO-SAM stacks in as a second row later with zero reformatting.
"""
from __future__ import annotations

import json
import os
from typing import Any


def upsert_metrics(out_dir: str, name: str, entry: dict[str, Any]) -> dict[str, Any]:
    path = os.path.join(out_dir, "metrics.json")
    data: dict[str, Any] = {"systems": {}}
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        data.setdefault("systems", {})
    data["systems"][name] = entry
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def _fmt(v, spec=".3f"):
    return format(v, spec) if isinstance(v, (int, float)) else "n/a"


def write_report(out_dir: str, data: dict[str, Any]) -> str:
    """Regenerate report.md: one fixed-column table, one row per system."""
    rpe_key = None  # first (smallest) RPE delta present, shown in the table
    for entry in data["systems"].values():
        deltas = sorted(float(k) for k in entry.get("rpe", {}))
        if deltas:
            rpe_key = deltas[0]
            break

    if rpe_key is not None:
        header = (f"| System | ATE trans RMSE [m] | ATE rot RMSE [deg] | "
                  f"RPE@{rpe_key:g}s trans [m] | RPE@{rpe_key:g}s rot [deg] | drift [m/m] | "
                  f"ANEES | median NEES | 95% coverage [%] | Consistency | matched/dropped |")
    else:
        header = ("| System | ATE trans RMSE [m] | ATE rot RMSE [deg] | ANEES | median NEES | "
                  "95% coverage [%] | Consistency | matched/dropped |")
    lines = [
        "# SLAM evaluation report",
        "",
        "Same bag, same ground truth, same metric engine (`slam_eval`) for every row.",
        "",
        header,
        "|" + "---|" * (header.count("|") - 1),
    ]
    for name, e in data["systems"].items():
        ate_e, nees_e = e.get("ate", {}), e.get("nees")
        rpe_e = e.get("rpe", {}).get(f"{rpe_key:g}") if rpe_key is not None else None
        cells = [
            name,
            _fmt(ate_e.get("trans_rmse_m")),
            _fmt(ate_e.get("rot_rmse_deg"), ".2f"),
        ]
        if rpe_key is not None:
            cells += [
                _fmt(rpe_e.get("trans_rmse_m")) if rpe_e else "n/a",
                _fmt(rpe_e.get("rot_rmse_deg"), ".2f") if rpe_e else "n/a",
                _fmt(rpe_e.get("drift_per_m"), ".4f") if rpe_e else "n/a",
            ]
        cells += [
            _fmt(nees_e.get("anees"), ".2f") if nees_e else "n/a",
            _fmt(nees_e.get("median"), ".2f") if nees_e else "n/a",
            _fmt(nees_e.get("coverage_95_pct"), ".1f") if nees_e else "n/a",
            nees_e.get("verdict", "n/a") if nees_e else "n/a",
            f"{e['association']['matched']}/{e['association']['dropped']}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Run details", ""]
    for name, e in data["systems"].items():
        run = e.get("run", {})
        detail = (f"- **{name}**: align={run.get('align')}"
                  + (f" (sim3 scale={e['ate'].get('scale'):.4f} — scale-corrected, flatters "
                     "metric sensors)" if run.get("align") == "sim3" else "")
                  + f", max_dt={run.get('max_dt')}s, cov_scale={run.get('cov_scale')}, "
                  f"RPE deltas={run.get('rpe_deltas')}s.")
        if e.get("rpe"):
            sweep = "; ".join(
                f"Δ={d}s: {v['trans_rmse_m']:.3f} m / {v['rot_rmse_deg']:.2f}° "
                f"({v['drift_per_m']:.4f} m/m, n={v['count']})"
                for d, v in sorted(e["rpe"].items(), key=lambda kv: float(kv[0])))
            detail += f"\n  - RPE sweep: {sweep}"
        if e.get("nees"):
            n = e["nees"]
            detail += (f"\n  - NEES: n={n['count']} (PD-skipped {n['n_skipped_pd']}), "
                       f"ANEES 95% band [{n['anees_lo']:.2f}, {n['anees_hi']:.2f}].")
        lines.append(detail)

    lines += [
        "",
        "## Caveats (apply to every row on `bags/slam_loop_02`)",
        "",
        "- **Ground truth is PX4 EKF2 output, not independent Gazebo truth** "
        "(EVAL-06 open): all accuracy numbers measure consistency with EKF2. "
        "Our system seeds NDT with an EKF2-derived initial guess (loosely coupled, "
        "one-way); LIO-SAM tightly couples the same IMU — whose orientation field "
        "the bridge (`imu_bridge_liosam`) fills from EKF2's attitude, the very "
        "estimator producing this GT — so both rows are flattered, differently.",
        "- **`ndt_cov_scale_factor = 435` was NEES-calibrated on this same bag** "
        "(`analysis/calibration_log.md`): our consistency row is in-sample until "
        "cross-validated on a second bag.",
        "- **LIO-SAM NEES is N/A (no covariance published)**: stock LIO-SAM leaves "
        "the 6x6 pose covariance on `/lio_sam/mapping/odometry` all-zero (verified "
        "empirically, LIO-SAM-04 run 2026-07-12: max |entry| = 0.0 over 118 msgs; "
        "the only covariance use in its source is a 0/1 degeneracy flag on "
        "`odometry_incremental` covariance[0]). Its consistency columns are `n/a`, "
        "not a claim of perfect confidence; scoring NEES would require patching its "
        "iSAM2 marginal extraction.",
        "- **Loop closures on this bag: ours 19, LIO-SAM 0 (structural)**: LIO-SAM's "
        "stock temporal gate `historyKeyframeSearchTimeDiff = 30 s` exceeds the "
        "~24 s trajectory, so its distance-based detector can never fire. In a "
        "diagnostic run with that gate lowered to 2 s (comparable to our "
        "`loop_min_age_difference = 10` keyframes ≈ 2 s) its ICP verifier accepted "
        "22 closures — the scored `lio_sam` row uses the stock gate.",
        "- **LIO-SAM handicaps on this bag**: deskew disabled (cloud has no "
        "per-point time) and `useImuHeadingInitialization: false` starts its map "
        "frame ~2.8° off EKF2's ENU yaw — the SE3 Umeyama ATE alignment absorbs "
        "the yaw offset, raw-frame metrics would not.",
        "",
    ]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
