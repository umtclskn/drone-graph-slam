"""metrics.json (merged across runs) + report.md (one table row per system).

Each engine run scores ONE system and upserts its entry into
<out>/metrics.json, then regenerates <out>/report.md from every entry — so
LIO-SAM stacks in as a second row later with zero reformatting.

Provenance (source bag + ground-truth definition) is carried per entry in
`run.bag` / `run.gt_label` (set by the CLI, defaulting to the ground-truth
file's `#` header). The report states it per row and never claims rows share a
bag or a ground truth unless they actually do — EVAL-06 showed that swapping
EKF2 GT for independent Gazebo truth moves ATE by 3x, so a wrong global label
is a wrong result, not a cosmetic one.
"""
from __future__ import annotations

import json
import os
from typing import Any

_UNSPECIFIED = "unspecified"


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


def _inputs(entry: dict[str, Any]) -> tuple[str, str]:
    """(bag, ground-truth label) for one row; entries written before EVAL-07
    carry neither, and are reported as unspecified rather than assumed."""
    run = entry.get("run", {})
    return run.get("bag") or _UNSPECIFIED, run.get("gt_label") or _UNSPECIFIED


def _provenance(systems: dict[str, Any]) -> list[str]:
    """Intro block: bag + GT stated per row, shared only when genuinely shared."""
    pairs = {name: _inputs(e) for name, e in systems.items()}
    bags = {b for b, _ in pairs.values()}
    gts = {g for _, g in pairs.values()}
    if len(bags) == 1 and len(gts) == 1:
        bag, gt = next(iter(pairs.values()))
        return [f"All rows: bag `{bag}`, ground truth **{gt}**, "
                "same metric engine (`slam_eval`)."]
    varies = " and ".join(w for w, n in (("bag", len(bags)), ("ground truth", len(gts)))
                          if n > 1)
    lines = [f"Same metric engine (`slam_eval`) for every row, but the **{varies}** differs "
             "between rows — inputs per row below. Rows are comparable to each other only "
             "where both match.", ""]
    lines += [f"- `{n}`: bag `{b}`, ground truth **{g}**" for n, (b, g) in pairs.items()]
    return lines


def _caveats(systems: dict[str, Any]) -> list[str]:
    """Only the caveats that apply to the rows actually present."""
    def names(sel):
        return ", ".join(f"`{n}`" for n in sel)

    ekf2 = [n for n, e in systems.items() if "ekf2" in _inputs(e)[1].lower()]
    scored_cov = [n for n, e in systems.items() if e.get("nees")]
    no_cov = [n for n, e in systems.items() if not e.get("nees")]
    lio = [n for n in systems if n.startswith("lio_sam")]
    lio_loop02 = [n for n in lio if "slam_loop_02" in _inputs(systems[n])[0]]

    out = []
    if ekf2:
        out.append(
            f"- **EKF2-derived ground truth ({names(ekf2)})**: PX4 EKF2 output is an "
            "estimate with its own error, not independent simulator truth, so those "
            "accuracy numbers measure *consistency with EKF2*. Our system seeds NDT "
            "with an EKF2-derived initial guess (loosely coupled, one-way); LIO-SAM "
            "tightly couples the same IMU — whose orientation field the bridge "
            "(`imu_bridge_liosam`) fills from EKF2's attitude, the very estimator "
            "producing that GT — so both are flattered, differently. EVAL-06 measured "
            "the size of it: holding bag and estimate fixed, swapping EKF2 GT for "
            "independent Gazebo model-state GT drops ATE 0.602 m -> 0.203 m.")
    if scored_cov:
        out.append(
            f"- **`ndt_cov_scale_factor = 435` is not a validated calibration "
            f"({names(scored_cov)})**: it was NEES-fitted on `bags/slam_loop_02` "
            "against EKF2 GT (`analysis/calibration_log.md`). The follow-up "
            "comparison (EVAL-08, `analysis/slam_eval_loop03/COMPARISON.md`) refined "
            "EVAL-06's single-run overconfidence verdict: residual overconfidence is "
            "real but **small** once the bootstrap keyframe-#0 frame-definition "
            "artifact is excluded (ANEES ~1.05, median NEES ~1.06–1.34, vs EVAL-06's "
            "raw single-run 5.65), and NEES on raw error is not run-stable — so **alpha "
            "is not calibratable from a single run** (this sharpens rather than reverses "
            "EVAL-06). Consistency columns are reported with alpha unchanged; read them "
            "as an uncalibrated diagnostic, not a passing check.")
    if no_cov:
        detail = ""
        if lio:
            detail = (" For LIO-SAM this is structural: stock LIO-SAM leaves the 6x6 "
                      "pose covariance on `/lio_sam/mapping/odometry` all-zero (verified "
                      "empirically, LIO-SAM-04 run 2026-07-12: max |entry| = 0.0 over 118 "
                      "msgs; the only covariance use in its source is a 0/1 degeneracy "
                      "flag on `odometry_incremental` covariance[0]), so scoring NEES "
                      "would require patching its iSAM2 marginal extraction.")
        out.append(
            f"- **`n/a` consistency columns mean no covariance was supplied "
            f"({names(no_cov)})** — not a claim of perfect confidence.{detail}")
    if lio_loop02:
        out.append(
            "- **Loop closures on `bags/slam_loop_02`: ours 19, LIO-SAM 0 "
            f"({names(lio_loop02)}), structurally**: LIO-SAM's stock temporal gate "
            "`historyKeyframeSearchTimeDiff = 30 s` exceeds the ~24 s trajectory, so "
            "its distance-based detector can never fire. In a diagnostic run with that "
            "gate lowered to 2 s (comparable to our `loop_min_age_difference = 10` "
            "keyframes ~ 2 s) its ICP verifier accepted 22 closures — the scored row "
            "uses the stock gate.")
    if lio:
        out.append(
            f"- **LIO-SAM handicaps ({names(lio)})**: deskew disabled (the recorded "
            "cloud has no per-point time) and `useImuHeadingInitialization: false` "
            "starts its map frame ~2.8 deg off the ENU yaw — the SE3 Umeyama ATE "
            "alignment absorbs the yaw offset, raw-frame metrics would not.")
    return out


def write_report(out_dir: str, data: dict[str, Any]) -> str:
    """Regenerate report.md: one fixed-column table, one row per system."""
    systems = data["systems"]
    rpe_key = None  # first (smallest) RPE delta present, shown in the table
    for entry in systems.values():
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
    lines = ["# SLAM evaluation report", ""] + _provenance(systems) + [
        "",
        header,
        "|" + "---|" * (header.count("|") - 1),
    ]
    for name, e in systems.items():
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
    for name, e in systems.items():
        run = e.get("run", {})
        bag, gt_label = _inputs(e)
        detail = (f"- **{name}**: bag=`{bag}`, ground truth={gt_label}, "
                  f"align={run.get('align')}"
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

    lines += ["", "## Caveats", ""] + _caveats(systems) + [""]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
