#!/usr/bin/env python3
"""NDT-14: ROS-free unit tests for the replay-harness pure logic.

Covers the two things the harness has to get right regardless of any rosbag:
  1. the baseline YAML (config/ndt14_baseline.yaml) parses into the right values;
  2. the pass/fail evaluation logic is correct on mock results (boundary cases,
     the missing-GT case, and the node-log parser that feeds it).

No ROS, no rosbag, no subprocesses: the harness module lazy-imports rclpy only
inside main()/the collector, so importing it here stays ROS-free.
"""
import os
import sys

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_PKG, "scripts"))

import ndt14_replay_harness as h  # noqa: E402

_BASELINE_YAML = os.path.join(_PKG, "config", "ndt14_baseline.yaml")


# --------------------------------------------------------------------------- #
# 1. baseline YAML parse                                                       #
# --------------------------------------------------------------------------- #

def test_baseline_yaml_parses_to_documented_values():
    b = h.load_baseline(_BASELINE_YAML)
    assert b.min_published_odom == 100
    assert b.max_degenerate_ratio == pytest.approx(0.15)
    assert b.max_mean_translation_err_m == pytest.approx(0.70)
    assert b.ekf2_prior_connected is True


def test_baseline_yaml_types():
    b = h.load_baseline(_BASELINE_YAML)
    assert isinstance(b.min_published_odom, int)
    assert isinstance(b.max_degenerate_ratio, float)
    assert isinstance(b.ekf2_prior_connected, bool)


def test_load_baseline_roundtrip_from_dict(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text(yaml.safe_dump({"ndt14_baseline": {
        "min_published_odom": 42,
        "max_degenerate_ratio": 0.2,
        "max_mean_translation_err_m": 1.0,
        "ekf2_prior_connected": False,
    }}))
    b = h.load_baseline(str(p))
    assert b.min_published_odom == 42
    assert b.max_degenerate_ratio == pytest.approx(0.2)
    assert b.ekf2_prior_connected is False


def test_load_baseline_missing_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"ndt14_baseline": {"min_published_odom": 1}}))
    with pytest.raises(KeyError):
        h.load_baseline(str(p))


# --------------------------------------------------------------------------- #
# 2. pass/fail logic on mock results                                           #
# --------------------------------------------------------------------------- #

def _baseline():
    return h.Baseline(min_published_odom=100, max_degenerate_ratio=0.15,
                      max_mean_translation_err_m=0.5, ekf2_prior_connected=True)


def _good_results():
    return h.HarnessResults(
        published_odom=120, reliable_count=112, degenerate_count=8,
        total_verdicts=120, ekf2_connected=True, non_identity_guess_count=110,
        max_guess_t_m=0.21, mean_translation_err_m=0.18, matched_gt_pairs=120)


def _checks_by_name(checks):
    return {c.name: c for c in checks}


def test_all_pass_on_good_results():
    checks = h.evaluate(_good_results(), _baseline())
    assert h.overall_pass(checks) is True
    assert all(c.passed for c in checks)


def test_too_few_odom_fails():
    r = _good_results()
    r.published_odom = 99
    checks = _checks_by_name(h.evaluate(r, _baseline()))
    assert checks["published_odom"].passed is False
    assert h.overall_pass(list(checks.values())) is False


def test_published_odom_boundary_is_inclusive():
    r = _good_results()
    r.published_odom = 100  # exactly the minimum must pass (>=)
    checks = _checks_by_name(h.evaluate(r, _baseline()))
    assert checks["published_odom"].passed is True


def test_degenerate_ratio_just_over_fails():
    r = _good_results()
    r.total_verdicts = 100
    r.degenerate_count = 15  # 0.15 is NOT < 0.15 -> fail
    checks = _checks_by_name(h.evaluate(r, _baseline()))
    assert checks["degenerate_ratio"].passed is False


def test_degenerate_ratio_just_under_passes():
    r = _good_results()
    r.total_verdicts = 100
    r.degenerate_count = 14  # 0.14 < 0.15 -> pass
    checks = _checks_by_name(h.evaluate(r, _baseline()))
    assert checks["degenerate_ratio"].passed is True


def test_zero_verdicts_fails_degenerate_check():
    r = _good_results()
    r.total_verdicts = 0
    r.degenerate_count = 0
    checks = _checks_by_name(h.evaluate(r, _baseline()))
    assert checks["degenerate_ratio"].passed is False


def test_missing_gt_pairs_fails_translation_check():
    r = _good_results()
    r.mean_translation_err_m = None
    r.matched_gt_pairs = 0
    checks = _checks_by_name(h.evaluate(r, _baseline()))
    assert checks["mean_translation_err_m"].passed is False
    assert h.overall_pass(list(checks.values())) is False


def test_translation_error_over_bound_fails():
    r = _good_results()
    r.mean_translation_err_m = 0.6
    checks = _checks_by_name(h.evaluate(r, _baseline()))
    assert checks["mean_translation_err_m"].passed is False


def test_ekf2_not_connected_fails():
    r = _good_results()
    r.ekf2_connected = False
    checks = _checks_by_name(h.evaluate(r, _baseline()))
    assert checks["ekf2_prior_connected"].passed is False
    assert h.overall_pass(list(checks.values())) is False


def test_failed_report_names_the_failing_check():
    r = _good_results()
    r.published_odom = 10
    checks = h.evaluate(r, _baseline())
    report = h.format_report(r, _baseline(), checks, {})
    assert "FAIL" in report
    assert "published_odom" in report
    # the failing check's detail must explain why (observed vs needed)
    failing = [c for c in checks if not c.passed]
    assert failing and "need" in failing[0].detail


# --------------------------------------------------------------------------- #
# node-log parser + error metric (feed the logic above)                        #
# --------------------------------------------------------------------------- #

_SAMPLE_LOG = """\
[INFO] ndt_frontend up. LiDAR '/x500/lidar_3d/points' ...
[INFO] EKF2 prior connected on '/odometry/ekf2' (first odometry received).
[INFO] NDT: converged=1 iters=12 fitness=-0.834 -> Reliable | moved 0.345 m / 2.10 deg | prior=identity (guess t=0.000 m)
[INFO] NDT: converged=1 iters=14 fitness=-0.901 -> Reliable | moved 0.402 m / 1.80 deg | prior=EKF2 (guess t=0.175 m)
[INFO] NDT: converged=1 iters=9 fitness=-0.450 -> Degenerate | moved 0.380 m / 0.50 deg | prior=EKF2 (guess t=0.210 m)
[INFO] NDT: converged=0 iters=35 fitness=-0.100 -> NotConverged | moved 0.050 m / 0.10 deg | prior=EKF2 (guess t=0.190 m)
"""


def test_parse_node_log_counts_verdicts_and_prior():
    r = h.HarnessResults()
    h.parse_node_log(_SAMPLE_LOG, r)
    assert r.total_verdicts == 4
    assert r.reliable_count == 2
    assert r.degenerate_count == 1
    assert r.ekf2_connected is True
    assert r.non_identity_guess_count == 3  # the three EKF2 guesses > 1e-4
    assert r.max_guess_t_m == pytest.approx(0.210)
    # identity-prior registration counts as "no prior" for the fitness comparison
    assert r.mean_fitness_no_prior == pytest.approx(-0.834)
    assert r.mean_fitness_with_prior is not None


def test_parse_node_log_requires_nonidentity_for_connected():
    log = ("[INFO] EKF2 prior connected on '/odometry/ekf2'.\n"
           "[INFO] NDT: converged=1 iters=5 fitness=-0.5 -> Reliable | "
           "moved 0.4 m / 1.0 deg | prior=identity (guess t=0.000 m)\n")
    r = h.HarnessResults()
    h.parse_node_log(log, r)
    # connection logged but no non-identity guess -> not "really" connected
    assert r.ekf2_connected is False


def test_mean_origin_anchored_error_is_zero_for_identical_tracks():
    # Same relative motion, but offset to a different absolute origin: the
    # origin-anchor must cancel the offset -> zero error.
    ndt = [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)]
    gt = [(0.0, 5.0, 5.0, 5.0), (1.0, 6.0, 5.0, 5.0), (2.0, 7.0, 5.0, 5.0)]
    err, pairs = h.mean_origin_anchored_error(ndt, gt)
    assert pairs == 3
    assert err == pytest.approx(0.0, abs=1e-9)


def test_mean_origin_anchored_error_detects_drift():
    ndt = [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)]
    gt = [(0.0, 0.0, 0.0, 0.0), (1.0, 1.5, 0.0, 0.0)]  # gt moves 1.5, ndt moves 1.0
    err, pairs = h.mean_origin_anchored_error(ndt, gt)
    assert pairs == 2
    # first pair anchored => 0; second => 0.5 m; mean = 0.25
    assert err == pytest.approx(0.25)


def test_mean_origin_anchored_error_ignores_trailing_orientation_fields():
    # Extra fields after z must be ignored (function takes position only).
    ndt = [(0.0, 0.0, 0.0, 0.0, 9, 9), (1.0, 1.0, 0.0, 0.0, 9, 9)]
    gt = [(0.0, 0.0, 0.0, 0.0, 1, 1), (1.0, 1.0, 0.0, 0.0, 1, 1)]
    err, pairs = h.mean_origin_anchored_error(ndt, gt)
    assert pairs == 2
    assert err == pytest.approx(0.0, abs=1e-9)


def test_mean_origin_anchored_error_empty():
    assert h.mean_origin_anchored_error([], []) == (None, 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
