#!/usr/bin/env python3
"""NDT-14 replay test harness.

Replays the canonical bag (`bags/slam_loop_02`) through the full NDT front-end and
asserts the results stay within recorded regression bounds. This is the artifact
that locks Sprint 2's "reproducible from a recorded bag" DoD and catches a
front-end regression after a refactor.

Replay recipe (learned from PX4-01 / EVAL-01), started in this order, all on
sim time, bag played WITHOUT ``--clock`` (the bag carries a recorded sim-time
``/clock``):

  1. ``ground_truth_bridge``      (EVAL-01)  -> /ground_truth/pose
  2. ``ekf2_odometry_adapter``    (PX4-01)   -> /odometry/ekf2  (the NDT-08 prior)
  3. ``ndt_frontend_node``        (NDT-13)   -> /ndt_frontend/ndt_odom
  4. ``ros2 bag play bags/slam_loop_02 --qos-profile-overrides-path ...``

What it measures and checks against ``config/ndt14_baseline.yaml``:
  * published_odom            >= min_published_odom
  * degenerate_ratio          <  max_degenerate_ratio          (from NDT-11 verdicts)
  * mean_translation_err_m    <  max_mean_translation_err_m     (vs GT, origin-anchored)
  * ekf2_prior_connected      == baseline                       (REAL bag data, NDT-08)

It also reports the EKF2-prior findings the story asks for (was /odometry/ekf2
received, were the initial guesses non-identity, and the fitness of prior vs
no-prior registrations) into ``analysis/ndt14_report.txt``.

YAGNI: no trajectory compounding, no formal ATE/RPE -- the mean translation
error is a crude origin-anchored sanity bound. The full ATE/RPE slice is EVAL-02
(Sprint 2 #7).

Usage (from the workspace root, after sourcing install/setup.bash):
    python3 src/drone-graph-slam/graph_slam/scripts/ndt14_replay_harness.py
    python3 .../ndt14_replay_harness.py --bag bags/slam_loop_02 --output analysis/ndt14_report.txt

The orchestration (subprocess launch, rclpy collection) is only reached from
main(); the pure baseline-parsing and pass/fail logic at the top of this module
is ROS-free and unit-tested by test/test_ndt14_harness.py.
"""
from __future__ import annotations

import argparse
import bisect
import datetime
import math
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import yaml

# --------------------------------------------------------------------------- #
# Pure, ROS-free logic (imported and exercised by the unit test).             #
# --------------------------------------------------------------------------- #

EPS_NON_IDENTITY_M = 1e-4  # guess translation above this counts as "non-identity"


@dataclass
class Baseline:
    """Regression bounds parsed from config/ndt14_baseline.yaml."""

    min_published_odom: int
    max_degenerate_ratio: float
    max_mean_translation_err_m: float
    ekf2_prior_connected: bool


def load_baseline(path: str) -> Baseline:
    """Parse the baseline YAML. Raises KeyError if a required key is missing."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    b = doc["ndt14_baseline"]
    return Baseline(
        min_published_odom=int(b["min_published_odom"]),
        max_degenerate_ratio=float(b["max_degenerate_ratio"]),
        max_mean_translation_err_m=float(b["max_mean_translation_err_m"]),
        ekf2_prior_connected=bool(b["ekf2_prior_connected"]),
    )


@dataclass
class HarnessResults:
    """Everything the harness observed during one replay run."""

    published_odom: int = 0
    reliable_count: int = 0
    degenerate_count: int = 0
    total_verdicts: int = 0
    ekf2_connected: bool = False
    non_identity_guess_count: int = 0
    max_guess_t_m: float = 0.0
    # mean origin-anchored ||ndt_odom - GT||; None if no GT pairs were matched.
    mean_translation_err_m: Optional[float] = None
    matched_gt_pairs: int = 0
    # Informational: mean NDT fitness with vs without the EKF2 prior (NDT-08 value).
    mean_fitness_with_prior: Optional[float] = None
    mean_fitness_no_prior: Optional[float] = None

    @property
    def degenerate_ratio(self) -> float:
        if self.total_verdicts == 0:
            return 0.0
        return self.degenerate_count / self.total_verdicts


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def evaluate(results: HarnessResults, baseline: Baseline) -> list[CheckResult]:
    """Compare observed results against the baseline. Pure, deterministic."""
    checks: list[CheckResult] = []

    checks.append(CheckResult(
        name="published_odom",
        passed=results.published_odom >= baseline.min_published_odom,
        detail=(f"observed {results.published_odom} ndt_odom messages, "
                f"need >= {baseline.min_published_odom}"),
    ))

    deg_ok = (results.total_verdicts > 0
              and results.degenerate_ratio < baseline.max_degenerate_ratio)
    checks.append(CheckResult(
        name="degenerate_ratio",
        passed=deg_ok,
        detail=(f"observed {results.degenerate_count}/{results.total_verdicts} "
                f"= {results.degenerate_ratio:.3f}, need < "
                f"{baseline.max_degenerate_ratio} "
                f"(0 verdicts => fail)" if results.total_verdicts == 0
                else f"observed {results.degenerate_count}/{results.total_verdicts} "
                     f"= {results.degenerate_ratio:.3f}, "
                     f"need < {baseline.max_degenerate_ratio}"),
    ))

    if results.mean_translation_err_m is None:
        checks.append(CheckResult(
            name="mean_translation_err_m",
            passed=False,
            detail="no ndt_odom/ground_truth pairs matched (GT bridge running?)",
        ))
    else:
        checks.append(CheckResult(
            name="mean_translation_err_m",
            passed=results.mean_translation_err_m < baseline.max_mean_translation_err_m,
            detail=(f"observed {results.mean_translation_err_m:.3f} m over "
                    f"{results.matched_gt_pairs} matched pairs, need < "
                    f"{baseline.max_mean_translation_err_m} m "
                    f"(crude origin-anchored bound, not ATE)"),
        ))

    checks.append(CheckResult(
        name="ekf2_prior_connected",
        passed=results.ekf2_connected == baseline.ekf2_prior_connected,
        detail=(f"connected={results.ekf2_connected} "
                f"(non-identity guesses: {results.non_identity_guess_count}, "
                f"max guess t={results.max_guess_t_m:.3f} m), "
                f"baseline expects {baseline.ekf2_prior_connected}"),
    ))

    return checks


def overall_pass(checks: list[CheckResult]) -> bool:
    return all(c.passed for c in checks)


def format_report(results: HarnessResults, baseline: Baseline,
                  checks: list[CheckResult], meta: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("NDT-14 replay test harness report")
    lines.append("=" * 72)
    lines.append(f"generated   : {meta.get('timestamp', '')}")
    lines.append(f"bag         : {meta.get('bag', '')}")
    lines.append(f"baseline    : {meta.get('baseline_path', '')}")
    lines.append(f"node log    : {meta.get('node_log', '')}")
    lines.append("")

    verdict = "PASS" if overall_pass(checks) else "FAIL"
    lines.append(f"OVERALL: {verdict}")
    lines.append("-" * 72)
    for c in checks:
        tag = "PASS" if c.passed else "FAIL"
        lines.append(f"  [{tag}] {c.name}: {c.detail}")
    lines.append("-" * 72)
    lines.append("")

    lines.append("Observed numbers")
    lines.append(f"  published ndt_odom      : {results.published_odom}")
    lines.append(f"  NDT verdicts (total)    : {results.total_verdicts}")
    lines.append(f"    Reliable              : {results.reliable_count}")
    lines.append(f"    Degenerate            : {results.degenerate_count}")
    lines.append(f"    degenerate ratio      : {results.degenerate_ratio:.3f}")
    if results.mean_translation_err_m is None:
        lines.append("  mean translation err    : n/a (no GT pairs matched)")
    else:
        lines.append(f"  mean translation err    : {results.mean_translation_err_m:.3f} m "
                     f"({results.matched_gt_pairs} pairs vs /ground_truth/pose)")
    lines.append("")

    lines.append("EKF2 prior findings (NDT-08 on REAL bag data, the harness's new value)")
    lines.append(f"  /odometry/ekf2 received : {results.ekf2_connected}")
    lines.append(f"  non-identity guesses    : {results.non_identity_guess_count}")
    lines.append(f"  max initial-guess |t|   : {results.max_guess_t_m:.3f} m")
    fp = results.mean_fitness_with_prior
    fn = results.mean_fitness_no_prior
    lines.append(f"  mean fitness w/ prior   : {fp:.3f}" if fp is not None
                 else "  mean fitness w/ prior   : n/a")
    lines.append(f"  mean fitness no prior   : {fn:.3f}" if fn is not None
                 else "  mean fitness no prior   : n/a")
    if fp is not None and fn is not None:
        lines.append(f"  fitness delta (no-pr - w/pr): {fn - fp:+.3f} "
                     "(lower fitness = better fit; negative delta => prior helps)")
    lines.append("")
    lines.append("Note: mean translation error is a crude origin-anchored sanity bound")
    lines.append("(both tracks anchored at their first matched position; odom and map")
    lines.append("are both ENU and ~aligned at takeoff), NOT a formal ATE/RPE (no")
    lines.append("whole-trajectory Umeyama fit, no RPE). Full ATE/RPE is EVAL-02 (#7).")
    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Node-log parsing (ROS-free: operates on plain text).                        #
# --------------------------------------------------------------------------- #

_VERDICT_RE = re.compile(r"->\s+(Reliable|NotConverged|PoorFit|Degenerate)\b")
_FITNESS_RE = re.compile(r"fitness=(-?\d+(?:\.\d+)?)")
_PRIOR_RE = re.compile(r"prior=(\w+)\s+\(guess t=(-?\d+(?:\.\d+)?)\s*m\)")
_CONNECTED_RE = re.compile(r"EKF2 prior connected")


def parse_node_log(text: str, results: HarnessResults) -> None:
    """Fill verdict counts, EKF2-connection and prior/fitness stats from the log."""
    fitness_with_prior: list[float] = []
    fitness_no_prior: list[float] = []

    if _CONNECTED_RE.search(text):
        results.ekf2_connected = True

    for line in text.splitlines():
        verdict_m = _VERDICT_RE.search(line)
        if not verdict_m:
            continue
        results.total_verdicts += 1
        verdict = verdict_m.group(1)
        if verdict == "Reliable":
            results.reliable_count += 1
        elif verdict == "Degenerate":
            results.degenerate_count += 1

        prior_m = _PRIOR_RE.search(line)
        fitness_m = _FITNESS_RE.search(line)
        fitness = float(fitness_m.group(1)) if fitness_m else None
        if prior_m:
            prior_type = prior_m.group(1)
            guess_t = float(prior_m.group(2))
            used_prior = (prior_type == "EKF2") and (guess_t > EPS_NON_IDENTITY_M)
            if used_prior:
                results.non_identity_guess_count += 1
                results.max_guess_t_m = max(results.max_guess_t_m, guess_t)
            if fitness is not None:
                (fitness_with_prior if used_prior else fitness_no_prior).append(fitness)
        elif fitness is not None:
            fitness_no_prior.append(fitness)

    if fitness_with_prior:
        results.mean_fitness_with_prior = sum(fitness_with_prior) / len(fitness_with_prior)
    if fitness_no_prior:
        results.mean_fitness_no_prior = sum(fitness_no_prior) / len(fitness_no_prior)

    # EKF2 prior is only "connected" for our purposes if it actually drove a
    # non-identity guess on real data (not just a socket connection).
    results.ekf2_connected = results.ekf2_connected and results.non_identity_guess_count >= 1


def mean_origin_anchored_error(
        ndt: "list[tuple]", gt: "list[tuple]") -> "tuple[Optional[float], int]":
    """Mean ||(ndt_i - ndt_0) - (gt_i - gt_0)|| over nearest-stamp matched pairs.

    Both trajectories are anchored at their first matched POSITION (single rigid
    translation offset). This works because the front-end's `odom` frame and the
    GT `map` frame are both ENU and ~aligned at takeoff, so the residual is
    front-end drift, not a frame mismatch. (A full SE(3) start-anchor was tried
    and measured WORSE here -- GT's real takeoff roll/pitch/yaw rotates its
    motion away from the already-aligned odom frame.)

    Deliberately NOT a formal ATE: no whole-trajectory Umeyama fit, no
    relative-pose (RPE) error -- just a coarse drift-sanity bound. The real
    ATE/RPE slice is EVAL-02 (Sprint 2 #7). Each sample is (t, x, y, z, ...);
    any orientation fields after z are ignored.
    """
    if not ndt or not gt:
        return None, 0
    gt_sorted = sorted(gt, key=lambda r: r[0])
    gt_times = [r[0] for r in gt_sorted]

    def nearest_gt(t: float) -> "tuple":
        i = bisect.bisect_left(gt_times, t)
        if i == 0:
            return gt_sorted[0]
        if i >= len(gt_sorted):
            return gt_sorted[-1]
        before, after = gt_sorted[i - 1], gt_sorted[i]
        return after if (after[0] - t) < (t - before[0]) else before

    ndt_sorted = sorted(ndt, key=lambda r: r[0])
    n0 = ndt_sorted[0]
    g0 = nearest_gt(n0[0])
    errs: list[float] = []
    for sample in ndt_sorted:
        _, gx, gy, gz = nearest_gt(sample[0])[:4]
        dx = (sample[1] - n0[1]) - (gx - g0[1])
        dy = (sample[2] - n0[2]) - (gy - g0[2])
        dz = (sample[3] - n0[3]) - (gz - g0[3])
        errs.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return (sum(errs) / len(errs), len(errs)) if errs else (None, 0)


# --------------------------------------------------------------------------- #
# Orchestration (rclpy + subprocesses) -- only reached from main().           #
# --------------------------------------------------------------------------- #


@dataclass
class _Proc:
    name: str
    popen: object
    log_path: Optional[str] = None
    log_file: object = None


def _start(name, cmd, log_path=None):
    """Launch a subprocess in its own process group (clean PID-group shutdown)."""
    import subprocess
    log_file = open(log_path, "w") if log_path else None
    out = log_file if log_file else subprocess.DEVNULL
    popen = subprocess.Popen(
        cmd,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group -> killpg by PID, never pkill
    )
    print(f"[harness] started {name} (pid {popen.pid}): {' '.join(cmd)}")
    return _Proc(name=name, popen=popen, log_path=log_path, log_file=log_file)


def _stop(proc: "_Proc", grace_s: float = 5.0) -> None:
    """Stop a process by its PID/process-group (SIGINT -> SIGTERM -> SIGKILL).

    Uses the process-group id from the PID we own -- NOT pkill (PX4-01 lesson).
    """
    popen = proc.popen
    if popen.poll() is not None:
        if proc.log_file:
            proc.log_file.close()
        return
    try:
        pgid = os.getpgid(popen.pid)
    except ProcessLookupError:
        pgid = None
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                popen.send_signal(sig)
        except ProcessLookupError:
            break
        try:
            popen.wait(timeout=grace_s)
            break
        except Exception:
            continue
    print(f"[harness] stopped {proc.name} (pid {popen.pid})")
    if proc.log_file:
        proc.log_file.close()


def _collect_via_rclpy(stop_after_s, on_ready):
    """Spin a small node collecting ndt_odom + GT poses until told to stop.

    Returns (ndt_samples, gt_samples) as lists of (t_sec, x, y, z). rclpy is
    imported lazily so importing this module stays ROS-free for the unit test.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped

    ndt_samples: list = []
    gt_samples: list = []

    rclpy.init()
    node = Node("ndt14_collector")
    qos = QoSProfile(depth=2000, reliability=ReliabilityPolicy.RELIABLE,
                     history=HistoryPolicy.KEEP_LAST)

    def on_odom(msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        ndt_samples.append((t, p.x, p.y, p.z))

    def on_gt(msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.position
        gt_samples.append((t, p.x, p.y, p.z))

    node.create_subscription(Odometry, "/ndt_frontend/ndt_odom", on_odom, qos)
    node.create_subscription(PoseStamped, "/ground_truth/pose", on_gt, qos)

    on_ready()  # collector is up; orchestrator may now launch the bag

    deadline = time.monotonic() + stop_after_s
    while rclpy.ok() and time.monotonic() < deadline and not on_ready.stop:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()
    return ndt_samples, gt_samples


class _Ready:
    """Tiny callable flag the collector calls when subscriptions are live."""

    def __init__(self):
        self._ready = False
        self.stop = False

    def __call__(self):
        self._ready = True

    @property
    def ready(self):
        return self._ready


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_baseline = os.path.normpath(os.path.join(here, "..", "config", "ndt14_baseline.yaml"))

    ap = argparse.ArgumentParser(description="NDT-14 replay test harness")
    ap.add_argument("--bag", default="bags/slam_loop_02")
    ap.add_argument("--baseline", default=default_baseline)
    ap.add_argument("--qos-overrides", default="bags/px4_qos_overrides.yaml")
    ap.add_argument("--params-file", default=None,
                    help="ndt_frontend slam_params.yaml (default: installed share copy)")
    ap.add_argument("--output", default="analysis/ndt14_report.txt")
    ap.add_argument("--node-log", default="analysis/ndt14_ndt_frontend.log")
    ap.add_argument("--bag-timeout", type=float, default=240.0,
                    help="max seconds to wait for `ros2 bag play` to finish")
    ap.add_argument("--settle", type=float, default=4.0,
                    help="seconds to wait for nodes to come up before playing the bag")
    ap.add_argument("--drain", type=float, default=4.0,
                    help="seconds to keep collecting after the bag finishes")
    ap.add_argument("--no-ekf2-prior", action="store_true",
                    help="skip the ekf2_odometry_adapter so the front-end runs with "
                         "identity guesses -- the A/B reference for NDT-08's value. "
                         "(The ekf2_prior_connected check is EXPECTED to fail in this "
                         "mode; use it only to measure no-prior fitness.)")
    args = ap.parse_args()

    baseline = load_baseline(args.baseline)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.node_log)), exist_ok=True)

    params_file = args.params_file
    if params_file is None:
        try:
            from ament_index_python.packages import get_package_share_directory
            params_file = os.path.join(
                get_package_share_directory("graph_slam"), "config", "slam_params.yaml")
        except Exception:
            params_file = None

    import threading

    ready = _Ready()
    collected: dict = {}

    def run_collector():
        collected["ndt"], collected["gt"] = _collect_via_rclpy(
            stop_after_s=args.bag_timeout + args.settle + args.drain + 30.0,
            on_ready=ready)

    collector_thread = threading.Thread(target=run_collector, daemon=True)
    collector_thread.start()
    # Wait for the collector's subscriptions to be live before launching anything.
    for _ in range(100):
        if ready.ready:
            break
        time.sleep(0.1)

    procs: list = []
    try:
        procs.append(_start(
            "ground_truth_bridge",
            ["ros2", "run", "px4_offboard", "ground_truth_bridge",
             "--ros-args", "-p", "use_sim_time:=true"]))
        if args.no_ekf2_prior:
            print("[harness] --no-ekf2-prior: NOT launching ekf2_odometry_adapter "
                  "(front-end will use identity guesses).")
        else:
            procs.append(_start(
                "ekf2_odometry_adapter",
                ["ros2", "run", "px4_offboard", "ekf2_odometry_adapter",
                 "--ros-args", "-p", "use_sim_time:=true"]))
        ndt_cmd = ["ros2", "run", "graph_slam", "ndt_frontend_node", "--ros-args"]
        if params_file and os.path.exists(params_file):
            ndt_cmd += ["--params-file", params_file]
        ndt_cmd += ["-p", "use_sim_time:=true", "-p", "publish_debug_clouds:=false"]
        procs.append(_start("ndt_frontend_node", ndt_cmd, log_path=args.node_log))

        print(f"[harness] waiting {args.settle:.0f}s for nodes to come up ...")
        time.sleep(args.settle)

        bag_cmd = ["ros2", "bag", "play", args.bag,
                   "--qos-profile-overrides-path", args.qos_overrides]
        bag = _start("bag_play", bag_cmd)
        print(f"[harness] playing bag (timeout {args.bag_timeout:.0f}s) ...")
        try:
            bag.popen.wait(timeout=args.bag_timeout)
        except Exception:
            print("[harness] WARNING: bag play exceeded timeout; stopping it.")
        _stop(bag)

        print(f"[harness] draining {args.drain:.0f}s of trailing messages ...")
        time.sleep(args.drain)
    finally:
        # Stop the live nodes by PID/process-group, in reverse order.
        for proc in reversed(procs):
            _stop(proc)
        ready.stop = True
        collector_thread.join(timeout=15.0)

    # ---- analyze ----------------------------------------------------------- #
    results = HarnessResults()
    ndt_samples = collected.get("ndt", [])
    gt_samples = collected.get("gt", [])
    results.published_odom = len(ndt_samples)
    err, pairs = mean_origin_anchored_error(ndt_samples, gt_samples)
    results.mean_translation_err_m = err
    results.matched_gt_pairs = pairs

    try:
        with open(args.node_log) as f:
            parse_node_log(f.read(), results)
    except FileNotFoundError:
        print(f"[harness] WARNING: node log {args.node_log} not found.")

    checks = evaluate(results, baseline)
    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "bag": args.bag,
        "baseline_path": args.baseline,
        "node_log": args.node_log,
    }
    report = format_report(results, baseline, checks, meta)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)
    print(f"[harness] report written to {args.output}")
    return 0 if overall_pass(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
