// L5-17: ROS-free raw-IMU window buffer (LIO-SAM's imuQueOpt pattern).
//
// Fixes the arrival-time window bug: samples are buffered keyed by their own
// header stamp, and each preintegration window is cut at the ndt_odom message's
// HEADER stamp (the time the ImuFactor's poses actually live at), not at
// whatever had accumulated by the time the callback happened to run. dt comes
// from consecutive sample stamps, so the result is a deterministic function of
// the stamped stream — independent of executor scheduling (this is what removes
// the non-deterministic IndeterminantLinearSystemException crash).
//
// Boundary policy: SNAP to the nearest stamp (LIO-SAM style, plan revision
// 2026-07-30) — samples at/after the cut stay buffered and the leftover
// [last-consumed-stamp, cut] time is attributed to the NEXT window through that
// window's first dt. No interpolation, no ZOH split.
//
// Pure algorithm class: no ROS, no PX4 — unit-testable without a simulator
// (see test_imu_window_buffer). Stamps must already be in the SLAM clock domain
// (sim time on replay); the clock reconciliation is the bridge's job (L5-17i,
// px4_offboard/imu_bridge.py), never this class's.
#pragma once

#include <cstddef>
#include <deque>

#include "graph_slam/imu/imu_preintegrator.hpp"

namespace graph_slam::imu {

// What one window cut did — the caller folds `gap_detected` into the existing
// no-IMU off-nominal fallback (L5-17e) so a gap-contaminated interval never
// ships an ImuFactor built across an unphysical dead-reckoned stretch.
struct WindowIntegrationResult {
  std::size_t integrated{0};   // samples folded into the preintegrator
  bool gap_detected{false};    // some consumed dt exceeded max_dt_gap_s
  double max_dt_s{0.0};        // largest consumed dt [s]
  double max_dt_stamp_s{0.0};  // stamp of the sample after that largest dt
};

class ImuWindowBuffer {
 public:
  // Buffer one sample. Samples are expected in non-decreasing stamp order (the
  // bridge publishes strictly monotonic stamps); an out-of-order stamp yields
  // dt<=0 at integration time and is skipped by ImuPreintegrator's guard.
  void push(const ImuSample& sample) { queue_.push_back(sample); }

  // Drop every buffered sample with stamp <= anchor_stamp_s and re-arm the dt
  // anchor there. Called at bootstrap: pre-X0 IMU precedes the first node and
  // is discarded; the first consumed sample's dt then spans [X0, its stamp].
  void reset(double anchor_stamp_s);

  // Drop buffered samples older than stamp_s WITHOUT touching the dt anchor.
  // Pre-bootstrap housekeeping only (bounds the queue while no window is
  // being cut yet).
  void pruneOlderThan(double stamp_s);

  // The LIO-SAM window cut: pop and integrate every buffered sample whose
  // stamp precedes cut_time_s; dt = sample stamp − previous consumed stamp
  // (or − the reset anchor for the first). A dt above max_dt_gap_s is still
  // integrated (some estimate beats none) but reported via gap_detected so the
  // caller can fall back for the whole interval. Samples at/after cut_time_s
  // stay buffered for the next window.
  WindowIntegrationResult integrateUpTo(double cut_time_s, double max_dt_gap_s,
                                        ImuPreintegrator& preint);

  std::size_t size() const { return queue_.size(); }
  double lastIntegratedStamp() const { return last_stamp_s_; }

 private:
  std::deque<ImuSample> queue_;
  // dt anchor: stamp of the last consumed sample (LIO-SAM's lastImuT_opt), or
  // the reset anchor right after reset(). Negative = never anchored.
  double last_stamp_s_{-1.0};
};

}  // namespace graph_slam::imu
