// L5-17: ROS-free tests for ImuWindowBuffer — the header-stamp window cut.
//
// The regression under test: the shipped L5-01/L5-03 intake cut each keyframe's
// preintegration window at message ARRIVAL (whatever had accumulated by the time
// the callback ran), while the ImuFactor constrains poses at HEADER stamps. Here
// we assert the buffer's output is a deterministic function of the stamped
// stream alone — independent of when/how integrateUpTo is called — plus the
// LIO-SAM boundary snap, the dt≤0 guard, and the L5-17e gap report.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "graph_slam/imu/imu_preintegrator.hpp"
#include "graph_slam/imu/imu_window_buffer.hpp"

namespace {

using graph_slam::imu::ImuPreintegrator;
using graph_slam::imu::ImuSample;
using graph_slam::imu::ImuWindowBuffer;
using graph_slam::imu::PreintegrationConfig;
using graph_slam::imu::WindowIntegrationResult;

constexpr double kRate = 250.0;           // nominal sample rate [Hz]
constexpr double kDt = 1.0 / kRate;       // nominal period [s]
constexpr double kMaxGap = 0.02;          // L5-17e default threshold [s]

// A hovering stream: gravity-cancelling specific force, zero gyro. Values do
// not matter for the windowing logic; stamps do.
std::vector<ImuSample> hoverStream(double t0, std::size_t n, double dt = kDt) {
  std::vector<ImuSample> out;
  out.reserve(n);
  for (std::size_t k = 0; k < n; ++k) {
    ImuSample s;
    s.stamp_s = t0 + static_cast<double>(k + 1) * dt;
    s.accel = Eigen::Vector3d(0.0, 0.0, 9.8);
    s.gyro = Eigen::Vector3d::Zero();
    out.push_back(s);
  }
  return out;
}

void expectPimEqual(const ImuPreintegrator& a, const ImuPreintegrator& b) {
  EXPECT_EQ(a.count(), b.count());
  EXPECT_NEAR(a.deltaTij(), b.deltaTij(), 1e-12);
  EXPECT_NEAR((a.deltaPij() - b.deltaPij()).norm(), 0.0, 1e-12);
  EXPECT_NEAR((a.deltaVij() - b.deltaVij()).norm(), 0.0, 1e-12);
  EXPECT_NEAR((a.covariance() - b.covariance()).norm(), 0.0, 1e-15);
}

// The direct regression test for the arrival-cut bug: the window content must
// depend only on which STAMPS precede the cut, not on how many samples had
// already been pushed when the cut ran (i.e. not on callback scheduling).
TEST(ImuWindowBuffer, WindowIsIndependentOfArrivalTiming) {
  const auto stream = hoverStream(0.0, 100);  // stamps 0.004 … 0.400
  const double cut1 = 0.2005;                 // 50 samples precede it
  const double cut2 = 0.4005;                 // the rest precede it

  // Arm A: "on-time" delivery — only the samples stamped before cut1 have
  // arrived when the first cut runs.
  PreintegrationConfig cfg;
  ImuPreintegrator pim_a(cfg);
  ImuWindowBuffer buf_a;
  buf_a.reset(0.0);
  for (std::size_t k = 0; k < 50; ++k) buf_a.push(stream[k]);
  const auto ra1 = buf_a.integrateUpTo(cut1, kMaxGap, pim_a);
  ImuPreintegrator pim_a2(cfg);
  for (std::size_t k = 50; k < stream.size(); ++k) buf_a.push(stream[k]);
  const auto ra2 = buf_a.integrateUpTo(cut2, kMaxGap, pim_a2);

  // Arm B: "late" callback — every sample is already buffered when the first
  // cut finally runs (this is exactly what front-end publish latency does).
  ImuPreintegrator pim_b(cfg);
  ImuWindowBuffer buf_b;
  buf_b.reset(0.0);
  for (const auto& s : stream) buf_b.push(s);
  const auto rb1 = buf_b.integrateUpTo(cut1, kMaxGap, pim_b);
  ImuPreintegrator pim_b2(cfg);
  const auto rb2 = buf_b.integrateUpTo(cut2, kMaxGap, pim_b2);

  EXPECT_EQ(ra1.integrated, rb1.integrated);
  EXPECT_EQ(ra2.integrated, rb2.integrated);
  EXPECT_EQ(ra1.integrated, 50u);
  EXPECT_EQ(ra2.integrated, 50u);
  expectPimEqual(pim_a, pim_b);
  expectPimEqual(pim_a2, pim_b2);
  EXPECT_FALSE(ra1.gap_detected || ra2.gap_detected || rb1.gap_detected ||
               rb2.gap_detected);
}

// Chunked vs single-shot calling cadence (executor jitter) yields the same PIM.
TEST(ImuWindowBuffer, DeterministicUnderVariedCallCadence) {
  const auto stream = hoverStream(10.0, 250);
  PreintegrationConfig cfg;

  ImuPreintegrator pim_once(cfg);
  ImuWindowBuffer buf_once;
  buf_once.reset(10.0);
  for (const auto& s : stream) buf_once.push(s);
  buf_once.integrateUpTo(11.0005, kMaxGap, pim_once);

  ImuPreintegrator pim_chunked(cfg);
  ImuWindowBuffer buf_chunked;
  buf_chunked.reset(10.0);
  for (const auto& s : stream) buf_chunked.push(s);
  // Cut at every "odom message" (~10 Hz) instead of once at the keyframe: the
  // accumulated PIM at the final cut must be identical.
  for (double cut = 10.1005; cut < 11.0006; cut += 0.1) {
    buf_chunked.integrateUpTo(cut, kMaxGap, pim_chunked);
  }
  expectPimEqual(pim_once, pim_chunked);
}

// LIO-SAM boundary snap: the sample straddling the cut stays buffered, and the
// leftover time reaches the NEXT window via that window's first dt — no window
// time is lost or double-counted across consecutive cuts.
TEST(ImuWindowBuffer, BoundarySnapCarriesRemainderToNextWindow) {
  const auto stream = hoverStream(0.0, 6);  // stamps 0.004 … 0.024
  PreintegrationConfig cfg;
  ImuWindowBuffer buf;
  buf.reset(0.0);
  for (const auto& s : stream) buf.push(s);

  ImuPreintegrator w1(cfg);
  const auto r1 = buf.integrateUpTo(0.0105, kMaxGap, w1);  // consumes .004, .008
  EXPECT_EQ(r1.integrated, 2u);
  EXPECT_NEAR(w1.deltaTij(), 0.008, 1e-12);        // snapped to the last stamp
  EXPECT_EQ(buf.size(), 4u);                        // straddler not consumed
  EXPECT_NEAR(buf.lastIntegratedStamp(), 0.008, 1e-12);

  ImuPreintegrator w2(cfg);
  const auto r2 = buf.integrateUpTo(0.0195, kMaxGap, w2);  // consumes .012, .016
  EXPECT_EQ(r2.integrated, 2u);
  // First dt of window 2 spans from the last consumed stamp (0.008), so the
  // [0.008, 0.012] remainder is attributed here — total time is conserved.
  EXPECT_NEAR(w2.deltaTij(), 0.008, 1e-12);
  EXPECT_NEAR(w1.deltaTij() + w2.deltaTij(), 0.016, 1e-12);
}

// L5-17e gap report: a ~134 ms stamp gap (the known slam_loop_03 PX4 step) is
// still integrated but flagged; nominal 4 ms jitter is not.
TEST(ImuWindowBuffer, GapPolicyFlagsLargeDtButStillIntegrates) {
  PreintegrationConfig cfg;
  ImuWindowBuffer buf;
  buf.reset(0.0);
  ImuSample a;  // nominal
  a.stamp_s = 0.004;
  a.accel = Eigen::Vector3d(0, 0, 9.8);
  ImuSample b = a;  // after a 134.6 ms gap
  b.stamp_s = 0.004 + 0.1346;
  ImuSample c = a;  // nominal again
  c.stamp_s = b.stamp_s + 0.004;
  buf.push(a);
  buf.push(b);
  buf.push(c);

  ImuPreintegrator pim(cfg);
  const auto r = buf.integrateUpTo(0.2, kMaxGap, pim);
  EXPECT_EQ(r.integrated, 3u);  // the gap sample IS integrated…
  EXPECT_TRUE(r.gap_detected);  // …but the interval is flagged for fallback
  EXPECT_NEAR(r.max_dt_s, 0.1346, 1e-9);
  EXPECT_NEAR(r.max_dt_stamp_s, b.stamp_s, 1e-12);
  EXPECT_NEAR(pim.deltaTij(), 0.004 + 0.1346 + 0.004, 1e-12);

  // Nominal stream: no flag.
  ImuWindowBuffer buf2;
  buf2.reset(0.0);
  for (const auto& s : hoverStream(0.0, 50)) buf2.push(s);
  ImuPreintegrator pim2(cfg);
  const auto r2 = buf2.integrateUpTo(1.0, kMaxGap, pim2);
  EXPECT_EQ(r2.integrated, 50u);
  EXPECT_FALSE(r2.gap_detected);
}

// Duplicate stamps (sim clock not advanced) are consumed without integrating
// and without throwing — the structural fix for the dt<=0 crash.
TEST(ImuWindowBuffer, DuplicateStampsAreConsumedNotIntegrated) {
  PreintegrationConfig cfg;
  ImuWindowBuffer buf;
  buf.reset(0.0);
  ImuSample a;
  a.stamp_s = 0.004;
  a.accel = Eigen::Vector3d(0, 0, 9.8);
  ImuSample dup = a;  // identical stamp
  ImuSample b = a;
  b.stamp_s = 0.008;
  buf.push(a);
  buf.push(dup);
  buf.push(b);

  ImuPreintegrator pim(cfg);
  const auto r = buf.integrateUpTo(0.1, kMaxGap, pim);
  EXPECT_EQ(r.integrated, 2u);  // dup skipped by the dt<=0 guard
  EXPECT_EQ(buf.size(), 0u);    // …but consumed (never wedges the queue)
  EXPECT_NEAR(pim.deltaTij(), 0.008, 1e-12);
}

// Bootstrap semantics: reset() drops pre-anchor samples (they precede X0) and
// the first consumed sample's dt spans from the anchor.
TEST(ImuWindowBuffer, ResetDropsPreAnchorSamplesAndAnchorsDt) {
  PreintegrationConfig cfg;
  ImuWindowBuffer buf;
  for (const auto& s : hoverStream(0.0, 10)) buf.push(s);  // 0.004 … 0.040
  buf.reset(0.020);                                        // bootstrap at t=0.020
  EXPECT_EQ(buf.size(), 5u);                               // 0.024 … 0.040 remain

  ImuPreintegrator pim(cfg);
  const auto r = buf.integrateUpTo(0.030, kMaxGap, pim);
  EXPECT_EQ(r.integrated, 2u);                 // 0.024, 0.028
  EXPECT_NEAR(pim.deltaTij(), 0.008, 1e-12);   // first dt = 0.024 − 0.020
}

// pruneOlderThan bounds the pre-bootstrap queue without touching the anchor.
TEST(ImuWindowBuffer, PruneOlderThanBoundsQueue) {
  ImuWindowBuffer buf;
  for (const auto& s : hoverStream(0.0, 10)) buf.push(s);
  buf.pruneOlderThan(0.021);
  EXPECT_EQ(buf.size(), 5u);
  EXPECT_LT(buf.lastIntegratedStamp(), 0.0);  // anchor untouched (never armed)
}

// AC [1]: W_ij = { samples | T_i <= t_k < T_j }. Samples at/after T_j stay
// buffered; samples at/before T_i are dropped by reset(); nothing outside the
// half-open interval is ever integrated into the edge's PIM.
TEST(ImuWindowBuffer, WindowMatchesHalfOpenKeyframeInterval) {
  // Stamps: 0.00, 0.01, …, 0.10. Keyframes at T_i=0.02, T_j=0.07 → expect
  // {0.03, 0.04, 0.05, 0.06} (4 samples), with 0.07 held for the next edge.
  PreintegrationConfig cfg;
  ImuWindowBuffer buf;
  for (int k = 0; k <= 10; ++k) {
    ImuSample s;
    s.stamp_s = 0.01 * k;
    s.accel = Eigen::Vector3d(0, 0, 9.8);
    buf.push(s);
  }
  buf.reset(0.02);  // T_i — drops stamps <= 0.02
  EXPECT_EQ(buf.size(), 8u);  // 0.03 … 0.10

  ImuPreintegrator pim(cfg);
  const auto r = buf.integrateUpTo(0.07, kMaxGap, pim);  // T_j
  EXPECT_EQ(r.integrated, 4u);
  EXPECT_NEAR(pim.deltaTij(), 0.04, 1e-12);  // 0.03−0.02 + 3×0.01
  EXPECT_EQ(buf.size(), 4u);                 // 0.07 … 0.10 remain
  EXPECT_NEAR(buf.lastIntegratedStamp(), 0.06, 1e-12);
}

// AC: constant-accel stream through the header-stamp window reproduces the
// closed-form NavState to <1e-3 (matches L5-01d's ImuPreintegrator expectation,
// but routed via ImuWindowBuffer so the window cut itself is in the path).
TEST(ImuWindowBuffer, ConstAccelThroughHeaderWindowMatchesAnalytic) {
  constexpr double kGLocal = 9.81;
  PreintegrationConfig cfg;
  cfg.gravity = kGLocal;

  const Eigen::Vector3d a_world(0.3, -0.2, 0.5);
  const Eigen::Vector3d v0(1.0, 0.5, -0.25);
  const Eigen::Vector3d measured_accel = a_world + Eigen::Vector3d(0, 0, kGLocal);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();

  constexpr double T_i = 0.0;
  constexpr double T_j = 1.0;
  constexpr double dt = 0.001;  // 1 kHz — same density as L5-01d
  ImuWindowBuffer buf;
  // Samples strictly inside [T_i, T_j): first at dt, last at T_j - dt.
  for (double t = T_i + dt; t < T_j - 1e-12; t += dt) {
    ImuSample s;
    s.stamp_s = t;
    s.accel = measured_accel;
    s.gyro = gyro;
    buf.push(s);
  }
  buf.reset(T_i);

  ImuPreintegrator preint(cfg);
  const auto r = buf.integrateUpTo(T_j, /*max_dt_gap_s=*/1.0, preint);
  EXPECT_EQ(r.integrated, 999u);  // dt … 0.999
  EXPECT_FALSE(r.gap_detected);
  EXPECT_NEAR(preint.deltaTij(), T_j - T_i - dt, 1e-9);  // snaps to last stamp

  // Predict over the snapped interval length Δ = T_j - dt (boundary snap).
  const double Delta = preint.deltaTij();
  const gtsam::NavState state_i(gtsam::Rot3(), gtsam::Point3(0, 0, 0), v0);
  const gtsam::NavState pred =
      preint.predict(state_i, gtsam::imuBias::ConstantBias());
  const Eigen::Vector3d expected_p = v0 * Delta + 0.5 * a_world * Delta * Delta;
  const Eigen::Vector3d expected_v = v0 + a_world * Delta;
  EXPECT_LT((pred.position() - gtsam::Point3(expected_p)).norm(), 1e-3);
  EXPECT_LT((pred.velocity() - expected_v).norm(), 1e-3);
  EXPECT_LT(gtsam::Rot3::Logmap(pred.attitude()).norm(), 1e-9);
}

}  // namespace
