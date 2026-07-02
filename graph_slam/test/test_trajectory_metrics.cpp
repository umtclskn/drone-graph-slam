// EVAL-02 (Sprint 2 full slice): ATE/RPE over known trajectories (ROS-free).
// Each case builds an analytic est/gt pair so the expected RMSE is computable by
// hand: perfect tracking -> ~0; a constant per-step drift -> a known ATE; a
// constant relative offset every step -> a known RPE. Pure Eigen, no ROS/PCL.

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>

#include "graph_slam/eval/trajectory_metrics.hpp"

namespace {

using graph_slam::absoluteTrajectoryError;
using graph_slam::relativePoseError;
using graph_slam::Trajectory;

Eigen::Isometry3f xyz(float x, float y, float z) {
  Eigen::Isometry3f t = Eigen::Isometry3f::Identity();
  t.translation() = Eigen::Vector3f(x, y, z);
  return t;
}

// A straight ground-truth path along +x, one metre per step.
Trajectory straightGt(std::size_t n) {
  Trajectory gt;
  for (std::size_t i = 0; i < n; ++i) {
    gt.push_back(xyz(static_cast<float>(i), 0.0F, 0.0F));
  }
  return gt;
}

// Perfect tracking: est == gt -> ATE and RPE both vanish.
TEST(TrajectoryMetricsTest, PerfectTrackingIsZero) {
  const Trajectory gt = straightGt(20);
  const Trajectory est = gt;

  const auto ate = absoluteTrajectoryError(est, gt);
  EXPECT_EQ(ate.count, 20U);
  EXPECT_LT(ate.trans_rmse_m, 1e-5F);
  EXPECT_LT(ate.rot_rmse_deg, 1e-4F);

  const auto rpe = relativePoseError(est, gt, 5);
  EXPECT_EQ(rpe.count, 15U);
  EXPECT_LT(rpe.trans_rmse_m, 1e-5F);
  EXPECT_LT(rpe.rot_rmse_deg, 1e-4F);
}

// A pure constant offset on every pose (est = gt shifted +1 m in y) is removed by
// first-pose alignment, so ATE is 0; RPE (frame-independent) is also 0 because the
// relative motion is identical.
TEST(TrajectoryMetricsTest, ConstantOffsetRemovedByAlignment) {
  const Trajectory gt = straightGt(10);
  Trajectory est;
  for (const auto& p : gt) {
    est.push_back(xyz(0.0F, 1.0F, 0.0F) * p);
  }

  const auto ate = absoluteTrajectoryError(est, gt);
  EXPECT_LT(ate.trans_rmse_m, 1e-5F);

  const auto rpe = relativePoseError(est, gt, 3);
  EXPECT_LT(rpe.trans_rmse_m, 1e-5F);
}

// Constant per-step y-drift: pose i is off by i*d in y after the first pose is
// pinned. ATE(trans) = d * sqrt(mean_i i^2) over i=0..n-1.
TEST(TrajectoryMetricsTest, LinearDriftMatchesClosedForm) {
  const std::size_t n = 11;
  const float d = 0.05F;
  const Trajectory gt = straightGt(n);
  Trajectory est;
  for (std::size_t i = 0; i < n; ++i) {
    est.push_back(xyz(static_cast<float>(i), d * static_cast<float>(i), 0.0F));
  }

  double sq = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double e = static_cast<double>(d) * static_cast<double>(i);
    sq += e * e;
  }
  const auto expected = static_cast<float>(std::sqrt(sq / static_cast<double>(n)));

  const auto ate = absoluteTrajectoryError(est, gt);
  EXPECT_NEAR(ate.trans_rmse_m, expected, 1e-5F);
}

// Each est step over-shoots the gt step by a fixed 0.1 m in x; RPE over delta=1
// must recover exactly 0.1 m, independent of where on the path it is measured.
TEST(TrajectoryMetricsTest, RelativeStepErrorRecovered) {
  const std::size_t n = 8;
  const Trajectory gt = straightGt(n);
  Trajectory est;
  est.push_back(xyz(0.0F, 0.0F, 0.0F));
  for (std::size_t i = 1; i < n; ++i) {
    est.push_back(xyz(1.1F * static_cast<float>(i), 0.0F, 0.0F));
  }

  const auto rpe = relativePoseError(est, gt, 1);
  EXPECT_EQ(rpe.count, n - 1);
  EXPECT_NEAR(rpe.trans_rmse_m, 0.1F, 1e-5F);
  EXPECT_LT(rpe.rot_rmse_deg, 1e-4F);
}

// A known constant yaw error per pose survives alignment of pose 0 and shows up as
// a rotation ATE; a 10 deg yaw on every pose but the first -> rot RMSE check.
TEST(TrajectoryMetricsTest, RotationErrorReported) {
  const std::size_t n = 5;
  const Trajectory gt = straightGt(n);
  const float deg10 = 10.0F * static_cast<float>(M_PI) / 180.0F;
  Trajectory est = gt;
  for (std::size_t i = 1; i < n; ++i) {
    est[i].linear() = Eigen::AngleAxisf(deg10, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  }

  // poses 1..4 each carry 10 deg, pose 0 carries 0 -> RMSE = 10*sqrt(4/5).
  const float expected = 10.0F * std::sqrt(4.0F / 5.0F);
  const auto ate = absoluteTrajectoryError(est, gt);
  EXPECT_NEAR(ate.rot_rmse_deg, expected, 1e-3F);
}

// Defensive contract: mismatched lengths / too-large delta -> count 0, no crash.
TEST(TrajectoryMetricsTest, DegenerateInputsReturnZeroCount) {
  EXPECT_EQ(absoluteTrajectoryError(straightGt(3), straightGt(4)).count, 0U);
  EXPECT_EQ(absoluteTrajectoryError({}, {}).count, 0U);
  EXPECT_EQ(relativePoseError(straightGt(3), straightGt(3), 5).count, 0U);
  EXPECT_EQ(relativePoseError(straightGt(3), straightGt(3), 0).count, 0U);
}

}  // namespace
