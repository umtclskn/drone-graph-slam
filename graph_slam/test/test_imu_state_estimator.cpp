// DUAL-GRAPH: unit tests for the side-graph IMU state estimator.
//
// The estimator is the LIO-SAM `imuPreintegration.cpp` port: a small iSAM2 with
// keys X/V/B where the LiDAR pose enters as an ABSOLUTE PriorFactor<Pose3>.
// These tests pin the four behaviours the dual-graph wiring depends on:
// bootstrap, the absolute prior actually pulling the IMU chain, bias
// observability, and the two self-healing paths (periodic reset, failure).
#include "graph_slam/imu/imu_state_estimator.hpp"

#include <cmath>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

namespace {

using graph_slam::imu::ImuStateEstimator;
using graph_slam::imu::ImuStateEstimatorConfig;

constexpr double kG = 9.81;
constexpr double kImuDt = 1.0 / 250.0;  // the x500 bag's raw IMU rate

ImuStateEstimatorConfig makeConfig() {
  ImuStateEstimatorConfig c;  // LIO-SAM sigma defaults + L5-04 bootstrap priors
  c.preint.gravity = kG;
  return c;
}

// Feed `seconds` of a stationary-hover IMU stream: specific force cancels
// gravity exactly, no rotation. `extra_accel` is a constant sensor bias added on
// top (what the estimator should learn).
void feedHover(ImuStateEstimator& est, double seconds,
               const Eigen::Vector3d& extra_accel = Eigen::Vector3d::Zero()) {
  const Eigen::Vector3d accel = Eigen::Vector3d(0.0, 0.0, kG) + extra_accel;
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const int steps = static_cast<int>(std::round(seconds / kImuDt));
  for (int k = 0; k < steps; ++k) {
    est.addImuSample(accel, gyro, kImuDt);
  }
}

gtsam::Pose3 atX(double x) { return {gtsam::Rot3(), gtsam::Point3(x, 0.0, 0.0)}; }

// Bootstrap seeds X(0)/V(0)/B(0) at the given pose, zero velocity, zero bias.
TEST(ImuStateEstimator, BootstrapSeedsAtGivenPose) {
  ImuStateEstimator est(makeConfig());
  EXPECT_FALSE(est.initialized());

  const gtsam::Pose3 start(gtsam::Rot3::Yaw(0.3), gtsam::Point3(1.0, -2.0, 0.5));
  est.initialize(start);

  EXPECT_TRUE(est.initialized());
  EXPECT_EQ(est.key(), 1u);
  EXPECT_TRUE(est.latest().state.pose().equals(start, 1e-9));
  EXPECT_LT(est.latest().state.velocity().norm(), 1e-9);
  EXPECT_LT(est.latest().bias.vector().norm(), 1e-9);
}

// A correction with no IMU accumulated cannot build an ImuFactor: it is skipped,
// leaving the chain untouched (the off-nominal empty-interval case).
TEST(ImuStateEstimator, CorrectionWithoutImuIsSkipped) {
  ImuStateEstimator est(makeConfig());
  est.initialize(gtsam::Pose3());

  EXPECT_FALSE(est.correct(atX(0.5)).has_value());
  EXPECT_EQ(est.key(), 1u);
  EXPECT_EQ(est.correctionCount(), 0u);
  EXPECT_TRUE(est.initialized());
}

// The LiDAR pose enters as an ABSOLUTE prior, so a correction that disagrees
// with the IMU prediction must drag the new node toward the LiDAR pose — while
// the (stiff, short-interval) IMU chain keeps it from jumping the whole way.
// This is the behaviour the single-graph BetweenFactor arrangement cannot express.
TEST(ImuStateEstimator, AbsolutePriorPullsTheImuChain) {
  ImuStateEstimator est(makeConfig());
  est.initialize(gtsam::Pose3());

  feedHover(est, 0.2);  // IMU says "did not move"
  const auto out = est.correct(atX(0.5));  // LiDAR says "moved 0.5 m in +x"
  ASSERT_TRUE(out.has_value());

  const double x = out->state.pose().translation().x();
  EXPECT_GT(x, 0.0) << "absolute prior did not pull the chain at all";
  EXPECT_LT(x, 0.5) << "IMU chain exerted no counter-pull";
  EXPECT_GT(out->state.velocity().x(), 0.0);
  EXPECT_GT(out->prior_chi2, 0.0);
  EXPECT_NEAR(out->delta_t_s, 0.2, 1e-6);
  EXPECT_EQ(est.key(), 2u);
  EXPECT_EQ(est.correctionCount(), 1u);
}

// Bias observability: a stationary drone whose accelerometer reads a constant
// +0.05 m/s^2 offset in x. The IMU alone would dead-reckon away; the absolute
// prior holds the pose still, so the mismatch has to be absorbed by the bias.
TEST(ImuStateEstimator, BiasConvergesOnConstantAccelOffset) {
  ImuStateEstimatorConfig config = makeConfig();
  config.reset_key_interval = 0;  // isolate convergence from the periodic reset
  ImuStateEstimator est(config);
  est.initialize(gtsam::Pose3());

  const Eigen::Vector3d true_bias(0.05, 0.0, 0.0);
  for (int k = 0; k < 200; ++k) {  // 200 x 0.1 s = 20 s at rest
    feedHover(est, 0.1, true_bias);
    ASSERT_TRUE(est.correct(gtsam::Pose3()).has_value()) << "correction " << k;
  }

  const Eigen::Vector3d estimated = est.latest().bias.accelerometer();
  EXPECT_NEAR(estimated.x(), true_bias.x(), 0.01);
  EXPECT_LT(std::abs(estimated.y()), 0.01);
  EXPECT_LT(est.latest().state.pose().translation().norm(), 0.05);
  EXPECT_EQ(est.failureCount(), 0u);
}

// The periodic marginals-as-priors reset keeps the side graph O(1). It must be
// invisible in the state: the estimate stays continuous across a reset.
TEST(ImuStateEstimator, PeriodicResetKeepsStateContinuous) {
  ImuStateEstimatorConfig config = makeConfig();
  config.reset_key_interval = 5;
  ImuStateEstimator est(config);
  est.initialize(gtsam::Pose3());

  gtsam::Pose3 previous;
  for (int k = 0; k < 12; ++k) {
    feedHover(est, 0.1);
    const auto out = est.correct(gtsam::Pose3());
    ASSERT_TRUE(out.has_value()) << "correction " << k;
    // No jump at the reset boundary (the drone is stationary throughout).
    EXPECT_LT((out->state.pose().translation() - previous.translation()).norm(), 0.05)
        << "state jumped at correction " << k;
    previous = out->state.pose();
  }
  EXPECT_GE(est.resetCount(), 2u);
  EXPECT_LE(est.key(), 6u);  // key cycles back to 1 instead of growing to 13
  EXPECT_EQ(est.correctionCount(), 12u);
}

// Divergence guard: an implausible optimized velocity drops the estimator out of
// its initialized state, and the NEXT correction re-bootstraps at its LiDAR pose
// (LIO-SAM's resetParams + systemInitialized branch).
TEST(ImuStateEstimator, FailureDetectionReinitializes) {
  ImuStateEstimatorConfig config = makeConfig();
  config.max_velocity_mps = 0.001;  // any real motion trips it
  ImuStateEstimator est(config);
  est.initialize(gtsam::Pose3());

  feedHover(est, 0.2);
  EXPECT_FALSE(est.correct(atX(0.5)).has_value());
  EXPECT_FALSE(est.initialized());
  EXPECT_EQ(est.failureCount(), 1u);

  const gtsam::Pose3 recovery = atX(2.0);
  const auto out = est.correct(recovery);
  ASSERT_TRUE(out.has_value());
  EXPECT_TRUE(out->state.pose().equals(recovery, 1e-9));
  EXPECT_TRUE(est.initialized());
}

}  // namespace
