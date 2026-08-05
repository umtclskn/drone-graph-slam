// L5-21: unit tests for the NDT↔IMU χ² consistency gate + Huber factor builder.
#include <cmath>
#include <limits>

#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>

#include "graph_slam/graph/ndt_consistency_gate.hpp"
#include "graph_slam/graph/odometry_factor_builder.hpp"

using graph_slam::graph::ConsistencyAction;
using graph_slam::graph::ConsistencyGateConfig;
using graph_slam::graph::ConsistencyGateMode;
using graph_slam::graph::Matrix6;
using graph_slam::graph::OdometryFactorBuilder;
using graph_slam::graph::evaluateNdtConsistency;
using graph_slam::graph::kDefaultConsistencyChi2Threshold;
using graph_slam::graph::poseBlockFromPreintCov;

namespace {

Matrix6 isotropicPoseSigma(double s) {
  return Matrix6::Identity() * (s * s);
}

TEST(NdtConsistencyGateTest, PoseBlockExtractsRotPosFromPreintCov) {
  Eigen::Matrix<double, 9, 9> cov9 = Eigen::Matrix<double, 9, 9>::Zero();
  cov9(0, 0) = 1.0;
  cov9(3, 3) = 4.0;
  cov9(0, 3) = cov9(3, 0) = 0.5;
  cov9(6, 6) = 99.0;  // velocity block must be dropped
  const Matrix6 pose = poseBlockFromPreintCov(cov9);
  EXPECT_DOUBLE_EQ(pose(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(pose(3, 3), 4.0);
  EXPECT_DOUBLE_EQ(pose(0, 3), 0.5);
  EXPECT_DOUBLE_EQ(pose(5, 5), 0.0);
  EXPECT_DOUBLE_EQ(pose(2, 2), 0.0);
}

TEST(NdtConsistencyGateTest, DefaultConfigIsDisabled) {
  ConsistencyGateConfig cfg;  // production default
  EXPECT_FALSE(cfg.enabled);
  EXPECT_EQ(cfg.mode, ConsistencyGateMode::Strict);
}

TEST(NdtConsistencyGateTest, GateDisabledReturnsNoneAndKeepsSigma) {
  ConsistencyGateConfig cfg;
  cfg.enabled = false;
  const gtsam::Pose3 a;  // identity
  const gtsam::Pose3 b(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const Matrix6 sigma = isotropicPoseSigma(0.1);
  const auto r = evaluateNdtConsistency(a, b, sigma, isotropicPoseSigma(0.05), cfg);
  EXPECT_EQ(r.action, ConsistencyAction::None);
  EXPECT_TRUE(r.add_ndt);
  EXPECT_NEAR((r.sigma_ndt_used - sigma).norm(), 0.0, 1e-15);
  EXPECT_DOUBLE_EQ(r.chi2, -1.0);
}

TEST(NdtConsistencyGateTest, ConsistentPosesAcceptNearDof) {
  // Identical poses → ξ=0 → χ²=0 → Accept.
  ConsistencyGateConfig cfg;
  cfg.enabled = true;
  cfg.mode = ConsistencyGateMode::Strict;
  cfg.chi2_threshold = kDefaultConsistencyChi2Threshold;
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.1, -0.05, 0.02),
                          gtsam::Point3(1.0, 2.0, 0.5));
  const Matrix6 sigma = isotropicPoseSigma(0.05);
  const auto r = evaluateNdtConsistency(pose, pose, sigma, sigma, cfg);
  EXPECT_EQ(r.action, ConsistencyAction::Accept);
  EXPECT_TRUE(r.add_ndt);
  EXPECT_NEAR(r.chi2, 0.0, 1e-12);
  EXPECT_NEAR(r.innovation.norm(), 0.0, 1e-12);
}

TEST(NdtConsistencyGateTest, SmallConsistentOffsetHasChi2NearDofScale) {
  // ξ along x of 0.05 m with σ=0.05 on that axis → χ² contribution ≈ 1 per
  // that dof when Σ_x_pred is tiny (gate sees mostly Σ_ndt).
  ConsistencyGateConfig cfg;
  cfg.enabled = true;
  cfg.chi2_threshold = kDefaultConsistencyChi2Threshold;
  const gtsam::Pose3 x_pred;
  const gtsam::Pose3 x_ndt(gtsam::Rot3(), gtsam::Point3(0.05, 0.0, 0.0));
  const Matrix6 sigma_ndt = isotropicPoseSigma(0.05);
  const Matrix6 sigma_pred = isotropicPoseSigma(1e-6);  // ~exact IMU
  const auto r =
      evaluateNdtConsistency(x_pred, x_ndt, sigma_ndt, sigma_pred, cfg);
  EXPECT_EQ(r.action, ConsistencyAction::Accept);
  EXPECT_TRUE(r.add_ndt);
  // One translational sigma away → χ² ≈ 1 (other dofs ~0).
  EXPECT_NEAR(r.chi2, 1.0, 0.05);
}

TEST(NdtConsistencyGateTest, LargeInnovationStrictDrops) {
  ConsistencyGateConfig cfg;
  cfg.enabled = true;
  cfg.mode = ConsistencyGateMode::Strict;
  cfg.chi2_threshold = kDefaultConsistencyChi2Threshold;
  const gtsam::Pose3 x_pred;
  // 1 m offset with σ=0.05 → ~400 σ² → huge χ²
  const gtsam::Pose3 x_ndt(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const Matrix6 sigma = isotropicPoseSigma(0.05);
  const auto r = evaluateNdtConsistency(x_pred, x_ndt, sigma, sigma, cfg);
  EXPECT_EQ(r.action, ConsistencyAction::Drop);
  EXPECT_FALSE(r.add_ndt);
  EXPECT_GT(r.chi2, cfg.chi2_threshold);
}

TEST(NdtConsistencyGateTest, LargeInnovationSoftInflates) {
  ConsistencyGateConfig cfg;
  cfg.enabled = true;
  cfg.mode = ConsistencyGateMode::Soft;
  cfg.chi2_threshold = kDefaultConsistencyChi2Threshold;
  const gtsam::Pose3 x_pred;
  const gtsam::Pose3 x_ndt(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const Matrix6 sigma = isotropicPoseSigma(0.05);
  const auto r = evaluateNdtConsistency(x_pred, x_ndt, sigma, sigma, cfg);
  EXPECT_EQ(r.action, ConsistencyAction::Inflate);
  EXPECT_TRUE(r.add_ndt);
  EXPECT_GT(r.chi2, cfg.chi2_threshold);
  const double expected_scale = r.chi2 / cfg.chi2_threshold;
  EXPECT_NEAR((r.sigma_ndt_used - sigma * expected_scale).norm(), 0.0, 1e-9);
}

TEST(NdtConsistencyGateTest, HuberRobustFactorWrapsGaussian) {
  OdometryFactorBuilder builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  gtsam::Matrix66 sigma = gtsam::Matrix66::Identity() * 0.01;
  const gtsam::Pose3 delta(gtsam::Rot3(), gtsam::Point3(0.5, 0.0, 0.0));
  const auto factor =
      builder.create_robust_factor(x0, x1, delta, sigma, /*huber_k=*/1.345);
  ASSERT_NE(factor.noiseModel(), nullptr);
  const auto* robust =
      dynamic_cast<const gtsam::noiseModel::Robust*>(factor.noiseModel().get());
  ASSERT_NE(robust, nullptr);
  ASSERT_NE(robust->robust(), nullptr);
  const auto gaussian_factor = builder.create_factor(x0, x1, delta, sigma);
  EXPECT_EQ(
      dynamic_cast<const gtsam::noiseModel::Robust*>(gaussian_factor.noiseModel().get()),
      nullptr);
}

TEST(NdtConsistencyGateTest, ToStringCoversActions) {
  EXPECT_STREQ(graph_slam::graph::toString(ConsistencyAction::None), "none");
  EXPECT_STREQ(graph_slam::graph::toString(ConsistencyAction::Accept), "accept");
  EXPECT_STREQ(graph_slam::graph::toString(ConsistencyAction::Drop), "drop");
  EXPECT_STREQ(graph_slam::graph::toString(ConsistencyAction::Inflate), "inflate");
}

}  // namespace
