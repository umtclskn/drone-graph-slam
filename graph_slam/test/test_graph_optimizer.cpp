#include <gtest/gtest.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "graph_slam/graph/graph_bootstrap.hpp"
#include "graph_slam/graph/graph_optimizer.hpp"
#include "graph_slam/graph/odometry_factor_builder.hpp"

namespace {

using graph_slam::graph::GraphBootstrap;
using graph_slam::graph::GraphOptimizer;
using graph_slam::graph::OdometryFactorBuilder;

constexpr double kTol = 1e-6;

gtsam::SharedNoiseModel diagonalNoise(const gtsam::Vector6& sigmas) {
  return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::Matrix66 identitySigma(double scale) {
  return gtsam::Matrix66::Identity() * scale;
}

TEST(GraphOptimizerTest, BootstrapEstimateExists) {
  GraphOptimizer optimizer;
  const GraphBootstrap bootstrap;
  const gtsam::Pose3 x0_pose;
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.1));
  const auto boot = bootstrap.create(x0_pose, noise);
  const gtsam::Key x0 = gtsam::Symbol('x', 0);

  const auto* prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(boot.graph.at(0).get());
  ASSERT_NE(prior, nullptr);
  optimizer.add_prior(*prior, x0, boot.values.at<gtsam::Pose3>(x0));
  optimizer.update();

  EXPECT_TRUE(optimizer.estimate().exists(x0));
}

TEST(GraphOptimizerTest, PriorPoseRecovery) {
  GraphOptimizer optimizer;
  const GraphBootstrap bootstrap;
  const gtsam::Pose3 x0_pose(gtsam::Rot3::RzRyRx(0.1, 0.2, 0.3),
                             gtsam::Point3(1.0, 2.0, 3.0));
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  const auto boot = bootstrap.create(x0_pose, noise);
  const gtsam::Key x0 = gtsam::Symbol('x', 0);

  const auto* prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(boot.graph.at(0).get());
  ASSERT_NE(prior, nullptr);
  optimizer.add_prior(*prior, x0, boot.values.at<gtsam::Pose3>(x0));
  optimizer.update();

  const auto& recovered = optimizer.estimate().at<gtsam::Pose3>(x0);
  EXPECT_TRUE(recovered.equals(x0_pose, kTol));
}

TEST(GraphOptimizerTest, SingleOdometryFactor) {
  GraphOptimizer optimizer;
  const GraphBootstrap bootstrap;
  const OdometryFactorBuilder factor_builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  const auto sigma = identitySigma(0.01);

  const gtsam::Pose3 x0_pose;
  const auto boot = bootstrap.create(x0_pose, noise);
  const auto* prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(boot.graph.at(0).get());
  ASSERT_NE(prior, nullptr);
  optimizer.add_prior(*prior, x0, boot.values.at<gtsam::Pose3>(x0));
  optimizer.update();

  const gtsam::Pose3 delta(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const auto between = factor_builder.create_factor(x0, x1, delta, sigma);
  const gtsam::Pose3 x1_init = x0_pose.compose(delta);
  optimizer.add_odometry(between, x1, x1_init);
  optimizer.update();

  EXPECT_TRUE(optimizer.estimate().exists(x1));
}

TEST(GraphOptimizerTest, OdometryChain) {
  GraphOptimizer optimizer;
  const GraphBootstrap bootstrap;
  const OdometryFactorBuilder factor_builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Key x2 = gtsam::Symbol('x', 2);
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  const auto sigma = identitySigma(0.01);

  const gtsam::Pose3 x0_pose;
  const auto boot = bootstrap.create(x0_pose, noise);
  const auto* prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(boot.graph.at(0).get());
  ASSERT_NE(prior, nullptr);
  optimizer.add_prior(*prior, x0, boot.values.at<gtsam::Pose3>(x0));
  optimizer.update();

  const gtsam::Pose3 delta01(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const gtsam::Pose3 x1_init = x0_pose.compose(delta01);
  optimizer.add_odometry(factor_builder.create_factor(x0, x1, delta01, sigma), x1, x1_init);
  optimizer.update();

  const gtsam::Pose3 delta12(gtsam::Rot3(), gtsam::Point3(0.0, 2.0, 0.0));
  const gtsam::Pose3 x2_init = x1_init.compose(delta12);
  optimizer.add_odometry(factor_builder.create_factor(x1, x2, delta12, sigma), x2, x2_init);
  optimizer.update();

  EXPECT_TRUE(optimizer.estimate().exists(x0));
  EXPECT_TRUE(optimizer.estimate().exists(x1));
  EXPECT_TRUE(optimizer.estimate().exists(x2));
}

TEST(GraphOptimizerTest, EstimateConsistencyWithIdentityOdometry) {
  GraphOptimizer optimizer;
  const GraphBootstrap bootstrap;
  const OdometryFactorBuilder factor_builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Key x2 = gtsam::Symbol('x', 2);
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  const auto sigma = identitySigma(0.01);

  const gtsam::Pose3 x0_pose(gtsam::Rot3::RzRyRx(0.1, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
  const auto boot = bootstrap.create(x0_pose, noise);
  const auto* prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(boot.graph.at(0).get());
  ASSERT_NE(prior, nullptr);
  optimizer.add_prior(*prior, x0, boot.values.at<gtsam::Pose3>(x0));
  optimizer.update();

  const gtsam::Pose3 delta01(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const gtsam::Pose3 x1_expected = x0_pose.compose(delta01);
  optimizer.add_odometry(factor_builder.create_factor(x0, x1, delta01, sigma), x1, x1_expected);
  optimizer.update();

  const gtsam::Pose3 delta12(gtsam::Rot3(), gtsam::Point3(0.0, 1.0, 0.0));
  const gtsam::Pose3 x2_expected = x1_expected.compose(delta12);
  optimizer.add_odometry(factor_builder.create_factor(x1, x2, delta12, sigma), x2, x2_expected);
  optimizer.update();

  const auto est = optimizer.estimate();
  EXPECT_TRUE(est.at<gtsam::Pose3>(x0).equals(x0_pose, kTol));
  EXPECT_TRUE(est.at<gtsam::Pose3>(x1).equals(x1_expected, kTol));
  EXPECT_TRUE(est.at<gtsam::Pose3>(x2).equals(x2_expected, kTol));
}

TEST(GraphOptimizerTest, IncrementalUpdateMultipleTimes) {
  GraphOptimizer optimizer;
  const GraphBootstrap bootstrap;
  const OdometryFactorBuilder factor_builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Key x2 = gtsam::Symbol('x', 2);
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  const auto sigma = identitySigma(0.01);

  const gtsam::Pose3 x0_pose;
  const auto boot = bootstrap.create(x0_pose, noise);
  const auto* prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(boot.graph.at(0).get());
  ASSERT_NE(prior, nullptr);
  optimizer.add_prior(*prior, x0, boot.values.at<gtsam::Pose3>(x0));

  EXPECT_NO_THROW(optimizer.update());
  EXPECT_NO_THROW(optimizer.update());

  const gtsam::Pose3 delta01(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const gtsam::Pose3 x1_init = x0_pose.compose(delta01);
  optimizer.add_odometry(factor_builder.create_factor(x0, x1, delta01, sigma), x1, x1_init);
  EXPECT_NO_THROW(optimizer.update());

  const gtsam::Pose3 delta12(gtsam::Rot3(), gtsam::Point3(0.0, 1.0, 0.0));
  const gtsam::Pose3 x2_init = x1_init.compose(delta12);
  optimizer.add_odometry(factor_builder.create_factor(x1, x2, delta12, sigma), x2, x2_init);
  EXPECT_NO_THROW(optimizer.update());

  EXPECT_TRUE(optimizer.estimate().exists(x0));
  EXPECT_TRUE(optimizer.estimate().exists(x1));
  EXPECT_TRUE(optimizer.estimate().exists(x2));
}

TEST(GraphOptimizerTest, Contains) {
  GraphOptimizer optimizer;
  const GraphBootstrap bootstrap;
  const OdometryFactorBuilder factor_builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Key x_missing = gtsam::Symbol('x', 99);
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  const auto sigma = identitySigma(0.01);

  EXPECT_FALSE(optimizer.contains(x0));
  EXPECT_FALSE(optimizer.contains(x1));

  const gtsam::Pose3 x0_pose;
  const auto boot = bootstrap.create(x0_pose, noise);
  const auto* prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(boot.graph.at(0).get());
  ASSERT_NE(prior, nullptr);
  optimizer.add_prior(*prior, x0, boot.values.at<gtsam::Pose3>(x0));

  EXPECT_TRUE(optimizer.contains(x0));
  EXPECT_FALSE(optimizer.contains(x1));
  EXPECT_FALSE(optimizer.contains(x_missing));

  const gtsam::Pose3 delta(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  optimizer.add_odometry(
      factor_builder.create_factor(x0, x1, delta, sigma), x1, x0_pose.compose(delta));

  EXPECT_TRUE(optimizer.contains(x0));
  EXPECT_TRUE(optimizer.contains(x1));
  EXPECT_FALSE(optimizer.contains(x_missing));
}

TEST(GraphOptimizerTest, EmptyGraphUpdateIsNoOp) {
  GraphOptimizer optimizer;
  EXPECT_NO_THROW(optimizer.update());
  EXPECT_NO_THROW(optimizer.update());
  EXPECT_FALSE(optimizer.contains(gtsam::Symbol('x', 0)));
}

// EVAL-05 AC-2: marginalCovariance returns a valid full 6x6 Sigma_post.
TEST(GraphOptimizerTest, MarginalCovarianceIsValid) {
  GraphOptimizer optimizer;
  const GraphBootstrap bootstrap;
  const OdometryFactorBuilder factor_builder;
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  const auto sigma = identitySigma(0.02);

  const gtsam::Pose3 x0_pose;
  const auto boot = bootstrap.create(x0_pose, noise);
  const auto* prior =
      dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(boot.graph.at(0).get());
  ASSERT_NE(prior, nullptr);
  optimizer.add_prior(*prior, gtsam::Symbol('x', 0), boot.values.at<gtsam::Pose3>(gtsam::Symbol('x', 0)));
  optimizer.update();

  // Six keyframes total (x0 bootstrap + x1..x5), each a 1 m step along x.
  const gtsam::Pose3 delta(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  gtsam::Pose3 prev = x0_pose;
  for (std::size_t i = 1; i <= 5; ++i) {
    const gtsam::Key x_prev = gtsam::Symbol('x', i - 1);
    const gtsam::Key x_cur = gtsam::Symbol('x', i);
    const gtsam::Pose3 init = prev.compose(delta);
    optimizer.add_odometry(factor_builder.create_factor(x_prev, x_cur, delta, sigma),
                           x_cur, init);
    optimizer.update();
    prev = init;
  }

  const gtsam::Key latest = gtsam::Symbol('x', 5);
  const Eigen::Matrix<double, 6, 6> cov = optimizer.marginalCovariance(latest);

  EXPECT_EQ(cov.rows(), 6);
  EXPECT_EQ(cov.cols(), 6);
  EXPECT_TRUE(cov.isApprox(cov.transpose(), 1e-9)) << "covariance not symmetric";
  for (int i = 0; i < 6; ++i) {
    EXPECT_GT(cov(i, i), 0.0) << "non-positive diagonal entry " << i;
  }
  // Position block trace (GTSAM tangent order: indices 3..5).
  EXPECT_GT(cov(3, 3) + cov(4, 4) + cov(5, 5), 0.0);

  // Absent key → zero matrix (documented fallback).
  const Eigen::Matrix<double, 6, 6> absent =
      optimizer.marginalCovariance(gtsam::Symbol('x', 99));
  EXPECT_TRUE(absent.isZero(0.0)) << "absent key must return the zero matrix";
}

}  // namespace
