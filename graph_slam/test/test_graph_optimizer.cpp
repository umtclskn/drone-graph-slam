#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "graph_slam/graph/graph_bootstrap.hpp"
#include "graph_slam/graph/graph_optimizer.hpp"
#include "graph_slam/graph/odometry_factor_builder.hpp"
#include "graph_slam/imu/imu_preintegrator.hpp"

namespace {

using graph_slam::graph::GraphBootstrap;
using graph_slam::graph::GraphOptimizer;
using graph_slam::graph::NodeState;
using graph_slam::graph::OdometryFactorBuilder;
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

constexpr double kTol = 1e-6;

gtsam::SharedNoiseModel diagonalNoise(const gtsam::Vector6& sigmas) {
  return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::Matrix66 identitySigma(double scale) {
  return gtsam::Matrix66::Identity() * scale;
}

gtsam::SharedNoiseModel velocityNoise() {
  return gtsam::noiseModel::Isotropic::Sigma(3, 10.0);
}

gtsam::SharedNoiseModel biasNoise() {
  return gtsam::noiseModel::Isotropic::Sigma(6, 0.1);
}

// Bootstrap node 0 with the expanded L5-02 state (X/V/B + three priors).
void bootstrapInto(GraphOptimizer& optimizer, const gtsam::Pose3& x0_pose,
                   const gtsam::SharedNoiseModel& pose_noise) {
  const GraphBootstrap bootstrap;
  const auto boot =
      bootstrap.create(x0_pose, pose_noise, velocityNoise(), biasNoise());
  optimizer.add_factors(boot.graph, boot.values);
}

// L5-02 scaffolding for node i>0: with no IMU factor yet, V(i)/B(i) are held by
// weak priors (mirrors graph_backend_node::addVelocityBiasScaffolding).
void addScaffolding(GraphOptimizer& optimizer, std::size_t i) {
  const gtsam::Velocity3 zero_velocity = gtsam::Velocity3::Zero();
  const gtsam::imuBias::ConstantBias zero_bias;
  optimizer.add_prior(
      gtsam::PriorFactor<gtsam::Velocity3>(V(i), zero_velocity, velocityNoise()),
      V(i), zero_velocity);
  optimizer.add_prior(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
                          B(i), zero_bias, biasNoise()),
                      B(i), zero_bias);
}

TEST(GraphOptimizerTest, BootstrapEstimateExists) {
  GraphOptimizer optimizer;
  bootstrapInto(optimizer, gtsam::Pose3(),
                diagonalNoise(gtsam::Vector6::Constant(0.1)));
  optimizer.update();

  EXPECT_TRUE(optimizer.estimate().exists(X(0)));
  EXPECT_TRUE(optimizer.estimate().exists(V(0)));
  EXPECT_TRUE(optimizer.estimate().exists(B(0)));
}

TEST(GraphOptimizerTest, PriorPoseRecovery) {
  GraphOptimizer optimizer;
  const gtsam::Pose3 x0_pose(gtsam::Rot3::RzRyRx(0.1, 0.2, 0.3),
                             gtsam::Point3(1.0, 2.0, 3.0));
  bootstrapInto(optimizer, x0_pose, diagonalNoise(gtsam::Vector6::Constant(0.01)));
  optimizer.update();

  const auto& recovered = optimizer.estimate().at<gtsam::Pose3>(X(0));
  EXPECT_TRUE(recovered.equals(x0_pose, kTol));
}

TEST(GraphOptimizerTest, SingleOdometryFactor) {
  GraphOptimizer optimizer;
  const OdometryFactorBuilder factor_builder;
  const auto sigma = identitySigma(0.01);

  const gtsam::Pose3 x0_pose;
  bootstrapInto(optimizer, x0_pose, diagonalNoise(gtsam::Vector6::Constant(0.01)));
  optimizer.update();

  const gtsam::Pose3 delta(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const auto between = factor_builder.create_factor(X(0), X(1), delta, sigma);
  const gtsam::Pose3 x1_init = x0_pose.compose(delta);
  optimizer.add_odometry(between, X(1), x1_init);
  addScaffolding(optimizer, 1);
  optimizer.update();

  EXPECT_TRUE(optimizer.estimate().exists(X(1)));
  EXPECT_TRUE(optimizer.estimate().exists(V(1)));
  EXPECT_TRUE(optimizer.estimate().exists(B(1)));
}

TEST(GraphOptimizerTest, OdometryChain) {
  GraphOptimizer optimizer;
  const OdometryFactorBuilder factor_builder;
  const auto sigma = identitySigma(0.01);

  const gtsam::Pose3 x0_pose;
  bootstrapInto(optimizer, x0_pose, diagonalNoise(gtsam::Vector6::Constant(0.01)));
  optimizer.update();

  const gtsam::Pose3 delta01(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const gtsam::Pose3 x1_init = x0_pose.compose(delta01);
  optimizer.add_odometry(factor_builder.create_factor(X(0), X(1), delta01, sigma),
                         X(1), x1_init);
  addScaffolding(optimizer, 1);
  optimizer.update();

  const gtsam::Pose3 delta12(gtsam::Rot3(), gtsam::Point3(0.0, 2.0, 0.0));
  const gtsam::Pose3 x2_init = x1_init.compose(delta12);
  optimizer.add_odometry(factor_builder.create_factor(X(1), X(2), delta12, sigma),
                         X(2), x2_init);
  addScaffolding(optimizer, 2);
  optimizer.update();

  EXPECT_TRUE(optimizer.estimate().exists(X(0)));
  EXPECT_TRUE(optimizer.estimate().exists(X(1)));
  EXPECT_TRUE(optimizer.estimate().exists(X(2)));
}

TEST(GraphOptimizerTest, EstimateConsistencyWithIdentityOdometry) {
  GraphOptimizer optimizer;
  const OdometryFactorBuilder factor_builder;
  const auto sigma = identitySigma(0.01);

  const gtsam::Pose3 x0_pose(gtsam::Rot3::RzRyRx(0.1, 0.0, 0.0),
                             gtsam::Point3(0.0, 0.0, 0.0));
  bootstrapInto(optimizer, x0_pose, diagonalNoise(gtsam::Vector6::Constant(0.01)));
  optimizer.update();

  const gtsam::Pose3 delta01(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const gtsam::Pose3 x1_expected = x0_pose.compose(delta01);
  optimizer.add_odometry(factor_builder.create_factor(X(0), X(1), delta01, sigma),
                         X(1), x1_expected);
  addScaffolding(optimizer, 1);
  optimizer.update();

  const gtsam::Pose3 delta12(gtsam::Rot3(), gtsam::Point3(0.0, 1.0, 0.0));
  const gtsam::Pose3 x2_expected = x1_expected.compose(delta12);
  optimizer.add_odometry(factor_builder.create_factor(X(1), X(2), delta12, sigma),
                         X(2), x2_expected);
  addScaffolding(optimizer, 2);
  optimizer.update();

  const auto est = optimizer.estimate();
  EXPECT_TRUE(est.at<gtsam::Pose3>(X(0)).equals(x0_pose, kTol));
  EXPECT_TRUE(est.at<gtsam::Pose3>(X(1)).equals(x1_expected, kTol));
  EXPECT_TRUE(est.at<gtsam::Pose3>(X(2)).equals(x2_expected, kTol));
}

TEST(GraphOptimizerTest, IncrementalUpdateMultipleTimes) {
  GraphOptimizer optimizer;
  const OdometryFactorBuilder factor_builder;
  const auto sigma = identitySigma(0.01);

  const gtsam::Pose3 x0_pose;
  bootstrapInto(optimizer, x0_pose, diagonalNoise(gtsam::Vector6::Constant(0.01)));

  EXPECT_NO_THROW(optimizer.update());
  EXPECT_NO_THROW(optimizer.update());

  const gtsam::Pose3 delta01(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const gtsam::Pose3 x1_init = x0_pose.compose(delta01);
  optimizer.add_odometry(factor_builder.create_factor(X(0), X(1), delta01, sigma),
                         X(1), x1_init);
  addScaffolding(optimizer, 1);
  EXPECT_NO_THROW(optimizer.update());

  const gtsam::Pose3 delta12(gtsam::Rot3(), gtsam::Point3(0.0, 1.0, 0.0));
  const gtsam::Pose3 x2_init = x1_init.compose(delta12);
  optimizer.add_odometry(factor_builder.create_factor(X(1), X(2), delta12, sigma),
                         X(2), x2_init);
  addScaffolding(optimizer, 2);
  EXPECT_NO_THROW(optimizer.update());

  EXPECT_TRUE(optimizer.estimate().exists(X(0)));
  EXPECT_TRUE(optimizer.estimate().exists(X(1)));
  EXPECT_TRUE(optimizer.estimate().exists(X(2)));
}

TEST(GraphOptimizerTest, Contains) {
  GraphOptimizer optimizer;
  const OdometryFactorBuilder factor_builder;
  const auto sigma = identitySigma(0.01);

  EXPECT_FALSE(optimizer.contains(X(0)));
  EXPECT_FALSE(optimizer.contains(X(1)));

  const gtsam::Pose3 x0_pose;
  bootstrapInto(optimizer, x0_pose, diagonalNoise(gtsam::Vector6::Constant(0.01)));

  EXPECT_TRUE(optimizer.contains(X(0)));
  EXPECT_TRUE(optimizer.contains(V(0)));
  EXPECT_TRUE(optimizer.contains(B(0)));
  EXPECT_FALSE(optimizer.contains(X(1)));
  EXPECT_FALSE(optimizer.contains(X(99)));

  const gtsam::Pose3 delta(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  optimizer.add_odometry(factor_builder.create_factor(X(0), X(1), delta, sigma),
                         X(1), x0_pose.compose(delta));

  EXPECT_TRUE(optimizer.contains(X(0)));
  EXPECT_TRUE(optimizer.contains(X(1)));
  EXPECT_FALSE(optimizer.contains(X(99)));
}

TEST(GraphOptimizerTest, EmptyGraphUpdateIsNoOp) {
  GraphOptimizer optimizer;
  EXPECT_NO_THROW(optimizer.update());
  EXPECT_NO_THROW(optimizer.update());
  EXPECT_FALSE(optimizer.contains(X(0)));
}

// EVAL-05 AC-2: marginalCovariance returns a valid full 6x6 Sigma_post.
TEST(GraphOptimizerTest, MarginalCovarianceIsValid) {
  GraphOptimizer optimizer;
  const OdometryFactorBuilder factor_builder;
  const auto sigma = identitySigma(0.02);

  const gtsam::Pose3 x0_pose;
  bootstrapInto(optimizer, x0_pose, diagonalNoise(gtsam::Vector6::Constant(0.01)));
  optimizer.update();

  // Six keyframes total (x0 bootstrap + x1..x5), each a 1 m step along x.
  const gtsam::Pose3 delta(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  gtsam::Pose3 prev = x0_pose;
  for (std::size_t i = 1; i <= 5; ++i) {
    const gtsam::Pose3 init = prev.compose(delta);
    optimizer.add_odometry(factor_builder.create_factor(X(i - 1), X(i), delta, sigma),
                           X(i), init);
    addScaffolding(optimizer, i);
    optimizer.update();
    prev = init;
  }

  const Eigen::Matrix<double, 6, 6> cov = optimizer.marginalCovariance(X(5));

  EXPECT_EQ(cov.rows(), 6);
  EXPECT_EQ(cov.cols(), 6);
  EXPECT_TRUE(cov.isApprox(cov.transpose(), 1e-9)) << "covariance not symmetric";
  for (int i = 0; i < 6; ++i) {
    EXPECT_GT(cov(i, i), 0.0) << "non-positive diagonal entry " << i;
  }
  // Position block trace (GTSAM tangent order: indices 3..5).
  EXPECT_GT(cov(3, 3) + cov(4, 4) + cov(5, 5), 0.0);

  // Absent key → zero matrix (documented fallback).
  const Eigen::Matrix<double, 6, 6> absent = optimizer.marginalCovariance(X(99));
  EXPECT_TRUE(absent.isZero(0.0)) << "absent key must return the zero matrix";
}

// ---------------------------------------------------------------------------
// L5-02 acceptance tests: expanded (Pose3, Velocity3, imuBias) node model
// ---------------------------------------------------------------------------

// AC: a 3-node graph with X/V/B keys optimizes WITHOUT an IMU factor and stays
// well-posed (velocity/bias held by priors).
TEST(GraphOptimizerTest, ExpandedThreeNodeGraphIsWellPosedWithoutImuFactor) {
  GraphOptimizer optimizer;
  const OdometryFactorBuilder factor_builder;
  const auto sigma = identitySigma(0.01);

  const gtsam::Pose3 x0_pose;
  bootstrapInto(optimizer, x0_pose, diagonalNoise(gtsam::Vector6::Constant(0.001)));
  optimizer.update();

  gtsam::Pose3 prev = x0_pose;
  const gtsam::Pose3 delta(gtsam::Rot3::Rz(0.1), gtsam::Point3(1.0, 0.5, 0.0));
  for (std::size_t i = 1; i <= 2; ++i) {
    const gtsam::Pose3 init = prev.compose(delta);
    optimizer.add_odometry(factor_builder.create_factor(X(i - 1), X(i), delta, sigma),
                           X(i), init);
    addScaffolding(optimizer, i);
    ASSERT_NO_THROW(optimizer.update()) << "iSAM2 rejected node " << i;
    prev = init;
  }

  const auto est = optimizer.estimate();
  for (std::size_t i = 0; i <= 2; ++i) {
    EXPECT_TRUE(est.exists(X(i)));
    EXPECT_TRUE(est.exists(V(i)));
    EXPECT_TRUE(est.exists(B(i)));
    // Velocity/bias sit exactly at their (only) prior means.
    EXPECT_TRUE(est.at<gtsam::Velocity3>(V(i)).isZero(kTol));
    EXPECT_TRUE(est.at<gtsam::imuBias::ConstantBias>(B(i)).vector().isZero(kTol));
  }
  // Well-posed: consistent open chain (chi2 ~ 0), positive pose uncertainty.
  EXPECT_LT(optimizer.chi2(), kTol);
  EXPECT_GT(optimizer.marginalCovPositionTrace(X(2)), 0.0);
}

// nodeEstimate() returns the full per-node struct (L5-02 API).
TEST(GraphOptimizerTest, NodeEstimateReturnsStruct) {
  GraphOptimizer optimizer;
  const gtsam::Pose3 x0_pose(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.2),
                             gtsam::Point3(1.0, -2.0, 0.5));
  bootstrapInto(optimizer, x0_pose, diagonalNoise(gtsam::Vector6::Constant(0.01)));
  optimizer.update();

  const NodeState node = optimizer.nodeEstimate(0);
  EXPECT_TRUE(node.pose.equals(x0_pose, kTol));
  EXPECT_TRUE(node.velocity.isZero(kTol));
  EXPECT_TRUE(node.bias.vector().isZero(kTol));
  EXPECT_GT(node.covariance(3, 3), 0.0);
  EXPECT_TRUE(node.covariance.isApprox(node.covariance.transpose(), 1e-9));

  // Absent node -> defaults + zero covariance (documented convention).
  const NodeState absent = optimizer.nodeEstimate(42);
  EXPECT_TRUE(absent.covariance.isZero(0.0));
}

// The keystone regression at unit level: adding V/B (held only by their own
// priors, no factor coupling them to poses) must leave the POSE solution and
// its marginals numerically identical to a pose-only graph — including after a
// loop-closure cross-edge. This is the mechanism behind the L5-02e "ATE delta
// ≈ 0" bag AC.
TEST(GraphOptimizerTest, ExpandedNodesLeavePoseSolutionUnchanged) {
  const OdometryFactorBuilder factor_builder;
  const auto pose_noise = diagonalNoise(gtsam::Vector6::Constant(0.001));
  const auto sigma = identitySigma(0.05);

  GraphOptimizer pose_only;   // pre-L5-02 shape: X keys only
  GraphOptimizer expanded;    // L5-02 shape: X + V + B with scaffolding priors

  const gtsam::Pose3 x0_pose;
  pose_only.add_prior(gtsam::PriorFactor<gtsam::Pose3>(X(0), x0_pose, pose_noise),
                      X(0), x0_pose);
  bootstrapInto(expanded, x0_pose, pose_noise);
  pose_only.update();
  expanded.update();

  // A drifting square-ish chain x0..x4 so the loop closure has real work to do.
  const std::vector<gtsam::Pose3> deltas = {
      {gtsam::Rot3::Rz(0.02), gtsam::Point3(1.0, 0.0, 0.0)},
      {gtsam::Rot3::Rz(M_PI_2 + 0.03), gtsam::Point3(1.0, 0.1, 0.0)},
      {gtsam::Rot3::Rz(M_PI_2 - 0.01), gtsam::Point3(1.1, -0.05, 0.0)},
      {gtsam::Rot3::Rz(M_PI_2 + 0.02), gtsam::Point3(0.95, 0.05, 0.0)}};
  gtsam::Pose3 prev = x0_pose;
  for (std::size_t i = 0; i < deltas.size(); ++i) {
    const gtsam::Pose3 init = prev.compose(deltas[i]);
    const auto factor =
        factor_builder.create_factor(X(i), X(i + 1), deltas[i], sigma);
    pose_only.add_odometry(factor, X(i + 1), init);
    expanded.add_odometry(factor, X(i + 1), init);
    addScaffolding(expanded, i + 1);
    pose_only.update();
    expanded.update();
    prev = init;
  }

  // Loop closure x4 -> x0 claiming identity (pulls the drifted square shut).
  const gtsam::BetweenFactor<gtsam::Pose3> loop(
      X(4), X(0), gtsam::Pose3(),
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(0.01)));
  pose_only.add_loop_closure(loop);
  expanded.add_loop_closure(loop);
  pose_only.update();
  expanded.update();

  for (std::size_t i = 0; i <= 4; ++i) {
    const gtsam::Pose3 p_ref = pose_only.estimate().at<gtsam::Pose3>(X(i));
    const gtsam::Pose3 p_exp = expanded.estimate().at<gtsam::Pose3>(X(i));
    EXPECT_TRUE(p_exp.equals(p_ref, 1e-9))
        << "pose " << i << " diverged: ref " << p_ref << " vs expanded " << p_exp;

    const Eigen::Matrix<double, 6, 6> c_ref = pose_only.marginalCovariance(X(i));
    const Eigen::Matrix<double, 6, 6> c_exp = expanded.marginalCovariance(X(i));
    EXPECT_TRUE(c_exp.isApprox(c_ref, 1e-9))
        << "pose marginal " << i << " diverged (max diff "
        << (c_exp - c_ref).cwiseAbs().maxCoeff() << ")";
  }
  EXPECT_NEAR(pose_only.chi2(), expanded.chi2(), 1e-9);
}

// ---------------------------------------------------------------------------
// L5-03: IMU backbone factor injection (add_imu_keyframe / add_ndt_edge)
// ---------------------------------------------------------------------------

// Build a PIM for a constant world-acceleration a=(0.5,0,0), zero gyro, over 1 s
// at 1 kHz, starting/linearizing at zero bias. Analytic (identity attitude):
// deltaP = 0.5*a*T^2 = 0.25 m in x, deltaV = a*T = 0.5 m/s in x, deltaR = I.
graph_slam::imu::ImuPreintegrator makeConstantAccelPim() {
  graph_slam::imu::ImuPreintegrator preint{graph_slam::imu::PreintegrationConfig{},
                                           gtsam::imuBias::ConstantBias{}};
  // Measured specific force = a_world - g_world = (0.5,0,0) - (0,0,-9.8).
  const Eigen::Vector3d accel(0.5, 0.0, 9.8);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  for (int i = 0; i < 1000; ++i) {
    preint.integrate(accel, gyro, 0.001);
  }
  return preint;
}

// Tightly prior node 0 at (origin, zero velocity, zero bias) on the optimizer.
void priorNodeZero(GraphOptimizer& optimizer) {
  const gtsam::Pose3 x0;
  const gtsam::Velocity3 v0 = gtsam::Velocity3::Zero();
  const gtsam::imuBias::ConstantBias b0;
  optimizer.add_prior(
      gtsam::PriorFactor<gtsam::Pose3>(
          X(0), x0, diagonalNoise(gtsam::Vector6::Constant(1e-4))),
      X(0), x0);
  optimizer.add_prior(
      gtsam::PriorFactor<gtsam::Velocity3>(
          V(0), v0, gtsam::noiseModel::Isotropic::Sigma(3, 1e-4)),
      V(0), v0);
  optimizer.add_prior(
      gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
          B(0), b0, gtsam::noiseModel::Isotropic::Sigma(6, 1e-4)),
      B(0), b0);
}

// AC (L5-03b/c): an ImuFactor + bias BetweenFactor added via add_imu_keyframe pull
// node 1's pose+velocity to the analytic dead-reckoned state (node 0 tightly
// priored), and bias stays at zero. Closed-form check of the factor wiring.
TEST(GraphOptimizerTest, ImuFactorConstrainsPoseAndVelocity) {
  GraphOptimizer optimizer;
  priorNodeZero(optimizer);
  optimizer.update();

  const gtsam::imuBias::ConstantBias b0;
  graph_slam::imu::ImuPreintegrator preint = makeConstantAccelPim();
  const gtsam::PreintegratedImuMeasurements& pim = preint.finish();
  const gtsam::NavState pred =
      preint.predict(gtsam::NavState(gtsam::Pose3(), gtsam::Velocity3::Zero()), b0);

  const gtsam::ImuFactor imu_factor(X(0), V(0), X(1), V(1), B(0), pim);
  const gtsam::BetweenFactor<gtsam::imuBias::ConstantBias> bias_factor(
      B(0), B(1), gtsam::imuBias::ConstantBias{},
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(1e-3)));
  optimizer.add_imu_keyframe(X(1), pred.pose(), V(1), pred.velocity(), B(1), b0,
                             imu_factor, bias_factor);
  optimizer.update();

  const auto est = optimizer.estimate();
  ASSERT_TRUE(est.exists(X(1)));
  ASSERT_TRUE(est.exists(V(1)));
  ASSERT_TRUE(est.exists(B(1)));

  const gtsam::Pose3 x1 = est.at<gtsam::Pose3>(X(1));
  const gtsam::Velocity3 v1 = est.at<gtsam::Velocity3>(V(1));
  EXPECT_NEAR(x1.translation().x(), 0.25, 1e-3);
  EXPECT_NEAR(x1.translation().y(), 0.0, 1e-3);
  EXPECT_NEAR(x1.translation().z(), 0.0, 1e-3);
  EXPECT_NEAR(v1.x(), 0.5, 1e-3);
  EXPECT_NEAR(v1.y(), 0.0, 1e-3);
  EXPECT_NEAR(v1.z(), 0.0, 1e-3);
  EXPECT_TRUE(x1.rotation().equals(gtsam::Rot3(), 1e-3));
  EXPECT_TRUE(est.at<gtsam::imuBias::ConstantBias>(B(1)).vector().isZero(1e-3));
  // Consistent measurement → tiny ImuFactor residual.
  EXPECT_LT(imu_factor.error(est), 1e-3);
}

// AC (L5-03d): the NDT BetweenFactor<Pose3> (add_ndt_edge, factor-only) coexists
// with the IMU backbone on the same edge without inserting duplicate values; an
// agreeing NDT measurement leaves the solution consistent (low chi2).
TEST(GraphOptimizerTest, ImuAndNdtEdgesCoexist) {
  GraphOptimizer optimizer;
  priorNodeZero(optimizer);
  optimizer.update();

  const gtsam::imuBias::ConstantBias b0;
  graph_slam::imu::ImuPreintegrator preint = makeConstantAccelPim();
  const gtsam::PreintegratedImuMeasurements& pim = preint.finish();
  const gtsam::NavState pred =
      preint.predict(gtsam::NavState(gtsam::Pose3(), gtsam::Velocity3::Zero()), b0);

  const gtsam::ImuFactor imu_factor(X(0), V(0), X(1), V(1), B(0), pim);
  const gtsam::BetweenFactor<gtsam::imuBias::ConstantBias> bias_factor(
      B(0), B(1), gtsam::imuBias::ConstantBias{},
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(1e-3)));
  optimizer.add_imu_keyframe(X(1), pred.pose(), V(1), pred.velocity(), B(1), b0,
                             imu_factor, bias_factor);

  // Independent NDT measurement agreeing with the IMU prediction: 0.25 m along x.
  const OdometryFactorBuilder factor_builder;
  const gtsam::Pose3 ndt_delta(gtsam::Rot3(), gtsam::Point3(0.25, 0.0, 0.0));
  optimizer.add_ndt_edge(
      factor_builder.create_factor(X(0), X(1), ndt_delta, identitySigma(0.01)));
  optimizer.update();

  const auto est = optimizer.estimate();
  // Node 1 exists exactly once (no duplicate value insertion).
  EXPECT_TRUE(est.exists(X(1)));
  EXPECT_TRUE(est.exists(V(1)));
  EXPECT_NEAR(est.at<gtsam::Pose3>(X(1)).translation().x(), 0.25, 2e-3);
  // Two agreeing constraints → consistent graph (small error).
  EXPECT_LT(optimizer.chi2(), 1.0);
}

}  // namespace
