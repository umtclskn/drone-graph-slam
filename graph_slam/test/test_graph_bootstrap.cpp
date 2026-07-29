#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/PriorFactor.h>

#include "graph_slam/graph/graph_bootstrap.hpp"
#include "graph_slam/graph/graph_optimizer.hpp"

namespace {

using graph_slam::graph::GraphBootstrap;
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

gtsam::SharedNoiseModel diagonalNoise(const gtsam::Vector6& sigmas) {
  return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::SharedNoiseModel velocityNoise() {
  return gtsam::noiseModel::Isotropic::Sigma(3, 10.0);
}

gtsam::SharedNoiseModel biasNoise() {
  return gtsam::noiseModel::Isotropic::Sigma(6, 0.1);
}

graph_slam::graph::BootstrapResult makeBootstrap(
    const gtsam::Pose3& pose,
    const gtsam::SharedNoiseModel& pose_noise = diagonalNoise(
        gtsam::Vector6::Constant(0.1))) {
  const GraphBootstrap bootstrap;
  return bootstrap.create(pose, pose_noise, velocityNoise(), biasNoise());
}

const gtsam::PriorFactor<gtsam::Pose3>* asPosePrior(
    const gtsam::NonlinearFactorGraph& graph, std::size_t i) {
  return dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(graph.at(i).get());
}

// L5-02 node model: bootstrap creates X(0)/V(0)/B(0), each with its own prior.
TEST(GraphBootstrapTest, GraphContainsExactlyThreeFactors) {
  const auto result = makeBootstrap(gtsam::Pose3());
  EXPECT_EQ(result.graph.size(), 3U);
}

TEST(GraphBootstrapTest, ValuesContainExactlyPoseVelocityBias) {
  const auto result = makeBootstrap(gtsam::Pose3());
  EXPECT_EQ(result.values.size(), 3U);
  EXPECT_TRUE(result.values.exists(X(0)));
  EXPECT_TRUE(result.values.exists(V(0)));
  EXPECT_TRUE(result.values.exists(B(0)));
}

TEST(GraphBootstrapTest, CorrectPoseInsertion) {
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.1, 0.2, 0.3),
                          gtsam::Point3(1.0, 2.0, 3.0));
  const auto result = makeBootstrap(pose);
  EXPECT_TRUE(result.values.at<gtsam::Pose3>(X(0)).equals(pose));
}

TEST(GraphBootstrapTest, VelocityAndBiasSeededAtZero) {
  const auto result = makeBootstrap(gtsam::Pose3());
  EXPECT_TRUE(result.values.at<gtsam::Velocity3>(V(0)).isZero(0.0));
  const auto& bias = result.values.at<gtsam::imuBias::ConstantBias>(B(0));
  EXPECT_TRUE(bias.vector().isZero(0.0));
}

TEST(GraphBootstrapTest, CorrectPriorFactorMean) {
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.4, 0.5, 0.6),
                          gtsam::Point3(0.5, -1.0, 2.5));
  const auto result = makeBootstrap(pose);
  const auto* prior = asPosePrior(result.graph, 0);
  ASSERT_NE(prior, nullptr);
  EXPECT_TRUE(prior->prior().equals(pose));
}

TEST(GraphBootstrapTest, CorrectPriorFactorTypesAndKeys) {
  const auto result = makeBootstrap(gtsam::Pose3());
  const auto* pose_prior = asPosePrior(result.graph, 0);
  const auto* vel_prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Velocity3>*>(
      result.graph.at(1).get());
  const auto* bias_prior =
      dynamic_cast<const gtsam::PriorFactor<gtsam::imuBias::ConstantBias>*>(
          result.graph.at(2).get());
  ASSERT_NE(pose_prior, nullptr);
  ASSERT_NE(vel_prior, nullptr);
  ASSERT_NE(bias_prior, nullptr);
  EXPECT_EQ(pose_prior->key(), X(0));
  EXPECT_EQ(vel_prior->key(), V(0));
  EXPECT_EQ(bias_prior->key(), B(0));
}

TEST(GraphBootstrapTest, CorrectPriorFactorNoise) {
  const auto pose_noise = diagonalNoise(
      (gtsam::Vector6() << 0.01, 0.02, 0.03, 0.04, 0.05, 0.06).finished());
  const auto vel_noise = velocityNoise();
  const auto bias_noise = biasNoise();
  const GraphBootstrap bootstrap;
  const auto result =
      bootstrap.create(gtsam::Pose3(), pose_noise, vel_noise, bias_noise);
  EXPECT_TRUE(asPosePrior(result.graph, 0)->noiseModel()->equals(*pose_noise));
  const auto* vel_prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Velocity3>*>(
      result.graph.at(1).get());
  const auto* bias_prior =
      dynamic_cast<const gtsam::PriorFactor<gtsam::imuBias::ConstantBias>*>(
          result.graph.at(2).get());
  EXPECT_TRUE(vel_prior->noiseModel()->equals(*vel_noise));
  EXPECT_TRUE(bias_prior->noiseModel()->equals(*bias_noise));
}

TEST(GraphBootstrapTest, NonOriginPosePreservedExactly) {
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(1.1, -0.7, 0.3),
                          gtsam::Point3(1.0, 2.0, 3.0));
  const auto result = makeBootstrap(pose, diagonalNoise(gtsam::Vector6::Constant(0.05)));
  EXPECT_TRUE(result.values.at<gtsam::Pose3>(X(0)).equals(pose));
  const auto* prior = asPosePrior(result.graph, 0);
  ASSERT_NE(prior, nullptr);
  EXPECT_TRUE(prior->prior().equals(pose));
}

// L5-04d: with the production prior sigmas (bootstrap_* defaults), a single
// bootstrapped node is a fully constrained graph — iSAM2 accepts it (no
// factorless variable), converges with chi2 ~ 0 at the seeds, and X/V/B are
// all observable: each marginal covariance exists, is positive definite, and
// (single node, priors only) its sigmas equal the prior sigmas.
TEST(GraphBootstrapTest, SingleNodeGraphIsFullyConstrained) {
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.3),
                          gtsam::Point3(1.0, 2.0, 3.0));
  constexpr double kPoseSigma = 0.001;
  constexpr double kVelSigma = 0.1;
  constexpr double kAccelBiasSigma = 0.1;
  constexpr double kGyroBiasSigma = 0.01;
  const auto pose_noise = diagonalNoise(gtsam::Vector6::Constant(kPoseSigma));
  const auto vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, kVelSigma);
  gtsam::Vector6 bias_sigmas;  // ConstantBias tangent order: accel, then gyro
  bias_sigmas << kAccelBiasSigma, kAccelBiasSigma, kAccelBiasSigma,
      kGyroBiasSigma, kGyroBiasSigma, kGyroBiasSigma;
  const auto bias_noise = gtsam::noiseModel::Diagonal::Sigmas(bias_sigmas);

  const GraphBootstrap bootstrap;
  const auto boot = bootstrap.create(pose, pose_noise, vel_noise, bias_noise);

  // The production ingest path: iSAM2 must accept and converge at the seeds.
  graph_slam::graph::GraphOptimizer optimizer;
  optimizer.add_factors(boot.graph, boot.values);
  optimizer.update();
  EXPECT_LT(optimizer.chi2(), 1e-12);
  const graph_slam::graph::NodeState node = optimizer.nodeEstimate(0);
  EXPECT_TRUE(node.pose.equals(pose, 1e-9));
  EXPECT_TRUE(node.velocity.isZero(1e-12));
  EXPECT_TRUE(node.bias.vector().isZero(1e-12));

  // Observability: every marginal exists and is positive definite.
  const gtsam::Marginals marginals(boot.graph, boot.values);
  for (const gtsam::Key key : {X(0), V(0), B(0)}) {
    const gtsam::Matrix cov = marginals.marginalCovariance(key);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0)
        << "marginal for key " << gtsam::Symbol(key) << " is not PD";
  }

  // Single node + priors only -> marginal sigma == prior sigma, per block.
  const gtsam::Matrix vel_cov = marginals.marginalCovariance(V(0));
  EXPECT_NEAR(std::sqrt(vel_cov(0, 0)), kVelSigma, 1e-9);
  const gtsam::Matrix bias_cov = marginals.marginalCovariance(B(0));
  EXPECT_NEAR(std::sqrt(bias_cov(0, 0)), kAccelBiasSigma, 1e-9);
  EXPECT_NEAR(std::sqrt(bias_cov(5, 5)), kGyroBiasSigma, 1e-9);
  const gtsam::Matrix pose_cov = marginals.marginalCovariance(X(0));
  EXPECT_NEAR(std::sqrt(pose_cov(0, 0)), kPoseSigma, 1e-9);
}

TEST(GraphBootstrapTest, StatelessAcrossInvocations) {
  const auto first = makeBootstrap(gtsam::Pose3());
  const gtsam::Pose3 second_pose(gtsam::Rot3::RzRyRx(0.2, 0.0, 0.0),
                                 gtsam::Point3(5.0, 0.0, 0.0));
  const auto second = makeBootstrap(second_pose);

  EXPECT_EQ(first.graph.size(), 3U);
  EXPECT_EQ(first.values.size(), 3U);
  EXPECT_EQ(second.graph.size(), 3U);
  EXPECT_EQ(second.values.size(), 3U);

  EXPECT_TRUE(first.values.at<gtsam::Pose3>(X(0)).equals(gtsam::Pose3()));
  EXPECT_TRUE(second.values.at<gtsam::Pose3>(X(0)).equals(second_pose));
}

}  // namespace
