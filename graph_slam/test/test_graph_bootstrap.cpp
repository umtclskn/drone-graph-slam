#include <gtest/gtest.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/PriorFactor.h>

#include "graph_slam/graph/graph_bootstrap.hpp"

namespace {

using graph_slam::graph::GraphBootstrap;

gtsam::SharedNoiseModel diagonalNoise(const gtsam::Vector6& sigmas) {
  return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

const gtsam::PriorFactor<gtsam::Pose3>* asPrior(const gtsam::NonlinearFactorGraph& graph) {
  if (graph.size() != 1U) {
    return nullptr;
  }
  return dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(graph.at(0).get());
}

TEST(GraphBootstrapTest, GraphContainsExactlyOneFactor) {
  const GraphBootstrap bootstrap;
  const gtsam::Pose3 pose;
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.1));
  const auto result = bootstrap.create(pose, noise);
  EXPECT_EQ(result.graph.size(), 1U);
}

TEST(GraphBootstrapTest, ValuesContainsExactlyOnePose) {
  const GraphBootstrap bootstrap;
  const gtsam::Pose3 pose;
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.1));
  const auto result = bootstrap.create(pose, noise);
  EXPECT_EQ(result.values.size(), 1U);
}

TEST(GraphBootstrapTest, CorrectKeyCreation) {
  const GraphBootstrap bootstrap;
  const gtsam::Pose3 pose;
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.1));
  const auto result = bootstrap.create(pose, noise);
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  EXPECT_TRUE(result.values.exists(x0));
}

TEST(GraphBootstrapTest, CorrectPoseInsertion) {
  const GraphBootstrap bootstrap;
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.1, 0.2, 0.3),
                          gtsam::Point3(1.0, 2.0, 3.0));
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.1));
  const auto result = bootstrap.create(pose, noise);
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const auto& retrieved = result.values.at<gtsam::Pose3>(x0);
  EXPECT_TRUE(retrieved.equals(pose));
}

TEST(GraphBootstrapTest, CorrectPriorFactorMean) {
  const GraphBootstrap bootstrap;
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.4, 0.5, 0.6),
                          gtsam::Point3(0.5, -1.0, 2.5));
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.1));
  const auto result = bootstrap.create(pose, noise);
  const auto* prior = asPrior(result.graph);
  ASSERT_NE(prior, nullptr);
  EXPECT_TRUE(prior->prior().equals(pose));
}

TEST(GraphBootstrapTest, CorrectPriorFactorNoise) {
  const GraphBootstrap bootstrap;
  const gtsam::Pose3 pose;
  const auto noise = diagonalNoise((gtsam::Vector6() << 0.01, 0.02, 0.03, 0.04, 0.05, 0.06).finished());
  const auto result = bootstrap.create(pose, noise);
  const auto* prior = asPrior(result.graph);
  ASSERT_NE(prior, nullptr);
  EXPECT_TRUE(prior->noiseModel()->equals(*noise));
}

TEST(GraphBootstrapTest, NonOriginPosePreservedExactly) {
  const GraphBootstrap bootstrap;
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(1.1, -0.7, 0.3),
                          gtsam::Point3(1.0, 2.0, 3.0));
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.05));
  const auto result = bootstrap.create(pose, noise);

  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  EXPECT_TRUE(result.values.at<gtsam::Pose3>(x0).equals(pose));

  const auto* prior = asPrior(result.graph);
  ASSERT_NE(prior, nullptr);
  EXPECT_TRUE(prior->prior().equals(pose));
}

TEST(GraphBootstrapTest, StatelessAcrossInvocations) {
  const GraphBootstrap bootstrap;
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.1));

  const auto first = bootstrap.create(gtsam::Pose3(), noise);
  const auto second = bootstrap.create(
      gtsam::Pose3(gtsam::Rot3::RzRyRx(0.2, 0.0, 0.0), gtsam::Point3(5.0, 0.0, 0.0)),
      noise);

  EXPECT_EQ(first.graph.size(), 1U);
  EXPECT_EQ(first.values.size(), 1U);
  EXPECT_EQ(second.graph.size(), 1U);
  EXPECT_EQ(second.values.size(), 1U);

  const gtsam::Pose3 second_pose(gtsam::Rot3::RzRyRx(0.2, 0.0, 0.0), gtsam::Point3(5.0, 0.0, 0.0));
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  EXPECT_TRUE(first.values.at<gtsam::Pose3>(x0).equals(gtsam::Pose3()));
  EXPECT_TRUE(second.values.at<gtsam::Pose3>(x0).equals(second_pose));
}

}  // namespace
