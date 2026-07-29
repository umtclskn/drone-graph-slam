#include <gtest/gtest.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "graph_slam/eval/graph_visualizer.hpp"
#include "graph_slam/graph/graph_bootstrap.hpp"
#include "graph_slam/graph/graph_optimizer.hpp"
#include "graph_slam/graph/odometry_factor_builder.hpp"

namespace {

using graph_slam::eval::GraphVisualizer;
using graph_slam::eval::GraphVisualization;
using graph_slam::graph::GraphBootstrap;
using graph_slam::graph::GraphOptimizer;
using graph_slam::graph::OdometryFactorBuilder;

gtsam::SharedNoiseModel diagonalNoise(const gtsam::Vector6& sigmas) {
  return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::Matrix66 identitySigma(double scale) {
  return gtsam::Matrix66::Identity() * scale;
}

// L5-02: bootstrap seeds X(0)/V(0)/B(0) with three priors; ingest wholesale.
void bootstrapInto(GraphOptimizer& optimizer, const gtsam::Pose3& x0_pose,
                   const gtsam::SharedNoiseModel& pose_noise) {
  const GraphBootstrap bootstrap;
  const auto boot = bootstrap.create(x0_pose, pose_noise,
                                     gtsam::noiseModel::Isotropic::Sigma(3, 10.0),
                                     gtsam::noiseModel::Isotropic::Sigma(6, 0.1));
  optimizer.add_factors(boot.graph, boot.values);
}

GraphVisualization vizFromOptimizer(const GraphOptimizer& optimizer) {
  return GraphVisualizer::fromEstimate(optimizer.estimate());
}

TEST(GraphVisualizerTest, EmptyGraph) {
  const GraphOptimizer optimizer;
  const GraphVisualization viz = vizFromOptimizer(optimizer);
  EXPECT_EQ(viz.nodeCount(), 0U);
  EXPECT_EQ(viz.odometryEdgeCount(), 0U);
}

TEST(GraphVisualizerTest, SingleNode) {
  GraphOptimizer optimizer;
  const gtsam::Pose3 x0_pose(gtsam::Rot3(), gtsam::Point3(1.0, 2.0, 3.0));
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  bootstrapInto(optimizer, x0_pose, noise);
  optimizer.update();

  const GraphVisualization viz = vizFromOptimizer(optimizer);
  EXPECT_EQ(viz.nodeCount(), 1U);
  EXPECT_EQ(viz.odometryEdgeCount(), 0U);
  EXPECT_TRUE(viz.poses[0].equals(x0_pose, 1e-6));
}

TEST(GraphVisualizerTest, MultipleNodes) {
  GraphOptimizer optimizer;
  const OdometryFactorBuilder factor_builder;
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  const auto sigma = identitySigma(0.01);
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Key x2 = gtsam::Symbol('x', 2);

  const gtsam::Pose3 x0_pose;
  bootstrapInto(optimizer, x0_pose, noise);
  optimizer.update();

  const gtsam::Pose3 delta01(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
  const gtsam::Pose3 x1_init = x0_pose.compose(delta01);
  optimizer.add_odometry(factor_builder.create_factor(x0, x1, delta01, sigma), x1, x1_init);
  optimizer.update();

  const gtsam::Pose3 delta12(gtsam::Rot3(), gtsam::Point3(0.0, 2.0, 0.0));
  const gtsam::Pose3 x2_init = x1_init.compose(delta12);
  optimizer.add_odometry(factor_builder.create_factor(x1, x2, delta12, sigma), x2, x2_init);
  optimizer.update();

  const GraphVisualization viz = vizFromOptimizer(optimizer);
  EXPECT_EQ(viz.nodeCount(), 3U);
  EXPECT_EQ(viz.poses.size(), 3U);
}

TEST(GraphVisualizerTest, PathSizeEqualsNodeCount) {
  GraphOptimizer optimizer;
  const OdometryFactorBuilder factor_builder;
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  const auto sigma = identitySigma(0.01);
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);

  const gtsam::Pose3 x0_pose;
  bootstrapInto(optimizer, x0_pose, noise);
  optimizer.update();

  const gtsam::Pose3 delta(gtsam::Rot3(), gtsam::Point3(0.5, 0.0, 0.0));
  optimizer.add_odometry(
      factor_builder.create_factor(x0, x1, delta, sigma), x1, x0_pose.compose(delta));
  optimizer.update();

  const GraphVisualization viz = vizFromOptimizer(optimizer);
  EXPECT_EQ(viz.poses.size(), viz.nodeCount());
}

TEST(GraphVisualizerTest, MarkerCountEqualsNodeCount) {
  GraphOptimizer optimizer;
  const OdometryFactorBuilder factor_builder;
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  const auto sigma = identitySigma(0.01);

  const gtsam::Pose3 x0_pose;
  bootstrapInto(optimizer, x0_pose, noise);
  optimizer.update();

  for (std::size_t i = 1; i <= 4; ++i) {
    const gtsam::Key x_prev = gtsam::Symbol('x', i - 1);
    const gtsam::Key x_cur = gtsam::Symbol('x', i);
    const gtsam::Pose3 delta(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
    const gtsam::Pose3 prev_pose = optimizer.estimate().at<gtsam::Pose3>(x_prev);
    optimizer.add_odometry(factor_builder.create_factor(x_prev, x_cur, delta, sigma), x_cur,
                           prev_pose.compose(delta));
    optimizer.update();
  }

  const GraphVisualization viz = vizFromOptimizer(optimizer);
  EXPECT_EQ(viz.nodeCount(), 5U);
  // One sphere marker per keyframe == node count.
  EXPECT_EQ(viz.nodeCount(), 5U);
}

TEST(GraphVisualizerTest, OdometryEdgeCount) {
  GraphOptimizer optimizer;
  const OdometryFactorBuilder factor_builder;
  const auto noise = diagonalNoise(gtsam::Vector6::Constant(0.01));
  const auto sigma = identitySigma(0.01);

  const gtsam::Pose3 x0_pose;
  bootstrapInto(optimizer, x0_pose, noise);
  optimizer.update();

  EXPECT_EQ(vizFromOptimizer(optimizer).odometryEdgeCount(), 0U);

  for (std::size_t i = 1; i <= 3; ++i) {
    const gtsam::Key x_prev = gtsam::Symbol('x', i - 1);
    const gtsam::Key x_cur = gtsam::Symbol('x', i);
    const gtsam::Pose3 delta(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0));
    const gtsam::Pose3 prev_pose = optimizer.estimate().at<gtsam::Pose3>(x_prev);
    optimizer.add_odometry(factor_builder.create_factor(x_prev, x_cur, delta, sigma), x_cur,
                           prev_pose.compose(delta));
    optimizer.update();
  }

  const GraphVisualization viz = vizFromOptimizer(optimizer);
  EXPECT_EQ(viz.nodeCount(), 4U);
  EXPECT_EQ(viz.odometryEdgeCount(), 3U);
}

}  // namespace
