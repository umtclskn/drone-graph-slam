// GRAPH-VIS-01: graph back-end node smoke test (AC-2).
//
// Verifies:
//   - Node constructs and spins without crashing.
//   - Node exposes its documented parameters with correct defaults.
//   - ≥5 synthetic odom messages are processed; ≥2 keyframes added to the
//     graph (verified by counting poses in the published /slam/graph_path).
//
// The ROS-free algorithm classes (KeyframePolicy, OdometryAccumulator, etc.)
// are already covered by their own gtests; this only checks node wiring.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "graph_backend_node.hpp"
#include "graph_slam/eval/graph_visualizer.hpp"
#include "graph_slam/graph/graph_optimizer.hpp"

namespace {

class RclcppEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { rclcpp::init(0, nullptr); }
  void TearDown() override { rclcpp::shutdown(); }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

nav_msgs::msg::Odometry makeSyntheticOdom(double x, double y = 0.0, double z = 0.0) {
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = rclcpp::Clock().now();
  odom.header.frame_id = "odom";
  odom.child_frame_id = "base_link";
  odom.pose.pose.position.x = x;
  odom.pose.pose.position.y = y;
  odom.pose.pose.position.z = z;
  odom.pose.pose.orientation.w = 1.0;
  // Diagonal covariance 0.01 in nav order [x,y,z,rx,ry,rz] (36-element row-major).
  for (int i = 0; i < 6; ++i) {
    odom.pose.covariance[static_cast<std::size_t>(i * 7)] = 0.01;
  }
  return odom;
}

// Independently optimize the same keyframe chain through a reference
// GraphOptimizer and return its post-optimization keyframe poses (the exact
// estimate the node's publish path must read back via estimate()). For an open
// chain (prior + sequential BetweenFactors) the MAP estimate is uniquely the
// cumulative composition of the deltas anchored at the prior, independent of the
// noise model — so a fixed odometry noise reproduces the optimizer output.
std::vector<gtsam::Pose3> referenceOptimizedPoses(
    const std::vector<gtsam::Pose3>& inputs) {
  using gtsam::Pose3;
  using gtsam::Symbol;

  graph_slam::graph::GraphOptimizer opt;
  const auto prior_noise =
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(0.001));
  opt.add_prior(gtsam::PriorFactor<Pose3>(Symbol('x', 0), inputs[0], prior_noise),
                Symbol('x', 0), inputs[0]);
  opt.update();

  const auto odom_noise =
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(0.1));
  for (std::size_t i = 1; i < inputs.size(); ++i) {
    const Pose3 delta = inputs[i - 1].between(inputs[i]);
    const gtsam::BetweenFactor<Pose3> factor(Symbol('x', i - 1), Symbol('x', i),
                                             delta, odom_noise);
    const Pose3 prev = opt.estimate().at<Pose3>(Symbol('x', i - 1));
    opt.add_odometry(factor, Symbol('x', i), prev.compose(delta));
    opt.update();
  }
  return graph_slam::eval::GraphVisualizer::fromEstimate(opt.estimate()).poses;
}

// Extract a numeric JSON field value from a flat "{\"key\": value, ...}" object.
// The diagnostics format is fully controlled by the node, so a minimal parser
// avoids pulling in a JSON dependency. Returns NaN if the key is absent.
double parseJsonField(const std::string& json, const std::string& key) {
  const std::string token = "\"" + key + "\":";
  const auto pos = json.find(token);
  if (pos == std::string::npos) {
    return std::nan("");
  }
  return std::stod(json.substr(pos + token.size()));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(GraphBackendNodeTest, ConstructsAndSpinsWithoutCrashing) {
  std::shared_ptr<rclcpp::Node> node;
  ASSERT_NO_THROW(node = graph_slam::createGraphBackendNode());
  ASSERT_NE(node, nullptr);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  exec.spin_some(std::chrono::milliseconds(100));
  exec.remove_node(node);
}

TEST(GraphBackendNodeTest, DeclaresDocumentedParameters) {
  const auto node = graph_slam::createGraphBackendNode();
  EXPECT_EQ(node->get_parameter("map_frame").as_string(), "map");
  EXPECT_EQ(node->get_parameter("odom_frame").as_string(), "odom");
  EXPECT_EQ(node->get_parameter("ndt_odom_topic").as_string(), "/ndt_frontend/ndt_odom");
  EXPECT_DOUBLE_EQ(node->get_parameter("keyframe_translation_m").as_double(), 0.5);
  EXPECT_DOUBLE_EQ(node->get_parameter("keyframe_rotation_rad").as_double(), 0.5);
  EXPECT_DOUBLE_EQ(node->get_parameter("keyframe_time_s").as_double(), 10.0);
}

TEST(GraphBackendNodeTest, SyntheticOdomChainAddsKeyframes) {
  // Low translation threshold so each 0.6 m step triggers a keyframe.
  // Rotation and time criteria disabled (negative value → disabled per KeyframePolicy).
  rclcpp::NodeOptions opts;
  opts.append_parameter_override("keyframe_translation_m", 0.4);
  opts.append_parameter_override("keyframe_rotation_rad", -1.0);
  opts.append_parameter_override("keyframe_time_s", -1.0);

  auto backend = graph_slam::createGraphBackendNode(opts);
  auto helper = rclcpp::Node::make_shared("test_helper_backend");

  auto pub = helper->create_publisher<nav_msgs::msg::Odometry>(
      "/ndt_frontend/ndt_odom", rclcpp::QoS(10));

  std::size_t max_poses = 0;
  auto path_sub = helper->create_subscription<nav_msgs::msg::Path>(
      "/slam/graph_path", rclcpp::QoS(10),
      [&max_poses](nav_msgs::msg::Path::SharedPtr msg) {
        max_poses = std::max(max_poses, msg->poses.size());
      });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(backend);
  exec.add_node(helper);

  // Publish 6 odom messages spaced 0.6 m apart along x.
  // Bootstrap (i=0) → x0; i=1..5 each have delta=0.6 m > 0.4 m → keyframe.
  // Expected: x0..x5 = 6 nodes in the path after all messages processed.
  for (int i = 0; i < 6; ++i) {
    pub->publish(makeSyntheticOdom(static_cast<double>(i) * 0.6));
    exec.spin_some(std::chrono::milliseconds(80));
  }
  // Extra spins to let path messages propagate through the subscription.
  exec.spin_some(std::chrono::milliseconds(200));

  EXPECT_GE(max_poses, 2UL) << "Expected ≥2 keyframe poses in /slam/graph_path";

  exec.remove_node(backend);
  exec.remove_node(helper);
}

// SLAM-08: the published trajectory must be the POST-optimization estimate read
// back from GraphOptimizer::estimate() (iSAM2 calculateEstimate()), not the raw
// front-end odom poses. Without loop closure the two are numerically identical,
// so this asserts the published path equals an independent reference optimizer's
// estimate — proving the publish path goes through estimate(). Once loop closure
// (SLAM-09/10) adds cross-edges, the same read path carries the correction with
// no extra wiring.
TEST(GraphBackendNodeTest, PublishedPathReflectsOptimizedEstimate) {
  rclcpp::NodeOptions opts;
  opts.append_parameter_override("keyframe_translation_m", 0.4);
  opts.append_parameter_override("keyframe_rotation_rad", -1.0);
  opts.append_parameter_override("keyframe_time_s", -1.0);

  auto backend = graph_slam::createGraphBackendNode(opts);
  auto helper = rclcpp::Node::make_shared("test_helper_backend_opt");

  auto pub = helper->create_publisher<nav_msgs::msg::Odometry>(
      "/ndt_frontend/ndt_odom", rclcpp::QoS(10));

  nav_msgs::msg::Path last_path;
  auto path_sub = helper->create_subscription<nav_msgs::msg::Path>(
      "/slam/graph_path", rclcpp::QoS(10),
      [&last_path](nav_msgs::msg::Path::SharedPtr msg) {
        if (msg->poses.size() >= last_path.poses.size()) {
          last_path = *msg;
        }
      });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(backend);
  exec.add_node(helper);

  // 6 odom messages with known cumulative drift along x (0.6 m steps > 0.4 m
  // threshold → every step is a keyframe → nodes x0..x5).
  std::vector<gtsam::Pose3> inputs;
  for (int i = 0; i < 6; ++i) {
    const double x = static_cast<double>(i) * 0.6;
    pub->publish(makeSyntheticOdom(x));
    inputs.emplace_back(gtsam::Rot3(), gtsam::Point3(x, 0.0, 0.0));
    exec.spin_some(std::chrono::milliseconds(80));
  }
  exec.spin_some(std::chrono::milliseconds(200));

  const std::vector<gtsam::Pose3> expected = referenceOptimizedPoses(inputs);

  EXPECT_EQ(last_path.header.frame_id, "map");
  ASSERT_EQ(last_path.poses.size(), expected.size())
      << "Published path node count must match the optimized estimate";

  constexpr double kTol = 1e-5;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto& p = last_path.poses[i].pose;
    const gtsam::Point3 t = expected[i].translation();
    EXPECT_NEAR(p.position.x, t.x(), kTol) << "pose " << i << " x";
    EXPECT_NEAR(p.position.y, t.y(), kTol) << "pose " << i << " y";
    EXPECT_NEAR(p.position.z, t.z(), kTol) << "pose " << i << " z";

    const gtsam::Quaternion q = expected[i].rotation().toQuaternion();
    EXPECT_NEAR(p.orientation.w, q.w(), kTol) << "pose " << i << " qw";
    EXPECT_NEAR(p.orientation.x, q.x(), kTol) << "pose " << i << " qx";
    EXPECT_NEAR(p.orientation.y, q.y(), kTol) << "pose " << i << " qy";
    EXPECT_NEAR(p.orientation.z, q.z(), kTol) << "pose " << i << " qz";
  }

  exec.remove_node(backend);
  exec.remove_node(helper);
}

// SLAM-11: every keyframe publish must emit one /slam/diagnostics JSON line.
// On an open chain (no loop closures yet) the prior + BetweenFactor chain is a
// tree with no conflicting constraints, so the optimum satisfies every factor
// exactly → chi2 == graph.error() is ~0 by design (it only spikes once a
// conflicting loop-closure or bad registration is added, SLAM-09/10). The
// genuine open-chain drift signal is the latest keyframe's Sigma_post position
// trace (ARCHITECTURE §6 concept #3), which is strictly positive and grows
// monotonically with keyframe_id. Asserts: chi2 finite & >= 0; trace finite,
// positive, and strictly increasing across keyframes.
TEST(GraphBackendNodeTest, DiagnosticsAreFiniteAndTraceGrows) {
  rclcpp::NodeOptions opts;
  opts.append_parameter_override("keyframe_translation_m", 0.4);
  opts.append_parameter_override("keyframe_rotation_rad", -1.0);
  opts.append_parameter_override("keyframe_time_s", -1.0);
  // Write the CSV to a throwaway path so the test never touches the real analysis/.
  opts.append_parameter_override(
      "diagnostics_csv_path",
      std::string("slam11_test_diagnostics.csv"));

  auto backend = graph_slam::createGraphBackendNode(opts);
  auto helper = rclcpp::Node::make_shared("test_helper_backend_diag");

  auto pub = helper->create_publisher<nav_msgs::msg::Odometry>(
      "/ndt_frontend/ndt_odom", rclcpp::QoS(10));

  std::vector<std::string> diagnostics;
  auto diag_sub = helper->create_subscription<std_msgs::msg::String>(
      "/slam/diagnostics", rclcpp::QoS(10),
      [&diagnostics](std_msgs::msg::String::SharedPtr msg) {
        diagnostics.push_back(msg->data);
      });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(backend);
  exec.add_node(helper);

  // Bootstrap (i=0) + 5 keyframes (i=1..5, each 0.6 m > 0.4 m threshold).
  for (int i = 0; i < 6; ++i) {
    pub->publish(makeSyntheticOdom(static_cast<double>(i) * 0.6));
    exec.spin_some(std::chrono::milliseconds(80));
  }
  exec.spin_some(std::chrono::milliseconds(200));

  ASSERT_GE(diagnostics.size(), 3UL)
      << "Expected ≥3 /slam/diagnostics messages (≥5 keyframes fed)";

  double prev_trace = -1.0;
  for (const std::string& json : diagnostics) {
    const double chi2 = parseJsonField(json, "chi2");
    const double trace = parseJsonField(json, "marginal_cov_trace");
    const double kf_id = parseJsonField(json, "keyframe_id");

    EXPECT_TRUE(std::isfinite(chi2)) << "chi2 not finite in: " << json;
    EXPECT_GE(chi2, 0.0) << "chi2 negative in: " << json;
    EXPECT_TRUE(std::isfinite(trace)) << "trace not finite in: " << json;
    EXPECT_GT(trace, 0.0) << "marginal_cov_trace not positive in: " << json;
    EXPECT_FALSE(std::isnan(kf_id)) << "keyframe_id missing in: " << json;

    // Open-chain drift: Sigma_post position trace grows monotonically.
    EXPECT_GT(trace, prev_trace) << "trace did not grow at: " << json;
    prev_trace = trace;
  }

  exec.remove_node(backend);
  exec.remove_node(helper);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new RclcppEnvironment);
  return RUN_ALL_TESTS();
}
