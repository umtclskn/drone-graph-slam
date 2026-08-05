// NDT-13: minimal node-wrapper smoke test. The full pipeline (Preprocessor,
// QualityChecker, NdtVoxelGrid, NdtRegistrar, gate, Sigma_meas) is already
// covered ROS-free by the per-class story tests; this only checks that the node
// constructs, declares its parameters, wires its pub/subs, and spins without
// crashing. Construction goes through the createNdtFrontendNode() factory so we
// never pull in the executable's main().

#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <vector>

#include "ndt_frontend_node.hpp"

namespace {

using PointT = pcl::PointXYZI;

/// Dense cube with good 3D spread: passes the NDT-04 quality gate. `x_offset`
/// shifts every point along x, in the cloud's OWN local sensor frame — used by
/// the L5-07/08 submap tests to synthesize a static cube observed from a
/// sensor that moved: a real +d world translation makes the static scene
/// appear shifted by -d in the next scan's local frame.
sensor_msgs::msg::PointCloud2 makeSpreadCloud(const rclcpp::Time& stamp, float x_offset = 0.0F) {
  pcl::PointCloud<PointT> cloud;
  constexpr int kN = 8;
  constexpr float kSpan = 5.0F;
  cloud.reserve(static_cast<std::size_t>(kN * kN * kN));
  for (int i = 0; i < kN; ++i) {
    for (int j = 0; j < kN; ++j) {
      for (int k = 0; k < kN; ++k) {
        PointT p;
        p.x = x_offset + static_cast<float>(i) * kSpan / static_cast<float>(kN - 1);
        p.y = static_cast<float>(j) * kSpan / static_cast<float>(kN - 1);
        p.z = static_cast<float>(k) * kSpan / static_cast<float>(kN - 1);
        p.intensity = 1.0F;
        cloud.push_back(p);
      }
    }
  }
  cloud.width = cloud.size();
  cloud.height = 1;
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  msg.header.stamp = stamp;
  msg.header.frame_id = "lidar_link";
  return msg;
}

/// L5-18 guess source: one PX4 EKF2 odometry sample (as republished in ENU by
/// px4_offboard's ekf2_odometry_adapter), level and `x` metres along +x. The
/// node keys its buffer on this header stamp and never re-stamps it, so the
/// stamp passed here is exactly what the guess lookup sees.
nav_msgs::msg::Odometry makeEkf2Odom(const rclcpp::Time& stamp, double x) {
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = "odom";
  odom.child_frame_id = "base_link";
  odom.pose.pose.position.x = x;
  odom.pose.pose.orientation.w = 1.0;
  return odom;
}

/// Node options that make EVERY registration fail the NDT-11 gate (impossible
/// fitness threshold) and trigger publishing on tiny motion, so the tests can
/// observe the gate path deterministically.
rclcpp::NodeOptions rejectingGateOptions() {
  rclcpp::NodeOptions options;
  options.parameter_overrides({
      {"gate_max_fitness_score", -1e9},
      {"min_translation_m", 0.05},
      {"min_rotation_deg", 1.0},
      {"publish_debug_clouds", false},
  });
  return options;
}

// One process-wide rclcpp context for the test executable.
class RclcppEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { rclcpp::init(0, nullptr); }
  void TearDown() override { rclcpp::shutdown(); }
};

// Construct the node and spin it briefly: no scans arrive, so the callbacks never
// fire, but the constructor + executor path must not throw or crash.
TEST(NdtFrontendNodeTest, ConstructsAndSpinsWithoutCrashing) {
  std::shared_ptr<rclcpp::Node> node;
  ASSERT_NO_THROW(node = graph_slam::createNdtFrontendNode());
  ASSERT_NE(node, nullptr);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  exec.spin_some(std::chrono::milliseconds(100));  // returns immediately if idle
  exec.remove_node(node);
}

// The node must expose its documented parameters with the right defaults so the
// YAML (INFRA-02) can override them.
TEST(NdtFrontendNodeTest, DeclaresDocumentedParameters) {
  const auto node = graph_slam::createNdtFrontendNode();
  EXPECT_EQ(node->get_parameter("lidar_topic").as_string(), "/x500/lidar_3d/points");
  // L5-18: EKF2 is the guess source. L5-19 re-admits optimized_state for the
  // submap target only — the guess still never reads it.
  EXPECT_EQ(node->get_parameter("ekf2_topic").as_string(), "/odometry/ekf2");
  EXPECT_DOUBLE_EQ(node->get_parameter("ekf2_buffer_seconds").as_double(), 5.0);
  EXPECT_DOUBLE_EQ(node->get_parameter("ekf2_max_time_diff_s").as_double(), 0.1);
  EXPECT_FALSE(node->has_parameter("imu_topic"));
  EXPECT_EQ(node->get_parameter("optimized_state_topic").as_string(), "/slam/optimized_state");
  EXPECT_DOUBLE_EQ(node->get_parameter("min_translation_m").as_double(), 0.3);
  EXPECT_DOUBLE_EQ(node->get_parameter("min_rotation_deg").as_double(), 5.0);
  EXPECT_TRUE(node->get_parameter("publish_debug_clouds").as_bool());
  // L5-20: support-size + guess-delta gate knobs (defaults match the config struct).
  EXPECT_DOUBLE_EQ(node->get_parameter("gate_min_scored_fraction").as_double(), 0.3);
  EXPECT_DOUBLE_EQ(node->get_parameter("gate_max_guess_delta_t").as_double(), 0.5);
  EXPECT_DOUBLE_EQ(node->get_parameter("gate_max_guess_delta_rot").as_double(), 0.17);
  // L5-07/08/19: scan-to-submap target + its ablation flag + collapse ε.
  EXPECT_TRUE(node->get_parameter("submap_enabled").as_bool());
  EXPECT_EQ(node->get_parameter("submap_window_size").as_int(), 8);
  EXPECT_DOUBLE_EQ(node->get_parameter("submap_voxel_leaf").as_double(), 0.3);
  EXPECT_DOUBLE_EQ(node->get_parameter("submap_collapse_eps_m").as_double(), 0.2);
  EXPECT_DOUBLE_EQ(node->get_parameter("submap_collapse_eps_rad").as_double(), 0.1);
  // L5-10: coordinated loop-closure rebuild feed + its rebuild-threshold ε.
  EXPECT_EQ(node->get_parameter("optimized_state_batch_topic").as_string(),
            "/slam/optimized_state_batch");
  EXPECT_DOUBLE_EQ(node->get_parameter("submap_rebuild_eps_m").as_double(), 0.2);
  EXPECT_DOUBLE_EQ(node->get_parameter("submap_rebuild_eps_rad").as_double(), 0.1);
}

// L5-18d / L5-19: the guess must stay independent of the graph (EKF2 only).
// L5-19 re-admits `/slam/optimized_state` for the **submap target** — that is
// allowed — but `/slam/optimized_odom` must still not feed the front-end, and
// EKF2 must remain subscribed. A future change that routes the guess through
// optimized_state would still be a behavioural regression (covered by bag
// gates); structurally we encode "EKF2 present; optimized_odom absent".
TEST(NdtFrontendNodeTest, GuessSubscribesToEkf2SubmapMayUseOptimizedState) {
  const auto node = graph_slam::createNdtFrontendNode();
  const std::string ekf2_topic = node->get_parameter("ekf2_topic").as_string();
  const std::string opt_topic =
      node->get_parameter("optimized_state_topic").as_string();
  const std::string opt_batch_topic =
      node->get_parameter("optimized_state_batch_topic").as_string();

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  // Graph discovery is asynchronous even in-process: spin until this node's own
  // EKF2 subscription is visible, then the rest of its list is populated too.
  std::map<std::string, std::vector<std::string>> subs;
  for (int i = 0; i < 100 && subs.count(ekf2_topic) == 0; ++i) {
    exec.spin_some(std::chrono::milliseconds(20));
    subs = node->get_node_graph_interface()->get_subscriber_names_and_types_by_node(
        node->get_name(), node->get_namespace());
  }
  exec.remove_node(node);

  ASSERT_EQ(subs.count(ekf2_topic), 1U)
      << "front-end must subscribe to the EKF2 guess source " << ekf2_topic;
  EXPECT_EQ(subs.count(opt_topic), 1U)
      << "L5-19: submap pose feed " << opt_topic << " must be subscribed";
  EXPECT_EQ(subs.count(opt_batch_topic), 1U)
      << "L5-10: coordinated loop-closure feed " << opt_batch_topic
      << " must be subscribed";
  EXPECT_EQ(subs.count("/slam/optimized_odom"), 0U)
      << "back-end odometry must not feed the front-end";
}

// NDT-11 gate enforcement, no-prior branch: when every registration is rejected
// and no EKF2 sample ever arrived (so the guess has no source), only the
// bootstrap odom message may appear — the rejected step must NOT be published or
// advance the keyframe.
TEST(NdtFrontendNodeTest, GateRejectWithoutPriorSkipsScan) {
  const auto node = graph_slam::createNdtFrontendNode(rejectingGateOptions());
  auto helper = rclcpp::Node::make_shared("gate_test_helper_noprior");
  auto scan_pub = helper->create_publisher<sensor_msgs::msg::PointCloud2>(
      node->get_parameter("lidar_topic").as_string(), 10);
  std::vector<nav_msgs::msg::Odometry> received;
  auto odom_sub = helper->create_subscription<nav_msgs::msg::Odometry>(
      "/ndt_frontend/ndt_odom", 10,
      [&received](nav_msgs::msg::Odometry::ConstSharedPtr msg) { received.push_back(*msg); });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  exec.add_node(helper);
  auto pump = [&exec] {
    for (int i = 0; i < 30; ++i) {
      exec.spin_some(std::chrono::milliseconds(10));
    }
  };

  scan_pub->publish(makeSpreadCloud(node->now()));  // bootstrap keyframe -> odom #1
  pump();
  ASSERT_EQ(received.size(), 1U);

  scan_pub->publish(makeSpreadCloud(node->now()));  // rejected, no prior -> skipped
  pump();
  EXPECT_EQ(received.size(), 1U) << "rejected registration without a prior must not publish";
}

// L5-18 + L5-13a: a rejected registration is skipped even when a perfectly good
// EKF2 guess is available. Pre-L5-06 the front-end substituted the EKF2 delta
// here; L5-06c removed that fallback and L5-13a made "skip" the only path.
// L5-18 restores the independent guess but deliberately does NOT restore the
// fallback (recovery policy is L5-13's story), so the skip must still hold.
//
// The setup also exercises the whole L5-18 guess plumbing end to end — EKF2
// intake, the stamp-keyed buffer, and the two-stamp lookup — on a synthetic
// 0.5 m step along +x. The node runs on sim time driven by this test, so the
// stamps it matches on are exact and the result is deterministic rather than
// wall-clock dependent.
TEST(NdtFrontendNodeTest, GateRejectSkipsScanEvenWithEkf2Guess) {
  rclcpp::NodeOptions options = rejectingGateOptions();
  options.append_parameter_override("use_sim_time", true);
  const auto node = graph_slam::createNdtFrontendNode(options);

  auto helper = rclcpp::Node::make_shared("gate_test_helper_ekf2");
  auto clock_pub = helper->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
  auto scan_pub = helper->create_publisher<sensor_msgs::msg::PointCloud2>(
      node->get_parameter("lidar_topic").as_string(), 10);
  auto ekf2_pub = helper->create_publisher<nav_msgs::msg::Odometry>(
      node->get_parameter("ekf2_topic").as_string(), rclcpp::QoS(50));
  std::vector<nav_msgs::msg::Odometry> received;
  auto odom_sub = helper->create_subscription<nav_msgs::msg::Odometry>(
      "/ndt_frontend/ndt_odom", 10,
      [&received](nav_msgs::msg::Odometry::ConstSharedPtr msg) { received.push_back(*msg); });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  exec.add_node(helper);
  auto pump = [&exec] {
    for (int i = 0; i < 30; ++i) {
      exec.spin_some(std::chrono::milliseconds(10));
    }
  };
  double sim_t = 1000.0;
  auto tick = [&](double dt) {
    sim_t += dt;
    rosgraph_msgs::msg::Clock clock;
    clock.clock = rclcpp::Time(static_cast<int64_t>(sim_t * 1e9), RCL_ROS_TIME);
    clock_pub->publish(clock);
    pump();
  };

  tick(0.0);
  ASSERT_NEAR(node->now().seconds(), sim_t, 1e-6) << "node is not on the test's sim clock";

  ekf2_pub->publish(makeEkf2Odom(node->now(), 0.0));  // EKF2 at the keyframe stamp
  pump();
  scan_pub->publish(makeSpreadCloud(node->now()));    // bootstrap keyframe -> odom #1
  pump();
  ASSERT_EQ(received.size(), 1U);

  tick(0.5);
  ekf2_pub->publish(makeEkf2Odom(node->now(), 0.5));  // EKF2 says +0.5 m since the kf
  pump();
  scan_pub->publish(makeSpreadCloud(node->now()));    // rejected -> must be skipped
  pump();
  EXPECT_EQ(received.size(), 1U)
      << "a rejected registration must not publish the EKF2 guess as odometry";
}

// L5-07/08: node options for the submap scenario below — small motion
// thresholds so a real (gate-accepted) 0.5 m step advances the keyframe, and
// otherwise DEFAULT (non-rejecting) gate/quality thresholds, since this
// scenario needs genuine NDT acceptance rather than the forced-reject trick
// `rejectingGateOptions()` uses elsewhere in this file.
//
// L5-20: the synthetic 8³ cube only lands ~12 % of points in occupied voxels
// (sparse vs ndt_resolution=1.0), so `gate_min_scored_fraction` is relaxed here
// to exercise window growth rather than the support check (covered in
// test_registration_gate). Guess-delta stays at the code default (0.5 m / 0.17
// rad) — the synthetic EKF2 match is exact.
rclcpp::NodeOptions submapNodeOptions(bool submap_enabled, int window_size) {
  rclcpp::NodeOptions options;
  options.parameter_overrides({
      {"min_translation_m", 0.1},
      {"min_rotation_deg", 1.0},
      {"publish_debug_clouds", false},
      {"submap_enabled", submap_enabled},
      {"submap_window_size", window_size},
      {"submap_voxel_leaf", 0.3},
      {"use_sim_time", true},
      {"gate_min_scored_fraction", 0.05},
  });
  return options;
}

// L5-07/08: bootstrap + two real, gate-accepted keyframes, each a synthetic
// exact +0.5 m x-translation (the cube's local-frame points are shifted -0.5 m
// per step to represent that real motion — see makeSpreadCloud's doc comment —
// and EKF2 is fed the matching absolute poses 0.0/0.5/1.0 m at exactly the scan
// stamps, so the L5-18 two-stamp guess is a near-exact match and default gate
// thresholds accept). By the third scan the target (when submap_enabled) is the
// L5-07 submap merged from BOTH prior keyframes (window_size=2), exercising the
// merge/voxel-downsample path, not just the size-1 degenerate case — and, since
// L5-18, exercising it with window poses taken from the node's own NDT odometry
// chain rather than from any back-end feed.
std::vector<nav_msgs::msg::Odometry> runSubmapScenario(bool submap_enabled, int window_size) {
  constexpr double kInterval = 0.5;  // s between scans

  const rclcpp::NodeOptions options = submapNodeOptions(submap_enabled, window_size);
  const auto node = graph_slam::createNdtFrontendNode(options);

  auto helper = rclcpp::Node::make_shared("submap_test_helper");
  auto clock_pub = helper->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
  auto scan_pub = helper->create_publisher<sensor_msgs::msg::PointCloud2>(
      node->get_parameter("lidar_topic").as_string(), 10);
  auto ekf2_pub = helper->create_publisher<nav_msgs::msg::Odometry>(
      node->get_parameter("ekf2_topic").as_string(), rclcpp::QoS(50));
  std::vector<nav_msgs::msg::Odometry> received;
  auto odom_sub = helper->create_subscription<nav_msgs::msg::Odometry>(
      "/ndt_frontend/ndt_odom", 10,
      [&received](nav_msgs::msg::Odometry::ConstSharedPtr msg) { received.push_back(*msg); });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  exec.add_node(helper);
  auto pump = [&exec] {
    for (int i = 0; i < 30; ++i) {
      exec.spin_some(std::chrono::milliseconds(10));
    }
  };
  double sim_t = 2000.0;
  auto tick = [&](double dt) {
    sim_t += dt;
    rosgraph_msgs::msg::Clock clock;
    clock.clock = rclcpp::Time(static_cast<int64_t>(sim_t * 1e9), RCL_ROS_TIME);
    clock_pub->publish(clock);
    pump();
  };
  // One step of the flight: advance the clock, publish where EKF2 thinks the
  // drone now is, then the scan the LiDAR would have taken there. Both carry the
  // same stamp, so the guess lookup for this scan and for the previous keyframe
  // both land on an exact sample.
  auto step = [&](double ekf2_x, float cloud_offset) {
    ekf2_pub->publish(makeEkf2Odom(node->now(), ekf2_x));
    pump();
    scan_pub->publish(makeSpreadCloud(node->now(), cloud_offset));
    pump();
  };

  tick(0.0);
  step(0.0, 0.0F);  // bootstrap -> kf0

  tick(kInterval);
  step(0.5, -0.5F);  // EKF2 guess +0.5 m, real +0.5 m -> kf1

  tick(kInterval);
  step(1.0, -1.0F);  // another +0.5 m; registers against the (possibly
                     // 2-keyframe) submap

  return received;
}

TEST(NdtFrontendNodeTest, ScanToSubmapWindowGrowsAndIsDeterministic) {
  const auto run1 = runSubmapScenario(/*submap_enabled=*/true, /*window_size=*/2);
  const auto run2 = runSubmapScenario(/*submap_enabled=*/true, /*window_size=*/2);

  ASSERT_EQ(run1.size(), 3U) << "bootstrap + two real accepted keyframes expected";
  ASSERT_EQ(run2.size(), 3U);

  // L5-07d: given fixed clouds + poses, the built submap (and everything
  // downstream of it) is reproducible run to run.
  for (std::size_t i = 0; i < run1.size(); ++i) {
    EXPECT_NEAR(run1[i].pose.pose.position.x, run2[i].pose.pose.position.x, 1e-6) << "msg " << i;
    EXPECT_NEAR(run1[i].pose.pose.position.y, run2[i].pose.pose.position.y, 1e-6) << "msg " << i;
    EXPECT_NEAR(run1[i].pose.pose.position.z, run2[i].pose.pose.position.z, 1e-6) << "msg " << i;
  }

  // Each accepted keyframe recovered close to the true 0.5 m x-translation
  // baked into the synthetic clouds: registering against the merged 2-keyframe
  // submap (message 3) did not silently break registration.
  EXPECT_NEAR(run1[1].pose.pose.position.x, 0.5, 0.05);
  EXPECT_NEAR(run1[2].pose.pose.position.x, 1.0, 0.05);
}

TEST(NdtFrontendNodeTest, SubmapAblationDisabledStillRegistersCorrectly) {
  // L5-08a ablation flag: submap_enabled=false must fall back to the old
  // single-scan target end to end (no merge, no crash) and still recover the
  // same true translations as the submap-enabled run above.
  const auto run = runSubmapScenario(/*submap_enabled=*/false, /*window_size=*/2);

  ASSERT_EQ(run.size(), 3U);
  EXPECT_NEAR(run[1].pose.pose.position.x, 0.5, 0.05);
  EXPECT_NEAR(run[2].pose.pose.position.x, 1.0, 0.05);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new RclcppEnvironment);
  return RUN_ALL_TESTS();
}
