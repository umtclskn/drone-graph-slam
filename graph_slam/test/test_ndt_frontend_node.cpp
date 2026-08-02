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

#include <chrono>
#include <graph_slam_msgs/msg/optimized_state.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vector>

#include "ndt_frontend_node.hpp"

namespace {

using PointT = pcl::PointXYZI;

/// Dense cube with good 3D spread: passes the NDT-04 quality gate.
sensor_msgs::msg::PointCloud2 makeSpreadCloud(const rclcpp::Time& stamp) {
  pcl::PointCloud<PointT> cloud;
  constexpr int kN = 8;
  constexpr float kSpan = 5.0F;
  cloud.reserve(static_cast<std::size_t>(kN * kN * kN));
  for (int i = 0; i < kN; ++i) {
    for (int j = 0; j < kN; ++j) {
      for (int k = 0; k < kN; ++k) {
        PointT p;
        p.x = static_cast<float>(i) * kSpan / static_cast<float>(kN - 1);
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

/// L5-06 anchor for the front-end IMU predictor: an optimized state at the
/// origin moving at `vx` along +x, zero bias.
graph_slam_msgs::msg::OptimizedState makeOptimizedState(const rclcpp::Time& stamp,
                                                        double vx) {
  graph_slam_msgs::msg::OptimizedState state;
  state.header.stamp = stamp;
  state.header.frame_id = "map";
  state.pose.orientation.w = 1.0;
  state.velocity.x = vx;
  return state;
}

/// A stationary/constant-velocity IMU sample in ENU: the accelerometer measures
/// specific force, so a body with zero world acceleration reads +g on z.
sensor_msgs::msg::Imu makeLevelImu(const rclcpp::Time& stamp, double gravity) {
  sensor_msgs::msg::Imu imu;
  imu.header.stamp = stamp;  // ignored by the node, which re-stamps with its clock
  imu.header.frame_id = "base_link";
  imu.linear_acceleration.z = gravity;
  return imu;
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
  // L5-06: the guess inputs replaced ekf2_topic, which no longer exists.
  EXPECT_EQ(node->get_parameter("imu_topic").as_string(), "/imu/data");
  EXPECT_EQ(node->get_parameter("optimized_state_topic").as_string(),
            "/slam/optimized_state");
  EXPECT_FALSE(node->has_parameter("ekf2_topic"));
  EXPECT_DOUBLE_EQ(node->get_parameter("min_translation_m").as_double(), 0.3);
  EXPECT_DOUBLE_EQ(node->get_parameter("min_rotation_deg").as_double(), 5.0);
  EXPECT_TRUE(node->get_parameter("publish_debug_clouds").as_bool());
}

// NDT-11 gate enforcement, no-prior branch: when every registration is
// rejected and no optimized state ever arrived (so the L5-06 predictor has no
// anchor), only the bootstrap odom message may appear — the rejected step must
// NOT be published or advance the keyframe.
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

// L5-06c: a rejected registration is skipped even when the IMU predictor has a
// perfectly good prediction available. This replaces the old
// GateRejectFallsBackToEkf2Delta test: substituting the prior for the rejected
// NDT output was safe while the prior was EKF2 (an independent source) but
// became a feedback loop once the prior is the IMU prediction that the
// back-end's own optimized state re-anchors (see the comment at the gate).
//
// The setup also exercises the whole L5-06 plumbing end to end — /imu/data
// intake, the optimized-state anchor, and predict() — on a synthetic
// constant-velocity segment: anchored at the origin at 1 m/s along +x and fed
// 0.5 s of level IMU (zero world acceleration), so the prediction is a pure
// 0.5 m translation in x. The analytic accuracy of that prediction is covered
// closed-form by the L5-01d ImuPreintegrator tests; what is node-level here is
// that the node feeds it and no longer publishes it as a measurement.
//
// The node runs on sim time driven by this test, so the IMU dt it re-stamps
// with (bag-replay-clock-domain) is exact and the result is deterministic
// rather than wall-clock dependent.
TEST(NdtFrontendNodeTest, GateRejectSkipsScanEvenWithPrediction) {
  constexpr double kGravity = 9.8;
  constexpr double kVelocity = 1.0;  // m/s along +x
  constexpr double kStep = 0.05;     // s between IMU samples
  constexpr int kSamples = 10;       // -> 0.5 s of integration, i.e. 0.5 m in x

  rclcpp::NodeOptions options = rejectingGateOptions();
  options.append_parameter_override("use_sim_time", true);
  options.append_parameter_override("imu_gravity", kGravity);
  const auto node = graph_slam::createNdtFrontendNode(options);

  auto helper = rclcpp::Node::make_shared("gate_test_helper_prediction");
  auto clock_pub = helper->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
  auto scan_pub = helper->create_publisher<sensor_msgs::msg::PointCloud2>(
      node->get_parameter("lidar_topic").as_string(), 10);
  auto imu_pub = helper->create_publisher<sensor_msgs::msg::Imu>(
      node->get_parameter("imu_topic").as_string(), rclcpp::SensorDataQoS());
  auto state_pub = helper->create_publisher<graph_slam_msgs::msg::OptimizedState>(
      node->get_parameter("optimized_state_topic").as_string(), 10);
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

  state_pub->publish(makeOptimizedState(node->now(), kVelocity));  // anchor the predictor
  pump();
  imu_pub->publish(makeLevelImu(node->now(), kGravity));  // primes last_imu_stamp_
  pump();

  scan_pub->publish(makeSpreadCloud(node->now()));  // bootstrap; latches prediction
  pump();
  ASSERT_EQ(received.size(), 1U);

  for (int i = 0; i < kSamples; ++i) {
    tick(kStep);
    imu_pub->publish(makeLevelImu(node->now(), kGravity));
    pump();
  }

  scan_pub->publish(makeSpreadCloud(node->now()));  // rejected -> must be skipped
  pump();
  EXPECT_EQ(received.size(), 1U)
      << "a rejected registration must not publish the IMU prediction as odometry";
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new RclcppEnvironment);
  return RUN_ALL_TESTS();
}
