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
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
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
  EXPECT_EQ(node->get_parameter("ekf2_topic").as_string(), "/odometry/ekf2");
  EXPECT_DOUBLE_EQ(node->get_parameter("min_translation_m").as_double(), 0.3);
  EXPECT_DOUBLE_EQ(node->get_parameter("min_rotation_deg").as_double(), 5.0);
  EXPECT_TRUE(node->get_parameter("publish_debug_clouds").as_bool());
}

// NDT-11 gate enforcement, no-prior branch: when every registration is
// rejected and no EKF2 odometry ever arrived, only the bootstrap odom message
// may appear — the rejected step must NOT be published or advance the keyframe.
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

// NDT-11 gate enforcement, fallback branch: with an EKF2 prior available, a
// rejected registration publishes the EKF2 relative motion instead, carrying
// the geometry-agnostic fallback covariance (sigma_t=0.1 m, sigma_rot=0.01 rad).
TEST(NdtFrontendNodeTest, GateRejectFallsBackToEkf2Delta) {
  const auto node = graph_slam::createNdtFrontendNode(rejectingGateOptions());
  auto helper = rclcpp::Node::make_shared("gate_test_helper_fallback");
  auto scan_pub = helper->create_publisher<sensor_msgs::msg::PointCloud2>(
      node->get_parameter("lidar_topic").as_string(), 10);
  auto ekf2_pub = helper->create_publisher<nav_msgs::msg::Odometry>(
      node->get_parameter("ekf2_topic").as_string(), 10);
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

  ekf2_pub->publish(makeEkf2Odom(node->now(), 0.0));  // prior pose A
  pump();
  scan_pub->publish(makeSpreadCloud(node->now()));  // bootstrap; snapshots prior A
  pump();
  ASSERT_EQ(received.size(), 1U);

  ekf2_pub->publish(makeEkf2Odom(node->now(), 0.5));  // prior pose B: +0.5 m in x
  pump();
  scan_pub->publish(makeSpreadCloud(node->now()));  // rejected -> EKF2 delta fallback
  pump();
  ASSERT_EQ(received.size(), 2U) << "fallback step must be published";

  const auto& odom = received.back();
  EXPECT_NEAR(odom.pose.pose.position.x, 0.5, 1e-3);  // EKF2 delta, not NDT output
  EXPECT_NEAR(odom.pose.pose.position.y, 0.0, 1e-3);
  // nav covariance order [x,y,z,rx,ry,rz]: fallback diag(0.1^2 x3, 0.01^2 x3).
  // Sigma is stored in float (Matrix6f), so compare at float precision.
  EXPECT_NEAR(odom.pose.covariance[0], 0.01, 1e-8);   // x variance
  EXPECT_NEAR(odom.pose.covariance[21], 1e-4, 1e-9);  // rx variance
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new RclcppEnvironment);
  return RUN_ALL_TESTS();
}
