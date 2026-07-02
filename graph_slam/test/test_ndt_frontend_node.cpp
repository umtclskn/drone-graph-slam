// NDT-13: minimal node-wrapper smoke test. The full pipeline (Preprocessor,
// QualityChecker, NdtVoxelGrid, NdtRegistrar, gate, Sigma_meas) is already
// covered ROS-free by the per-class story tests; this only checks that the node
// constructs, declares its parameters, wires its pub/subs, and spins without
// crashing. Construction goes through the createNdtFrontendNode() factory so we
// never pull in the executable's main().

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "ndt_frontend_node.hpp"

namespace {

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

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new RclcppEnvironment);
  return RUN_ALL_TESTS();
}
