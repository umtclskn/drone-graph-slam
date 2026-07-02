// INFRA-01: verify the REP-105 TF chain map -> odom -> base_link is lookup-able
// once static stubs and the ndt_frontend odom->base_link broadcaster are running.
// Static map->odom and base_link->lidar_link are published in-test (same as launch).

#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <chrono>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "ndt_frontend_node.hpp"

namespace {

using PointT = pcl::PointXYZI;

class RclcppEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { rclcpp::init(0, nullptr); }
  void TearDown() override { rclcpp::shutdown(); }
};

/// Publishes the two static transforms that the launch file provides.
class StaticTfStubs : public rclcpp::Node {
 public:
  StaticTfStubs() : Node("static_tf_stubs") {
    broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    publishIdentity("map", "odom");
    publishIdentity("base_link", "lidar_link");
  }

 private:
  void publishIdentity(const std::string& parent, const std::string& child) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = now();
    tf.header.frame_id = parent;
    tf.child_frame_id = child;
    tf.transform.rotation.w = 1.0;
    broadcaster_->sendTransform(tf);
  }

  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> broadcaster_;
};

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

TEST(TfTreeTest, MapToBaseLinkLookupSucceeds) {
  auto static_tf = std::make_shared<StaticTfStubs>();
  auto frontend = graph_slam::createNdtFrontendNode();
  ASSERT_NE(frontend, nullptr);

  const std::string lidar_topic = frontend->get_parameter("lidar_topic").as_string();

  auto scan_pub_node = rclcpp::Node::make_shared("scan_publisher");
  auto scan_pub = scan_pub_node->create_publisher<sensor_msgs::msg::PointCloud2>(lidar_topic, 10);

  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(frontend->get_clock());
  tf2_ros::TransformListener tf_listener(*tf_buffer, frontend, false);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(static_tf);
  exec.add_node(frontend);
  exec.add_node(scan_pub_node);

  const auto scan = makeSpreadCloud(frontend->now());
  scan_pub->publish(scan);

  bool lookup_ok = false;
  for (int attempt = 0; attempt < 100 && !lookup_ok; ++attempt) {
    exec.spin_some(std::chrono::milliseconds(20));
    try {
      (void)tf_buffer->lookupTransform("map", "base_link", tf2::TimePointZero);
      lookup_ok = true;
    } catch (const tf2::TransformException&) {
      // odom->base_link arrives with the first accepted scan; keep spinning.
    }
  }

  EXPECT_TRUE(lookup_ok) << "map -> base_link TF chain incomplete";
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new RclcppEnvironment);
  return RUN_ALL_TESTS();
}
