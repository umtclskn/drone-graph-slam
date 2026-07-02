// GRAPH-VIS-01: publish optimized graph state from GraphOptimizer into RViz.
//
// Topics (map frame):
//   /slam/graph_path    nav_msgs/Path
//   /slam/keyframes     visualization_msgs/MarkerArray (SPHERE per keyframe)
//   /slam/graph_edges   visualization_msgs/Marker (LINE_LIST odometry chain)
//
// The algorithm class stays ROS-free; only THIS executable links ROS. The node
// holds a shared_ptr to the same GraphOptimizer instance the back-end uses so
// graph state is not duplicated.

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <chrono>
#include <cmath>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <tf2_ros/static_transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "graph_slam/eval/graph_visualizer.hpp"
#include "graph_slam/graph/graph_bootstrap.hpp"
#include "graph_slam/graph/graph_optimizer.hpp"
#include "graph_slam/graph/odometry_factor_builder.hpp"

namespace graph_slam {
namespace {

geometry_msgs::msg::Pose pose3ToMsg(const gtsam::Pose3& pose) {
  geometry_msgs::msg::Pose msg;
  const gtsam::Point3 t = pose.translation();
  msg.position.x = t.x();
  msg.position.y = t.y();
  msg.position.z = t.z();
  const gtsam::Quaternion q = pose.rotation().toQuaternion();
  msg.orientation.w = q.w();
  msg.orientation.x = q.x();
  msg.orientation.y = q.y();
  msg.orientation.z = q.z();
  return msg;
}

nav_msgs::msg::Path buildPath(const eval::GraphVisualization& viz, const std::string& frame_id,
                              const rclcpp::Time& stamp) {
  nav_msgs::msg::Path path;
  path.header.frame_id = frame_id;
  path.header.stamp = stamp;
  path.poses.reserve(viz.nodeCount());
  for (const gtsam::Pose3& pose : viz.poses) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header = path.header;
    ps.pose = pose3ToMsg(pose);
    path.poses.push_back(ps);
  }
  return path;
}

visualization_msgs::msg::MarkerArray buildKeyframeMarkers(const eval::GraphVisualization& viz,
                                                        const std::string& frame_id,
                                                        const rclcpp::Time& stamp,
                                                        double sphere_diameter) {
  visualization_msgs::msg::MarkerArray array;
  for (std::size_t i = 0; i < viz.nodeCount(); ++i) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = "keyframes";
    marker.id = static_cast<int32_t>(i);
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = pose3ToMsg(viz.poses[i]);
    marker.scale.x = sphere_diameter;
    marker.scale.y = sphere_diameter;
    marker.scale.z = sphere_diameter;
    marker.color.r = 0.2F;
    marker.color.g = 0.8F;
    marker.color.b = 1.0F;
    marker.color.a = 1.0F;
    array.markers.push_back(marker);
  }
  return array;
}

visualization_msgs::msg::Marker buildOdometryEdges(const eval::GraphVisualization& viz,
                                                   const std::string& frame_id,
                                                   const rclcpp::Time& stamp, float line_width) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = "odometry_edges";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = line_width;
  marker.color.r = 1.0F;
  marker.color.g = 1.0F;
  marker.color.b = 1.0F;
  marker.color.a = 0.9F;
  marker.points.reserve(viz.odometryEdgeCount() * 2);
  for (std::size_t i = 0; i + 1 < viz.nodeCount(); ++i) {
    geometry_msgs::msg::Point start;
    geometry_msgs::msg::Point end;
    const gtsam::Point3 t0 = viz.poses[i].translation();
    const gtsam::Point3 t1 = viz.poses[i + 1].translation();
    start.x = t0.x();
    start.y = t0.y();
    start.z = t0.z();
    end.x = t1.x();
    end.y = t1.y();
    end.z = t1.z();
    marker.points.push_back(start);
    marker.points.push_back(end);
  }
  return marker;
}

class GraphVisualizerNode : public rclcpp::Node {
 public:
  explicit GraphVisualizerNode(std::shared_ptr<graph::GraphOptimizer> optimizer)
      : rclcpp::Node("graph_visualizer"), optimizer_(std::move(optimizer)) {
    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 10.0);
    sphere_diameter_m_ = declare_parameter<double>("sphere_diameter_m", 0.15);
    edge_line_width_m_ = declare_parameter<double>("edge_line_width_m", 0.03);
    const bool self_test = declare_parameter<bool>("self_test", false);
    // Bare `ros2 run` has no TF tree; publish identity world->map so RViz can use
    // Fixed Frame `map`. Disable when a sim/back-end already publishes map/odom.
    const bool publish_tf = declare_parameter<bool>("publish_tf", true);

    path_pub_ = create_publisher<nav_msgs::msg::Path>("/slam/graph_path", 10);
    keyframes_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>("/slam/keyframes", 10);
    edges_pub_ = create_publisher<visualization_msgs::msg::Marker>("/slam/graph_edges", 10);

    const auto period = std::chrono::duration<double>(1.0 / publish_rate_hz_);
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        [this]() { publishVisualization(); });

    RCLCPP_INFO(get_logger(),
                "graph_visualizer up. Publishing /slam/graph_path, /slam/keyframes, "
                "/slam/graph_edges in frame '%s' at %.1f Hz.",
                frame_id_.c_str(), publish_rate_hz_);

    if (publish_tf) {
      static_tf_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = now();
      tf.header.frame_id = "world";
      tf.child_frame_id = frame_id_;
      tf.transform.rotation.w = 1.0;
      static_tf_->sendTransform(tf);
      RCLCPP_INFO(get_logger(),
                  "Publishing static identity TF world -> %s (set RViz Fixed Frame to '%s').",
                  frame_id_.c_str(), frame_id_.c_str());
    }

    if (self_test) {
      self_test_step_ = 0;
      self_test_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() { runSelfTestStep(); });
      RCLCPP_INFO(get_logger(), "self_test enabled: growing a synthetic odometry chain.");
    }
  }

 private:
  void publishVisualization() {
    const auto stamp = now();
    const eval::GraphVisualization viz =
        eval::GraphVisualizer::fromEstimate(optimizer_->estimate());

    path_pub_->publish(buildPath(viz, frame_id_, stamp));
    keyframes_pub_->publish(
        buildKeyframeMarkers(viz, frame_id_, stamp, sphere_diameter_m_));
    edges_pub_->publish(buildOdometryEdges(viz, frame_id_, stamp,
                                          static_cast<float>(edge_line_width_m_)));
  }

  void runSelfTestStep() {
    using graph::GraphBootstrap;
    using graph::OdometryFactorBuilder;

    constexpr int kMaxNodes = 8;
    if (self_test_step_ > kMaxNodes) {
      return;
    }

    const auto noise =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(0.01));
    const gtsam::Matrix66 sigma = gtsam::Matrix66::Identity() * 0.01;
    const GraphBootstrap bootstrap;
    const OdometryFactorBuilder factor_builder;

    if (self_test_step_ == 0) {
      const gtsam::Pose3 x0_pose;
      const auto boot = bootstrap.create(x0_pose, noise);
      const auto* prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(
          boot.graph.at(0).get());
      if (prior == nullptr) {
        RCLCPP_ERROR(get_logger(), "self_test bootstrap failed.");
        return;
      }
      const gtsam::Key x0 = gtsam::Symbol('x', 0);
      optimizer_->add_prior(*prior, x0, boot.values.at<gtsam::Pose3>(x0));
      optimizer_->update();
    } else {
      const std::size_t prev = static_cast<std::size_t>(self_test_step_ - 1);
      const std::size_t cur = static_cast<std::size_t>(self_test_step_);
      const gtsam::Key x_prev = gtsam::Symbol('x', prev);
      const gtsam::Key x_cur = gtsam::Symbol('x', cur);
      const double angle = static_cast<double>(self_test_step_) * M_PI / 8.0;
      const gtsam::Pose3 delta(gtsam::Rot3::Rz(angle),
                               gtsam::Point3(1.0 * std::cos(angle), 1.0 * std::sin(angle), 0.0));
      const gtsam::Pose3 x_prev_pose = optimizer_->estimate().at<gtsam::Pose3>(x_prev);
      const gtsam::Pose3 x_cur_init = x_prev_pose.compose(delta);
      optimizer_->add_odometry(factor_builder.create_factor(x_prev, x_cur, delta, sigma),
                               x_cur, x_cur_init);
      optimizer_->update();
    }

    ++self_test_step_;
  }

  std::string frame_id_;
  double publish_rate_hz_ = 10.0;
  double sphere_diameter_m_ = 0.15;
  double edge_line_width_m_ = 0.03;
  int self_test_step_ = 0;

  std::shared_ptr<graph::GraphOptimizer> optimizer_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr keyframes_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr edges_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr self_test_timer_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_;
};

}  // namespace

std::shared_ptr<rclcpp::Node> createGraphVisualizerNode(
    std::shared_ptr<graph::GraphOptimizer> optimizer) {
  return std::make_shared<GraphVisualizerNode>(std::move(optimizer));
}

}  // namespace graph_slam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto optimizer = std::make_shared<graph_slam::graph::GraphOptimizer>();
  rclcpp::spin(graph_slam::createGraphVisualizerNode(optimizer));
  rclcpp::shutdown();
  return 0;
}
