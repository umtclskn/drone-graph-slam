// EVAL-03: source/target/aligned overlay as a live ROS2 node.
//
// Flow: subscribe to a LiDAR cloud; on the std_srvs/Trigger service `~/capture`
// snapshot the current scan as the TARGET; the NEXT incoming scan becomes the
// SOURCE; register source->target with the graph_slam pipeline (Preprocessor ->
// NdtVoxelGrid -> NdtRegistrar) and publish source (red) / target (green) /
// aligned = T*source (blue) for RViz, plus a CSV for the matplotlib 2D view.
//
// graph_slam's algorithm classes stay ROS-free; only THIS executable links ROS.

#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <string>

#include "graph_slam/ndt_registrar.hpp"
#include "graph_slam/ndt_voxel_grid.hpp"
#include "graph_slam/point_types.hpp"
#include "graph_slam/preprocessor.hpp"
#include "graph_slam/registration_gate.hpp"

namespace graph_slam {
namespace {

/// A closed 5x5x3 m room with ~1 cm surface jitter, used by `self_test` so the
/// demo runs (and aligns well) without a bag.
CloudPtr makeRoom() {
  auto cloud = std::make_shared<Cloud>();
  std::mt19937 rng(42);
  std::normal_distribution<float> jitter(0.0F, 0.01F);
  const auto add = [&](float x, float y, float z) {
    PointT p;
    p.x = x + jitter(rng);
    p.y = y + jitter(rng);
    p.z = z + jitter(rng);
    p.intensity = 1.0F;
    cloud->push_back(p);
  };
  constexpr float kStep = 0.1F;
  for (float x = 0.0F; x <= 5.0F + 1e-4F; x += kStep) {
    for (float y = 0.0F; y <= 5.0F + 1e-4F; y += kStep) {
      add(x, y, 0.0F);
      add(x, y, 3.0F);
    }
  }
  for (float x = 0.0F; x <= 5.0F + 1e-4F; x += kStep) {
    for (float z = 0.0F; z <= 3.0F + 1e-4F; z += kStep) {
      add(x, 0.0F, z);
      add(x, 5.0F, z);
    }
  }
  for (float y = 0.0F; y <= 5.0F + 1e-4F; y += kStep) {
    for (float z = 0.0F; z <= 3.0F + 1e-4F; z += kStep) {
      add(0.0F, y, z);
      add(5.0F, y, z);
    }
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  return cloud;
}

class Eval03OverlayNode : public rclcpp::Node {
 public:
  Eval03OverlayNode() : rclcpp::Node("eval03_overlay") {
    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/x500/lidar_3d/points");
    csv_path_ = declare_parameter<std::string>("csv_path", "eval03_overlay.csv");
    frame_id_ = declare_parameter<std::string>("frame_id", "lidar");
    plot_on_capture_ = declare_parameter<bool>("plot_on_capture", true);
    voxel_leaf_ = declare_parameter<double>("voxel_leaf", 0.2);
    ndt_resolution_ = declare_parameter<double>("ndt_resolution", 1.0);
    // NDT optimiser + gate knobs (INFRA-02); defaults equal the struct defaults.
    ndt_cfg_.max_iterations = declare_parameter<int>("ndt_max_iterations", ndt_cfg_.max_iterations);
    ndt_cfg_.step_epsilon = declare_parameter<double>("ndt_step_epsilon", ndt_cfg_.step_epsilon);
    ndt_cfg_.outlier_ratio = declare_parameter<double>("ndt_outlier_ratio", ndt_cfg_.outlier_ratio);
    ndt_cfg_.regularization =
        declare_parameter<double>("ndt_regularization", ndt_cfg_.regularization);
    gate_cfg_.max_fitness_score =
        declare_parameter<double>("gate_max_fitness_score", gate_cfg_.max_fitness_score);
    gate_cfg_.min_hessian_eigenvalue =
        declare_parameter<double>("gate_min_hessian_eigenvalue", gate_cfg_.min_hessian_eigenvalue);
    gate_cfg_.max_condition_number =
        declare_parameter<double>("gate_max_condition_number", gate_cfg_.max_condition_number);
    // Set frame_id to the frame your scans are published in. Only publish our own
    // static TF when nothing else does (e.g. replaying a bag with no /tf); with a
    // live sim that already publishes the lidar frame, set publish_tf:=false.
    const bool publish_tf = declare_parameter<bool>("publish_tf", true);
    const bool self_test = declare_parameter<bool>("self_test", false);

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { onScan(msg); });
    pub_source_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/source", 1);
    pub_target_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/target", 1);
    pub_aligned_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/aligned", 1);
    capture_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/capture",
        [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
               std::shared_ptr<std_srvs::srv::Trigger::Response> res) { onCapture(req, res); });

    // Optionally publish a static identity TF for the overlay frame so RViz has a
    // frame to render in when nothing else publishes it (e.g. a bag with no /tf).
    // With a live sim that already publishes the lidar frame, set publish_tf:=false
    // to avoid a duplicate TF authority.
    if (publish_tf) {
      static_tf_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = now();
      tf.header.frame_id = "world";
      tf.child_frame_id = frame_id_;
      tf.transform.rotation.w = 1.0;  // identity
      static_tf_->sendTransform(tf);
    }

    RCLCPP_INFO(get_logger(),
                "eval03_overlay up. Subscribing '%s'. Call '%s/capture' then the NEXT scan "
                "registers; overlay -> RViz topics + CSV '%s'.",
                lidar_topic_.c_str(), get_fully_qualified_name(), csv_path_.c_str());

    if (self_test) {
      self_test_timer_ = create_wall_timer(std::chrono::milliseconds(500), [this]() {
        self_test_timer_->cancel();  // one-shot
        runSelfTest();
      });
    }
  }

 private:
  void onScan(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
    auto cloud = std::make_shared<Cloud>();
    pcl::fromROSMsg(*msg, *cloud);
    latest_ = cloud;
    if (awaiting_source_) {
      awaiting_source_ = false;
      RCLCPP_INFO(get_logger(), "Source scan received (%zu pts) - registering.", cloud->size());
      runOverlay(cloud, target_);
    }
  }

  void onCapture(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                 std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
    if (!latest_ || latest_->empty()) {
      res->success = false;
      res->message = "No scan received yet on " + lidar_topic_;
      RCLCPP_WARN(get_logger(), "%s", res->message.c_str());
      return;
    }
    target_ = std::make_shared<Cloud>(*latest_);  // snapshot at the call moment
    awaiting_source_ = true;
    res->success = true;
    res->message =
        "Captured target (" + std::to_string(target_->size()) + " pts); waiting for next scan.";
    RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
  }

  // Preprocess both raw scans (live path), then register + publish.
  void runOverlay(const CloudConstPtr& source_raw, const CloudConstPtr& target_raw) {
    PreprocessConfig pre_cfg;
    pre_cfg.voxel_leaf = static_cast<float>(voxel_leaf_);
    const Preprocessor pre{pre_cfg};
    emitOverlay(pre.process(source_raw), pre.process(target_raw));
  }

  // Register two already-preprocessed clouds, then publish source/target/aligned.
  void emitOverlay(const CloudPtr& source, const CloudPtr& target) {
    NdtGridConfig grid_cfg;
    grid_cfg.resolution = ndt_resolution_;
    NdtVoxelGrid grid{grid_cfg};
    grid.build(target);

    const RegistrationResult result = NdtRegistrar{ndt_cfg_}.align(grid, source);
    const RegistrationStatus verdict = evaluateRegistration(result, gate_cfg_);
    RCLCPP_INFO(get_logger(), "NDT: converged=%d iters=%d fitness=%.3f -> %s",
                static_cast<int>(result.converged), result.iterations, result.fitness_score,
                toString(verdict));

    auto aligned = std::make_shared<Cloud>();
    pcl::transformPointCloud(*source, *aligned, result.transform);  // aligned = T * source

    publishCloud(*pub_source_, source);
    publishCloud(*pub_target_, target);
    publishCloud(*pub_aligned_, aligned);
    writeCsv(source, target, aligned);
    maybePlot();
  }

  // All three overlay clouds share one frame (`frame_id_`); the launch publishes a
  // static TF for it so RViz has a fixed frame to render in.
  void publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>& pub,
                    const CloudConstPtr& cloud) {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.frame_id = frame_id_;
    msg.header.stamp = now();
    pub.publish(msg);
  }

  void writeCsv(const CloudConstPtr& source, const CloudConstPtr& target,
                const CloudConstPtr& aligned) {
    std::ofstream f(csv_path_);
    if (!f) {
      RCLCPP_WARN(get_logger(), "Could not open CSV '%s' for writing.", csv_path_.c_str());
      return;
    }
    f << "label,x,y,z\n";
    const auto dump = [&f](const char* label, const CloudConstPtr& c) {
      for (const auto& p : c->points) {
        f << label << ',' << p.x << ',' << p.y << ',' << p.z << '\n';
      }
    };
    dump("source", source);
    dump("target", target);
    dump("aligned", aligned);
    RCLCPP_INFO(get_logger(), "Wrote overlay CSV: %s", csv_path_.c_str());
  }

  void maybePlot() {
    if (!plot_on_capture_) {
      return;
    }
    std::string script;
    try {
      script =
          ament_index_cpp::get_package_share_directory("graph_slam") + "/scripts/eval03_plot.py";
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Cannot locate plot script: %s", e.what());
      return;
    }
    const std::string cmd = "python3 '" + script + "' '" + csv_path_ + "' &";
    if (std::system(cmd.c_str()) != 0) {
      RCLCPP_WARN(get_logger(), "Plot launch failed; run manually: python3 %s %s", script.c_str(),
                  csv_path_.c_str());
    }
  }

  void runSelfTest() {
    RCLCPP_INFO(get_logger(), "self_test: registering a synthetic room pair (known 0.08 m shift).");
    // Target = preprocessed room; source = a clean rigid transform of it, so the
    // synthetic demo aligns crisply (avoids voxel-binning aliasing of the lattice;
    // real, noisy scans don't alias). The live path preprocesses both raw scans.
    PreprocessConfig pre_cfg;
    pre_cfg.voxel_leaf = static_cast<float>(voxel_leaf_);
    const CloudPtr target = Preprocessor{pre_cfg}.process(makeRoom());
    Eigen::Matrix4f motion = Eigen::Matrix4f::Identity();
    motion.block<3, 1>(0, 3) = Eigen::Vector3f(0.08F, -0.05F, 0.04F);  // small in-basin motion
    auto source = std::make_shared<Cloud>();
    pcl::transformPointCloud(*target, *source, motion);
    emitOverlay(source, target);
  }

  // params
  std::string lidar_topic_;
  std::string csv_path_;
  std::string frame_id_;
  bool plot_on_capture_ = true;
  double voxel_leaf_ = 0.2;
  double ndt_resolution_ = 1.0;
  NdtConfig ndt_cfg_;
  RegistrationGateConfig gate_cfg_;

  // state
  CloudPtr latest_;
  CloudPtr target_;
  bool awaiting_source_ = false;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_source_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_target_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_aligned_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr capture_srv_;
  rclcpp::TimerBase::SharedPtr self_test_timer_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_;
};

}  // namespace
}  // namespace graph_slam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<graph_slam::Eval03OverlayNode>());
  rclcpp::shutdown();
  return 0;
}
