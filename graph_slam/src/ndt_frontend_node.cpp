// NDT-13: the whole NDT front-end pipeline as ONE rclcpp node.
//
// Per scan (ARCHITECTURE §4 order):
//   LiDAR -> Preprocessor -> QualityChecker (NDT-04, reject -> log + skip)
//         -> NdtVoxelGrid (target = current keyframe) -> NdtRegistrar.align
//            (NDT-08 EKF2 prior as initial guess) -> NDT-11 gate verdict
//         -> NDT-12 Sigma_meas -> publish ~/ndt_odom.
//
// Keyframe policy (simple, YAGNI): the first accepted scan is the keyframe; each
// later scan is registered against it, but ~/ndt_odom is only published (and the
// keyframe advanced) once the measured motion exceeds a translation/rotation
// threshold. Standing still -> no new keyframe -> no new odom message.
//
// The algorithm classes stay ROS-free; only THIS executable links ROS. No map
// management, no loop closure. TF (INFRA-01): odom->base_link here; map->odom and
// base_link->lidar_link are static launch-file broadcasters.

#include "ndt_frontend_node.hpp"

#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <memory>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <tf2_ros/transform_broadcaster.h>
// NOTE: header paths are FLAT (graph_slam/foo.hpp) except the two already migrated
// to ARCHITECTURE §3 subdirs (preprocess/, eval/). The package layout is only
// half-migrated; finishing it is a separate cleanup story (see NDT-13 log). These
// includes match where the files actually live today.
#include "graph_slam/initial_guess.hpp"
#include "graph_slam/ndt_registrar.hpp"
#include "graph_slam/ndt_voxel_grid.hpp"
#include "graph_slam/point_types.hpp"
#include "graph_slam/preprocess/quality_checker.hpp"
#include "graph_slam/preprocessor.hpp"
#include "graph_slam/registration_gate.hpp"
#include "graph_slam/registration_result.hpp"

namespace graph_slam {
namespace {

Eigen::Matrix4f poseToMatrix(const geometry_msgs::msg::Pose& p) {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  const Eigen::Quaternionf q(
      static_cast<float>(p.orientation.w), static_cast<float>(p.orientation.x),
      static_cast<float>(p.orientation.y), static_cast<float>(p.orientation.z));
  m.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
  m(0, 3) = static_cast<float>(p.position.x);
  m(1, 3) = static_cast<float>(p.position.y);
  m(2, 3) = static_cast<float>(p.position.z);
  return m;
}

geometry_msgs::msg::Pose matrixToPose(const Eigen::Matrix4f& m) {
  geometry_msgs::msg::Pose p;
  const Eigen::Quaternionf q(Eigen::Matrix3f(m.block<3, 3>(0, 0)));
  p.position.x = m(0, 3);
  p.position.y = m(1, 3);
  p.position.z = m(2, 3);
  p.orientation.w = q.w();
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  return p;
}

/// Rotation angle (rad) encoded by the 3x3 block of an SE(3) matrix.
double rotationAngle(const Eigen::Matrix4f& m) {
  const double trace = m.block<3, 3>(0, 0).cast<double>().trace();
  return std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
}

/// Reorder Sigma_meas (GTSAM Pose3 order [rx,ry,rz,x,y,z]) into the nav_msgs
/// covariance order [x,y,z,rx,ry,rz], row-major 6x6.
std::array<double, 36> sigmaToNavCovariance(const Matrix6f& sigma) {
  constexpr std::array<int, 6> kPerm{3, 4, 5, 0, 1, 2};  // nav index -> sigma index
  std::array<double, 36> cov{};
  for (int a = 0; a < 6; ++a) {
    for (int b = 0; b < 6; ++b) {
      cov[6 * a + b] = static_cast<double>(sigma(kPerm[a], kPerm[b]));
    }
  }
  return cov;
}

class NdtFrontendNode : public rclcpp::Node {
 public:
  NdtFrontendNode() : rclcpp::Node("ndt_frontend") {
    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/x500/lidar_3d/points");
    ekf2_topic_ = declare_parameter<std::string>("ekf2_topic", "/odometry/ekf2");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    min_translation_m_ = declare_parameter<double>("min_translation_m", 0.3);
    min_rotation_deg_ = declare_parameter<double>("min_rotation_deg", 5.0);
    publish_debug_clouds_ = declare_parameter<bool>("publish_debug_clouds", true);
    // Pipeline knobs (INFRA-02); defaults equal the struct defaults, so a YAML edit
    // changes behaviour without a rebuild.
    pre_cfg_.voxel_leaf = static_cast<float>(declare_parameter<double>("voxel_leaf", 0.2));
    grid_cfg_.resolution = declare_parameter<double>("ndt_resolution", grid_cfg_.resolution);
    ndt_cfg_.max_iterations = declare_parameter<int>("ndt_max_iterations", ndt_cfg_.max_iterations);
    ndt_cfg_.step_epsilon = declare_parameter<double>("ndt_step_epsilon", ndt_cfg_.step_epsilon);
    ndt_cfg_.outlier_ratio = declare_parameter<double>("ndt_outlier_ratio", ndt_cfg_.outlier_ratio);
    ndt_cfg_.regularization =
        declare_parameter<double>("ndt_regularization", ndt_cfg_.regularization);
    ndt_cfg_.hessian_lambda =
        declare_parameter<double>("ndt_hessian_lambda", ndt_cfg_.hessian_lambda);
    ndt_cfg_.cov_scale_factor =
        declare_parameter<double>("ndt_cov_scale_factor", ndt_cfg_.cov_scale_factor);
    ndt_cfg_.sigma_rot_fallback =
        declare_parameter<double>("ndt_sigma_rot_fallback", ndt_cfg_.sigma_rot_fallback);
    ndt_cfg_.sigma_t_fallback =
        declare_parameter<double>("ndt_sigma_t_fallback", ndt_cfg_.sigma_t_fallback);
    ndt_cfg_.sigma_max_condition =
        declare_parameter<double>("ndt_sigma_max_condition", ndt_cfg_.sigma_max_condition);
    gate_cfg_.max_fitness_score =
        declare_parameter<double>("gate_max_fitness_score", gate_cfg_.max_fitness_score);
    gate_cfg_.min_hessian_eigenvalue =
        declare_parameter<double>("gate_min_hessian_eigenvalue", gate_cfg_.min_hessian_eigenvalue);
    gate_cfg_.max_condition_number =
        declare_parameter<double>("gate_max_condition_number", gate_cfg_.max_condition_number);
    quality_cfg_.min_points = declare_parameter<int>("quality_min_points", quality_cfg_.min_points);
    quality_cfg_.min_spread_eigenvalue = declare_parameter<double>(
        "quality_min_spread_eigenvalue", quality_cfg_.min_spread_eigenvalue);

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("~/ndt_odom", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    if (publish_debug_clouds_) {
      scan_raw_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/scan_raw", 1);
      scan_target_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/scan_target", 1);
    }
    scan_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { onScan(msg); });
    ekf2_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        ekf2_topic_, rclcpp::SensorDataQoS(),
        [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { onEkf2(msg); });

    RCLCPP_INFO(get_logger(),
                "ndt_frontend up. LiDAR '%s', EKF2 prior '%s'. keyframe at >%.2f m / >%.1f deg; "
                "debug_clouds=%d. Publishing %s/ndt_odom.",
                lidar_topic_.c_str(), ekf2_topic_.c_str(), min_translation_m_, min_rotation_deg_,
                static_cast<int>(publish_debug_clouds_), get_fully_qualified_name());
  }

 private:
  void onEkf2(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
    latest_odom_ = poseToMatrix(msg->pose.pose);
    if (!have_odom_) {
      have_odom_ = true;
      RCLCPP_INFO(get_logger(), "EKF2 prior connected on '%s' (first odometry received).",
                  ekf2_topic_.c_str());
    }
  }

  void onScan(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
    auto raw = std::make_shared<Cloud>();
    pcl::fromROSMsg(*msg, *raw);
    const CloudPtr scan = Preprocessor{pre_cfg_}.process(raw);

    // NDT-04 input gate: drop unusable scans BEFORE NDT runs.
    const QualityResult quality = QualityChecker{quality_cfg_}.check(scan);
    if (!quality.accepted) {
      RCLCPP_WARN(get_logger(), "scan rejected by quality gate (%s); skipping.",
                  quality.reason.c_str());
      return;
    }
    publishCloud(scan_raw_pub_, scan, msg->header);

    // First accepted scan bootstraps the keyframe at the odom origin.
    if (!have_keyframe_) {
      setKeyframe(scan, Eigen::Matrix4f::Identity(), msg->header);
      publishOdom(Eigen::Matrix4f::Identity(), Matrix6f::Zero(), msg->header.stamp);
      return;
    }

    // NDT-08: seed with the EKF2 relative-motion prior when odometry is available.
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    const bool used_prior = have_odom_ && have_keyframe_odom_;
    if (used_prior) {
      init_guess = relativePoseGuess(odom_at_keyframe_, latest_odom_);
    }

    const RegistrationResult result =
        NdtRegistrar{ndt_cfg_}.align(*keyframe_grid_, scan, init_guess);
    const RegistrationStatus verdict = evaluateRegistration(result, gate_cfg_);
    const double moved_m = result.transform.block<3, 1>(0, 3).norm();
    const double turned_deg = rotationAngle(result.transform) * 180.0 / M_PI;
    RCLCPP_INFO(get_logger(),
                "NDT: converged=%d iters=%d fitness=%.3f -> %s | moved %.3f m / %.2f deg | "
                "prior=%s (guess t=%.3f m)",
                static_cast<int>(result.converged), result.iterations, result.fitness_score,
                toString(verdict), moved_m, turned_deg, used_prior ? "EKF2" : "identity",
                init_guess.block<3, 1>(0, 3).norm());

    // Keyframe policy: only emit + advance the keyframe on sufficient motion.
    if (moved_m < min_translation_m_ && turned_deg < min_rotation_deg_) {
      return;
    }
    const Eigen::Matrix4f world_from_current = world_from_keyframe_ * result.transform;
    publishOdom(world_from_current, result.sigma_meas, msg->header.stamp);
    setKeyframe(scan, world_from_current, msg->header);
  }

  // Promote `scan` to the active keyframe: rebuild the NDT target grid, store its
  // world pose, and snapshot the EKF2 pose at this instant for the next prior.
  void setKeyframe(const CloudPtr& scan, const Eigen::Matrix4f& world_pose,
                   const std_msgs::msg::Header& header) {
    keyframe_grid_ = std::make_unique<NdtVoxelGrid>(grid_cfg_);
    keyframe_grid_->build(scan);
    world_from_keyframe_ = world_pose;
    have_keyframe_ = true;
    if (have_odom_) {
      odom_at_keyframe_ = latest_odom_;
      have_keyframe_odom_ = true;
    }
    publishCloud(scan_target_pub_, scan, header);
  }

  void publishOdom(const Eigen::Matrix4f& world_from_base, const Matrix6f& sigma,
                   const builtin_interfaces::msg::Time& stamp) {
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose = matrixToPose(world_from_base);
    odom.pose.covariance = sigmaToNavCovariance(sigma);
    odom_pub_->publish(odom);

    // INFRA-01: odom->base_link TF, stamped from the odom message (not node clock).
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = odom_frame_;
    tf.child_frame_id = base_frame_;
    tf.transform.translation.x = odom.pose.pose.position.x;
    tf.transform.translation.y = odom.pose.pose.position.y;
    tf.transform.translation.z = odom.pose.pose.position.z;
    tf.transform.rotation = odom.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf);
  }

  void publishCloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
                    const CloudConstPtr& cloud, const std_msgs::msg::Header& header) {
    if (!pub) {  // debug clouds disabled
      return;
    }
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header = header;
    pub->publish(msg);
  }

  // params
  std::string lidar_topic_;
  std::string ekf2_topic_;
  std::string odom_frame_;
  std::string base_frame_;
  double min_translation_m_ = 0.3;
  double min_rotation_deg_ = 5.0;
  bool publish_debug_clouds_ = true;
  PreprocessConfig pre_cfg_;
  NdtGridConfig grid_cfg_;
  NdtConfig ndt_cfg_;
  RegistrationGateConfig gate_cfg_;
  QualityConfig quality_cfg_;

  // state
  std::unique_ptr<NdtVoxelGrid> keyframe_grid_;
  Eigen::Matrix4f world_from_keyframe_ = Eigen::Matrix4f::Identity();
  bool have_keyframe_ = false;
  Eigen::Matrix4f latest_odom_ = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f odom_at_keyframe_ = Eigen::Matrix4f::Identity();
  bool have_odom_ = false;
  bool have_keyframe_odom_ = false;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ekf2_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_raw_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_target_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace

std::shared_ptr<rclcpp::Node> createNdtFrontendNode() {
  return std::make_shared<NdtFrontendNode>();
}

}  // namespace graph_slam
