// NDT-13: the whole NDT front-end pipeline as ONE rclcpp node.
//
// Per scan (ARCHITECTURE §4 order):
//   LiDAR -> Preprocessor -> QualityChecker (NDT-04, reject -> log + skip)
//         -> NdtVoxelGrid (target = current keyframe) -> NdtRegistrar.align
//            (L5-06 IMU-predicted initial guess) -> NDT-11 gate ENFORCED:
//            non-Reliable -> predicted-delta fallback (or skip when no prediction)
//         -> NDT-12 Sigma_meas -> publish ~/ndt_odom.
//
// L5-06 initial guess: the EKF2 delta is gone. This node runs a SECOND instance
// of the ROS-free graph_slam::imu::ImuPreintegrator (the back-end owns the first
// one, for the ImuFactor) purely as a dead-reckoning predictor: it integrates
// /imu/data continuously and is re-seeded from the back-end's optimized
// (pose, velocity, bias) on every /slam/optimized_state message. The NDT guess
// is the relative pose between the prediction latched at the current keyframe
// and the prediction at this scan — LIO-SAM's imuIntegratorImu_ role.
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
#include <optional>
#include <utility>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <graph_slam_msgs/msg/optimized_state.hpp>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <tf2_ros/transform_broadcaster.h>
// NOTE: header paths are FLAT (graph_slam/foo.hpp) except the two already migrated
// to ARCHITECTURE §3 subdirs (preprocess/, eval/). The package layout is only
// half-migrated; finishing it is a separate cleanup story (see NDT-13 log). These
// includes match where the files actually live today.
#include "graph_slam/imu/imu_preintegrator.hpp"
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

/// L5-06: the IMU predictor works in gtsam::Pose3 (double, GTSAM NavState); the
/// NDT pipeline works in Eigen::Matrix4f. One conversion at the boundary.
Eigen::Matrix4f gtsamPoseToMatrix(const gtsam::Pose3& pose) {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = pose.rotation().matrix().cast<float>();
  m.block<3, 1>(0, 3) = pose.translation().cast<float>();
  return m;
}

Eigen::Vector3d toEigen(const geometry_msgs::msg::Vector3& v) {
  return {v.x, v.y, v.z};
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
  explicit NdtFrontendNode(rclcpp::NodeOptions options)
      : rclcpp::Node("ndt_frontend", options) {
    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/x500/lidar_3d/points");
    // L5-06: the guess inputs — raw IMU to dead-reckon with, and the back-end's
    // optimized state to re-seed from. Same /imu/data the back-end consumes
    // (sensor_msgs/Imu from px4_offboard/imu_bridge.py, FLU base_link) — a
    // second subscriber to a standard message keeps graph_slam PX4-agnostic.
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/imu/data");
    optimized_state_topic_ =
        declare_parameter<std::string>("optimized_state_topic", "/slam/optimized_state");
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
    // L5-06: the predictor reads the SAME imu_* keys as the back-end's
    // preintegrator (slam_params.yaml applies them to every node via `/**`), so
    // both integrators share one noise/gravity definition by construction.
    predictor_cfg_.accel_noise_sigma =
        declare_parameter<double>("imu_accel_noise_sigma", predictor_cfg_.accel_noise_sigma);
    predictor_cfg_.gyro_noise_sigma =
        declare_parameter<double>("imu_gyro_noise_sigma", predictor_cfg_.gyro_noise_sigma);
    predictor_cfg_.integration_sigma =
        declare_parameter<double>("imu_integration_sigma", predictor_cfg_.integration_sigma);
    predictor_cfg_.gravity = declare_parameter<double>("imu_gravity", predictor_cfg_.gravity);
    predictor_.emplace(predictor_cfg_);

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("~/ndt_odom", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    if (publish_debug_clouds_) {
      scan_raw_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/scan_raw", 1);
      scan_target_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/scan_target", 1);
    }
    scan_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { onScan(msg); });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) { onImu(msg); });
    optimized_state_sub_ = create_subscription<graph_slam_msgs::msg::OptimizedState>(
        optimized_state_topic_, rclcpp::QoS(10),
        [this](graph_slam_msgs::msg::OptimizedState::ConstSharedPtr msg) {
          onOptimizedState(msg);
        });

    RCLCPP_INFO(get_logger(),
                "ndt_frontend up. LiDAR '%s', IMU guess from '%s' anchored on '%s'. "
                "keyframe at >%.2f m / >%.1f deg; debug_clouds=%d. Publishing %s/ndt_odom.",
                lidar_topic_.c_str(), imu_topic_.c_str(), optimized_state_topic_.c_str(),
                min_translation_m_, min_rotation_deg_,
                static_cast<int>(publish_debug_clouds_), get_fully_qualified_name());
  }

 private:
  // L5-06: fold one IMU sample into the predictor. Like the back-end (L5-01a) the
  // sample is re-stamped with the node clock, because px4_offboard/imu_bridge.py
  // forwards PX4 wall-clock stamps that cannot be differenced against the
  // sim-time scan/optimized-state stream (bag-replay-clock-domain). Only the
  // consecutive-sample dt is used; ImuPreintegrator drops dt<=0 itself.
  void onImu(const sensor_msgs::msg::Imu::ConstSharedPtr& msg) {
    const double stamp_s = now().seconds();
    if (have_anchor_ && last_imu_stamp_ >= 0.0) {
      predictor_->integrate(
          Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y,
                          msg->linear_acceleration.z),
          Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y,
                          msg->angular_velocity.z),
          stamp_s - last_imu_stamp_);
    }
    last_imu_stamp_ = stamp_s;
  }

  // L5-06: re-seed the predictor from the back-end's freshest optimized keyframe
  // state. The accumulated interval is cleared and the bias becomes the new
  // linearization point, exactly as the back-end does at each keyframe — so from
  // here the predictor dead-reckons off an optimized (pose, velocity, bias)
  // instead of compounding its own drift.
  void onOptimizedState(const graph_slam_msgs::msg::OptimizedState::ConstSharedPtr& msg) {
    const gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(
        msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y,
        msg->pose.orientation.z);
    const gtsam::Point3 position(msg->pose.position.x, msg->pose.position.y,
                                 msg->pose.position.z);
    anchor_state_ = gtsam::NavState(rotation, position, toEigen(msg->velocity));
    anchor_bias_ = gtsam::imuBias::ConstantBias(toEigen(msg->accel_bias),
                                                toEigen(msg->gyro_bias));
    predictor_->reset(anchor_bias_);
    const bool first = !have_anchor_;
    have_anchor_ = true;

    // Re-latch the keyframe end of the guess onto the fresh anchor. The back-end
    // only makes a keyframe when an ~/ndt_odom message arrives, and the
    // front-end only emits one when it sets a keyframe — so this optimized state
    // IS the back-end's take on the CURRENT keyframe, and the anchor is where
    // that keyframe sits in the new chain. Without this, the next guess would
    // difference a pose from the pre-reset chain against one from the post-reset
    // chain and inject the whole dead-reckoning error plus the graph's pose
    // correction into the seed. (What it does drop is the IMU between the
    // keyframe's scan time and this message's arrival — the front-end->back-end
    // round trip, ~27 ms; that is the L5-17 header-stamp window issue, out of
    // scope here.)
    if (have_keyframe_) {
      predicted_at_keyframe_ = predictedPose();
    }
    if (first) {
      RCLCPP_INFO(get_logger(), "IMU predictor anchored on '%s' (first optimized state).",
                  optimized_state_topic_.c_str());
    }
  }

  // The predictor's pose at the newest integrated IMU sample, in the anchor's
  // (map) frame. Empty until the first optimized state arrives. Only ever
  // consumed as a DIFFERENCE of two such poses, so the map/odom frame offset and
  // the anchor's publish latency cancel out of the NDT guess.
  std::optional<Eigen::Matrix4f> predictedPose() const {
    if (!have_anchor_) {
      return std::nullopt;
    }
    return gtsamPoseToMatrix(predictor_->predict(anchor_state_, anchor_bias_).pose());
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

    // The IMU prediction for THIS scan; also latched as the keyframe reference if
    // this scan becomes the keyframe, so both ends of the guess come from the
    // same predictor.
    const std::optional<Eigen::Matrix4f> predicted = predictedPose();

    // First accepted scan bootstraps the keyframe at the odom origin.
    if (!have_keyframe_) {
      setKeyframe(scan, Eigen::Matrix4f::Identity(), predicted, msg->header);
      publishOdom(Eigen::Matrix4f::Identity(), Matrix6f::Zero(), msg->header.stamp);
      return;
    }

    // L5-06: seed NDT with the IMU-predicted motion since the keyframe. Both
    // poses come from the same predictor chain, so this is the dead-reckoned
    // keyframe->scan transform; identity until the predictor has an anchor.
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    const bool used_prior = predicted && predicted_at_keyframe_;
    if (used_prior) {
      init_guess = relativePoseGuess(*predicted_at_keyframe_, *predicted);
    }

    const RegistrationResult result =
        NdtRegistrar{ndt_cfg_}.align(*keyframe_grid_, scan, init_guess);
    const RegistrationStatus verdict = evaluateRegistration(result, gate_cfg_);

    // NDT-11 gate ENFORCEMENT: a rejected registration must not enter the
    // odometry chain (a confident wrong factor wrecks the graph), so skip the
    // scan and leave the keyframe where it is; the next scan is retried
    // immediately.
    //
    // L5-06c deletes the substitute-the-prior fallback that used to run here.
    // It was safe only while the prior was EKF2 — an INDEPENDENT source. With
    // the IMU-predicted guess it closes a positive feedback loop: the guess is
    // published as if it were measured motion -> the back-end optimizes on it
    // -> the corrupted optimized state re-anchors the very predictor that
    // produced the guess. Measured on slam_loop_03: a drifting 0.36 m guess
    // during a hover was published as real motion and the loop diverged to
    // ATE 1203 m within 1 s. Choosing a better recovery than "skip" is L5-13.
    if (verdict != RegistrationStatus::Reliable) {
      RCLCPP_WARN(get_logger(), "NDT rejected (%s); scan skipped (guess t=%.3f m).",
                  toString(verdict), init_guess.block<3, 1>(0, 3).norm());
      return;
    }
    const Eigen::Matrix4f& delta = result.transform;
    const Matrix6f& sigma = result.sigma_meas;

    const double moved_m = delta.block<3, 1>(0, 3).norm();
    const double turned_deg = rotationAngle(delta) * 180.0 / M_PI;
    RCLCPP_INFO(get_logger(),
                "NDT: converged=%d iters=%d fitness=%.3f -> %s | moved %.3f m / %.2f deg | "
                "prior=%s (guess t=%.3f m / %.2f deg)",
                static_cast<int>(result.converged), result.iterations, result.fitness_score,
                toString(verdict), moved_m, turned_deg, used_prior ? "IMU" : "identity",
                init_guess.block<3, 1>(0, 3).norm(),
                rotationAngle(init_guess) * 180.0 / M_PI);

    // Keyframe policy: only emit + advance the keyframe on sufficient motion.
    if (moved_m < min_translation_m_ && turned_deg < min_rotation_deg_) {
      return;
    }
    const Eigen::Matrix4f world_from_current = world_from_keyframe_ * delta;
    publishOdom(world_from_current, sigma, msg->header.stamp);
    setKeyframe(scan, world_from_current, predicted, msg->header);
  }

  // Promote `scan` to the active keyframe: rebuild the NDT target grid, store its
  // world pose, and latch the IMU prediction at this instant as the reference
  // end of the next scan's guess.
  void setKeyframe(const CloudPtr& scan, const Eigen::Matrix4f& world_pose,
                   const std::optional<Eigen::Matrix4f>& predicted,
                   const std_msgs::msg::Header& header) {
    keyframe_grid_ = std::make_unique<NdtVoxelGrid>(grid_cfg_);
    keyframe_grid_->build(scan);
    world_from_keyframe_ = world_pose;
    have_keyframe_ = true;
    predicted_at_keyframe_ = predicted;
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
  std::string imu_topic_;
  std::string optimized_state_topic_;
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
  imu::PreintegrationConfig predictor_cfg_;

  // state
  std::unique_ptr<NdtVoxelGrid> keyframe_grid_;
  Eigen::Matrix4f world_from_keyframe_ = Eigen::Matrix4f::Identity();
  bool have_keyframe_ = false;

  // L5-06 IMU predictor: a second ImuPreintegrator (the back-end owns the one
  // that feeds the ImuFactor) dead-reckoning from the latest optimized state.
  // last_imu_stamp_ gives consecutive-sample dt and stays continuous across
  // anchor resets — reset() clears the accumulated interval, not the clock.
  std::optional<imu::ImuPreintegrator> predictor_;
  gtsam::NavState anchor_state_;
  gtsam::imuBias::ConstantBias anchor_bias_;
  bool have_anchor_ = false;
  double last_imu_stamp_ = -1.0;
  std::optional<Eigen::Matrix4f> predicted_at_keyframe_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<graph_slam_msgs::msg::OptimizedState>::SharedPtr
      optimized_state_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_raw_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_target_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace

std::shared_ptr<rclcpp::Node> createNdtFrontendNode(rclcpp::NodeOptions options) {
  return std::make_shared<NdtFrontendNode>(std::move(options));
}

}  // namespace graph_slam
