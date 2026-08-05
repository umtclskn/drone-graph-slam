// NDT-13: the whole NDT front-end pipeline as ONE rclcpp node.
//
// Per scan (ARCHITECTURE §4 order):
//   LiDAR -> Preprocessor -> QualityChecker (NDT-04, reject -> log + skip)
//         -> NdtVoxelGrid (target = L5-07/08 submap, see below) -> NdtRegistrar.align
//            (L5-18 EKF2-sourced initial guess) -> NDT-11 gate ENFORCED:
//            non-Reliable -> skip the scan (recovery policy is L5-13)
//         -> NDT-12 Sigma_meas -> publish ~/ndt_odom.
//
// L5-18 initial guess (reverses L5-06): the guess is the relative motion between
// two PX4 EKF2 poses, T_guess = T_ekf2(t_keyframe)^-1 * T_ekf2(t_scan), looked up
// by header stamp from a time-ordered buffer of /odometry/ekf2. EKF2 is fed by
// IMU inside PX4 firmware and consumes NO graph_slam output, so it cannot form a
// `guess -> NDT -> backend_state -> guess` cycle; L5-06's back-end-anchored IMU
// predictor could, and measurably regressed the front-end (L5-06d: ATE
// 0.710 -> 1.036 m; XY diagnosis: XY 0.192 -> 0.647 m). EKF2's own unbounded
// indoor drift does not matter here because only a DIFFERENCE over one
// keyframe->scan interval is used, so slow drift cancels.
// This node holds no IMU integrator. EKF2 is a guess and only a guess: not a
// graph factor, not a gate fallback, not a bootstrap prior. L5-19 re-subscribes
// to /slam/optimized_state solely to pose the submap target — never the guess.
//
// graph_slam stays PX4-agnostic: /odometry/ekf2 is standard nav_msgs/Odometry,
// published by px4_offboard/ekf2_odometry_adapter.py (PX4-01), which is the one
// sanctioned px4_msgs boundary (ARCHITECTURE §3/§11). Its header stamp is already
// in the SLAM clock domain (sim time under use_sim_time), so this node uses that
// stamp verbatim and never re-stamps with its own clock.
//
// L5-07/08/19 scan-to-submap target (LIO-SAM's extractSurroundingKeyFrames,
// adapted): the last `submap_window_size` keyframe clouds are held in
// `submap_keyframes_`. New entries start with this node's NDT pose
// (`world_from_keyframe_`, unrefined); `/slam/optimized_state` later replaces
// each matching entry's pose by exact stamp/id (L5-19). rebuildTargetGrid()
// fuses only same-source poses (never refined+unrefined together). The NDT
// initial guess stays on EKF2 and never reads optimized_state (L5-18).
// `submap_enabled=false` degrades to the old single-scan target. Sliding-window
// eviction is fixed-N (YAML `submap_window_size`); N tuning is L5-09.
//
// L5-10 loop-closure rebuild: on `/slam/optimized_state_batch` (published once
// per accepted closure, covering the whole re-optimized keyframe range),
// onOptimizedStateBatch() applies every entry to the window in ONE atomic pass
// (applyOptimizedPoseBatch) — never the collapse-inducing N-independent-
// publishes path L5-12 measured and reverted — and rebuilds the target grid
// once if the max pose shift crosses (`submap_rebuild_eps_m`,
// `submap_rebuild_eps_rad`). L5-19e's per-entry collapse-on-jump guard is
// unchanged and still applies to the ordinary single-keyframe
// `/slam/optimized_state` path above.
//
// Keyframe policy (simple, YAGNI): the first accepted scan is the keyframe; each
// later scan is registered against the target, but ~/ndt_odom is only published
// (and the keyframe advanced) once the measured motion exceeds a translation/
// rotation threshold. Standing still -> no new keyframe -> no new odom message.
//
// The algorithm classes stay ROS-free; only THIS executable links ROS. No map
// management, no loop closure. TF (INFRA-01): odom->base_link here; map->odom and
// base_link->lidar_link are static launch-file broadcasters.

#include "ndt_frontend_node.hpp"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <tf2_ros/transform_broadcaster.h>
#include <vector>
// NOTE: header paths are FLAT (graph_slam/foo.hpp) except the two already migrated
// to ARCHITECTURE §3 subdirs (preprocess/, eval/). The package layout is only
// half-migrated; finishing it is a separate cleanup story (see NDT-13 log). These
// includes match where the files actually live today.
#include <graph_slam_msgs/msg/optimized_state.hpp>
#include <graph_slam_msgs/msg/optimized_state_batch.hpp>

#include "graph_slam/initial_guess.hpp"
#include "graph_slam/ndt_registrar.hpp"
#include "graph_slam/ndt_voxel_grid.hpp"
#include "graph_slam/point_types.hpp"
#include "graph_slam/preprocess/quality_checker.hpp"
#include "graph_slam/preprocessor.hpp"
#include "graph_slam/registration_gate.hpp"
#include "graph_slam/registration_result.hpp"
#include "graph_slam/submap_pose_policy.hpp"

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

/// L5-18: EKF2 arrives as a ROS pose; the NDT pipeline works in Eigen::Matrix4f.
/// One conversion at the boundary.
Eigen::Matrix4f poseToMatrix(const geometry_msgs::msg::Pose& p) {
  const Eigen::Quaternionf q(static_cast<float>(p.orientation.w), static_cast<float>(p.orientation.x),
                             static_cast<float>(p.orientation.y), static_cast<float>(p.orientation.z));
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
  m(0, 3) = static_cast<float>(p.position.x);
  m(1, 3) = static_cast<float>(p.position.y);
  m(2, 3) = static_cast<float>(p.position.z);
  return m;
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

/// L5-18: one buffered PX4 EKF2 pose, keyed by the message's ORIGINAL header
/// stamp. The stamp is used verbatim — never re-stamped with the node clock —
/// because px4_offboard's ekf2_odometry_adapter already publishes in the SLAM
/// clock domain (sim time under use_sim_time), the domain the scan headers use.
struct Ekf2Sample {
  double stamp_s = 0.0;
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
};

class NdtFrontendNode : public rclcpp::Node {
 public:
  explicit NdtFrontendNode(rclcpp::NodeOptions options)
      : rclcpp::Node("ndt_frontend", options) {
    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/x500/lidar_3d/points");
    // L5-18: the ONLY motion input — PX4 EKF2 odometry, republished as standard
    // nav_msgs/Odometry (ENU) by px4_offboard/ekf2_odometry_adapter, so graph_slam
    // sees no px4_msgs. Independent of the graph by construction: EKF2 runs inside
    // PX4 firmware and consumes nothing this package publishes.
    ekf2_topic_ = declare_parameter<std::string>("ekf2_topic", "/odometry/ekf2");
    ekf2_buffer_seconds_ =
        declare_parameter<double>("ekf2_buffer_seconds", ekf2_buffer_seconds_);
    ekf2_max_time_diff_s_ =
        declare_parameter<double>("ekf2_max_time_diff_s", ekf2_max_time_diff_s_);
    // L5-19: submap pose feed only — never used for the NDT initial guess.
    optimized_state_topic_ =
        declare_parameter<std::string>("optimized_state_topic", "/slam/optimized_state");
    // L5-10: coordinated batch feed, published once per accepted loop closure —
    // see onOptimizedStateBatch() for why this is a SEPARATE topic from the
    // per-keyframe one above rather than N single-entry messages.
    optimized_state_batch_topic_ = declare_parameter<std::string>(
        "optimized_state_batch_topic", "/slam/optimized_state_batch");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    min_translation_m_ = declare_parameter<double>("min_translation_m", 0.3);
    min_rotation_deg_ = declare_parameter<double>("min_rotation_deg", 5.0);
    publish_debug_clouds_ = declare_parameter<bool>("publish_debug_clouds", true);
    // L5-07/08: scan-to-submap target. submap_enabled=false is the ablation flag
    // (reproduces the pre-L5-07 single-scan target exactly). window_size = N
    // keyframe clouds held; voxel_leaf downsamples the merged submap cloud.
    submap_enabled_ = declare_parameter<bool>("submap_enabled", submap_enabled_);
    submap_window_size_ = declare_parameter<int>("submap_window_size", submap_window_size_);
    submap_voxel_leaf_ = declare_parameter<double>("submap_voxel_leaf", submap_voxel_leaf_);
    submap_collapse_eps_m_ =
        declare_parameter<double>("submap_collapse_eps_m", submap_collapse_eps_m_);
    submap_collapse_eps_rad_ =
        declare_parameter<double>("submap_collapse_eps_rad", submap_collapse_eps_rad_);
    // L5-10: loop-closure coordinated-rebuild thresholds. A batch's max pose
    // shift among already-refined window entries must exceed EITHER of these
    // before rebuildTargetGrid() is called again (thrashing guard — most
    // closures nudge already-accurate poses by a few mm/mrad).
    submap_rebuild_eps_m_ =
        declare_parameter<double>("submap_rebuild_eps_m", submap_rebuild_eps_m_);
    submap_rebuild_eps_rad_ =
        declare_parameter<double>("submap_rebuild_eps_rad", submap_rebuild_eps_rad_);
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
    gate_cfg_.min_scored_fraction =
        declare_parameter<double>("gate_min_scored_fraction", gate_cfg_.min_scored_fraction);
    gate_cfg_.max_guess_delta_t =
        declare_parameter<double>("gate_max_guess_delta_t", gate_cfg_.max_guess_delta_t);
    gate_cfg_.max_guess_delta_rot =
        declare_parameter<double>("gate_max_guess_delta_rot", gate_cfg_.max_guess_delta_rot);
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
        ekf2_topic_, rclcpp::QoS(50),
        [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { onEkf2(msg); });
    // L5-19: submap pose refinement only. The NDT guess path never reads this.
    optimized_state_sub_ = create_subscription<graph_slam_msgs::msg::OptimizedState>(
        optimized_state_topic_, rclcpp::QoS(10),
        [this](graph_slam_msgs::msg::OptimizedState::ConstSharedPtr msg) {
          onOptimizedState(msg);
        });
    // L5-10: coordinated loop-closure submap rebuild. Separate topic/handler
    // from the per-keyframe one above — see onOptimizedStateBatch().
    optimized_state_batch_sub_ =
        create_subscription<graph_slam_msgs::msg::OptimizedStateBatch>(
            optimized_state_batch_topic_, rclcpp::QoS(10),
            [this](graph_slam_msgs::msg::OptimizedStateBatch::ConstSharedPtr msg) {
              onOptimizedStateBatch(msg);
            });

    RCLCPP_INFO(get_logger(),
                "ndt_frontend up. LiDAR '%s', NDT guess from EKF2 '%s' (buffer %.1f s, "
                "match tol %.0f ms; guess independent of graph). submap poses from '%s'. "
                "keyframe at >%.2f m / >%.1f deg; debug_clouds=%d. target=%s (N=%d, "
                "leaf=%.2f m). Publishing %s/ndt_odom.",
                lidar_topic_.c_str(), ekf2_topic_.c_str(), ekf2_buffer_seconds_,
                ekf2_max_time_diff_s_ * 1e3, optimized_state_topic_.c_str(),
                min_translation_m_, min_rotation_deg_,
                static_cast<int>(publish_debug_clouds_),
                submap_enabled_ ? "scan-to-submap" : "scan-to-scan", submap_window_size_,
                submap_voxel_leaf_, get_fully_qualified_name());
  }

 private:
  // L5-19: refine exactly one submap window entry by stamp/id. Guess path untouched.
  void onOptimizedState(const graph_slam_msgs::msg::OptimizedState::ConstSharedPtr& msg) {
    if (!submap_enabled_ || submap_keyframes_.empty()) {
      return;
    }
    OptimizedPoseUpdate update;
    update.stamp_s = rclcpp::Time(msg->header.stamp).seconds();
    update.keyframe_id = msg->keyframe_id;
    update.pose = poseToMatrix(msg->pose);
    const ApplyOptimizedResult applied = applyOptimizedPose(
        submap_keyframes_, update, submap_collapse_eps_m_, submap_collapse_eps_rad_);
    if (applied.status == ApplyOptimizedStatus::NoMatch) {
      return;
    }
    if (applied.status == ApplyOptimizedStatus::Collapsed) {
      RCLCPP_INFO(get_logger(),
                  "submap collapsed to 1 keyframe after refined pose jump "
                  "(stamp %.3f, id %d).",
                  update.stamp_s, update.keyframe_id);
    }
    rebuildTargetGrid();
  }

  // L5-10: a coordinated batch of pose updates from ONE accepted loop closure
  // — every keyframe in the back-end's affected range, applied as a single
  // atomic pass (applyOptimizedPoseBatch), never as N independent
  // applyOptimizedPose() calls. The window's clouds are never discarded here:
  // poses move, the grid rebuilds once (only if the max shift crosses the
  // configured ε — thrashing guard), same window size before and after.
  void onOptimizedStateBatch(
      const graph_slam_msgs::msg::OptimizedStateBatch::ConstSharedPtr& msg) {
    if (!submap_enabled_ || submap_keyframes_.empty()) {
      return;
    }
    std::vector<OptimizedPoseUpdate> updates;
    updates.reserve(msg->states.size());
    for (const auto& state : msg->states) {
      OptimizedPoseUpdate update;
      update.stamp_s = rclcpp::Time(state.header.stamp).seconds();
      update.keyframe_id = state.keyframe_id;
      update.pose = poseToMatrix(state.pose);
      updates.push_back(update);
    }

    const std::size_t window_size_before = submap_keyframes_.size();
    const BatchApplyResult applied = applyOptimizedPoseBatch(
        submap_keyframes_, updates, submap_rebuild_eps_m_, submap_rebuild_eps_rad_);
    if (applied.matched_count == 0) {
      RCLCPP_DEBUG(get_logger(),
                  "loop-closure batch: 0/%zu updates matched the active submap "
                  "window (window outside the affected range); no work done.",
                  updates.size());
      return;
    }

    if (applied.should_rebuild) {
      rebuildTargetGrid();
    }
    RCLCPP_INFO(get_logger(),
                "L5-10 loop-closure submap update: %zu/%zu updates matched, max "
                "shift %.3f m / %.3f rad, window %zu -> %zu entries, rebuild=%d.",
                applied.matched_count, updates.size(), applied.max_shift_m,
                applied.max_shift_rad, window_size_before, submap_keyframes_.size(),
                static_cast<int>(applied.should_rebuild));
  }

  // L5-18a: buffer one PX4 EKF2 pose, keyed by its ORIGINAL header stamp (the
  // adapter already publishes in the SLAM clock domain, so no re-stamping here —
  // and clock reconciliation never belongs inside graph_slam). Samples arrive in
  // stamp order, so the deque stays sorted and a plain prune from the front
  // bounds it to the horizon.
  void onEkf2(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
    Ekf2Sample sample;
    sample.stamp_s = rclcpp::Time(msg->header.stamp).seconds();
    sample.pose = poseToMatrix(msg->pose.pose);
    if (!ekf2_buffer_.empty() && sample.stamp_s < ekf2_buffer_.back().stamp_s) {
      return;  // out-of-order sample: would break the sorted-buffer invariant
    }
    ekf2_buffer_.push_back(sample);
    while (!ekf2_buffer_.empty() &&
           (sample.stamp_s - ekf2_buffer_.front().stamp_s) > ekf2_buffer_seconds_) {
      ekf2_buffer_.pop_front();
    }
    if (!logged_first_ekf2_) {
      logged_first_ekf2_ = true;
      RCLCPP_INFO(get_logger(), "first EKF2 sample on '%s' (stamp %.3f s).",
                  ekf2_topic_.c_str(), sample.stamp_s);
    }
  }

  // L5-18b: the buffered EKF2 pose nearest `stamp_s`. Nearest-stamp snap, like
  // the rest of this project's time matching — no interpolation (deliberate; see
  // the L5-17 audit). Absent when the buffer is empty or the closest sample is
  // farther away than the match tolerance, in which case the caller falls back to
  // an identity guess rather than seeding NDT with a stale pose.
  std::optional<Eigen::Matrix4f> ekf2PoseAt(double stamp_s) const {
    if (ekf2_buffer_.empty()) {
      return std::nullopt;
    }
    const auto after = std::lower_bound(
        ekf2_buffer_.begin(), ekf2_buffer_.end(), stamp_s,
        [](const Ekf2Sample& s, double t) { return s.stamp_s < t; });

    const Ekf2Sample* best = nullptr;
    if (after == ekf2_buffer_.end()) {
      best = &ekf2_buffer_.back();
    } else if (after == ekf2_buffer_.begin()) {
      best = &(*after);
    } else {
      const Ekf2Sample& before = *std::prev(after);
      best = (stamp_s - before.stamp_s) <= (after->stamp_s - stamp_s) ? &before : &(*after);
    }
    if (std::abs(best->stamp_s - stamp_s) > ekf2_max_time_diff_s_) {
      return std::nullopt;
    }
    return best->pose;
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

    // Scan time = the LiDAR message's own header stamp (the true measurement
    // instant), which is what both ends of the EKF2 guess are looked up against.
    const double scan_stamp_s = rclcpp::Time(msg->header.stamp).seconds();

    // First accepted scan bootstraps the keyframe at the odom origin.
    if (!have_keyframe_) {
      setKeyframe(scan, Eigen::Matrix4f::Identity(), scan_stamp_s, msg->header);
      publishOdom(Eigen::Matrix4f::Identity(), Matrix6f::Zero(), msg->header.stamp);
      return;
    }

    // L5-18b: seed NDT with the EKF2 motion over the keyframe->scan interval,
    // T_guess = T_ekf2(t_keyframe)^-1 * T_ekf2(t_scan). Both ends are looked up
    // from the SAME buffer at scan time (rather than latching the keyframe end
    // when the keyframe was made), so the keyframe end can use an EKF2 sample
    // that only arrived afterwards and is therefore a closer stamp match.
    // Differencing two poses cancels EKF2's slow absolute drift; identity when
    // either end has no sample within the match tolerance.
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    const std::optional<Eigen::Matrix4f> ekf2_at_keyframe = ekf2PoseAt(keyframe_stamp_s_);
    const std::optional<Eigen::Matrix4f> ekf2_at_scan = ekf2PoseAt(scan_stamp_s);
    const bool used_prior = ekf2_at_keyframe && ekf2_at_scan;
    if (used_prior) {
      init_guess = relativePoseGuess(*ekf2_at_keyframe, *ekf2_at_scan);
    }

    const RegistrationResult aligned =
        NdtRegistrar{ndt_cfg_}.align(*target_grid_, scan, init_guess);
    // L5-20: tell the gate whether init_guess was a real EKF2 prior. Identity
    // fallback is "no information", so PriorInconsistent must not fire on it.
    RegistrationResult result = aligned;
    result.have_prior_guess = used_prior;
    const RegistrationStatus verdict = evaluateRegistration(result, gate_cfg_);

    // NDT-11 gate ENFORCEMENT: a rejected registration must not enter the
    // odometry chain (a confident wrong factor wrecks the graph), so skip the
    // scan and leave the keyframe where it is; the next scan is retried
    // immediately.
    //
    // L5-06c deleted the substitute-the-prior fallback that used to run here,
    // because with the back-end-anchored IMU guess it closed a positive feedback
    // loop (a 0.36 m hover dead-reckoning guess published as measured motion
    // diverged slam_loop_03 to ATE 1203 m within 1 s). L5-18 makes the prior an
    // INDEPENDENT source again, which would make such a fallback safe — but
    // whether "substitute the prior" beats "skip" is L5-13's recovery-policy
    // call, not this story's, so the skip behaviour is left exactly as it is.
    if (verdict != RegistrationStatus::Reliable) {
      const double support = result.total_points > 0
                                 ? static_cast<double>(result.scored_points) /
                                       static_cast<double>(result.total_points)
                                 : 0.0;
      const Eigen::Matrix4f guess_delta =
          relativePoseGuess(result.initial_guess, result.transform);
      RCLCPP_WARN(get_logger(),
                  "NDT rejected (%s); scan skipped (guess t=%.3f m, support=%.2f, "
                  "delta_t=%.3f m / %.2f deg).",
                  toString(verdict), init_guess.block<3, 1>(0, 3).norm(), support,
                  guess_delta.block<3, 1>(0, 3).norm(),
                  rotationAngle(guess_delta) * 180.0 / M_PI);
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
                toString(verdict), moved_m, turned_deg, used_prior ? "EKF2" : "identity",
                init_guess.block<3, 1>(0, 3).norm(),
                rotationAngle(init_guess) * 180.0 / M_PI);

    // Keyframe policy: only emit + advance the keyframe on sufficient motion.
    if (moved_m < min_translation_m_ && turned_deg < min_rotation_deg_) {
      return;
    }
    const Eigen::Matrix4f world_from_current = world_from_keyframe_ * delta;
    publishOdom(world_from_current, sigma, msg->header.stamp);
    setKeyframe(scan, world_from_current, scan_stamp_s, msg->header);
  }

  // Promote `scan` to the active keyframe: push it as an unrefined (front-end NDT)
  // submap entry, evict beyond the window size, rebuild the fused NDT target grid,
  // store the world pose, and record this scan's stamp as the keyframe end of the
  // next EKF2 guess. Backend OptimizedState later refines the matching entry by
  // stamp (L5-19).
  void setKeyframe(const CloudPtr& scan, const Eigen::Matrix4f& world_pose,
                   double stamp_s, const std_msgs::msg::Header& header) {
    world_from_keyframe_ = world_pose;
    have_keyframe_ = true;
    keyframe_stamp_s_ = stamp_s;

    SubmapKeyframe entry;
    entry.cloud = scan;
    entry.pose = world_pose;
    entry.stamp_s = stamp_s;
    entry.keyframe_id = -1;
    entry.refined = false;
    submap_keyframes_.push_back(std::move(entry));
    while (submap_keyframes_.size() > static_cast<std::size_t>(submap_window_size_)) {
      submap_keyframes_.pop_front();
    }
    rebuildTargetGrid();

    publishCloud(scan_target_pub_, scan, header);
  }

  // L5-07c/L5-08a/L5-19d: fuse the sliding window into ONE NDT target grid.
  // Only same-source poses are fused (planSubmapFusion); mixed refined/unrefined
  // windows degrade to the newest cloud alone. Transforms use relativePoseGuess
  // on the selected poses. submap_enabled_=false or a size-1 window both degrade
  // to the old single-scan target.
  void rebuildTargetGrid() {
    target_grid_ = std::make_unique<NdtVoxelGrid>(grid_cfg_);
    if (submap_keyframes_.empty()) {
      return;
    }
    if (!submap_enabled_) {
      target_grid_->build(submap_keyframes_.back().cloud);
      return;
    }

    const SubmapFusionPlan plan = planSubmapFusion(submap_keyframes_);
    if (plan.rejected_mixed_source) {
      RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
                            "submap refused mixed refined/unrefined fusion; "
                            "using single-newest until backend catches up.");
    }
    if (plan.mode == SubmapFusionMode::SingleNewest || plan.indices.size() <= 1) {
      target_grid_->build(submap_keyframes_.back().cloud);
      return;
    }

    const Eigen::Matrix4f pose_ref = submap_keyframes_[plan.indices.back()].pose;
    auto merged = std::make_shared<Cloud>();
    for (const std::size_t idx : plan.indices) {
      const auto& kf = submap_keyframes_[idx];
      Cloud transformed;
      pcl::transformPointCloud(*kf.cloud, transformed, relativePoseGuess(pose_ref, kf.pose));
      *merged += transformed;
    }

    pcl::VoxelGrid<PointT> voxel;
    voxel.setInputCloud(merged);
    const auto leaf = static_cast<float>(submap_voxel_leaf_);
    voxel.setLeafSize(leaf, leaf, leaf);
    auto downsampled = std::make_shared<Cloud>();
    voxel.filter(*downsampled);
    target_grid_->build(downsampled);
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
  double ekf2_buffer_seconds_ = 5.0;      // s, EKF2 pose buffer horizon
  double ekf2_max_time_diff_s_ = 0.1;     // s, max |sample - target| for a match
  std::string optimized_state_topic_;     // L5-19: submap pose feed only
  std::string optimized_state_batch_topic_;  // L5-10: coordinated loop-closure feed
  std::string odom_frame_;
  std::string base_frame_;
  double min_translation_m_ = 0.3;
  double min_rotation_deg_ = 5.0;
  bool publish_debug_clouds_ = true;
  bool submap_enabled_ = true;      // L5-07/08 ablation flag; false = scan-to-scan
  int submap_window_size_ = 8;      // N keyframe clouds held in the submap window
  double submap_voxel_leaf_ = 0.3;  // m, merged submap cloud downsample leaf
  double submap_collapse_eps_m_ = 0.2;    // L5-19e: collapse window on refined jump
  double submap_collapse_eps_rad_ = 0.1;  // L5-19e: ~5.7 deg
  double submap_rebuild_eps_m_ = 0.2;     // L5-10: loop-closure batch rebuild threshold
  double submap_rebuild_eps_rad_ = 0.1;   // L5-10: ~5.7 deg
  PreprocessConfig pre_cfg_;
  NdtGridConfig grid_cfg_;
  NdtConfig ndt_cfg_;
  RegistrationGateConfig gate_cfg_;
  QualityConfig quality_cfg_;

  // state
  std::unique_ptr<NdtVoxelGrid> target_grid_;
  Eigen::Matrix4f world_from_keyframe_ = Eigen::Matrix4f::Identity();
  bool have_keyframe_ = false;
  // Header stamp of the scan that became the current keyframe — the t_keyframe
  // end of the L5-18 EKF2 guess.
  double keyframe_stamp_s_ = 0.0;

  // L5-07/19 submap sliding window: last submap_window_size_ keyframe clouds.
  // Poses start unrefined (NDT chain) and are replaced by backend-optimized
  // poses matched by stamp/id (rebuildTargetGrid via planSubmapFusion).
  std::deque<SubmapKeyframe> submap_keyframes_;

  // L5-18 EKF2 guess source: time-ordered PX4 EKF2 poses keyed by header stamp.
  // Independent of the graph; optimized_state never seeds the guess.
  std::deque<Ekf2Sample> ekf2_buffer_;
  bool logged_first_ekf2_ = false;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ekf2_sub_;
  rclcpp::Subscription<graph_slam_msgs::msg::OptimizedState>::SharedPtr optimized_state_sub_;
  rclcpp::Subscription<graph_slam_msgs::msg::OptimizedStateBatch>::SharedPtr
      optimized_state_batch_sub_;
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
