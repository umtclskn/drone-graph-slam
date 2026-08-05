// GRAPH-VIS-01: GTSAM graph back-end node.
//
// Subscribes to /ndt_frontend/ndt_odom (nav_msgs/Odometry published by
// NDTFrontendNode) and drives the full back-end pipeline on every message:
//   KeyframePolicy  →  OdometryAccumulator  →  GraphBootstrap / OdometryFactorBuilder
//   →  GraphOptimizer (iSAM2)  →  publish visualisation + TF correction.
//
// Topics subscribed:
//   <ndt_odom_topic>           nav_msgs/Odometry  (default: /ndt_frontend/ndt_odom)
//
// Topics published:
//   /slam/graph_path           nav_msgs/Path          (optimized keyframe trajectory)
//   /slam/keyframes            visualization_msgs/MarkerArray  (SPHERE per keyframe)
//   /slam/graph_edges          visualization_msgs/Marker       (LINE_LIST odom chain)
//   /slam/optimized_odom       nav_msgs/Odometry  (latest optimized pose)
//   /slam/optimized_state      graph_slam_msgs/OptimizedState  (L5-06/L5-19:
//                              pose + velocity + bias + keyframe_id; front-end
//                              uses it for submap poses only — guess is EKF2)
//
// TF published (dynamic, updated on each keyframe):
//   map  →  odom   (ARCHITECTURE §5 REP-105 graph correction)

#include "graph_backend_node.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/PriorFactor.h>

#include <gtsam/slam/BetweenFactor.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <graph_slam_msgs/msg/optimized_state.hpp>
#include <graph_slam_msgs/msg/optimized_state_batch.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "graph_slam/eval/graph_visualizer.hpp"
#include "graph_slam/graph/graph_bootstrap.hpp"
#include "graph_slam/imu/imu_preintegrator.hpp"
#include "graph_slam/imu/imu_state_estimator.hpp"
#include "graph_slam/imu/imu_window_buffer.hpp"
#include "graph_slam/graph/graph_optimizer.hpp"
#include "graph_slam/graph/keyframe_policy.hpp"
#include "graph_slam/graph/ndt_consistency_gate.hpp"
#include "graph_slam/graph/odometry_accumulator.hpp"
#include "graph_slam/graph/odometry_factor_builder.hpp"
#include "graph_slam/loop/loop_closure_candidate_finder.hpp"
#include "graph_slam/loop/loop_closure_submap.hpp"
#include "graph_slam/loop/loop_closure_verifier.hpp"
#include "graph_slam/point_types.hpp"

namespace graph_slam {
namespace {

// L5-02 node model: every keyframe i carries X(i) pose, V(i) velocity, B(i)
// IMU bias. X(i) is byte-identical to the old Symbol('x', i), so pose keys
// (and every consumer that walks them) are unchanged.
namespace sym = gtsam::symbol_shorthand;

// ---------------------------------------------------------------------------
// Type conversion helpers
// ---------------------------------------------------------------------------

gtsam::Pose3 navPoseToGtsam(const geometry_msgs::msg::Pose& p) {
  const gtsam::Rot3 r = gtsam::Rot3::Quaternion(
      p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
  return {r, gtsam::Point3(p.position.x, p.position.y, p.position.z)};
}

// nav order: [x, y, z, rx, ry, rz]  ↔  GTSAM tangent order: [rx, ry, rz, x, y, z].
// The permutation {3,4,5,0,1,2} is self-inverse, so the same code handles both
// directions (ARCHITECTURE §6 — Sigma_meas is stored in GTSAM order).
gtsam::Matrix66 navCovToGtsam(const std::array<double, 36>& cov) {
  constexpr std::array<int, 6> kPerm{3, 4, 5, 0, 1, 2};
  gtsam::Matrix66 sigma;
  for (int a = 0; a < 6; ++a) {
    for (int b = 0; b < 6; ++b) {
      sigma(a, b) = cov[static_cast<std::size_t>(6 * kPerm[a] + kPerm[b])];
    }
  }
  return sigma;
}

// KeyframePolicy works in Eigen; GTSAM accumulated delta lives in gtsam::Pose3.
graph::Pose gtsamPoseToEigen(const gtsam::Pose3& pose) {
  graph::Pose t = graph::Pose::Identity();
  const gtsam::Point3 p = pose.translation();
  t.translation() = Eigen::Vector3d(p.x(), p.y(), p.z());
  t.linear() = pose.rotation().matrix();
  return t;
}

// gtsam::Pose3 <-> Eigen::Isometry3d, for handing the loop-closure verifier an
// initial guess and reading back its recovered relative pose (SLAM-10).
Eigen::Isometry3d gtsamPoseToIsometry(const gtsam::Pose3& pose) {
  Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
  iso.linear() = pose.rotation().matrix();
  const gtsam::Point3 t = pose.translation();
  iso.translation() = Eigen::Vector3d(t.x(), t.y(), t.z());
  return iso;
}

gtsam::Pose3 isometryToGtsamPose(const Eigen::Isometry3d& iso) {
  return {gtsam::Rot3(iso.linear()), gtsam::Point3(iso.translation())};
}

geometry_msgs::msg::Vector3 toVector3Msg(const Eigen::Vector3d& v) {
  geometry_msgs::msg::Vector3 out;
  out.x = v.x();
  out.y = v.y();
  out.z = v.z();
  return out;
}

geometry_msgs::msg::Pose gtsamPoseToMsg(const gtsam::Pose3& pose) {
  geometry_msgs::msg::Pose p;
  const gtsam::Point3 t = pose.translation();
  p.position.x = t.x();
  p.position.y = t.y();
  p.position.z = t.z();
  const gtsam::Quaternion q = pose.rotation().toQuaternion();
  p.orientation.w = q.w();
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  return p;
}

// ---------------------------------------------------------------------------
// Visualization builders (mirror of graph_visualizer_node.cpp helpers)
// ---------------------------------------------------------------------------

nav_msgs::msg::Path buildPath(const eval::GraphVisualization& viz,
                               const std::string& frame_id,
                               const rclcpp::Time& stamp) {
  nav_msgs::msg::Path path;
  path.header.frame_id = frame_id;
  path.header.stamp = stamp;
  path.poses.reserve(viz.nodeCount());
  for (const gtsam::Pose3& pose : viz.poses) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header = path.header;
    ps.pose = gtsamPoseToMsg(pose);
    path.poses.push_back(ps);
  }
  return path;
}

visualization_msgs::msg::MarkerArray buildKeyframeMarkers(
    const eval::GraphVisualization& viz, const std::string& frame_id,
    const rclcpp::Time& stamp, double sphere_diameter, double label_height) {
  visualization_msgs::msg::MarkerArray array;
  for (std::size_t i = 0; i < viz.nodeCount(); ++i) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame_id;
    m.header.stamp = stamp;
    m.ns = "keyframes";
    m.id = static_cast<int32_t>(i);
    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose = gtsamPoseToMsg(viz.poses[i]);
    m.scale.x = sphere_diameter;
    m.scale.y = sphere_diameter;
    m.scale.z = sphere_diameter;
    m.color.r = 0.2F;
    m.color.g = 0.8F;
    m.color.b = 1.0F;
    m.color.a = 1.0F;
    array.markers.push_back(m);

    // "x{id}" text label, sitting just above the sphere (separate namespace so it
    // never collides with the SPHERE's (ns,id) and can be toggled independently).
    visualization_msgs::msg::Marker label;
    label.header.frame_id = frame_id;
    label.header.stamp = stamp;
    label.ns = "keyframe_labels";
    label.id = static_cast<int32_t>(i);
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose = gtsamPoseToMsg(viz.poses[i]);
    label.pose.position.z += sphere_diameter + 0.5 * label_height;
    label.scale.z = label_height;  // text height [m]
    label.color.r = 1.0F;
    label.color.g = 1.0F;
    label.color.b = 1.0F;
    label.color.a = 1.0F;
    label.text = "x" + std::to_string(i);
    array.markers.push_back(label);
  }
  return array;
}

// Loop-closure overlay: a red LINE_LIST tying each looped keyframe pair
// (match — query) plus a "LC xq->xm" text label at the edge midpoint. Stable ids
// (edges only ever accumulate), so no DELETEALL needed.
visualization_msgs::msg::MarkerArray buildLoopClosureMarkers(
    const eval::GraphVisualization& viz,
    const std::vector<std::pair<int, int>>& loop_edges, const std::string& frame_id,
    const rclcpp::Time& stamp, float line_width, double label_height) {
  visualization_msgs::msg::MarkerArray array;

  visualization_msgs::msg::Marker lines;
  lines.header.frame_id = frame_id;
  lines.header.stamp = stamp;
  lines.ns = "loop_edges";
  lines.id = 0;
  lines.type = visualization_msgs::msg::Marker::LINE_LIST;
  lines.action = visualization_msgs::msg::Marker::ADD;
  lines.scale.x = line_width;
  lines.color.r = 1.0F;
  lines.color.g = 0.1F;
  lines.color.b = 0.1F;
  lines.color.a = 0.9F;

  for (std::size_t i = 0; i < loop_edges.size(); ++i) {
    const auto m = static_cast<std::size_t>(loop_edges[i].first);   // match
    const auto q = static_cast<std::size_t>(loop_edges[i].second);  // query
    if (m >= viz.nodeCount() || q >= viz.nodeCount()) {
      continue;
    }
    const gtsam::Point3 pm = viz.poses[m].translation();
    const gtsam::Point3 pq = viz.poses[q].translation();
    geometry_msgs::msg::Point a;
    geometry_msgs::msg::Point b;
    a.x = pm.x(); a.y = pm.y(); a.z = pm.z();
    b.x = pq.x(); b.y = pq.y(); b.z = pq.z();
    lines.points.push_back(a);
    lines.points.push_back(b);

    visualization_msgs::msg::Marker label;
    label.header.frame_id = frame_id;
    label.header.stamp = stamp;
    label.ns = "loop_labels";
    label.id = static_cast<int32_t>(i);
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose.position.x = 0.5 * (a.x + b.x);
    label.pose.position.y = 0.5 * (a.y + b.y);
    label.pose.position.z = 0.5 * (a.z + b.z) + label_height;
    label.pose.orientation.w = 1.0;
    label.scale.z = 1.2 * label_height;
    label.color.r = 1.0F;
    label.color.g = 0.3F;
    label.color.b = 0.3F;
    label.color.a = 1.0F;
    label.text = "LC x" + std::to_string(q) + "->x" + std::to_string(m);
    array.markers.push_back(label);
  }
  array.markers.push_back(lines);
  return array;
}

visualization_msgs::msg::Marker buildOdometryEdges(const eval::GraphVisualization& viz,
                                                    const std::string& frame_id,
                                                    const rclcpp::Time& stamp,
                                                    float line_width) {
  visualization_msgs::msg::Marker m;
  m.header.frame_id = frame_id;
  m.header.stamp = stamp;
  m.ns = "odometry_edges";
  m.id = 0;
  m.type = visualization_msgs::msg::Marker::LINE_LIST;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.scale.x = line_width;
  m.color.r = 1.0F;
  m.color.g = 1.0F;
  m.color.b = 1.0F;
  m.color.a = 0.9F;
  m.points.reserve(viz.odometryEdgeCount() * 2);
  for (std::size_t i = 0; i + 1 < viz.nodeCount(); ++i) {
    geometry_msgs::msg::Point start;
    geometry_msgs::msg::Point end;
    const gtsam::Point3 t0 = viz.poses[i].translation();
    const gtsam::Point3 t1 = viz.poses[i + 1].translation();
    start.x = t0.x(); start.y = t0.y(); start.z = t0.z();
    end.x = t1.x(); end.y = t1.y(); end.z = t1.z();
    m.points.push_back(start);
    m.points.push_back(end);
  }
  return m;
}

// EVAL-05: one Sigma_post ellipsoid per keyframe. For each node take the 3x3
// POSITION block of the 6x6 marginal (GTSAM tangent order [rx,ry,rz,x,y,z] →
// position is the bottom-right block, indices 3..5), eigendecompose it, and emit
// a SPHERE oriented by the eigenvectors with semi-axes scaled to the 95%
// 3-DOF confidence radius (chi2_3dof(0.95) ≈ 7.815). Green for normal
// keyframes, cyan for keyframes inside a closed loop (Sigma_post shrinks there).
// A leading DELETEALL clears stale markers so the array fully refreshes.
visualization_msgs::msg::MarkerArray buildCovarianceEllipsoids(
    const eval::GraphVisualization& viz,
    const std::vector<Eigen::Matrix<double, 6, 6>>& covs,
    const std::set<std::size_t>& loop_affected, const std::string& frame_id,
    const rclcpp::Time& stamp, double marker_scale) {
  constexpr double kChi2_3dof_95 = 7.815;
  constexpr double kMinEigenvalue = 1e-9;  // keep RViz scales strictly positive

  visualization_msgs::msg::MarkerArray array;
  // DELETEALL clears every marker regardless of namespace, so leave its ns empty:
  // sharing (ns="covariance", id=0) with the first ellipsoid makes RViz warn
  // about a duplicate (ns,id) pair within the same MarkerArray.
  visualization_msgs::msg::Marker clear;
  clear.header.frame_id = frame_id;
  clear.header.stamp = stamp;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  array.markers.push_back(clear);

  for (std::size_t i = 0; i < viz.nodeCount(); ++i) {
    const Eigen::Matrix3d pos_block = covs[i].block<3, 3>(3, 3);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(pos_block);
    Eigen::Vector3d evals = solver.eigenvalues();
    Eigen::Matrix3d evecs = solver.eigenvectors();
    if (evecs.determinant() < 0.0) {
      evecs.col(0) *= -1.0;  // ensure a right-handed (proper) rotation
    }
    const Eigen::Quaterniond q(evecs);

    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame_id;
    m.header.stamp = stamp;
    m.ns = "covariance";
    m.id = static_cast<int32_t>(i);
    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.action = visualization_msgs::msg::Marker::ADD;
    const gtsam::Point3 t = viz.poses[i].translation();
    m.pose.position.x = t.x();
    m.pose.position.y = t.y();
    m.pose.position.z = t.z();
    m.pose.orientation.w = q.w();
    m.pose.orientation.x = q.x();
    m.pose.orientation.y = q.y();
    m.pose.orientation.z = q.z();
    // marker_scale is a DISPLAY-ONLY exaggeration so the (cm-scale) 95% ellipsoids
    // are visible in a 20 m room. It does not touch the logged covariance — the
    // CSV / eval05_plot.py keep the true Sigma_post for any quantitative claim.
    const double axis = marker_scale * 2.0;
    m.scale.x = axis * std::sqrt(kChi2_3dof_95 * std::max(evals(0), kMinEigenvalue));
    m.scale.y = axis * std::sqrt(kChi2_3dof_95 * std::max(evals(1), kMinEigenvalue));
    m.scale.z = axis * std::sqrt(kChi2_3dof_95 * std::max(evals(2), kMinEigenvalue));
    const bool looped = loop_affected.count(i) > 0;
    m.color.r = 0.0F;
    m.color.g = 1.0F;
    m.color.b = looped ? 1.0F : 0.0F;       // cyan if inside a loop, else green
    m.color.a = looped ? 0.8F : 0.6F;
    array.markers.push_back(m);
  }
  return array;
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

class GraphBackendNode : public rclcpp::Node {
 public:
  explicit GraphBackendNode(rclcpp::NodeOptions options)
      : rclcpp::Node("graph_backend", options),
        kf_policy_(graph::KeyframePolicyConfig{}) {  // placeholder, overwritten below
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    ndt_odom_topic_ =
        declare_parameter<std::string>("ndt_odom_topic", "/ndt_frontend/ndt_odom");
    sphere_diameter_m_ = declare_parameter<double>("sphere_diameter_m", 0.15);
    edge_line_width_m_ = declare_parameter<double>("edge_line_width_m", 0.03);
    // EVAL-05: display multiplier on the Sigma_post 95% ellipsoids. 1.0 = the
    // statistically exact 95% ellipsoid; does NOT affect the logged covariance.
    // (Was 5.0 to make the pre-calibration cm-scale ellipsoids visible; after the
    // NEES calibration Sigma_post is physically correct, so no exaggeration is
    // needed — set <1.0 only to de-clutter, e.g. 0.5 for ~1-sigma-ish markers.)
    covariance_marker_scale_ =
        declare_parameter<double>("covariance_marker_scale", 1.0);
    // EVAL-05: height [m] of the "x{id}" keyframe + "LC ..." loop-closure text
    // labels in RViz (readable in a ~20 m room).
    label_height_m_ = declare_parameter<double>("label_height_m", 0.4);
    // EVAL-05: full Sigma_post log (ARCHITECTURE §9 columns) — supersedes the
    // minimal SLAM-11 CSV. The /slam/diagnostics JSON topic is unchanged.
    diagnostics_csv_path_ = declare_parameter<std::string>(
        "diagnostics_csv_path", "analysis/eval05_covariance_log.csv");
    gt_topic_ =
        declare_parameter<std::string>("ground_truth_topic", "/ground_truth/pose");
    latest_gt_.pose.orientation.w = 1.0;  // valid identity quat until first GT msg

    // L5-01: IMU intake + preintegration params. The wrapper is ROS-free; the node
    // owns the subscription, the sim-time re-stamp, and the bounded buffer.
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/imu/data");
    imu_buffer_seconds_ = declare_parameter<double>("imu_buffer_seconds", 5.0);
    preint_config_.accel_noise_sigma = declare_parameter<double>(
        "imu_accel_noise_sigma", preint_config_.accel_noise_sigma);
    preint_config_.gyro_noise_sigma = declare_parameter<double>(
        "imu_gyro_noise_sigma", preint_config_.gyro_noise_sigma);
    preint_config_.accel_bias_rw_sigma = declare_parameter<double>(
        "imu_accel_bias_rw_sigma", preint_config_.accel_bias_rw_sigma);
    preint_config_.gyro_bias_rw_sigma = declare_parameter<double>(
        "imu_gyro_bias_rw_sigma", preint_config_.gyro_bias_rw_sigma);
    preint_config_.integration_sigma = declare_parameter<double>(
        "imu_integration_sigma", preint_config_.integration_sigma);
    preint_config_.gravity =
        declare_parameter<double>("imu_gravity", preint_config_.gravity);
    // L5-17e: inter-sample header-stamp gap above which a keyframe interval is
    // treated as off-nominal (falls back to NDT+scaffolding for that one edge).
    // Nominal IMU period is ~4 ms (250 Hz); 0.02 s = 5x nominal, well under the
    // known ~134 ms slam_loop_03 PX4 stamp step.
    imu_max_dt_gap_s_ = declare_parameter<double>("imu_max_dt_gap_s", 0.02);
    // L5-01e: rolling ~1 s preintegration-vs-GT self-check (off in production).
    imu_diagnostic_enabled_ = declare_parameter<bool>("imu_diagnostic_enabled", false);
    imu_diagnostic_window_s_ = declare_parameter<double>("imu_diagnostic_window_s", 1.0);
    // L5-03 ablation toggles (both default true = full LiDAR-inertial backbone).
    //   imu_factor_enabled=false → pre-L5-03 path (NDT edge + weak V/B scaffolding
    //     priors) = NDT-only, reproduces L5-02/pre-L5.
    //   ndt_factor_enabled=false → IMU-only (drop the NDT BetweenFactor; node j held
    //     by the ImuFactor alone; requires IMU present in each interval).
    imu_factor_enabled_ = declare_parameter<bool>("imu_factor_enabled", true);
    ndt_factor_enabled_ = declare_parameter<bool>("ndt_factor_enabled", true);
    // L5-21: χ² consistency gate + Huber on NDT BetweenFactors (single-graph only).
    consistency_gate_.enabled =
        declare_parameter<bool>("ndt_consistency_gate_enabled", false);
    const std::string mode_str =
        declare_parameter<std::string>("ndt_consistency_gate_mode", "strict");
    consistency_gate_.mode = (mode_str == "soft") ? graph::ConsistencyGateMode::Soft
                                                  : graph::ConsistencyGateMode::Strict;
    consistency_gate_.chi2_threshold = declare_parameter<double>(
        "ndt_consistency_chi2_threshold", graph::kDefaultConsistencyChi2Threshold);
    consistency_gate_.huber_k =
        declare_parameter<double>("ndt_huber_k", graph::kDefaultHuberK);
    if (mode_str != "strict" && mode_str != "soft") {
      RCLCPP_WARN(get_logger(),
                  "ndt_consistency_gate_mode='%s' unknown; using strict.",
                  mode_str.c_str());
    }
    // L5-03: build the persistent preintegrator from the (now fully-populated)
    // config; it is reset at the bootstrap keyframe and after every keyframe.
    preint_.emplace(preint_config_);

    // DUAL-GRAPH ablation flag (src/DUAL_GRAPH_ANALYSIS.md §5). false = today's
    // single graph, byte-identical. true = LIO-SAM's split: the pose graph goes
    // Pose3-only (NDT BetweenFactor + loop closures, no ImuFactor, no V/B keys)
    // and the IMU chain moves into a SEPARATE small iSAM2 (ImuStateEstimator)
    // where each NDT pose enters as an absolute PriorFactor<Pose3>. That side
    // graph corrects on EVERY ndt_odom message (~10 Hz, LIO-SAM's
    // "every mapping publish" cadence) and is the source of
    // /slam/optimized_state, i.e. of the front-end's IMU-predicted guess.
    dual_graph_enabled_ = declare_parameter<bool>("dual_graph_enabled", false);
    dual_correction_rot_sigma_ =
        declare_parameter<double>("dual_correction_rot_sigma", 0.05);
    dual_correction_pos_sigma_ =
        declare_parameter<double>("dual_correction_pos_sigma", 0.1);
    dual_reset_key_interval_ =
        declare_parameter<int>("dual_reset_key_interval", 100);

    // L5-02 scaffolding prior noise for the per-node V(i)/B(i) values (weak,
    // held at zero; see addVelocityBiasScaffolding). Deliberately weak and
    // SEPARATE from the L5-04 bootstrap priors: they only hold otherwise-
    // factorless variables on the NDT-only / no-IMU path without moving the
    // pose solution (L5-02e's byte-identical guarantee).
    scaffold_velocity_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, 10.0);
    scaffold_bias_noise_ = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);

    // L5-04 bootstrap priors: the sigmas for the X(0)/V(0)/B(0) PriorFactors.
    // Pose: tight, at the first NDT-odom pose — explicitly NOT EKF2. Velocity:
    // near zero — the canonical bags bootstrap pre-takeoff at rest (verified on
    // slam_loop_03); widen if a bag ever bootstraps mid-flight. Bias: zero-mean
    // with separate accel/gyro sigmas, wide enough to admit the ~0.036 m/s^2
    // accel bias the optimizer settles at on slam_loop_03 (L5-03 journal) —
    // LIO-SAM's priorBiasNoise sigma 1e-3 would fight that at ~36 sigma.
    bootstrap_pose_sigma_ = declare_parameter<double>("bootstrap_pose_sigma", 0.001);
    bootstrap_velocity_sigma_ =
        declare_parameter<double>("bootstrap_velocity_sigma", 0.1);
    bootstrap_accel_bias_sigma_ =
        declare_parameter<double>("bootstrap_accel_bias_sigma", 0.1);
    bootstrap_gyro_bias_sigma_ =
        declare_parameter<double>("bootstrap_gyro_bias_sigma", 0.01);
    bootstrap_pose_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
        gtsam::Vector6::Constant(bootstrap_pose_sigma_));
    bootstrap_velocity_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(3, bootstrap_velocity_sigma_);
    gtsam::Vector6 bias_sigmas;  // ConstantBias tangent order: accel, then gyro
    bias_sigmas << bootstrap_accel_bias_sigma_, bootstrap_accel_bias_sigma_,
        bootstrap_accel_bias_sigma_, bootstrap_gyro_bias_sigma_,
        bootstrap_gyro_bias_sigma_, bootstrap_gyro_bias_sigma_;
    bootstrap_bias_noise_ = gtsam::noiseModel::Diagonal::Sigmas(bias_sigmas);

    // DUAL-GRAPH: the side estimator shares the IMU noise and the L5-04 prior
    // sigmas with the single-graph path, so the two arms differ only in graph
    // ARCHITECTURE. Its correction sigmas are the one genuinely new constant
    // (LIO-SAM's hand-fixed correctionNoise — Sigma_prop is a relative-edge
    // covariance and is not a valid absolute prior; see DUAL_GRAPH_ANALYSIS §4.2).
    if (dual_graph_enabled_) {
      imu::ImuStateEstimatorConfig est_config;
      est_config.preint = preint_config_;
      est_config.prior_pose_sigma = bootstrap_pose_sigma_;
      est_config.prior_velocity_sigma = bootstrap_velocity_sigma_;
      est_config.prior_accel_bias_sigma = bootstrap_accel_bias_sigma_;
      est_config.prior_gyro_bias_sigma = bootstrap_gyro_bias_sigma_;
      est_config.correction_rot_sigma = dual_correction_rot_sigma_;
      est_config.correction_pos_sigma = dual_correction_pos_sigma_;
      est_config.reset_key_interval = dual_reset_key_interval_;
      estimator_.emplace(est_config);
    }

    const double kf_dist = declare_parameter<double>("keyframe_translation_m", 0.5);
    const double kf_angle = declare_parameter<double>("keyframe_rotation_rad", 0.5);
    const double kf_time = declare_parameter<double>("keyframe_time_s", 10.0);
    kf_policy_ = graph::KeyframePolicy(graph::KeyframePolicyConfig{kf_dist, kf_angle, kf_time});

    // SLAM-09: loop-closure candidate finder (detection only; SLAM-10 verifies).
    const double lc_dist = declare_parameter<double>("loop_min_distance_m", 0.5);
    const int lc_age = declare_parameter<int>("loop_min_age_difference", 10);
    finder_ = LoopClosureCandidateFinder(
        LoopClosureCandidateFinder::Params{lc_dist, lc_age});

    // SLAM-10: loop-closure verifier (NDT between candidate keyframe clouds).
    scan_topic_ =
        declare_parameter<std::string>("scan_topic", "/ndt_frontend/scan_raw");
    LoopClosureVerifier::Params verify_params;
    verify_params.min_ndt_score =
        declare_parameter<double>("loop_min_ndt_score", verify_params.min_ndt_score);
    verify_params.min_sigma_eigenvalue = declare_parameter<double>(
        "loop_min_sigma_eigenvalue", verify_params.min_sigma_eigenvalue);
    verify_params.max_condition_number = declare_parameter<double>(
        "loop_max_condition_number", verify_params.max_condition_number);
    verify_params.grid_config.resolution =
        declare_parameter<double>("ndt_resolution", verify_params.grid_config.resolution);
    // EVAL-05 calibration: the loop-closure BetweenFactor noise is also derived
    // from measurementCovariance, so the verifier's Sigma_meas must use the same
    // scale as the front-end odometry edges (else looped keyframes stay
    // overconfident). Reads the same ndt_cov_scale_factor key.
    verify_params.ndt_config.cov_scale_factor = declare_parameter<double>(
        "ndt_cov_scale_factor", verify_params.ndt_config.cov_scale_factor);
    verifier_ = LoopClosureVerifier(verify_params);

    // L5-12: frozen historical submap target. `match_key`'s NDT target is fused
    // from keyframes [match_index-w, match_index+w] (backend-optimized poses),
    // not the single match cloud, for more support and a wider convergence basin
    // (ARCHITECTURE §8) — same reasoning as the front-end's L5-07/08 scan-to-submap.
    loop_submap_window_ = declare_parameter<int>("loop_submap_window", 2);
    loop_submap_voxel_leaf_ = declare_parameter<double>("loop_submap_voxel_leaf", 0.3);

    // L5-12: χ² Mahalanobis gate + Huber kernel on the loop-closure factor
    // (single-graph path only — see the design note in L5_MICROSTORIES.md §L5-12).
    // Unlike the odometry gate (default OFF, L5-21), this defaults ON: rejecting a
    // loop-closure candidate is risk-free (the graph simply continues without that
    // edge, same as the pre-existing score/eigenvalue/condition-number rejections),
    // so there is no cascade to weigh against the extra outlier protection.
    loop_consistency_gate_.enabled =
        declare_parameter<bool>("loop_chi2_gate_enabled", true);
    loop_consistency_gate_.mode = graph::ConsistencyGateMode::Strict;
    loop_consistency_gate_.chi2_threshold =
        declare_parameter<double>("loop_chi2_threshold", 16.811893829770927);  // χ²_0.99(6)
    loop_consistency_gate_.huber_k =
        declare_parameter<double>("loop_huber_k", graph::kDefaultHuberK);

    // L5-10: on an accepted closure, publish ONE coordinated batch covering the
    // affected keyframe range so the front-end's submap can follow the shifted
    // poses without the L5-12-measured collapse (see publishOptimizedStateBatch
    // + processLoopClosures). Ablation flag for the ON/OFF bag comparison this
    // story's AC asks for.
    optimized_state_batch_topic_ = declare_parameter<std::string>(
        "optimized_state_batch_topic", "/slam/optimized_state_batch");
    loop_submap_rebuild_enabled_ =
        declare_parameter<bool>("loop_submap_rebuild_enabled", true);

    path_pub_ = create_publisher<nav_msgs::msg::Path>("/slam/graph_path", 10);
    keyframes_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>("/slam/keyframes", 10);
    edges_pub_ = create_publisher<visualization_msgs::msg::Marker>("/slam/graph_edges", 10);
    loop_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/slam/loop_closures", 10);
    optimized_odom_pub_ =
        create_publisher<nav_msgs::msg::Odometry>("/slam/optimized_odom", 10);
    // L5-06: typed companion of /slam/optimized_odom carrying the full L5-02
    // node state (pose + velocity + bias) the front-end predictor resets from.
    optimized_state_pub_ = create_publisher<graph_slam_msgs::msg::OptimizedState>(
        "/slam/optimized_state", 10);
    // L5-10: coordinated loop-closure submap-rebuild feed (see
    // publishOptimizedStateBatch). Separate topic from the per-keyframe one
    // above so the front-end can tell "one closure's worth of updates,
    // applied together" apart from the ordinary newest-keyframe refine.
    optimized_state_batch_pub_ = create_publisher<graph_slam_msgs::msg::OptimizedStateBatch>(
        optimized_state_batch_topic_, 10);
    diagnostics_pub_ =
        create_publisher<std_msgs::msg::String>("/slam/diagnostics", 10);
    // EVAL-05: live Sigma_post ellipsoids + GT trajectory.
    ellipsoids_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/slam/covariance_ellipsoids", 10);
    gt_path_pub_ = create_publisher<nav_msgs::msg::Path>("/slam/gt_path", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        ndt_odom_topic_, rclcpp::QoS(10),
        [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { onNdtOdom(msg); });
    // SLAM-10: buffer the preprocessed scan so each keyframe can keep its cloud
    // for later NDT verification (the front-end publishes the same cloud it feeds
    // into registration on ~/scan_raw).
    scan_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        scan_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { onScan(msg); });
    // EVAL-05: EVAL-01 ground-truth pose (ENU, frame map) for the CSV gt_* columns
    // and the /slam/gt_path overlay.
    gt_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        gt_topic_, rclcpp::QoS(50),
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) { onGroundTruth(msg); });
    // L5-01: raw IMU (sensor_msgs/Imu from px4_offboard/imu_bridge.py, ENU/FLU).
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) { onImu(msg); });

    RCLCPP_INFO(get_logger(),
                "graph_backend up. Subscribing to '%s'. KF thresholds: %.2f m / %.2f rad / "
                "%.1f s. Publishing /slam/graph_path, /slam/keyframes, /slam/graph_edges, "
                "/slam/optimized_odom, /slam/optimized_state. TF: %s → %s.",
                ndt_odom_topic_.c_str(), kf_dist, kf_angle, kf_time, map_frame_.c_str(),
                odom_frame_.c_str());
    if (dual_graph_enabled_) {
      RCLCPP_INFO(get_logger(),
                  "GRAPH ARCHITECTURE: DUAL (LIO-SAM style). Pose graph = NDT "
                  "BetweenFactor + loop closures only (Pose3-only nodes); IMU chain in a "
                  "separate iSAM2 corrected per ndt_odom by an absolute PriorFactor<Pose3> "
                  "(sigma %.3g rad / %.3g m, reset every %d keys). /slam/optimized_state "
                  "comes from that side graph.",
                  dual_correction_rot_sigma_, dual_correction_pos_sigma_,
                  dual_reset_key_interval_);
    } else {
      RCLCPP_INFO(get_logger(),
                  "GRAPH ARCHITECTURE: SINGLE (ImuFactor + NDT BetweenFactor on the same "
                  "X(i)/V(i)/B(i) nodes). imu_factor_enabled=%d ndt_factor_enabled=%d.",
                  static_cast<int>(imu_factor_enabled_),
                  static_cast<int>(ndt_factor_enabled_));
    }
  }

 private:
  // L5-01: a node-clock-stamped GT pose sample, used only by the L5-01e diagnostic
  // (NavState seed + finite-difference velocity). Declared before the methods whose
  // signatures reference it.
  struct GtSample {
    double stamp_s{0.0};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    gtsam::Rot3 rotation;
  };

  // Keep the latest preprocessed scan so the next keyframe can snapshot it for
  // loop-closure verification (SLAM-10).
  void onScan(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
    auto cloud = std::make_shared<Cloud>();
    pcl::fromROSMsg(*msg, *cloud);
    latest_cloud_ = cloud;
    have_cloud_ = true;
  }

  // EVAL-05: keep the latest GT pose (for the per-event CSV gt_* columns) and
  // accumulate the full GT trajectory for the /slam/gt_path overlay.
  void onGroundTruth(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg) {
    latest_gt_ = *msg;
    have_gt_ = true;
    geometry_msgs::msg::PoseStamped ps = *msg;
    ps.header.frame_id = map_frame_;
    gt_path_.poses.push_back(ps);

    // L5-01e: keep a short GT pose history for the IMU-diagnostic NavState seed.
    // Re-stamp with the node clock (sim time under use_sim_time) so it shares the
    // exact clock domain as the re-stamped IMU buffer (bag-replay-clock-domain).
    if (imu_diagnostic_enabled_) {
      GtSample g;
      g.stamp_s = now().seconds();
      g.position = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y,
                                   msg->pose.position.z);
      g.rotation = gtsam::Rot3::Quaternion(msg->pose.orientation.w, msg->pose.orientation.x,
                                           msg->pose.orientation.y, msg->pose.orientation.z);
      gt_history_.push_back(g);
      while (!gt_history_.empty() &&
             (g.stamp_s - gt_history_.front().stamp_s) > imu_buffer_seconds_) {
        gt_history_.pop_front();
      }
    }
  }

  // L5-17: buffer one IMU sample keyed by its HEADER stamp. Post-L5-17i the
  // bridge (px4_offboard/imu_bridge.py) already publishes /imu/data in the SLAM
  // sim-time domain (node clock under use_sim_time; OS-time pass-through on
  // real hardware), so the header stamp is directly comparable to the
  // ndt_odom/GT stamps and the old now() re-stamp is gone. The intake only
  // BUFFERS — integration happens in onNdtOdom, where the window is cut at the
  // odom message's header stamp (LIO-SAM's buffer-then-cut, imuQueOpt pattern).
  void onImu(const sensor_msgs::msg::Imu::ConstSharedPtr& msg) {
    imu::ImuSample s;
    s.stamp_s = rclcpp::Time(msg->header.stamp).seconds();
    s.accel = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y,
                              msg->linear_acceleration.z);
    s.gyro = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y,
                             msg->angular_velocity.z);
    // L5-01e diagnostic buffer (off in production) — untouched path, now fed the
    // header stamp (same sim-time domain as the gt_history_ re-stamp, within the
    // diagnostic's 0.15 s matching tolerance).
    imu_buffer_.push_back(s);
    while (!imu_buffer_.empty() &&
           (s.stamp_s - imu_buffer_.front().stamp_s) > imu_buffer_seconds_) {
      imu_buffer_.pop_front();
    }

    if (imu_factor_enabled_) {
      if (dual_graph_enabled_ && estimator_) {
        // DUAL-GRAPH: the side estimator keeps its per-sample immediate feed;
        // dt now comes from consecutive header stamps (deterministic given the
        // stream) instead of node-clock arrival deltas.
        if (bootstrapped_) {
          if (last_dual_imu_stamp_ >= 0.0) {
            estimator_->addImuSample(s.accel, s.gyro,
                                     s.stamp_s - last_dual_imu_stamp_);
          }
          last_dual_imu_stamp_ = s.stamp_s;
        }
      } else {
        // Single graph (the L5 backbone): buffer only. Pushing unconditionally
        // (even pre-bootstrap) means samples stamped after X0 but delivered
        // before the bootstrap callback are not lost; bootstrap's
        // imu_window_.reset() drops everything stamped at/before X0.
        imu_window_.push(s);
        if (!bootstrapped_) {
          imu_window_.pruneOlderThan(s.stamp_s - imu_buffer_seconds_);
        }
      }
    }

    if (imu_diagnostic_enabled_) {
      maybeRunImuDiagnostic(s.stamp_s);
    }
  }

  // L5-01e: once ~imu_diagnostic_window_s of IMU has elapsed, preintegrate that
  // window and report drift vs the independent Gazebo GT + covariance growth.
  void maybeRunImuDiagnostic(double t_now) {
    if (diag_window_start_t_ < 0.0) {
      diag_window_start_t_ = t_now;  // open the first window
      return;
    }
    if (t_now - diag_window_start_t_ < imu_diagnostic_window_s_) {
      return;
    }
    runImuDiagnostic(diag_window_start_t_, t_now);
    diag_window_start_t_ = t_now;  // slide the window
  }

  // Nearest GT sample to time t (within a 0.15 s tolerance, else none).
  std::optional<GtSample> gtNearest(double t) const {
    std::optional<GtSample> best;
    double best_dt = 0.15;
    for (const GtSample& g : gt_history_) {
      const double dt = std::abs(g.stamp_s - t);
      if (dt < best_dt) {
        best_dt = dt;
        best = g;
      }
    }
    return best;
  }

  // GT nav-frame velocity at time t by central difference of the position history
  // over a ~0.1 s baseline (the /ground_truth/odom twist is empty — pose-only
  // bridge). A fixed baseline is robust to the per-sample jitter that the
  // node-clock re-stamp adds; zero if history does not cover t±h.
  Eigen::Vector3d gtVelocity(double t) const {
    constexpr double h = 0.05;  // half-baseline [s]
    const std::optional<GtSample> a = gtNearest(t - h);
    const std::optional<GtSample> b = gtNearest(t + h);
    if (a && b && b->stamp_s > a->stamp_s) {
      return (b->position - a->position) / (b->stamp_s - a->stamp_s);
    }
    return Eigen::Vector3d::Zero();
  }

  // Preintegrate the buffered IMU over [win_start, win_end], predict from the GT
  // NavState at win_start, and log position/rotation error vs GT at win_end plus
  // the covariance-trace growth (mid-window vs end). Validates gravity sign, frame,
  // and noise before any IMU factor is added (L5-03).
  void runImuDiagnostic(double win_start, double win_end) {
    const std::optional<GtSample> gt_i = gtNearest(win_start);
    const std::optional<GtSample> gt_j = gtNearest(win_end);
    if (!gt_i || !gt_j) {
      return;  // no GT coverage for this window yet
    }

    imu::ImuPreintegrator preint(preint_config_);
    std::size_t n = 0;
    std::size_t half = 0;
    double cov_trace_mid = 0.0;
    // First count how many samples fall in the window (for the mid-window capture).
    std::size_t in_window = 0;
    for (const imu::ImuSample& s : imu_buffer_) {
      if (s.stamp_s >= win_start && s.stamp_s <= win_end) {
        ++in_window;
      }
    }
    half = in_window / 2;

    bool have_prev = false;
    double prev_t = win_start;
    for (const imu::ImuSample& s : imu_buffer_) {
      if (s.stamp_s < win_start || s.stamp_s > win_end) {
        continue;
      }
      if (have_prev) {
        preint.integrate(s.accel, s.gyro, s.stamp_s - prev_t);
      }
      prev_t = s.stamp_s;
      have_prev = true;
      ++n;
      if (n == half && preint.count() > 0) {
        cov_trace_mid = preint.covariance().trace();
      }
    }
    if (preint.count() == 0) {
      return;
    }

    const Eigen::Vector3d v_i = gtVelocity(win_start);
    const gtsam::NavState state_i(gt_i->rotation, gtsam::Point3(gt_i->position), v_i);
    const gtsam::NavState pred = preint.predict(state_i, gtsam::imuBias::ConstantBias());

    const double pos_err = (Eigen::Vector3d(pred.position()) - gt_j->position).norm();
    const double rot_err_deg =
        gtsam::Rot3::Logmap(pred.attitude().between(gt_j->rotation)).norm() * 180.0 / M_PI;
    const double gt_dist = (gt_j->position - gt_i->position).norm();
    const double gt_speed = gt_dist / std::max(win_end - win_start, 1e-6);
    const double cov_trace_end = preint.covariance().trace();

    RCLCPP_INFO(
        get_logger(),
        "IMU-diag [%.2f→%.2f s] n=%zu dt=%.3f | GT speed %.3f m/s (moved %.3f m), "
        "seed |v0|=%.3f m/s | pos_err %.4f m, rot_err %.4f deg | cov_trace mid→end "
        "%.3e→%.3e",
        win_start, win_end, preint.count(), preint.deltaTij(), gt_speed, gt_dist,
        v_i.norm(), pos_err, rot_err_deg, cov_trace_mid, cov_trace_end);
  }

  void onNdtOdom(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
    const rclcpp::Time current_stamp(msg->header.stamp);
    const gtsam::Pose3 current_pose = navPoseToGtsam(msg->pose.pose);

    std::array<double, 36> cov_arr{};
    std::copy(msg->pose.covariance.begin(), msg->pose.covariance.end(), cov_arr.begin());
    const gtsam::Matrix66 sigma_meas = navCovToGtsam(cov_arr);

    // -----------------------------------------------------------------------
    // Bootstrap: first message → insert X(0)/V(0)/B(0) + their PriorFactors
    // (L5-02 node model; L5-04 principled priors: pose tight at the first
    // NDT-odom pose — not EKF2 — velocity near zero, bias zero-mean)
    // -----------------------------------------------------------------------
    if (!bootstrapped_) {
      if (dual_graph_enabled_ && estimator_) {
        // DUAL-GRAPH: the pose graph is Pose3-only here — no V(0)/B(0) to hold,
        // because no factor in it ever touches velocity or bias. The IMU chain
        // is bootstrapped in the side graph instead, at the same first NDT pose.
        optimizer_.add_prior(
            gtsam::PriorFactor<gtsam::Pose3>(sym::X(0), current_pose,
                                             bootstrap_pose_noise_),
            sym::X(0), current_pose);
        estimator_->initialize(current_pose);
      } else {
        const graph::GraphBootstrap bootstrap;
        const auto boot = bootstrap.create(current_pose, bootstrap_pose_noise_,
                                           bootstrap_velocity_noise_,
                                           bootstrap_bias_noise_);
        optimizer_.add_factors(boot.graph, boot.values);
      }
      optimizer_.update();
      last_pose_ = current_pose;
      last_kf_stamp_ = current_stamp;
      kf_index_ = 0;
      bootstrapped_ = true;
      // L5-03/L5-17: start the IMU backbone at X(0) with zero bias and re-anchor
      // the window buffer at X0's HEADER stamp, so the [X0,X1] interval
      // integrates exactly from the time X0 lives at (any sample stamped at or
      // before it precedes X0 and is discarded).
      if (preint_) {
        preint_->reset(gtsam::imuBias::ConstantBias{});
      }
      imu_window_.reset(current_stamp.seconds());
      imu_gap_in_interval_ = false;
      last_dual_imu_stamp_ = -1.0;
      storeKeyframeCloud(sym::X(0));
      publishAll(current_stamp);
      if (dual_graph_enabled_) {
        // First anchor for the front-end predictor (publishAll leaves the
        // optimized state to the side estimator on this path).
        publishOptimizedState(current_stamp);
      }
      emitDiagnostics(current_stamp);
      // L5-04: log the full initial state + prior sigmas (clean-init evidence).
      RCLCPP_INFO(get_logger(),
                  "Graph bootstrapped at x0 (%.2f, %.2f, %.2f) [first NDT-odom pose, "
                  "prior sigma %.4g] | v0 = (0, 0, 0) m/s [prior sigma %.4g m/s] | "
                  "b0 = 0 [prior sigma accel %.4g m/s^2, gyro %.4g rad/s]; bias chain "
                  "starts at B(0), first BetweenFactor<imuBias> lands at x1.",
                  current_pose.translation().x(), current_pose.translation().y(),
                  current_pose.translation().z(), bootstrap_pose_sigma_,
                  bootstrap_velocity_sigma_, bootstrap_accel_bias_sigma_,
                  bootstrap_gyro_bias_sigma_);
      return;
    }

    // -----------------------------------------------------------------------
    // Accumulate scan-to-scan step (ARCHITECTURE §6 concept #2)
    // -----------------------------------------------------------------------
    const gtsam::Pose3 delta_step = last_pose_.inverse().compose(current_pose);
    accumulator_.add_measurement(delta_step, sigma_meas);
    last_pose_ = current_pose;

    // -----------------------------------------------------------------------
    // L5-17c: cut the IMU preintegration window at THIS message's header stamp
    // (the time its pose lives at), LIO-SAM style — never at callback arrival.
    // Cutting on every ndt_odom (not just keyframes) keeps the deque short; the
    // accumulated PIM at a keyframe is identical either way because dt chains
    // continuously across cuts. A dt gap (L5-17e) is latched until the next
    // keyframe, which then falls back to the NDT+scaffolding path.
    // -----------------------------------------------------------------------
    if (!dual_graph_enabled_ && imu_factor_enabled_ && preint_) {
      const imu::WindowIntegrationResult win = imu_window_.integrateUpTo(
          current_stamp.seconds(), imu_max_dt_gap_s_, *preint_);
      if (win.gap_detected) {
        imu_gap_in_interval_ = true;
        RCLCPP_WARN(get_logger(),
                    "IMU header-stamp gap %.1f ms (> %.1f ms) ending at t=%.3f s; "
                    "the keyframe interval containing it will fall back to "
                    "NDT+scaffolding (no ImuFactor across the gap).",
                    win.max_dt_s * 1e3, imu_max_dt_gap_s_ * 1e3,
                    win.max_dt_stamp_s);
      }
    }

    // -----------------------------------------------------------------------
    // DUAL-GRAPH: correct the side IMU graph on EVERY ndt_odom message, not just
    // at keyframes (LIO-SAM corrects once per mapping publish, not once per
    // mapping keyframe — DUAL_GRAPH_ANALYSIS §1.4). The NDT pose enters there as
    // an absolute prior; the optimized (pose, velocity, bias) that comes back is
    // what re-anchors the front-end's IMU predictor.
    // -----------------------------------------------------------------------
    if (dual_graph_enabled_ && estimator_) {
      const std::optional<imu::ImuStateEstimate> corrected =
          estimator_->correct(current_pose);
      if (corrected) {
        publishOptimizedState(current_stamp);
      } else if (estimator_->failureCount() != last_estimator_failures_) {
        last_estimator_failures_ = estimator_->failureCount();
        RCLCPP_WARN(get_logger(),
                    "IMU side graph diverged (failure #%zu); re-initializing at the "
                    "next NDT pose.",
                    last_estimator_failures_);
      }
    }

    // -----------------------------------------------------------------------
    // Keyframe check
    // -----------------------------------------------------------------------
    const auto acc = accumulator_.current();
    const graph::Pose delta_eigen = gtsamPoseToEigen(acc.delta);
    const double elapsed_s = (current_stamp - last_kf_stamp_).seconds();

    if (!kf_policy_.is_keyframe_due(delta_eigen, elapsed_s)) {
      return;
    }

    // -----------------------------------------------------------------------
    // Keyframe triggered: add BetweenFactor, update iSAM2, reset accumulator
    // -----------------------------------------------------------------------
    const std::size_t prev_idx = kf_index_;
    ++kf_index_;
    const gtsam::Key x_prev = sym::X(prev_idx);
    const gtsam::Key x_cur = sym::X(kf_index_);

    const gtsam::Values est_before = optimizer_.estimate();
    const gtsam::Pose3 prev_pose = est_before.at<gtsam::Pose3>(x_prev);

    // The NDT relative-pose measurement for this edge (always built from the
    // accumulated scan-to-scan delta; added as an independent factor per L5-03d).
    // L5-21 may replace the noise model (Huber) and/or inflate Σ before insert.
    const graph::OdometryFactorBuilder factor_builder;

    // Use the IMU backbone only if enabled AND this interval actually carried IMU
    // (count>0 ⇒ deltaTij>0) AND no header-stamp gap contaminated it (L5-17e).
    // An off-nominal interval falls back to the NDT+scaffolding path so node j
    // is never left unconstrained.
    const bool use_imu = !dual_graph_enabled_ && imu_factor_enabled_ && preint_ &&
                         preint_->count() > 0 && !imu_gap_in_interval_;
    bool add_ndt = use_imu ? ndt_factor_enabled_ : true;
    last_consist_chi2_ = -1.0;
    last_consist_action_ = graph::ConsistencyAction::None;

    gtsam::Matrix66 sigma_for_factor = acc.covariance;
    // Placeholder; overwritten below once we know whether Huber applies.
    gtsam::BetweenFactor<gtsam::Pose3> ndt_factor =
        factor_builder.create_factor(x_prev, x_cur, acc.delta, sigma_for_factor);

    if (dual_graph_enabled_) {
      // DUAL-GRAPH: the mapping graph is Pose3-only — the NDT relative edge and
      // (later) loop closures, nothing else. No ImuFactor crosses these nodes,
      // so a mis-measured preintegration can no longer drag the pose solution or
      // its Sigma_post; the IMU lives entirely in the side graph and reaches the
      // map only indirectly, by improving the next scan-match guess.
      // L5-21 gate is single-graph only (no co-located IMU prediction here).
      ndt_factor =
          factor_builder.create_factor(x_prev, x_cur, acc.delta, acc.covariance);
      const gtsam::Pose3 x_cur_init = prev_pose.compose(acc.delta);
      optimizer_.add_odometry(ndt_factor, x_cur, x_cur_init);
      optimizer_.update();

      const gtsam::Values est_after = optimizer_.estimate();
      last_ndt_chi2_ = 2.0 * ndt_factor.error(est_after);
      last_imu_chi2_ = -1.0;
      last_bias_chi2_ = -1.0;
      last_opt_bias_ = estimator_ ? estimator_->latest().bias
                                  : gtsam::imuBias::ConstantBias{};
    } else if (use_imu) {
      const gtsam::Velocity3 prev_vel =
          est_before.at<gtsam::Velocity3>(sym::V(prev_idx));
      const gtsam::imuBias::ConstantBias prev_bias =
          est_before.at<gtsam::imuBias::ConstantBias>(sym::B(prev_idx));
      const gtsam::PreintegratedImuMeasurements& pim = preint_->finish();

      // Initial guesses: NDT-composed pose (best) when the NDT edge is present,
      // else the IMU-predicted pose; IMU-predicted velocity; bias carried from i.
      const gtsam::NavState predicted =
          preint_->predict(gtsam::NavState(prev_pose, prev_vel), prev_bias);
      const gtsam::Pose3 x_pred = predicted.pose();
      const gtsam::Pose3 x_ndt = prev_pose.compose(acc.delta);

      // L5-21: χ² consistency gate before inserting the NDT BetweenFactor.
      if (ndt_factor_enabled_ && consistency_gate_.enabled) {
        const graph::Matrix6 sigma_x_pred =
            graph::poseBlockFromPreintCov(preint_->covariance());
        const graph::ConsistencyGateResult gate = graph::evaluateNdtConsistency(
            x_pred, x_ndt, acc.covariance, sigma_x_pred, consistency_gate_);
        last_consist_chi2_ = gate.chi2;
        last_consist_action_ = gate.action;
        sigma_for_factor = gate.sigma_ndt_used;
        if (!gate.add_ndt) {
          add_ndt = false;
          ++consist_drop_count_;
        } else if (gate.action == graph::ConsistencyAction::Inflate) {
          ++consist_inflate_count_;
        }
      }

      const gtsam::Pose3 x_cur_init = add_ndt ? x_ndt : x_pred;
      const gtsam::Velocity3 v_cur_init = predicted.velocity();

      // Backbone: ImuFactor (pose+velocity) + separate bias random-walk factor.
      const gtsam::ImuFactor imu_factor(x_prev, sym::V(prev_idx), x_cur,
                                        sym::V(kf_index_), sym::B(prev_idx), pim);
      const gtsam::BetweenFactor<gtsam::imuBias::ConstantBias> bias_factor(
          sym::B(prev_idx), sym::B(kf_index_), gtsam::imuBias::ConstantBias{},
          biasRandomWalkNoise(pim.deltaTij()));

      optimizer_.add_imu_keyframe(x_cur, x_cur_init, sym::V(kf_index_), v_cur_init,
                                  sym::B(kf_index_), prev_bias, imu_factor, bias_factor);
      if (add_ndt) {
        // Gate ON → Huber robust kernel on the (possibly inflated) Σ.
        // Gate OFF → pure Gaussian (ablation baseline, byte-comparable to L5-17).
        ndt_factor =
            consistency_gate_.enabled
                ? factor_builder.create_robust_factor(x_prev, x_cur, acc.delta,
                                                       sigma_for_factor,
                                                       consistency_gate_.huber_k)
                : factor_builder.create_factor(x_prev, x_cur, acc.delta,
                                               sigma_for_factor);
        optimizer_.add_ndt_edge(ndt_factor);
      }
      optimizer_.update();

      // Diagnostics (L5-03e): per-edge chi2 = 2*factor.error on the fresh estimate.
      const gtsam::Values est_after = optimizer_.estimate();
      last_imu_chi2_ = 2.0 * imu_factor.error(est_after);
      last_bias_chi2_ = 2.0 * bias_factor.error(est_after);
      last_ndt_chi2_ = add_ndt ? 2.0 * ndt_factor.error(est_after) : -1.0;

      // Reset preintegration to node j's optimized bias for the next interval.
      last_opt_bias_ = est_after.at<gtsam::imuBias::ConstantBias>(sym::B(kf_index_));
      preint_->reset(last_opt_bias_);
    } else {
      // Pre-L5-03 path (NDT-only / no-IMU fallback): NDT edge inserts X(j);
      // V/B held by weak scaffolding priors. No IMU prediction → gate skips.
      ndt_factor =
          factor_builder.create_factor(x_prev, x_cur, acc.delta, acc.covariance);
      const gtsam::Pose3 x_cur_init = prev_pose.compose(acc.delta);
      optimizer_.add_odometry(ndt_factor, x_cur, x_cur_init);
      addVelocityBiasScaffolding(kf_index_);
      optimizer_.update();

      const gtsam::Values est_after = optimizer_.estimate();
      last_ndt_chi2_ = 2.0 * ndt_factor.error(est_after);
      last_imu_chi2_ = -1.0;
      last_bias_chi2_ = -1.0;
      last_opt_bias_ = est_after.at<gtsam::imuBias::ConstantBias>(sym::B(kf_index_));
      // L5-17e: an interval that fell back (gap-contaminated or empty) must not
      // leak its partial/contaminated PIM into the next edge.
      if (!dual_graph_enabled_ && imu_factor_enabled_ && preint_) {
        preint_->reset(last_opt_bias_);
      }
    }
    imu_gap_in_interval_ = false;  // L5-17e: the gap verdict is per keyframe interval
    accumulator_.reset();
    last_kf_stamp_ = current_stamp;
    storeKeyframeCloud(x_cur);

    publishAll(current_stamp);
    emitDiagnostics(current_stamp);
    RCLCPP_INFO(get_logger(), "KF x%zu added (%.2f, %.2f, %.2f) | total nodes: %zu.",
                kf_index_, current_pose.translation().x(), current_pose.translation().y(),
                current_pose.translation().z(), kf_index_ + 1);
    if (use_imu) {
      const gtsam::Vector6 b = last_opt_bias_.vector();
      RCLCPP_INFO(get_logger(),
                  "  edge x%zu->x%zu | chi2 imu %.3f (dof 9) ndt %.3f (dof 6) "
                  "bias %.3e (dof 6) | consist χ² %.3f (%s) drops=%zu inflate=%zu "
                  "| bias acc [%.4f %.4f %.4f] gyro [%.4f %.4f %.4f]",
                  prev_idx, kf_index_, last_imu_chi2_, last_ndt_chi2_, last_bias_chi2_,
                  last_consist_chi2_, graph::toString(last_consist_action_),
                  consist_drop_count_, consist_inflate_count_, b(0), b(1), b(2), b(3),
                  b(4), b(5));
    }
    if (dual_graph_enabled_ && estimator_) {
      const imu::ImuStateEstimate& e = estimator_->latest();
      const gtsam::Vector6 b = e.bias.vector();
      RCLCPP_INFO(get_logger(),
                  "  edge x%zu->x%zu | ndt chi2 %.3f (dof 6) | side graph: %zu "
                  "corrections (%zu resets, %zu failures), chi2 imu %.3f prior %.3f, "
                  "|v| %.3f m/s, bias acc [%.4f %.4f %.4f] gyro [%.4f %.4f %.4f]",
                  prev_idx, kf_index_, last_ndt_chi2_, estimator_->correctionCount(),
                  estimator_->resetCount(), estimator_->failureCount(), e.imu_chi2,
                  e.prior_chi2, e.state.velocity().norm(), b(0), b(1), b(2), b(3), b(4),
                  b(5));
    }

    // SLAM-10: verify any loop-closure candidates for the new keyframe and, on a
    // strong + well-conditioned NDT match, add a BetweenFactor that re-optimizes
    // the whole graph (map->odom jumps, Sigma_post shrinks for the looped poses).
    processLoopClosures(current_stamp);
  }

  // L5-03: LIO-SAM's bias random-walk noise for the BetweenFactor<imuBias> over an
  // interval of dt seconds — sigmas = sqrt(dt) * [accBiasRW x3, gyrBiasRW x3]
  // (ConstantBias tangent order is accelerometer-then-gyroscope). Mirrors LIO-SAM's
  // noiseModelBetweenBias scaling exactly (keeps the ours-vs-LIO-SAM comparison a
  // controlled one).
  gtsam::SharedNoiseModel biasRandomWalkNoise(double dt) const {
    const double s = std::sqrt(std::max(dt, 1e-9));
    gtsam::Vector6 sigmas;
    sigmas << preint_config_.accel_bias_rw_sigma * s,
        preint_config_.accel_bias_rw_sigma * s,
        preint_config_.accel_bias_rw_sigma * s,
        preint_config_.gyro_bias_rw_sigma * s,
        preint_config_.gyro_bias_rw_sigma * s,
        preint_config_.gyro_bias_rw_sigma * s;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
  }

  // L5-02 scaffolding: used only on the NDT-only / no-IMU path now (L5-03). A new
  // node's V(i)/B(i) would be factorless and iSAM2 would reject them — hold each
  // with a weak prior at zero. No factor couples the velocity/bias block to the
  // pose block, so on that path the pose estimate, its marginals, and chi2 are
  // unchanged vs pre-L5-02. When the IMU backbone is active the ImuFactor +
  // BetweenFactor<imuBias> replace these priors.
  void addVelocityBiasScaffolding(std::size_t index) {
    const gtsam::Velocity3 zero_velocity = gtsam::Velocity3::Zero();
    const gtsam::imuBias::ConstantBias zero_bias;
    optimizer_.add_prior(
        gtsam::PriorFactor<gtsam::Velocity3>(sym::V(index), zero_velocity,
                                             scaffold_velocity_noise_),
        sym::V(index), zero_velocity);
    optimizer_.add_prior(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
                             sym::B(index), zero_bias, scaffold_bias_noise_),
                         sym::B(index), zero_bias);
  }

  // Snapshot the buffered scan as this keyframe's cloud (SLAM-10). If no scan has
  // arrived yet (e.g. unit tests feed odometry only) the keyframe simply has no
  // cloud and is skipped during verification.
  void storeKeyframeCloud(gtsam::Key key) {
    if (have_cloud_ && latest_cloud_) {
      keyframe_clouds_[key] = latest_cloud_;
    }
  }

  // Publish visualisation topics + optimized_odom + map→odom TF.
  void publishAll(const rclcpp::Time& stamp) {
    const gtsam::Values est = optimizer_.estimate();
    const eval::GraphVisualization viz = eval::GraphVisualizer::fromEstimate(est);

    path_pub_->publish(buildPath(viz, map_frame_, stamp));
    keyframes_pub_->publish(buildKeyframeMarkers(viz, map_frame_, stamp,
                                                 sphere_diameter_m_, label_height_m_));
    edges_pub_->publish(
        buildOdometryEdges(viz, map_frame_, stamp,
                           static_cast<float>(edge_line_width_m_)));
    loop_pub_->publish(buildLoopClosureMarkers(
        viz, loop_edges_, map_frame_, stamp,
        static_cast<float>(2.0 * edge_line_width_m_), label_height_m_));

    // EVAL-05: live Sigma_post ellipsoids (one per keyframe) + GT path overlay.
    std::vector<Eigen::Matrix<double, 6, 6>> covs;
    covs.reserve(viz.nodeCount());
    for (std::size_t i = 0; i < viz.nodeCount(); ++i) {
      covs.push_back(optimizer_.marginalCovariance(sym::X(i)));
    }
    ellipsoids_pub_->publish(buildCovarianceEllipsoids(
        viz, covs, loop_affected_, map_frame_, stamp, covariance_marker_scale_));

    gt_path_.header.frame_id = map_frame_;
    gt_path_.header.stamp = stamp;
    gt_path_pub_->publish(gt_path_);

    if (viz.poses.empty()) {
      return;
    }

    // Latest optimized keyframe pose in map frame.
    const gtsam::Pose3 latest_map = viz.poses.back();

    // /slam/optimized_odom: latest graph estimate in map frame.
    nav_msgs::msg::Odometry opt_odom;
    opt_odom.header.stamp = stamp;
    opt_odom.header.frame_id = map_frame_;
    opt_odom.child_frame_id = "base_link";
    opt_odom.pose.pose = gtsamPoseToMsg(latest_map);
    optimized_odom_pub_->publish(opt_odom);

    // L5-06: the same instant as a typed OptimizedState (pose + velocity + bias)
    // — the front-end's IMU predictor re-seeds from it. On the single-graph path
    // it is published here, per keyframe; on the dual-graph path the side
    // estimator owns it and publishes per correction (see onNdtOdom).
    if (!dual_graph_enabled_) {
      publishOptimizedState(stamp);
    }

    // map → odom TF: T_map_odom = T_map_base_optimized * T_odom_base_raw⁻¹
    // (ARCHITECTURE §5 REP-105). At keyframe time, last_pose_ == raw odom pose
    // for the same physical position as latest_map, so this gives the
    // accumulated drift correction.
    const gtsam::Pose3 T_map_odom = latest_map.compose(last_pose_.inverse());

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = map_frame_;
    tf.child_frame_id = odom_frame_;
    const gtsam::Point3 t = T_map_odom.translation();
    tf.transform.translation.x = t.x();
    tf.transform.translation.y = t.y();
    tf.transform.translation.z = t.z();
    const gtsam::Quaternion q = T_map_odom.rotation().toQuaternion();
    tf.transform.rotation.w = q.w();
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf_broadcaster_->sendTransform(tf);
  }

  // L5-06 / DUAL-GRAPH: publish the typed (pose, velocity, bias) the front-end
  // predictor re-anchors on. Single graph → the newest keyframe node of the pose
  // graph. Dual graph → the side IMU estimator's freshest optimized state (the
  // pose graph has no V/B keys there, and the estimator is both fresher and
  // published ~3x more often, once per NDT correction rather than per keyframe).
  // The consumer only ever differences two predictions, so the map/odom frame
  // offset of this pose cancels out of the NDT guess (ndt_frontend_node.cpp:294-303).
  //
  // `index`: absent (default) = the newest keyframe (kf_index_), the normal
  // per-keyframe/per-correction call and the only path that may read the
  // dual-graph side estimator. An explicit index (used by
  // publishOptimizedStateBatch, L5-10) always reads the pose graph via
  // buildOptimizedStateMsg — the dual-graph estimator has no per-index history.
  void publishOptimizedState(const rclcpp::Time& stamp,
                             std::optional<std::size_t> index = std::nullopt) {
    if (!index && dual_graph_enabled_ && estimator_ && estimator_->initialized()) {
      const imu::ImuStateEstimate& e = estimator_->latest();
      graph_slam_msgs::msg::OptimizedState opt_state;
      opt_state.header.stamp = stamp;
      opt_state.header.frame_id = map_frame_;
      opt_state.keyframe_id = static_cast<int32_t>(kf_index_);
      opt_state.pose = gtsamPoseToMsg(e.state.pose());
      opt_state.velocity = toVector3Msg(e.state.velocity());
      opt_state.accel_bias = toVector3Msg(e.bias.accelerometer());
      opt_state.gyro_bias = toVector3Msg(e.bias.gyroscope());
      optimized_state_pub_->publish(opt_state);
      return;
    }
    optimized_state_pub_->publish(buildOptimizedStateMsg(index.value_or(kf_index_), stamp));
  }

  // Pose-graph-only OptimizedState builder for a SINGLE index (never the
  // dual-graph side estimator — that only has a "latest" state, not per-index
  // history), used by publishOptimizedState's single-graph branch. Goes
  // through nodeEstimate(), which also computes a marginal covariance this
  // message never carries; that one-off cost is fine for a single index but
  // NOT for a range, so publishOptimizedStateBatch (L5-10) builds its
  // messages directly from one shared `optimizer_.estimate()` snapshot
  // instead of calling this in a loop — see that method's own comment.
  [[nodiscard]] graph_slam_msgs::msg::OptimizedState buildOptimizedStateMsg(
      std::size_t idx, const rclcpp::Time& stamp) const {
    graph_slam_msgs::msg::OptimizedState opt_state;
    opt_state.header.stamp = stamp;
    opt_state.header.frame_id = map_frame_;
    // L5-19: front-end matches submap window entries by stamp (primary) or this id.
    opt_state.keyframe_id = static_cast<int32_t>(idx);
    const graph::NodeState node = optimizer_.nodeEstimate(idx);
    opt_state.pose = gtsamPoseToMsg(node.pose);
    opt_state.velocity = toVector3Msg(node.velocity);
    opt_state.accel_bias = toVector3Msg(node.bias.accelerometer());
    opt_state.gyro_bias = toVector3Msg(node.bias.gyroscope());
    return opt_state;
  }

  // L5-10: after an accepted loop closure re-optimizes every keyframe in
  // [lo, hi], publish ONE OptimizedStateBatch message covering that whole
  // range so the front-end applies it as a single atomic update
  // (applyOptimizedPoseBatch) instead of N independent
  // /slam/optimized_state publishes — L5-12 measured that the latter
  // repeatedly triggers the per-entry collapse guard (L5-19e) and stalls NDT
  // registration (97-99/111 keyframes vs 111/111). Cheap: this only fires on
  // the rare accepted-closure event, and entries outside the front-end's
  // small active window are simply not matched (no-ops there).
  // Returns the number of entries published, for the loop_closure diagnostics
  // row/JSON.
  // NOTE: reads ONE `optimizer_.estimate()` snapshot (a single
  // isam2_.calculateEstimate() call) and indexes into it directly, rather than
  // calling nodeEstimate() per index — nodeEstimate additionally computes a
  // marginal covariance (via calculateEstimate() + a Bayes-tree solve) that
  // OptimizedState.msg has no field for and this batch never uses; doing that
  // per index would be O(range size) wasted marginal solves on a range that
  // can span 80+ keyframes.
  std::size_t publishOptimizedStateBatch(const rclcpp::Time& stamp, std::size_t lo,
                                         std::size_t hi) {
    const gtsam::Values est = optimizer_.estimate();
    graph_slam_msgs::msg::OptimizedStateBatch batch;
    batch.header.stamp = stamp;
    batch.header.frame_id = map_frame_;
    batch.states.reserve(hi - lo + 1);
    for (std::size_t idx = lo; idx <= hi; ++idx) {
      const gtsam::Key key = sym::X(idx);
      if (!est.exists(key)) {
        continue;  // should not happen on the live pipeline (a real keyframe index)
      }
      graph_slam_msgs::msg::OptimizedState opt_state;
      opt_state.header.stamp = stamp;
      opt_state.header.frame_id = map_frame_;
      opt_state.keyframe_id = static_cast<int32_t>(idx);
      opt_state.pose = gtsamPoseToMsg(est.at<gtsam::Pose3>(key));
      if (est.exists(sym::V(idx))) {
        opt_state.velocity = toVector3Msg(est.at<gtsam::Velocity3>(sym::V(idx)));
      }
      if (est.exists(sym::B(idx))) {
        const auto& bias = est.at<gtsam::imuBias::ConstantBias>(sym::B(idx));
        opt_state.accel_bias = toVector3Msg(bias.accelerometer());
        opt_state.gyro_bias = toVector3Msg(bias.gyroscope());
      }
      batch.states.push_back(std::move(opt_state));
    }
    optimized_state_batch_pub_->publish(batch);
    return batch.states.size();
  }

  // SLAM-11: publish optimizer health (chi2 + latest Sigma_post position-trace)
  // to /slam/diagnostics (JSON line) and append one row to the CSV log.
  void emitDiagnostics(const rclcpp::Time& stamp) {
    const gtsam::Key latest_key = sym::X(kf_index_);
    const gtsam::Values est = optimizer_.estimate();
    const double chi2 = optimizer_.chi2();
    const double cov_trace = optimizer_.marginalCovPositionTrace(latest_key);
    // Keyframe count. (Was est.size() when nodes were pose-only; since L5-02
    // the estimate holds X+V+B per keyframe, so est.size() is 3x this.)
    const std::size_t num_factors = kf_index_ + 1;
    const auto keyframe_id = static_cast<int>(kf_index_);

    // SLAM-09: detect revisits of much-older keyframes (detection only; no factor
    // added — SLAM-10 verifies with NDT before closing the loop).
    const auto candidates = finder_.findCandidates(est, latest_key);
    std::ostringstream lc;
    lc << '[';
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      const LoopClosureCandidate& c = candidates[i];
      if (i != 0) {
        lc << ", ";
      }
      lc << "{\"query\": " << gtsam::Symbol(c.query_key).index()
         << ", \"match\": " << gtsam::Symbol(c.match_key).index()
         << ", \"dist\": " << c.distance_m << '}';
    }
    lc << ']';
    if (!candidates.empty()) {
      const LoopClosureCandidate& best = candidates.front();
      RCLCPP_INFO(get_logger(),
                  "Loop candidate: x%zu near x%zu (dist %.3f m); %zu total.",
                  static_cast<std::size_t>(gtsam::Symbol(best.query_key).index()),
                  static_cast<std::size_t>(gtsam::Symbol(best.match_key).index()),
                  best.distance_m, candidates.size());
    }

    // L5-03e: per-edge chi2 + optimized bias trajectory (accel-then-gyro).
    const gtsam::Vector3 ba = last_opt_bias_.accelerometer();
    const gtsam::Vector3 bg = last_opt_bias_.gyroscope();

    std::ostringstream json;
    json << std::setprecision(9) << "{\"t\": " << stamp.seconds()
         << ", \"keyframe_id\": " << keyframe_id
         << ", \"num_factors\": " << num_factors
         << ", \"num_values\": " << num_factors
         << ", \"chi2\": " << chi2
         << ", \"marginal_cov_trace\": " << cov_trace
         << ", \"imu_chi2\": " << last_imu_chi2_
         << ", \"ndt_chi2\": " << last_ndt_chi2_
         << ", \"bias_chi2\": " << last_bias_chi2_
         << ", \"consist_chi2\": " << last_consist_chi2_
         << ", \"consist_action\": \"" << graph::toString(last_consist_action_) << "\""
         << ", \"consist_drop_count\": " << consist_drop_count_
         << ", \"consist_inflate_count\": " << consist_inflate_count_
         << ", \"bias_acc\": [" << ba.x() << ", " << ba.y() << ", " << ba.z() << "]"
         << ", \"bias_gyro\": [" << bg.x() << ", " << bg.y() << ", " << bg.z() << "]";
    // DUAL-GRAPH: side-graph health. Additive fields, only on the dual path.
    if (dual_graph_enabled_ && estimator_) {
      const imu::ImuStateEstimate& e = estimator_->latest();
      json << ", \"dual_graph\": {\"corrections\": " << estimator_->correctionCount()
           << ", \"resets\": " << estimator_->resetCount()
           << ", \"failures\": " << estimator_->failureCount()
           << ", \"imu_chi2\": " << e.imu_chi2 << ", \"prior_chi2\": " << e.prior_chi2
           << ", \"bias_chi2\": " << e.bias_chi2
           << ", \"speed\": " << e.state.velocity().norm() << "}";
    }
    json << ", \"loop_candidates\": " << lc.str() << "}";

    std_msgs::msg::String diag_msg;
    diag_msg.data = json.str();
    diagnostics_pub_->publish(diag_msg);

    // EVAL-05 §9 row: optimized pose + velocity + bias (L5-05 schema v2) + GT +
    // full 6x6 Sigma_post upper triangle. Read through the L5-02 NodeState struct.
    writeCsvRow(stamp.seconds(), keyframe_id, "keyframe", currentNodeState(), "", "");
  }

  // The newest keyframe's state as the EVAL-05 CSV logs it. Pose and Sigma_post
  // always come from the mapping pose graph (the ARCHITECTURE §6 covariance chain
  // is untouched by the dual-graph split). Velocity and bias come from whichever
  // graph actually estimates them: the pose graph's V/B keys on the single path,
  // the side estimator on the dual path (where the pose graph has no V/B at all
  // and NodeState would read them back as zeros). Schema is unchanged — only the
  // provenance of est_v*/bias_* differs, per DUAL_GRAPH_ANALYSIS §4.2 risk 3.
  [[nodiscard]] graph::NodeState currentNodeState() const {
    graph::NodeState node = optimizer_.nodeEstimate(kf_index_);
    if (dual_graph_enabled_ && estimator_) {
      node.velocity = estimator_->latest().state.velocity();
      node.bias = estimator_->latest().bias;
    }
    return node;
  }

  // EVAL-05 / ARCHITECTURE §9 logging contract: one CSV row per diagnostic event.
  // `event` is "keyframe" or "loop_closure"; lc_from/lc_to are filled only for
  // closures. Columns (schema v2, L5-05): schema_version, t, keyframe_id, event,
  // est_x..est_qw, then the L5-02 node state est_vx,est_vy,est_vz (Velocity3) and
  // bias_ax,bias_ay,bias_az,bias_gx,bias_gy,bias_gz (imuBias, accel-then-gyro),
  // then gt_x..gt_qw, then the upper triangle of the 6x6 Sigma_post (pose marginal,
  // GTSAM tangent order rx,ry,rz,x,y,z), then lc_from, lc_to. The leading
  // schema_version column tags the file so downstream parsers reject a pre-L5 (v1)
  // CSV loudly instead of silently reading the wrong columns. Velocity/bias marginal
  // variances are intentionally NOT logged (NodeState carries only the pose
  // marginal; adding them is deferred — see the L5-05 design note).
  static constexpr int kEval05SchemaVersion = 2;
  void writeCsvRow(double t, int keyframe_id, const std::string& event,
                   const graph::NodeState& node, const std::string& lc_from,
                   const std::string& lc_to, int submap_batch_size = -1) {
    if (!csv_open_) {
      csv_.open(diagnostics_csv_path_, std::ios::out | std::ios::trunc);
      if (csv_.is_open()) {
        csv_ << "schema_version,t,keyframe_id,event,"
                "est_x,est_y,est_z,est_qx,est_qy,est_qz,est_qw,"
                "est_vx,est_vy,est_vz,"
                "bias_ax,bias_ay,bias_az,bias_gx,bias_gy,bias_gz,"
                "gt_x,gt_y,gt_z,gt_qx,gt_qy,gt_qz,gt_qw,"
                "cov_00,cov_01,cov_02,cov_03,cov_04,cov_05,"
                "cov_11,cov_12,cov_13,cov_14,cov_15,"
                "cov_22,cov_23,cov_24,cov_25,"
                "cov_33,cov_34,cov_35,"
                "cov_44,cov_45,cov_55,"
                "consist_chi2,consist_action,"
                "lc_from,lc_to,submap_batch_size\n";
        csv_open_ = true;
      } else {
        RCLCPP_WARN(get_logger(), "Could not open diagnostics CSV '%s' for writing.",
                    diagnostics_csv_path_.c_str());
      }
    }
    if (!csv_open_) {
      return;
    }

    const gtsam::Point3 et = node.pose.translation();
    const gtsam::Quaternion eq = node.pose.rotation().toQuaternion();
    const gtsam::Velocity3& v = node.velocity;
    const gtsam::Vector3 ba = node.bias.accelerometer();
    const gtsam::Vector3 bg = node.bias.gyroscope();
    const Eigen::Matrix<double, 6, 6>& cov = node.covariance;
    const geometry_msgs::msg::Pose& gt = latest_gt_.pose;

    csv_ << std::setprecision(9) << kEval05SchemaVersion << ',' << t << ','
         << keyframe_id << ',' << event << ',' << et.x() << ',' << et.y() << ','
         << et.z() << ',' << eq.x() << ',' << eq.y() << ',' << eq.z() << ',' << eq.w()
         << ',' << v.x() << ',' << v.y() << ',' << v.z() << ',' << ba.x() << ','
         << ba.y() << ',' << ba.z() << ',' << bg.x() << ',' << bg.y() << ',' << bg.z()
         << ',' << gt.position.x << ',' << gt.position.y << ',' << gt.position.z << ','
         << gt.orientation.x << ',' << gt.orientation.y << ',' << gt.orientation.z
         << ',' << gt.orientation.w;
    // Upper triangle of the symmetric 6x6 pose marginal: row r, columns r..5.
    for (int r = 0; r < 6; ++r) {
      for (int c = r; c < 6; ++c) {
        csv_ << ',' << cov(r, c);
      }
    }
    // L5-21: trailing consistency-gate columns (extra; schema_version stays 2 so
    // EVAL-05 parsers that require only v2 columns keep working).
    csv_ << ',' << last_consist_chi2_ << ',' << graph::toString(last_consist_action_);
    // L5-10: trailing submap-rebuild-batch size, same additive convention.
    // -1 for every "keyframe" row and every rejected/gated closure.
    csv_ << ',' << lc_from << ',' << lc_to << ',' << submap_batch_size << '\n';
    csv_.flush();
  }

  // L5-12: fuse keyframes [match_index-w, match_index+w] (backend-optimized
  // poses, clamped below query_index) into one "frozen historical submap" cloud
  // in match's frame — the NDT verification target, replacing the single
  // match_cloud with a richer target (more support -> wider convergence basin,
  // same reasoning as the front-end's L5-07/08 scan-to-submap). "Frozen" = built
  // fresh, once, from the CURRENT estimate at verify time; a distinct, one-shot
  // construction from the front-end's moving active submap (see the L5_MICRO-
  // STORIES.md design note).
  CloudPtr buildLoopClosureWindowTarget(gtsam::Key match_key, gtsam::Key query_key,
                                        const gtsam::Values& est) const {
    const auto match_index = static_cast<int>(gtsam::Symbol(match_key).index());
    const auto query_index = static_cast<int>(gtsam::Symbol(query_key).index());
    const int lo = std::max(0, match_index - loop_submap_window_);
    const int hi = std::min(match_index + loop_submap_window_, query_index - 1);

    std::vector<LoopSubmapEntry> window;
    for (int idx = lo; idx <= hi; ++idx) {
      const gtsam::Key key = sym::X(static_cast<std::size_t>(idx));
      const auto cloud_it = keyframe_clouds_.find(key);
      if (cloud_it == keyframe_clouds_.end() || !est.exists(key)) {
        continue;
      }
      window.push_back(LoopSubmapEntry{
          cloud_it->second, est.at<gtsam::Pose3>(key).matrix().cast<float>()});
    }

    const Eigen::Matrix4f pose_ref =
        est.at<gtsam::Pose3>(match_key).matrix().cast<float>();
    return buildLoopClosureTarget(window, pose_ref, loop_submap_voxel_leaf_);
  }

  // SLAM-10/L5-12: for each candidate of the newest keyframe, run NDT verification
  // against a frozen historical submap. Accept the closest one that passes
  // (candidates are distance-sorted) — adding its BetweenFactor, re-optimizing,
  // and re-publishing — then stop (the estimate has shifted, so the remaining
  // stale guesses are no longer trustworthy). Log every verdict to
  // /slam/diagnostics; reject leaves the graph untouched.
  void processLoopClosures(const rclcpp::Time& stamp) {
    const gtsam::Values est = optimizer_.estimate();
    const gtsam::Key latest_key = sym::X(kf_index_);
    const auto candidates = finder_.findCandidates(est, latest_key);

    for (const LoopClosureCandidate& c : candidates) {
      const auto query_it = keyframe_clouds_.find(c.query_key);
      const auto match_it = keyframe_clouds_.find(c.match_key);
      if (query_it == keyframe_clouds_.end() || match_it == keyframe_clouds_.end()) {
        continue;  // no stored cloud (should not happen on the live pipeline)
      }

      const gtsam::Pose3 T_world_match = est.at<gtsam::Pose3>(c.match_key);
      const gtsam::Pose3 T_world_query = est.at<gtsam::Pose3>(c.query_key);
      const Eigen::Isometry3d init_guess =
          gtsamPoseToIsometry(T_world_match.inverse().compose(T_world_query));

      const CloudPtr target = buildLoopClosureWindowTarget(c.match_key, c.query_key, est);
      const VerificationResult v = verifier_.verify(query_it->second, target, init_guess);

      const auto q = static_cast<int>(gtsam::Symbol(c.query_key).index());
      const auto m = static_cast<int>(gtsam::Symbol(c.match_key).index());

      if (!v.accepted) {
        logLoopClosure(stamp, q, m, v.ndt_score, false, v.reason);
        continue;
      }

      // L5-12: χ² Mahalanobis gate before trusting the match — compares the loop
      // measurement's implied pose for query (x_ndt = T_world_match (+) relative)
      // against query's CURRENT pose-graph belief (x_pred = T_world_query, its own
      // marginal covariance). On this single-graph path T_world_query already
      // reflects the joint IMU+NDT fit, so it doubles as "the IMU-predicted pose"
      // — there is no separate one to preintegrate across a 10+ keyframe gap.
      // Reuses evaluateNdtConsistency (L5-21) unchanged; scoped like L5-21 to the
      // single-graph path (no co-located marginal on the dual-graph Pose3-only path).
      const gtsam::Pose3 relative = isometryToGtsamPose(v.relative_pose);
      const gtsam::Pose3 x_ndt = T_world_match.compose(relative);
      const graph::Matrix6 sigma_ndt_raw = v.noise_model->covariance();
      graph::Matrix6 sigma_for_factor = sigma_ndt_raw;
      bool add_loop = true;
      double loop_chi2 = -1.0;
      if (!dual_graph_enabled_ && loop_consistency_gate_.enabled) {
        const graph::Matrix6 sigma_x_pred = optimizer_.marginalCovariance(c.query_key);
        const graph::ConsistencyGateResult gate = graph::evaluateNdtConsistency(
            T_world_query, x_ndt, sigma_ndt_raw, sigma_x_pred, loop_consistency_gate_);
        loop_chi2 = gate.chi2;
        sigma_for_factor = gate.sigma_ndt_used;
        if (!gate.add_ndt) {
          add_loop = false;
          ++loop_consist_drop_count_;
        } else if (gate.action == graph::ConsistencyAction::Inflate) {
          ++loop_consist_inflate_count_;
        }
      }

      if (!add_loop) {
        logLoopClosure(stamp, q, m, v.ndt_score, false, "chi2_inconsistent", loop_chi2);
        continue;
      }

      // Accepted: add the cross-edge (Huber-wrapped when the gate is enabled,
      // same construction the odometry edge uses — L5-21) and re-optimize.
      const graph::OdometryFactorBuilder factor_builder;
      const gtsam::BetweenFactor<gtsam::Pose3> factor =
          loop_consistency_gate_.enabled
              ? factor_builder.create_robust_factor(c.match_key, c.query_key, relative,
                                                    sigma_for_factor,
                                                    loop_consistency_gate_.huber_k)
              : factor_builder.create_factor(c.match_key, c.query_key, relative,
                                             sigma_for_factor);
      optimizer_.add_loop_closure(factor);
      optimizer_.update();
      // EVAL-05: every keyframe spanned by the loop (match..query) gets its
      // Sigma_post tightened, so mark them for the cyan ellipsoid colour.
      const auto lo = static_cast<std::size_t>(std::min(m, q));
      const auto hi = static_cast<std::size_t>(std::max(m, q));
      for (std::size_t k = lo; k <= hi; ++k) {
        loop_affected_.insert(k);
      }
      loop_edges_.emplace_back(m, q);  // {match, query} for the RViz loop overlay
      publishAll(stamp);
      // L5-10: re-optimizing pulled EVERY keyframe in [lo, hi], not just the
      // newest, so the front-end's active submap (L5-07/09) would otherwise
      // keep stale pre-closure poses for its other window entries. L5-12
      // measured that republishing /slam/optimized_state once per historical
      // keyframe here — N independent messages through the front-end's
      // existing per-entry applyOptimizedPose (L5-19) — collapses the active
      // submap window (L5-19e's collapse-on-jump guard fires once per shifted
      // entry) and stalls NDT registration for the rest of the run (97-99/111
      // keyframes vs 111/111 without it; see L5_MICROSTORIES.md §L5-12). The
      // fix is to publish the whole range as ONE coordinated batch instead —
      // the front-end applies every entry atomically (applyOptimizedPoseBatch)
      // and rebuilds its target grid at most once, never collapsing the window.
      std::size_t submap_batch_size = 0;
      if (loop_submap_rebuild_enabled_) {
        submap_batch_size = publishOptimizedStateBatch(stamp, lo, hi);
      }
      logLoopClosure(stamp, q, m, v.ndt_score, true, "", loop_chi2,
                     static_cast<int>(submap_batch_size));
      RCLCPP_INFO(get_logger(),
                  "LOOP CLOSURE accepted: x%d -> x%d (score %.3f, chi2 %.3f); graph "
                  "re-optimized.",
                  q, m, v.ndt_score, loop_chi2);
      return;  // one closure per keyframe; estimate has changed.
    }
  }

  // Publish a loop-closure verdict to /slam/diagnostics; on accept also append a
  // "loop_closure" CSV row (its marginal_cov_trace should step DOWN vs the
  // preceding keyframe row — ARCHITECTURE §9). `chi2_gate` is the L5-12 χ²
  // Mahalanobis-gate value (-1 when the gate did not run this candidate).
  // `submap_batch_size` (L5-10) is the number of keyframes published via
  // publishOptimizedStateBatch for this closure (-1 = not accepted /
  // loop_submap_rebuild_enabled_ off; this is the back-end's OWN record of
  // what it sent, not confirmation of what the front-end's separate process
  // did with it — see onOptimizedStateBatch's own log for that side).
  void logLoopClosure(const rclcpp::Time& stamp, int query, int match, double score,
                      bool accepted, const std::string& reason,
                      double chi2_gate = -1.0, int submap_batch_size = -1) {
    const gtsam::Key latest_key = sym::X(kf_index_);
    const double chi2 = optimizer_.chi2();
    const double cov_trace = optimizer_.marginalCovPositionTrace(latest_key);

    std::ostringstream json;
    json << std::setprecision(9) << "{\"t\": " << stamp.seconds()
         << ", \"keyframe_id\": " << static_cast<int>(kf_index_)
         << ", \"chi2\": " << chi2 << ", \"marginal_cov_trace\": " << cov_trace
         << ", \"loop_closure\": {\"query\": " << query << ", \"match\": " << match
         << ", \"score\": " << score << ", \"chi2_gate\": " << chi2_gate
         << ", \"accepted\": " << (accepted ? "true" : "false");
    if (!accepted) {
      json << ", \"reason\": \"" << reason << "\"";
    }
    if (accepted) {
      json << ", \"submap_batch_size\": " << submap_batch_size;
    }
    json << "}}";

    std_msgs::msg::String diag_msg;
    diag_msg.data = json.str();
    diagnostics_pub_->publish(diag_msg);

    if (accepted) {
      writeCsvRow(stamp.seconds(), static_cast<int>(kf_index_), "loop_closure",
                  currentNodeState(), std::to_string(match), std::to_string(query),
                  submap_batch_size);
    }
  }

  // Algorithm components (ROS-free)
  graph::KeyframePolicy kf_policy_;
  graph::OdometryAccumulator accumulator_;
  graph::GraphOptimizer optimizer_;
  LoopClosureCandidateFinder finder_;
  LoopClosureVerifier verifier_;

  // State
  bool bootstrapped_ = false;
  // L5-02: weak scaffolding priors holding V(i)/B(i) on the NDT-only path.
  gtsam::SharedNoiseModel scaffold_velocity_noise_;
  gtsam::SharedNoiseModel scaffold_bias_noise_;
  // L5-04: principled bootstrap priors for X(0)/V(0)/B(0) (sigmas kept for the
  // clean-init log line).
  gtsam::SharedNoiseModel bootstrap_pose_noise_;
  gtsam::SharedNoiseModel bootstrap_velocity_noise_;
  gtsam::SharedNoiseModel bootstrap_bias_noise_;
  double bootstrap_pose_sigma_ = 0.001;
  double bootstrap_velocity_sigma_ = 0.1;
  double bootstrap_accel_bias_sigma_ = 0.1;
  double bootstrap_gyro_bias_sigma_ = 0.01;
  gtsam::Pose3 last_pose_;
  rclcpp::Time last_kf_stamp_{0, 0, RCL_ROS_TIME};
  std::size_t kf_index_ = 0;

  // SLAM-10: per-keyframe clouds for NDT verification + latest buffered scan.
  std::unordered_map<gtsam::Key, CloudPtr> keyframe_clouds_;
  CloudPtr latest_cloud_;
  bool have_cloud_ = false;

  // EVAL-05: latest GT pose (CSV gt_* columns), accumulated GT trajectory, and
  // the set of keyframe indices spanned by an accepted loop closure (cyan ellipsoids).
  geometry_msgs::msg::PoseStamped latest_gt_;
  bool have_gt_ = false;
  nav_msgs::msg::Path gt_path_;
  std::set<std::size_t> loop_affected_;
  std::vector<std::pair<int, int>> loop_edges_;  // {match, query} per accepted closure

  // L5-01: IMU intake state (GtSample defined at the top of the private section).
  imu::PreintegrationConfig preint_config_;
  std::deque<imu::ImuSample> imu_buffer_;
  std::deque<GtSample> gt_history_;
  double diag_window_start_t_ = -1.0;

  // L5-03: persistent preintegrator for the IMU backbone (built from
  // preint_config_ in the ctor); only its PIM accumulation is reset() per
  // keyframe. L5-17: samples reach it through imu_window_, a raw deque keyed by
  // the message HEADER stamp — each window is cut at ndt_odom.header.stamp in
  // onNdtOdom (LIO-SAM's imuQueOpt pattern), never at callback arrival, and dt
  // comes from consecutive header stamps. imu_gap_in_interval_ latches L5-17e's
  // gap verdict until the next keyframe folds it into the no-IMU fallback.
  std::optional<imu::ImuPreintegrator> preint_;
  imu::ImuWindowBuffer imu_window_;
  bool imu_gap_in_interval_ = false;
  // DUAL-GRAPH immediate-feed dt anchor (header-stamp domain, L5-17): the side
  // estimator integrates per sample in onImu, so it needs its own previous-stamp
  // scalar. Kept separate from imu_window_ (single-graph path only).
  double last_dual_imu_stamp_ = -1.0;
  // DUAL-GRAPH: the side IMU graph (constructed only when dual_graph_enabled_).
  // It owns its own iSAM2, its own preintegrator, and the X/V/B chain; the
  // mapping optimizer_ above stays Pose3-only while it is alive.
  std::optional<imu::ImuStateEstimator> estimator_;
  std::size_t last_estimator_failures_ = 0;
  // Latest per-edge chi2 (2*factor.error) + optimized bias, for /slam/diagnostics
  // + the per-keyframe log. -1 = not applicable this keyframe (e.g. an NDT-only
  // edge has no IMU/bias chi2).
  double last_imu_chi2_ = -1.0;
  double last_ndt_chi2_ = -1.0;
  double last_bias_chi2_ = -1.0;
  // L5-21: last keyframe's consistency-gate verdict + running counters.
  double last_consist_chi2_ = -1.0;
  graph::ConsistencyAction last_consist_action_{graph::ConsistencyAction::None};
  std::size_t consist_drop_count_ = 0;
  std::size_t consist_inflate_count_ = 0;
  gtsam::imuBias::ConstantBias last_opt_bias_;
  // L5-12: same gate machinery, applied to loop-closure candidates instead of
  // odometry edges (Mahalanobis test against the query's current pose marginal).
  std::size_t loop_consist_drop_count_ = 0;
  std::size_t loop_consist_inflate_count_ = 0;

  // Params
  std::string map_frame_;
  std::string odom_frame_;
  std::string ndt_odom_topic_;
  std::string scan_topic_;
  std::string gt_topic_;
  std::string imu_topic_;
  double imu_buffer_seconds_ = 5.0;
  double imu_max_dt_gap_s_ = 0.02;
  bool imu_diagnostic_enabled_ = false;
  double imu_diagnostic_window_s_ = 1.0;
  bool imu_factor_enabled_ = true;
  bool ndt_factor_enabled_ = true;
  graph::ConsistencyGateConfig consistency_gate_;
  // L5-12: frozen historical submap window (half-width) + downsample leaf, and
  // the loop-closure χ² gate (own, stricter-by-default config; see ctor).
  int loop_submap_window_ = 2;
  double loop_submap_voxel_leaf_ = 0.3;
  graph::ConsistencyGateConfig loop_consistency_gate_;
  // L5-10: coordinated loop-closure submap-rebuild feed + its ablation flag.
  std::string optimized_state_batch_topic_;
  bool loop_submap_rebuild_enabled_ = true;
  bool dual_graph_enabled_ = false;
  double dual_correction_rot_sigma_ = 0.05;
  double dual_correction_pos_sigma_ = 0.1;
  int dual_reset_key_interval_ = 100;
  double sphere_diameter_m_ = 0.15;
  double edge_line_width_m_ = 0.03;
  double covariance_marker_scale_ = 1.0;
  double label_height_m_ = 0.4;
  std::string diagnostics_csv_path_;

  // SLAM-11 diagnostics CSV (RAII: closed on node destruction).
  std::ofstream csv_;
  bool csv_open_ = false;

  // ROS
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gt_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr keyframes_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr edges_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr loop_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr optimized_odom_pub_;
  rclcpp::Publisher<graph_slam_msgs::msg::OptimizedState>::SharedPtr optimized_state_pub_;
  rclcpp::Publisher<graph_slam_msgs::msg::OptimizedStateBatch>::SharedPtr
      optimized_state_batch_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr diagnostics_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ellipsoids_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr gt_path_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Factory (used by main and by the gtest smoke test)
// ---------------------------------------------------------------------------

std::shared_ptr<rclcpp::Node> createGraphBackendNode(rclcpp::NodeOptions options) {
  return std::make_shared<GraphBackendNode>(std::move(options));
}

}  // namespace graph_slam
