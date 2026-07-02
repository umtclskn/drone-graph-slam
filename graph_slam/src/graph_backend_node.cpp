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
#include <fstream>
#include <iomanip>
#include <memory>
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
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "graph_slam/eval/graph_visualizer.hpp"
#include "graph_slam/graph/graph_bootstrap.hpp"
#include "graph_slam/graph/graph_optimizer.hpp"
#include "graph_slam/graph/keyframe_policy.hpp"
#include "graph_slam/graph/odometry_accumulator.hpp"
#include "graph_slam/graph/odometry_factor_builder.hpp"
#include "graph_slam/loop/loop_closure_candidate_finder.hpp"
#include "graph_slam/loop/loop_closure_verifier.hpp"
#include "graph_slam/point_types.hpp"

namespace graph_slam {
namespace {

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

    path_pub_ = create_publisher<nav_msgs::msg::Path>("/slam/graph_path", 10);
    keyframes_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>("/slam/keyframes", 10);
    edges_pub_ = create_publisher<visualization_msgs::msg::Marker>("/slam/graph_edges", 10);
    loop_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/slam/loop_closures", 10);
    optimized_odom_pub_ =
        create_publisher<nav_msgs::msg::Odometry>("/slam/optimized_odom", 10);
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

    RCLCPP_INFO(get_logger(),
                "graph_backend up. Subscribing to '%s'. KF thresholds: %.2f m / %.2f rad / "
                "%.1f s. Publishing /slam/graph_path, /slam/keyframes, /slam/graph_edges, "
                "/slam/optimized_odom. TF: %s → %s.",
                ndt_odom_topic_.c_str(), kf_dist, kf_angle, kf_time, map_frame_.c_str(),
                odom_frame_.c_str());
  }

 private:
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
  }

  void onNdtOdom(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
    const rclcpp::Time current_stamp(msg->header.stamp);
    const gtsam::Pose3 current_pose = navPoseToGtsam(msg->pose.pose);

    std::array<double, 36> cov_arr{};
    std::copy(msg->pose.covariance.begin(), msg->pose.covariance.end(), cov_arr.begin());
    const gtsam::Matrix66 sigma_meas = navCovToGtsam(cov_arr);

    // -----------------------------------------------------------------------
    // Bootstrap: first message → insert x0 + PriorFactor
    // -----------------------------------------------------------------------
    if (!bootstrapped_) {
      const gtsam::Key x0 = gtsam::Symbol('x', 0);
      const auto prior_noise =
          gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(0.001));
      const graph::GraphBootstrap bootstrap;
      const auto boot = bootstrap.create(current_pose, prior_noise);
      const auto* prior =
          dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(boot.graph.at(0).get());
      optimizer_.add_prior(*prior, x0, boot.values.at<gtsam::Pose3>(x0));
      optimizer_.update();
      last_pose_ = current_pose;
      last_kf_stamp_ = current_stamp;
      kf_index_ = 0;
      bootstrapped_ = true;
      storeKeyframeCloud(x0);
      publishAll(current_stamp);
      emitDiagnostics(current_stamp);
      RCLCPP_INFO(get_logger(), "Graph bootstrapped at x0 (%.2f, %.2f, %.2f).",
                  current_pose.translation().x(), current_pose.translation().y(),
                  current_pose.translation().z());
      return;
    }

    // -----------------------------------------------------------------------
    // Accumulate scan-to-scan step (ARCHITECTURE §6 concept #2)
    // -----------------------------------------------------------------------
    const gtsam::Pose3 delta_step = last_pose_.inverse().compose(current_pose);
    accumulator_.add_measurement(delta_step, sigma_meas);
    last_pose_ = current_pose;

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
    const gtsam::Key x_prev = gtsam::Symbol('x', prev_idx);
    const gtsam::Key x_cur = gtsam::Symbol('x', kf_index_);

    const gtsam::Pose3 prev_optimized =
        optimizer_.estimate().at<gtsam::Pose3>(x_prev);
    const gtsam::Pose3 x_cur_init = prev_optimized.compose(acc.delta);

    const graph::OdometryFactorBuilder factor_builder;
    const auto factor =
        factor_builder.create_factor(x_prev, x_cur, acc.delta, acc.covariance);
    optimizer_.add_odometry(factor, x_cur, x_cur_init);
    optimizer_.update();
    accumulator_.reset();
    last_kf_stamp_ = current_stamp;
    storeKeyframeCloud(x_cur);

    publishAll(current_stamp);
    emitDiagnostics(current_stamp);
    RCLCPP_INFO(get_logger(), "KF x%zu added (%.2f, %.2f, %.2f) | total nodes: %zu.",
                kf_index_, current_pose.translation().x(), current_pose.translation().y(),
                current_pose.translation().z(), kf_index_ + 1);

    // SLAM-10: verify any loop-closure candidates for the new keyframe and, on a
    // strong + well-conditioned NDT match, add a BetweenFactor that re-optimizes
    // the whole graph (map->odom jumps, Sigma_post shrinks for the looped poses).
    processLoopClosures(current_stamp);
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
      covs.push_back(optimizer_.marginalCovariance(gtsam::Symbol('x', i)));
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

  // SLAM-11: publish optimizer health (chi2 + latest Sigma_post position-trace)
  // to /slam/diagnostics (JSON line) and append one row to the CSV log.
  void emitDiagnostics(const rclcpp::Time& stamp) {
    const gtsam::Key latest_key = gtsam::Symbol('x', kf_index_);
    const gtsam::Values est = optimizer_.estimate();
    const double chi2 = optimizer_.chi2();
    const double cov_trace = optimizer_.marginalCovPositionTrace(latest_key);
    const std::size_t num_factors = est.size();  // == num nodes
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

    std::ostringstream json;
    json << std::setprecision(9) << "{\"t\": " << stamp.seconds()
         << ", \"keyframe_id\": " << keyframe_id
         << ", \"num_factors\": " << num_factors
         << ", \"num_values\": " << num_factors
         << ", \"chi2\": " << chi2
         << ", \"marginal_cov_trace\": " << cov_trace
         << ", \"loop_candidates\": " << lc.str() << "}";

    std_msgs::msg::String diag_msg;
    diag_msg.data = json.str();
    diagnostics_pub_->publish(diag_msg);

    // EVAL-05 §9 row: optimized pose + GT + full 6x6 Sigma_post upper triangle.
    const gtsam::Pose3 est_pose = est.at<gtsam::Pose3>(latest_key);
    const Eigen::Matrix<double, 6, 6> cov = optimizer_.marginalCovariance(latest_key);
    writeCsvRow(stamp.seconds(), keyframe_id, "keyframe", est_pose, cov, "", "");
  }

  // EVAL-05 / ARCHITECTURE §9 logging contract: one CSV row per diagnostic event.
  // `event` is "keyframe" or "loop_closure"; lc_from/lc_to are filled only for
  // closures. Columns (exact §9 order): t, keyframe_id, event, est_x..est_qw,
  // gt_x..gt_qw, then the upper triangle of the 6x6 Sigma_post (GTSAM tangent
  // order rx,ry,rz,x,y,z), then lc_from, lc_to.
  void writeCsvRow(double t, int keyframe_id, const std::string& event,
                   const gtsam::Pose3& est, const Eigen::Matrix<double, 6, 6>& cov,
                   const std::string& lc_from, const std::string& lc_to) {
    if (!csv_open_) {
      csv_.open(diagnostics_csv_path_, std::ios::out | std::ios::trunc);
      if (csv_.is_open()) {
        csv_ << "t,keyframe_id,event,"
                "est_x,est_y,est_z,est_qx,est_qy,est_qz,est_qw,"
                "gt_x,gt_y,gt_z,gt_qx,gt_qy,gt_qz,gt_qw,"
                "cov_00,cov_01,cov_02,cov_03,cov_04,cov_05,"
                "cov_11,cov_12,cov_13,cov_14,cov_15,"
                "cov_22,cov_23,cov_24,cov_25,"
                "cov_33,cov_34,cov_35,"
                "cov_44,cov_45,cov_55,"
                "lc_from,lc_to\n";
        csv_open_ = true;
      } else {
        RCLCPP_WARN(get_logger(), "Could not open diagnostics CSV '%s' for writing.",
                    diagnostics_csv_path_.c_str());
      }
    }
    if (!csv_open_) {
      return;
    }

    const gtsam::Point3 et = est.translation();
    const gtsam::Quaternion eq = est.rotation().toQuaternion();
    const geometry_msgs::msg::Pose& gt = latest_gt_.pose;

    csv_ << std::setprecision(9) << t << ',' << keyframe_id << ',' << event << ','
         << et.x() << ',' << et.y() << ',' << et.z() << ',' << eq.x() << ',' << eq.y()
         << ',' << eq.z() << ',' << eq.w() << ',' << gt.position.x << ','
         << gt.position.y << ',' << gt.position.z << ',' << gt.orientation.x << ','
         << gt.orientation.y << ',' << gt.orientation.z << ',' << gt.orientation.w;
    // Upper triangle of the symmetric 6x6: row r, columns r..5.
    for (int r = 0; r < 6; ++r) {
      for (int c = r; c < 6; ++c) {
        csv_ << ',' << cov(r, c);
      }
    }
    csv_ << ',' << lc_from << ',' << lc_to << '\n';
    csv_.flush();
  }

  // SLAM-10: for each candidate of the newest keyframe, run NDT verification.
  // Accept the closest one that passes (candidates are distance-sorted) — adding
  // its BetweenFactor, re-optimizing, and re-publishing — then stop (the estimate
  // has shifted, so the remaining stale guesses are no longer trustworthy). Log
  // every verdict to /slam/diagnostics; reject leaves the graph untouched.
  void processLoopClosures(const rclcpp::Time& stamp) {
    const gtsam::Values est = optimizer_.estimate();
    const gtsam::Key latest_key = gtsam::Symbol('x', kf_index_);
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

      const VerificationResult v =
          verifier_.verify(query_it->second, match_it->second, init_guess);

      const auto q = static_cast<int>(gtsam::Symbol(c.query_key).index());
      const auto m = static_cast<int>(gtsam::Symbol(c.match_key).index());

      if (!v.accepted) {
        logLoopClosure(stamp, q, m, v.ndt_score, false, v.reason);
        continue;
      }

      // Accepted: add the cross-edge and re-optimize the whole graph.
      const gtsam::Pose3 relative = isometryToGtsamPose(v.relative_pose);
      const gtsam::BetweenFactor<gtsam::Pose3> factor(c.match_key, c.query_key, relative,
                                                      v.noise_model);
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
      logLoopClosure(stamp, q, m, v.ndt_score, true, "");
      RCLCPP_INFO(get_logger(),
                  "LOOP CLOSURE accepted: x%d -> x%d (score %.3f); graph re-optimized.",
                  q, m, v.ndt_score);
      return;  // one closure per keyframe; estimate has changed.
    }
  }

  // Publish a loop-closure verdict to /slam/diagnostics; on accept also append a
  // "loop_closure" CSV row (its marginal_cov_trace should step DOWN vs the
  // preceding keyframe row — ARCHITECTURE §9).
  void logLoopClosure(const rclcpp::Time& stamp, int query, int match, double score,
                      bool accepted, const std::string& reason) {
    const gtsam::Key latest_key = gtsam::Symbol('x', kf_index_);
    const double chi2 = optimizer_.chi2();
    const double cov_trace = optimizer_.marginalCovPositionTrace(latest_key);

    std::ostringstream json;
    json << std::setprecision(9) << "{\"t\": " << stamp.seconds()
         << ", \"keyframe_id\": " << static_cast<int>(kf_index_)
         << ", \"chi2\": " << chi2 << ", \"marginal_cov_trace\": " << cov_trace
         << ", \"loop_closure\": {\"query\": " << query << ", \"match\": " << match
         << ", \"score\": " << score
         << ", \"accepted\": " << (accepted ? "true" : "false");
    if (!accepted) {
      json << ", \"reason\": \"" << reason << "\"";
    }
    json << "}}";

    std_msgs::msg::String diag_msg;
    diag_msg.data = json.str();
    diagnostics_pub_->publish(diag_msg);

    if (accepted) {
      const gtsam::Pose3 est_pose = optimizer_.estimate().at<gtsam::Pose3>(latest_key);
      const Eigen::Matrix<double, 6, 6> cov = optimizer_.marginalCovariance(latest_key);
      writeCsvRow(stamp.seconds(), static_cast<int>(kf_index_), "loop_closure", est_pose,
                  cov, std::to_string(match), std::to_string(query));
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

  // Params
  std::string map_frame_;
  std::string odom_frame_;
  std::string ndt_odom_topic_;
  std::string scan_topic_;
  std::string gt_topic_;
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
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr keyframes_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr edges_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr loop_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr optimized_odom_pub_;
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
