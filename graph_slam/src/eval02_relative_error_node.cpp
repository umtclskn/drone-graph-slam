// EVAL-02: relative transform error (Sprint 1 slice) + dead-reckoned trajectory
// with ATE/RPE (Sprint 2 full slice) vs ground truth.
//
// For consecutive LiDAR scan pairs (i -> i+stride) the node registers them with
// the graph_slam pipeline (Preprocessor -> NdtVoxelGrid -> NdtRegistrar), looks
// up the ground-truth motion over the same interval from /ground_truth/pose
// (EVAL-01), and writes one CSV row per pair: translation error (m) + rotation
// error (deg) from graph_slam::compute, plus the convergence flag and NDT-11
// verdict (-> scripts/eval02_relative_error.py).
//
// Sprint 2 full slice: the same per-pair NDT transforms are CHAINED into a
// dead-reckoned NDT-odometry trajectory (P_0 = first GT pose, P_{k+1} = P_k *
// T_ndt) and compared against the GT trajectory. The node writes a per-scan
// trajectory CSV (est + GT pose) and reports ATE + RPE (graph_slam ROS-free core
// trajectory_metrics) -> scripts/eval02_trajectory.py for the 2D/3D/per-axis
// plots. This compounding is the front-end-only slice of SLAM-03 (SPRINTS.md);
// the full OdometryEstimator with covariance propagation is Sprint 3.
//
// Design: during replay the scan callback only PREPROCESSES + buffers (cheap),
// and GT is buffered; registration + exact nearest-stamp GT matching run in a
// finalize pass once the bag stops (idle watchdog). This decouples the heavy NDT
// solve from real-time, so GT delivery is never starved by a blocked executor.
//
// graph_slam's algorithm classes stay ROS-free; only THIS executable links ROS.

#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "graph_slam/eval/relative_error.hpp"
#include "graph_slam/eval/trajectory_metrics.hpp"
#include "graph_slam/initial_guess.hpp"
#include "graph_slam/ndt_registrar.hpp"
#include "graph_slam/ndt_voxel_grid.hpp"
#include "graph_slam/point_types.hpp"
#include "graph_slam/preprocessor.hpp"
#include "graph_slam/registration_gate.hpp"

namespace graph_slam {
namespace {

Eigen::Isometry3f poseToIsometry(const geometry_msgs::msg::Pose& p) {
  Eigen::Isometry3f t = Eigen::Isometry3f::Identity();
  t.translation() =
      Eigen::Vector3f(static_cast<float>(p.position.x), static_cast<float>(p.position.y),
                      static_cast<float>(p.position.z));
  const Eigen::Quaternionf q(
      static_cast<float>(p.orientation.w), static_cast<float>(p.orientation.x),
      static_cast<float>(p.orientation.y), static_cast<float>(p.orientation.z));
  t.linear() = q.normalized().toRotationMatrix();
  return t;
}

// "x,y,z,qx,qy,qz,qw" for a pose, the trajectory-CSV column convention.
std::string poseToCsv(const Eigen::Isometry3f& p) {
  const Eigen::Vector3f t = p.translation();
  const Eigen::Quaternionf q(p.linear());
  std::ostringstream os;
  os << t.x() << ',' << t.y() << ',' << t.z() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ','
     << q.w();
  return os.str();
}

class Eval02RelativeErrorNode : public rclcpp::Node {
 public:
  Eval02RelativeErrorNode() : rclcpp::Node("eval02_relative_error") {
    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/x500/lidar_3d/points");
    gt_topic_ = declare_parameter<std::string>("gt_topic", "/ground_truth/pose");
    csv_path_ = declare_parameter<std::string>("csv_path", "eval02_pairs.csv");
    // Sprint 2 full slice: per-scan dead-reckoned trajectory CSV (empty = skip)
    // and the index gap (in trajectory nodes) the RPE is computed over.
    traj_csv_path_ = declare_parameter<std::string>("traj_csv_path", "eval02_trajectory.csv");
    rpe_delta_ = static_cast<int>(declare_parameter<int>("rpe_delta", 10));
    voxel_leaf_ = declare_parameter<double>("voxel_leaf", 0.2);
    ndt_resolution_ = declare_parameter<double>("ndt_resolution", 1.0);
    gt_tol_ms_ = declare_parameter<double>("gt_tol_ms", 50.0);
    // NDT optimiser + gate knobs (INFRA-02). Defaults equal the struct defaults,
    // so a YAML edit changes behaviour without a rebuild.
    ndt_cfg_.max_iterations = declare_parameter<int>("ndt_max_iterations", ndt_cfg_.max_iterations);
    ndt_cfg_.step_epsilon = declare_parameter<double>("ndt_step_epsilon", ndt_cfg_.step_epsilon);
    ndt_cfg_.outlier_ratio = declare_parameter<double>("ndt_outlier_ratio", ndt_cfg_.outlier_ratio);
    ndt_cfg_.regularization =
        declare_parameter<double>("ndt_regularization", ndt_cfg_.regularization);
    // Sigma_meas knobs (NDT-12). Defaults equal the struct defaults.
    ndt_cfg_.hessian_lambda =
        declare_parameter<double>("ndt_hessian_lambda", ndt_cfg_.hessian_lambda);
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
    // Register pair (i, i+stride) for i = 0, stride, 2*stride, ...; stride=1 is
    // every consecutive pair. max_pairs caps the count (0 = all). idle_sec is how
    // long with no new scan before the finalize pass runs.
    stride_ = static_cast<int>(declare_parameter<int>("stride", 1));
    max_pairs_ = static_cast<int>(declare_parameter<int>("max_pairs", 0));
    idle_sec_ = declare_parameter<double>("idle_sec", 3.0);
    // Diagnostic A/B knob (NOT the default pipeline): seed align() with the EKF2
    // relative-motion prior (NDT-08) instead of identity. Default false keeps the
    // identity-guess behaviour every prior EVAL-02 result was produced with.
    use_ekf2_prior_ = declare_parameter<bool>("use_ekf2_prior", false);
    ekf2_topic_ = declare_parameter<std::string>("ekf2_topic", "/odometry/ekf2");

    gt_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        gt_topic_, rclcpp::QoS(rclcpp::KeepLast(2000)).reliable(),
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) { onGt(msg); });
    scan_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { onScan(msg); });
    if (use_ekf2_prior_) {
      ekf2_sub_ = create_subscription<nav_msgs::msg::Odometry>(
          ekf2_topic_, rclcpp::QoS(rclcpp::KeepLast(2000)).reliable(),
          [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { onEkf2(msg); });
    }
    watchdog_ = create_wall_timer(std::chrono::seconds(1), [this]() { onWatchdog(); });
    last_activity_ = Clock::now();

    RCLCPP_INFO(get_logger(),
                "eval02 up. Collecting scans '%s' + GT '%s'; finalize after %.0fs idle "
                "-> CSV '%s' (stride=%d, gt_tol=%.0f ms). init_guess=%s.",
                lidar_topic_.c_str(), gt_topic_.c_str(), idle_sec_, csv_path_.c_str(), stride_,
                gt_tol_ms_, use_ekf2_prior_ ? ekf2_topic_.c_str() : "identity");
  }

 private:
  using Clock = std::chrono::steady_clock;

  struct Scan {
    CloudPtr cloud;  // preprocessed
    double stamp;    // seconds
  };

  // One node of the dead-reckoned trajectory: the compounded NDT estimate at a
  // scan stamp, plus the nearest GT pose when one was matched within tolerance.
  struct TrajNode {
    double stamp = 0.0;
    Eigen::Isometry3f est = Eigen::Isometry3f::Identity();
    bool has_gt = false;
    Eigen::Isometry3f gt = Eigen::Isometry3f::Identity();
  };

  static double toSec(const builtin_interfaces::msg::Time& t) {
    return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
  }

  void onGt(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg) {
    gt_buf_.emplace_back(toSec(msg->header.stamp), poseToIsometry(msg->pose));
    last_activity_ = Clock::now();
  }

  // EKF2 relative-motion prior source (NDT-08), only subscribed when
  // use_ekf2_prior_ is set. Buffered like GT; the per-pair guess is built in
  // finalize() from the two nearest-stamp poses. Does NOT update last_activity_
  // so the idle watchdog tracks the LiDAR/GT replay, not this side channel.
  void onEkf2(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
    ekf2_buf_.emplace_back(toSec(msg->header.stamp), poseToIsometry(msg->pose.pose));
  }

  // Light: preprocess + buffer only, so the executor never blocks during replay.
  void onScan(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
    auto raw = std::make_shared<Cloud>();
    pcl::fromROSMsg(*msg, *raw);
    PreprocessConfig pre_cfg;
    pre_cfg.voxel_leaf = static_cast<float>(voxel_leaf_);
    scans_.push_back({Preprocessor{pre_cfg}.process(raw), toSec(msg->header.stamp)});
    last_activity_ = Clock::now();
  }

  void onWatchdog() {
    if (finalized_ || scans_.empty()) {
      return;
    }
    const double idle = std::chrono::duration<double>(Clock::now() - last_activity_).count();
    if (idle >= idle_sec_) {
      finalize();
    }
  }

  // Nearest buffered (stamp, pose) to `stamp`; reports the |dt| it matched at.
  // Shared by the GT buffer and the EKF2-prior buffer.
  static std::optional<std::pair<Eigen::Isometry3f, double>> nearestPose(
      const std::vector<std::pair<double, Eigen::Isometry3f>>& buf, double stamp) {
    if (buf.empty()) {
      return std::nullopt;
    }
    const auto best = std::min_element(
        buf.begin(), buf.end(), [stamp](const auto& a, const auto& b) {
          return std::abs(a.first - stamp) < std::abs(b.first - stamp);
        });
    return std::make_pair(best->second, std::abs(best->first - stamp));
  }

  void finalize() {
    finalized_ = true;
    std::ofstream csv(csv_path_);
    if (!csv) {
      RCLCPP_ERROR(get_logger(), "Cannot open CSV '%s' for writing.", csv_path_.c_str());
      return;
    }
    csv << "pair_id,stamp_i,stamp_j,t_err_m,r_err_deg,converged,verdict,fitness,"
           "gt_dt_i_ms,gt_dt_j_ms\n";

    const int stride = std::max(1, stride_);
    const double tol_s = gt_tol_ms_ * 1e-3;
    int pair_id = 0;
    int written = 0;
    int skipped = 0;
    int sigma_hessian = 0;   // NDT-12: Sigma_meas from the cost Hessian
    int sigma_fallback = 0;  // NDT-12: degenerate H -> diagonal fallback
    int prior_used = 0;      // pairs seeded with the EKF2 prior (NDT-08)
    int prior_missing = 0;   // use_ekf2_prior but no EKF2 in tol -> identity fallback

    // Sprint 2 full slice: dead-reckoned NDT-odometry trajectory. `est_pose` is the
    // running compounded estimate; `traj` holds one node per scan (est + matched GT).
    std::vector<TrajNode> traj;
    Eigen::Isometry3f est_pose = Eigen::Isometry3f::Identity();
    for (std::size_t i = 0; i + stride < scans_.size(); i += stride, ++pair_id) {
      if (max_pairs_ > 0 && written >= max_pairs_) {
        break;
      }
      const Scan& scan_i = scans_[i];
      const Scan& scan_j = scans_[i + stride];

      // Register prev(target) <- curr(source): align returns T mapping source(j)
      // onto target(i), i.e. T_rel = pose_i^{-1} * pose_j -- same convention as
      // T_gt = GT_i^{-1} * GT_j.
      NdtGridConfig grid_cfg;
      grid_cfg.resolution = ndt_resolution_;
      NdtVoxelGrid grid{grid_cfg};
      grid.build(scan_i.cloud);

      // NDT-08 A/B (use_ekf2_prior): seed align() with the EKF2 relative-motion
      // prior T = ekf2_i^{-1} * ekf2_j (same convention as align's output and
      // T_gt) when both endpoints have an EKF2 pose within tolerance; otherwise
      // fall back to identity for this pair. Default false -> always identity.
      Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
      if (use_ekf2_prior_) {
        const auto pr_i = nearestPose(ekf2_buf_, scan_i.stamp);
        const auto pr_j = nearestPose(ekf2_buf_, scan_j.stamp);
        if (pr_i && pr_j && pr_i->second <= tol_s && pr_j->second <= tol_s) {
          init_guess = relativePoseGuess(pr_i->first.matrix(), pr_j->first.matrix());
          ++prior_used;
        } else {
          ++prior_missing;
        }
      }
      const RegistrationResult res = NdtRegistrar{ndt_cfg_}.align(grid, scan_j.cloud, init_guess);
      const RegistrationStatus verdict = evaluateRegistration(res, gate_cfg_);

      // NDT-12: every accepted-or-not result now carries Sigma_meas. Surface which
      // path produced it (Hessian vs geometry-aware fallback) so a degenerate
      // scene is visible on a live `ros2 run`, not just inferable from the CSV.
      if (res.sigma_from_hessian) {
        ++sigma_hessian;
      } else {
        ++sigma_fallback;
        RCLCPP_WARN(get_logger(),
                    "pair %d: Sigma_meas FALLBACK (degenerate Hessian); diagonal "
                    "sigma_rot=%.3f rad, sigma_t=%.3f m used.",
                    pair_id, ndt_cfg_.sigma_rot_fallback, ndt_cfg_.sigma_t_fallback);
      }

      const auto gt_i = nearestPose(gt_buf_, scan_i.stamp);
      const auto gt_j = nearestPose(gt_buf_, scan_j.stamp);
      const bool i_has_gt = gt_i && gt_i->second <= tol_s;
      const bool j_has_gt = gt_j && gt_j->second <= tol_s;

      // Compound the trajectory regardless of the per-pair GT match: T_ndt maps
      // source(j) onto target(i) (T_rel = pose_i^{-1} pose_j), so pose_j =
      // pose_i * T_ndt. Seed pose 0 at the first node's GT when available so the
      // est and GT trajectories share frame 0 and ATE reads as pure drift.
      const Eigen::Isometry3f T_ndt(res.transform);
      if (traj.empty()) {
        est_pose = i_has_gt ? gt_i->first : Eigen::Isometry3f::Identity();
        traj.push_back({scan_i.stamp, est_pose, i_has_gt,
                        i_has_gt ? gt_i->first : Eigen::Isometry3f::Identity()});
      }
      est_pose = est_pose * T_ndt;
      traj.push_back({scan_j.stamp, est_pose, j_has_gt,
                      j_has_gt ? gt_j->first : Eigen::Isometry3f::Identity()});

      if (!i_has_gt || !j_has_gt) {
        ++skipped;
        continue;
      }

      const Eigen::Isometry3f T_gt = gt_i->first.inverse() * gt_j->first;
      const RelativeErrorResult err = compute(T_ndt, T_gt);

      csv << pair_id << ',' << scan_i.stamp << ',' << scan_j.stamp << ',' << err.t_err_m << ','
          << err.r_err_deg << ',' << (res.converged ? 1 : 0) << ',' << toString(verdict) << ','
          << res.fitness_score << ',' << gt_i->second * 1e3 << ',' << gt_j->second * 1e3 << '\n';
      ++written;
    }
    csv.flush();
    RCLCPP_INFO(get_logger(),
                "finalize: %d scans, %d GT poses -> %d pairs written, %d skipped (no GT in "
                "%.0f ms). CSV: %s",
                static_cast<int>(scans_.size()), static_cast<int>(gt_buf_.size()), written, skipped,
                gt_tol_ms_, csv_path_.c_str());
    RCLCPP_INFO(get_logger(), "NDT-12 Sigma_meas: %d from Hessian, %d fallback (of %d registered).",
                sigma_hessian, sigma_fallback, sigma_hessian + sigma_fallback);
    if (use_ekf2_prior_) {
      RCLCPP_INFO(get_logger(),
                  "NDT-08 prior: %d pairs seeded from '%s', %d fell back to identity "
                  "(no EKF2 within %.0f ms). %zu EKF2 poses buffered.",
                  prior_used, ekf2_topic_.c_str(), prior_missing, gt_tol_ms_, ekf2_buf_.size());
    }

    writeTrajectory(traj);
  }

  // Sprint 2 full slice: write the per-scan dead-reckoned trajectory and report
  // ATE/RPE against GT. ATE/RPE are computed by the ROS-free graph_slam core; only
  // the matched-GT subsequence feeds the metrics (est and GT must be index-aligned).
  void writeTrajectory(const std::vector<TrajNode>& traj) {
    if (traj_csv_path_.empty() || traj.empty()) {
      return;
    }

    Trajectory est_v;
    Trajectory gt_v;
    for (const auto& n : traj) {
      if (n.has_gt) {
        est_v.push_back(n.est);
        gt_v.push_back(n.gt);
      }
    }
    const AbsoluteTrajectoryError ate = absoluteTrajectoryError(est_v, gt_v);
    const RelativePoseError rpe =
        relativePoseError(est_v, gt_v, static_cast<std::size_t>(std::max(1, rpe_delta_)));

    std::ofstream csv(traj_csv_path_);
    if (!csv) {
      RCLCPP_ERROR(get_logger(), "Cannot open trajectory CSV '%s' for writing.",
                   traj_csv_path_.c_str());
      return;
    }
    // Summary as comment header lines (scripts/eval02_trajectory.py parses these).
    csv << "# ate_trans_m=" << ate.trans_rmse_m << '\n'
        << "# ate_rot_deg=" << ate.rot_rmse_deg << '\n'
        << "# rpe_trans_m=" << rpe.trans_rmse_m << '\n'
        << "# rpe_rot_deg=" << rpe.rot_rmse_deg << '\n'
        << "# rpe_delta=" << rpe.delta << '\n'
        << "# nodes=" << traj.size() << '\n'
        << "# nodes_with_gt=" << ate.count << '\n';
    csv << "idx,stamp,est_x,est_y,est_z,est_qx,est_qy,est_qz,est_qw,"
           "gt_x,gt_y,gt_z,gt_qx,gt_qy,gt_qz,gt_qw\n";
    int idx = 0;
    for (const auto& n : traj) {
      csv << idx++ << ',' << n.stamp << ',' << poseToCsv(n.est) << ',';
      if (n.has_gt) {
        csv << poseToCsv(n.gt);
      } else {
        csv << ",,,,,,";  // GT absent: leave the 7 GT columns empty
      }
      csv << '\n';
    }
    csv.flush();

    RCLCPP_INFO(get_logger(),
                "trajectory: %d nodes (%zu with GT) -> %s | ATE trans=%.3f m rot=%.3f deg | "
                "RPE(delta=%zu) trans=%.3f m rot=%.3f deg",
                static_cast<int>(traj.size()), ate.count, traj_csv_path_.c_str(),
                ate.trans_rmse_m, ate.rot_rmse_deg, rpe.delta, rpe.trans_rmse_m, rpe.rot_rmse_deg);
  }

  // params
  std::string lidar_topic_;
  std::string gt_topic_;
  std::string csv_path_;
  std::string traj_csv_path_;
  int rpe_delta_ = 10;
  double voxel_leaf_ = 0.2;
  double ndt_resolution_ = 1.0;
  double gt_tol_ms_ = 50.0;
  int stride_ = 1;
  int max_pairs_ = 0;
  double idle_sec_ = 3.0;
  bool use_ekf2_prior_ = false;
  std::string ekf2_topic_;
  NdtConfig ndt_cfg_;
  RegistrationGateConfig gate_cfg_;

  // state
  std::vector<Scan> scans_;
  std::vector<std::pair<double, Eigen::Isometry3f>> gt_buf_;
  std::vector<std::pair<double, Eigen::Isometry3f>> ekf2_buf_;
  Clock::time_point last_activity_;
  bool finalized_ = false;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gt_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ekf2_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
  rclcpp::TimerBase::SharedPtr watchdog_;
};

}  // namespace
}  // namespace graph_slam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<graph_slam::Eval02RelativeErrorNode>());
  rclcpp::shutdown();
  return 0;
}
