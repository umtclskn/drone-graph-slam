#pragma once

#include <Eigen/Geometry>

namespace graph_slam::graph {

// FLAG FOR HUMAN OWNER (project convention): no pose alias existed yet in the
// `graph` namespace or the io/ headers (point_types.hpp aliases are PCL *cloud*
// types, not poses), so SLAM-01 introduces one here. Eigen::Isometry3d keeps
// this policy GTSAM-free, as required. Later graph stories that live in GTSAM
// may prefer gtsam::Pose3 — if so, this is the single line to change.
using Pose = Eigen::Isometry3d;

/// Plain thresholds for the keyframe policy. A keyframe is due when ANY enabled
/// criterion is met-or-exceeded (OR semantics). Convention: a non-positive
/// threshold DISABLES that criterion. No ROS, no YAML — the (later) node owns
/// YAML->config wiring (ARCHITECTURE §7).
struct KeyframePolicyConfig {
  double translation_m = 0.0;  ///< distance threshold (metres); <= 0 disables.
  double rotation_rad = 0.0;   ///< geodesic-angle threshold (radians); <= 0 disables.
  double time_s = 0.0;         ///< elapsed-time threshold (seconds); <= 0 disables.
};

/// Decides whether a new keyframe is due from the relative pose accumulated
/// since the last keyframe and the elapsed time since it.
///
/// Stateless w.r.t. keyframes: it does NOT own or track the last-keyframe
/// pose/time. The front-end accumulator (SLAM-03) owns that reference and
/// restarts it at each keyframe (ARCHITECTURE §6 #2). It also does not
/// special-case the first frame — bootstrap (SLAM-04) creates node 0 explicitly;
/// this policy governs only subsequent keyframes.
class KeyframePolicy {
 public:
  explicit KeyframePolicy(KeyframePolicyConfig cfg) : cfg_(cfg) {}

  /// \param delta_since_last_kf relative transform from the last keyframe to the
  ///        current pose.
  /// \param elapsed_s seconds since the last keyframe.
  /// \return true iff any ENABLED criterion is met-or-exceeded (>=).
  [[nodiscard]] bool is_keyframe_due(const Pose& delta_since_last_kf, double elapsed_s) const {
    if (cfg_.translation_m > 0.0 &&
        delta_since_last_kf.translation().norm() >= cfg_.translation_m) {
      return true;
    }
    if (cfg_.rotation_rad > 0.0) {
      // Geodesic angle of the delta rotation, in [0, pi] — no Euler wraparound.
      const double angle = Eigen::AngleAxisd(delta_since_last_kf.rotation()).angle();
      if (angle >= cfg_.rotation_rad) {
        return true;
      }
    }
    return cfg_.time_s > 0.0 && elapsed_s >= cfg_.time_s;
  }

 private:
  KeyframePolicyConfig cfg_;
};

}  // namespace graph_slam::graph
