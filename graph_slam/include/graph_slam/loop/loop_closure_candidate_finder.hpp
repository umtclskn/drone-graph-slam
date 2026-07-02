#pragma once

#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Values.h>

namespace graph_slam {

/// One loop-closure candidate: the newest keyframe is spatially close to a much
/// older keyframe in the optimized estimate (ARCHITECTURE §8). Detection only —
/// SLAM-10 verifies candidates with NDT before adding a BetweenFactor.
struct LoopClosureCandidate {
  gtsam::Key query_key;   // current (newest) keyframe
  gtsam::Key match_key;   // older keyframe that is spatially close
  double distance_m;      // Euclidean distance between optimized positions
};

/// Distance/revisit-based loop-closure candidate finder (SLAM-09). Pure
/// geometry over the optimized pose graph — no descriptors, no ICP, no place
/// recognition (those are later upgrades). ROS-free.
class LoopClosureCandidateFinder {
 public:
  struct Params {
    double min_distance_m = 0.5;    // candidate only if dist < this
    int min_age_difference = 10;    // match_key must be at least this many
                                    // keyframes older than query_key (avoids
                                    // matching consecutive frames)
  };

  LoopClosureCandidateFinder() = default;
  explicit LoopClosureCandidateFinder(Params p);

  /// Given the full current optimized estimate and the newest keyframe key,
  /// return all candidates that satisfy the distance + age constraints, sorted
  /// ascending by distance. Empty vector if none (or query_key absent).
  [[nodiscard]] std::vector<LoopClosureCandidate> findCandidates(
      const gtsam::Values& estimate, gtsam::Key query_key) const;

 private:
  Params params_;
};

}  // namespace graph_slam
