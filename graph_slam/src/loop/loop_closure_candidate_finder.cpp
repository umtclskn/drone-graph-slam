#include "graph_slam/loop/loop_closure_candidate_finder.hpp"

#include <algorithm>
#include <cstdint>

#include <gtsam/inference/Symbol.h>

namespace graph_slam {

LoopClosureCandidateFinder::LoopClosureCandidateFinder(Params p) : params_(p) {}

std::vector<LoopClosureCandidate> LoopClosureCandidateFinder::findCandidates(
    const gtsam::Values& estimate, gtsam::Key query_key) const {
  std::vector<LoopClosureCandidate> candidates;
  if (!estimate.exists(query_key)) {
    return candidates;
  }

  const gtsam::Symbol query_sym(query_key);
  const auto query_index = static_cast<std::int64_t>(query_sym.index());
  const gtsam::Point3 query_pos = estimate.at<gtsam::Pose3>(query_key).translation();

  for (const gtsam::Key match_key : estimate.keys()) {
    const gtsam::Symbol match_sym(match_key);
    if (match_sym.chr() != query_sym.chr()) {
      continue;  // only compare poses of the same key family (e.g. 'x')
    }
    const auto match_index = static_cast<std::int64_t>(match_sym.index());
    if (query_index - match_index < params_.min_age_difference) {
      continue;  // too young (includes the query itself)
    }
    const gtsam::Point3 match_pos = estimate.at<gtsam::Pose3>(match_key).translation();
    const double dist = (query_pos - match_pos).norm();
    if (dist < params_.min_distance_m) {
      candidates.push_back({query_key, match_key, dist});
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const LoopClosureCandidate& a, const LoopClosureCandidate& b) {
              return a.distance_m < b.distance_m;
            });
  return candidates;
}

}  // namespace graph_slam
