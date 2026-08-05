#pragma once

// L5-12: fuse a small window of historical keyframes into one NDT verification
// target for loop closure ("frozen historical submap" — SPRINTS.md guiding
// decisions). Distinct from the front-end's active/sliding submap
// (submap_pose_policy.hpp): every pose here comes from the ONE current
// backend-optimized estimate, so there is no refined/unrefined provenance to
// reconcile — this is a plain transform-merge-downsample, not a fusion policy.

#include <vector>

#include <Eigen/Core>

#include "graph_slam/point_types.hpp"

namespace graph_slam {

/// One historical keyframe contributing to a loop-closure verification target:
/// its cloud (local sensor frame) and its current backend-optimized pose.
struct LoopSubmapEntry {
  CloudPtr cloud;
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
};

/// Fuse `window` into one cloud expressed in `pose_ref`'s frame — each entry
/// transformed by `relativePoseGuess(pose_ref, entry.pose)` (T_ref^-1 * T_k, the
/// same convention as the front-end's `rebuildTargetGrid()`), merged, then
/// voxel-downsampled at `voxel_leaf` metres. `pose_ref` is normally the loop
/// candidate's match pose, so the result is ready to become the NDT target
/// `LoopClosureVerifier::verify()` aligns the query cloud onto (ARCHITECTURE §8).
/// Empty `window` -> empty cloud. Entries with a null `cloud` are skipped.
[[nodiscard]] CloudPtr buildLoopClosureTarget(const std::vector<LoopSubmapEntry>& window,
                                              const Eigen::Matrix4f& pose_ref,
                                              double voxel_leaf);

}  // namespace graph_slam
