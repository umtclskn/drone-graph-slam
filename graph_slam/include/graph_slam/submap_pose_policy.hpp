#pragma once

// L5-19: same-source submap fusion policy (ROS-free).
//
// Submap relatives are T_rel = T_ref^{-1} · T_k. Mixing a backend-optimized pose
// with a front-end NDT bootstrap pose leaves uncanceled absolute error in T_rel.
// These helpers decide which window entries may be fused together and how an
// OptimizedState update is applied (exact stamp/id match only).

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "graph_slam/point_types.hpp"

namespace graph_slam {

/// One keyframe cloud in the sliding submap window.
struct SubmapKeyframe {
  CloudPtr cloud;
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  double stamp_s = 0.0;   // LiDAR / ndt_odom header stamp of this keyframe
  int keyframe_id = -1;   // backend kf_index_ once known; -1 until then
  bool refined = false;   // false = front-end NDT bootstrap; true = backend
};

enum class SubmapFusionMode {
  SingleNewest,    // degrade to the newest cloud alone (identity transform)
  FuseSameSource,  // fuse the filtered same-source subset
};

struct SubmapFusionPlan {
  SubmapFusionMode mode = SubmapFusionMode::SingleNewest;
  std::vector<std::size_t> indices;  // into the window; used for FuseSameSource
  bool rejected_mixed_source = false;
};

/// Build a fusion plan for `window`. Never returns indices that mix refined
/// with unrefined. If the newest entry is unrefined while any older entry is
/// already refined (bootstrap race), degrade to SingleNewest and set
/// `rejected_mixed_source`.
inline SubmapFusionPlan planSubmapFusion(const std::deque<SubmapKeyframe>& window) {
  SubmapFusionPlan plan;
  if (window.empty()) {
    return plan;
  }
  if (window.size() == 1) {
    plan.mode = SubmapFusionMode::SingleNewest;
    plan.indices = {0};
    return plan;
  }

  const bool ref_refined = window.back().refined;
  if (!ref_refined) {
    for (const auto& kf : window) {
      if (kf.refined) {
        plan.mode = SubmapFusionMode::SingleNewest;
        plan.indices = {window.size() - 1};
        plan.rejected_mixed_source = true;
        return plan;
      }
    }
  }

  plan.mode = SubmapFusionMode::FuseSameSource;
  for (std::size_t i = 0; i < window.size(); ++i) {
    if (window[i].refined == ref_refined) {
      plan.indices.push_back(i);
    } else {
      plan.rejected_mixed_source = true;
    }
  }
  if (plan.indices.size() <= 1) {
    plan.mode = SubmapFusionMode::SingleNewest;
    plan.indices = {window.size() - 1};
  }
  return plan;
}

struct OptimizedPoseUpdate {
  double stamp_s = 0.0;
  int keyframe_id = -1;
  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
};

enum class ApplyOptimizedStatus {
  NoMatch,
  Refined,
  Collapsed,  // already-refined entry jumped > ε → window collapsed to it
};

struct ApplyOptimizedResult {
  ApplyOptimizedStatus status = ApplyOptimizedStatus::NoMatch;
  std::optional<std::size_t> index;
};

inline double rotationAngle(const Eigen::Matrix4f& m) {
  const double trace = m.block<3, 3>(0, 0).cast<double>().trace();
  return std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
}

/// Match by exact stamp (primary, |Δt| < stamp_eps_s) else by keyframe_id (≥0).
/// Updates only that entry. If it was already refined and the pose jump exceeds
/// `collapse_eps_m` / `collapse_eps_rad`, collapse the window to that single
/// entry with the new pose (loop / large re-opt).
inline ApplyOptimizedResult applyOptimizedPose(std::deque<SubmapKeyframe>& window,
                                               const OptimizedPoseUpdate& update,
                                               double collapse_eps_m = 0.2,
                                               double collapse_eps_rad = 0.1,
                                               double stamp_eps_s = 1e-6) {
  ApplyOptimizedResult result;
  if (window.empty()) {
    return result;
  }

  std::optional<std::size_t> match;
  for (std::size_t i = 0; i < window.size(); ++i) {
    if (std::abs(window[i].stamp_s - update.stamp_s) < stamp_eps_s) {
      match = i;
      break;
    }
  }
  if (!match && update.keyframe_id >= 0) {
    for (std::size_t i = 0; i < window.size(); ++i) {
      if (window[i].keyframe_id == update.keyframe_id) {
        match = i;
        break;
      }
    }
  }
  if (!match) {
    return result;
  }

  result.index = *match;
  SubmapKeyframe& entry = window[*match];
  const bool was_refined = entry.refined;
  Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
  if (was_refined) {
    delta = entry.pose.inverse() * update.pose;
  }
  entry.pose = update.pose;
  entry.refined = true;
  if (update.keyframe_id >= 0) {
    entry.keyframe_id = update.keyframe_id;
  }

  if (was_refined) {
    const double jumped_m = delta.block<3, 1>(0, 3).norm();
    const double jumped_rad = rotationAngle(delta);
    if (jumped_m > collapse_eps_m || jumped_rad > collapse_eps_rad) {
      SubmapKeyframe keep = entry;
      window.clear();
      window.push_back(std::move(keep));
      result.status = ApplyOptimizedStatus::Collapsed;
      result.index = 0;
      return result;
    }
  }

  result.status = ApplyOptimizedStatus::Refined;
  return result;
}

/// L5-10: outcome of applying a coordinated batch of updates (one accepted
/// loop closure's worth) to the submap window in a single atomic pass.
struct BatchApplyResult {
  std::size_t matched_count = 0;  // window entries touched by this batch
  double max_shift_m = 0.0;       // largest translation jump among already-
                                   // refined matched entries
  double max_shift_rad = 0.0;     // largest rotation jump, same population
  bool should_rebuild = false;    // max shift crossed the rebuild threshold
};

/// Apply a coordinated batch of optimized-pose updates (e.g. every keyframe
/// touched by an accepted loop closure) as ONE atomic pass: every matching
/// window entry's pose is updated first, and only THEN does the caller decide
/// (via the returned `should_rebuild`) whether the maximum pose shift among
/// already-refined entries exceeds (`rebuild_eps_m`, `rebuild_eps_rad`) and
/// the target grid should be rebuilt.
///
/// This is the L5-10 replacement for calling `applyOptimizedPose` once per
/// update in a loop: N independent calls each run L5-19e's per-entry
/// collapse-on-jump guard in isolation, so a flood of independently-arriving
/// updates repeatedly discards all-but-the-latest entry and can strand the
/// window on a single stale keyframe far from the drone's current position
/// (measured, L5-12: 97-99/111 keyframes reached vs 111/111 without it). This
/// function never collapses the window -- it keeps every entry's cloud and
/// only moves poses, exactly matching this story's "rebuild from the new
/// poses" acceptance criterion.
///
/// Matching is by `keyframe_id` only (never stamp): a loop-closure batch
/// spans historical keyframes whose original scan stamp the back-end does not
/// retain, so every `OptimizedPoseUpdate.keyframe_id` in `updates` must be
/// >= 0. An update whose id does not match any window entry (outside the
/// active window, or not yet refined once) is a cheap no-op, same as before.
inline BatchApplyResult applyOptimizedPoseBatch(
    std::deque<SubmapKeyframe>& window, const std::vector<OptimizedPoseUpdate>& updates,
    double rebuild_eps_m = 0.2, double rebuild_eps_rad = 0.1) {
  BatchApplyResult result;
  if (window.empty() || updates.empty()) {
    return result;
  }

  for (const OptimizedPoseUpdate& update : updates) {
    if (update.keyframe_id < 0) {
      continue;
    }
    std::optional<std::size_t> match;
    for (std::size_t i = 0; i < window.size(); ++i) {
      if (window[i].keyframe_id == update.keyframe_id) {
        match = i;
        break;
      }
    }
    if (!match) {
      continue;
    }

    SubmapKeyframe& entry = window[*match];
    ++result.matched_count;
    if (entry.refined) {
      const Eigen::Matrix4f delta = entry.pose.inverse() * update.pose;
      result.max_shift_m =
          std::max(result.max_shift_m, static_cast<double>(delta.block<3, 1>(0, 3).norm()));
      result.max_shift_rad = std::max(result.max_shift_rad, rotationAngle(delta));
    }
    entry.pose = update.pose;
    entry.refined = true;
  }

  result.should_rebuild = result.matched_count > 0 &&
                          (result.max_shift_m > rebuild_eps_m ||
                           result.max_shift_rad > rebuild_eps_rad);
  return result;
}

}  // namespace graph_slam
