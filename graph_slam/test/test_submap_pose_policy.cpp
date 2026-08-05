#include <gtest/gtest.h>

#include <Eigen/Core>

#include <deque>
#include <memory>
#include <vector>

#include "graph_slam/point_types.hpp"
#include "graph_slam/submap_pose_policy.hpp"

namespace {

graph_slam::SubmapKeyframe makeEntry(double stamp_s, float x, bool refined,
                                     int keyframe_id = -1) {
  graph_slam::SubmapKeyframe kf;
  kf.cloud = std::make_shared<graph_slam::Cloud>();
  kf.pose = Eigen::Matrix4f::Identity();
  kf.pose(0, 3) = x;
  kf.stamp_s = stamp_s;
  kf.keyframe_id = keyframe_id;
  kf.refined = refined;
  return kf;
}

}  // namespace

// L5-19f: a window that mixes backend-refined poses with front-end-unrefined
// poses must NOT be fused together — planSubmapFusion rejects the mix.
TEST(SubmapPosePolicyTest, RejectsMixedRefinedAndUnrefinedFusion) {
  std::deque<graph_slam::SubmapKeyframe> window;
  window.push_back(makeEntry(1.0, 0.0F, /*refined=*/true, 0));
  window.push_back(makeEntry(2.0, 0.5F, /*refined=*/true, 1));
  window.push_back(makeEntry(3.0, 1.0F, /*refined=*/false, -1));  // newest unrefined

  const graph_slam::SubmapFusionPlan plan = graph_slam::planSubmapFusion(window);
  EXPECT_TRUE(plan.rejected_mixed_source);
  EXPECT_EQ(plan.mode, graph_slam::SubmapFusionMode::SingleNewest);
  ASSERT_EQ(plan.indices.size(), 1U);
  EXPECT_EQ(plan.indices.front(), 2U);
}

TEST(SubmapPosePolicyTest, FusesAllRefinedSameSource) {
  std::deque<graph_slam::SubmapKeyframe> window;
  window.push_back(makeEntry(1.0, 0.0F, true, 0));
  window.push_back(makeEntry(2.0, 0.5F, true, 1));
  window.push_back(makeEntry(3.0, 1.0F, true, 2));

  const graph_slam::SubmapFusionPlan plan = graph_slam::planSubmapFusion(window);
  EXPECT_FALSE(plan.rejected_mixed_source);
  EXPECT_EQ(plan.mode, graph_slam::SubmapFusionMode::FuseSameSource);
  ASSERT_EQ(plan.indices.size(), 3U);
  EXPECT_EQ(plan.indices[0], 0U);
  EXPECT_EQ(plan.indices[1], 1U);
  EXPECT_EQ(plan.indices[2], 2U);
}

TEST(SubmapPosePolicyTest, FusesAllUnrefinedBootstrap) {
  std::deque<graph_slam::SubmapKeyframe> window;
  window.push_back(makeEntry(1.0, 0.0F, false));
  window.push_back(makeEntry(2.0, 0.5F, false));

  const graph_slam::SubmapFusionPlan plan = graph_slam::planSubmapFusion(window);
  EXPECT_FALSE(plan.rejected_mixed_source);
  EXPECT_EQ(plan.mode, graph_slam::SubmapFusionMode::FuseSameSource);
  EXPECT_EQ(plan.indices.size(), 2U);
}

TEST(SubmapPosePolicyTest, ApplyOptimizedMatchesByStampOnly) {
  std::deque<graph_slam::SubmapKeyframe> window;
  window.push_back(makeEntry(1.0, 0.0F, false));
  window.push_back(makeEntry(2.0, 0.5F, false));
  window.push_back(makeEntry(3.0, 1.0F, false));

  graph_slam::OptimizedPoseUpdate update;
  update.stamp_s = 2.0;
  update.keyframe_id = 99;
  update.pose = Eigen::Matrix4f::Identity();
  update.pose(0, 3) = 7.0F;

  const auto result = graph_slam::applyOptimizedPose(window, update);
  EXPECT_EQ(result.status, graph_slam::ApplyOptimizedStatus::Refined);
  ASSERT_TRUE(result.index.has_value());
  EXPECT_EQ(*result.index, 1U);
  EXPECT_TRUE(window[1].refined);
  EXPECT_FLOAT_EQ(window[1].pose(0, 3), 7.0F);
  EXPECT_EQ(window[1].keyframe_id, 99);
  // Neighbours untouched.
  EXPECT_FALSE(window[0].refined);
  EXPECT_FALSE(window[2].refined);
  EXPECT_FLOAT_EQ(window[0].pose(0, 3), 0.0F);
  EXPECT_FLOAT_EQ(window[2].pose(0, 3), 1.0F);
}

TEST(SubmapPosePolicyTest, ApplyOptimizedDoesNotUseLatestOrNearest) {
  std::deque<graph_slam::SubmapKeyframe> window;
  window.push_back(makeEntry(1.0, 0.0F, false));
  window.push_back(makeEntry(2.0, 0.5F, false));
  window.push_back(makeEntry(3.0, 1.0F, false));

  graph_slam::OptimizedPoseUpdate update;
  update.stamp_s = 9.9;  // no match
  update.keyframe_id = -1;
  update.pose = Eigen::Matrix4f::Identity();
  update.pose(0, 3) = 99.0F;

  const auto result = graph_slam::applyOptimizedPose(window, update);
  EXPECT_EQ(result.status, graph_slam::ApplyOptimizedStatus::NoMatch);
  EXPECT_FALSE(result.index.has_value());
  EXPECT_FALSE(window.back().refined);
  EXPECT_FLOAT_EQ(window.back().pose(0, 3), 1.0F);
}

TEST(SubmapPosePolicyTest, LargeRefinedJumpCollapsesWindow) {
  std::deque<graph_slam::SubmapKeyframe> window;
  window.push_back(makeEntry(1.0, 0.0F, true, 0));
  window.push_back(makeEntry(2.0, 0.5F, true, 1));
  window.push_back(makeEntry(3.0, 1.0F, true, 2));

  graph_slam::OptimizedPoseUpdate update;
  update.stamp_s = 3.0;
  update.keyframe_id = 2;
  update.pose = Eigen::Matrix4f::Identity();
  update.pose(0, 3) = 1.0F + 0.5F;  // 0.5 m jump > default 0.2 m eps

  const auto result = graph_slam::applyOptimizedPose(window, update, 0.2, 0.1);
  EXPECT_EQ(result.status, graph_slam::ApplyOptimizedStatus::Collapsed);
  ASSERT_EQ(window.size(), 1U);
  EXPECT_TRUE(window.front().refined);
  EXPECT_FLOAT_EQ(window.front().pose(0, 3), 1.5F);
  EXPECT_DOUBLE_EQ(window.front().stamp_s, 3.0);
}

// L5-10: a loop closure shifts poses in the active submap window. All three
// matched entries are updated in one pass and the window is NEVER collapsed
// (unlike the per-entry applyOptimizedPose path) -- exactly the failure mode
// L5-12 measured and this story exists to avoid.
TEST(SubmapPosePolicyTest, BatchUpdatesAllMatchingEntriesWithoutCollapsing) {
  std::deque<graph_slam::SubmapKeyframe> window;
  window.push_back(makeEntry(1.0, 0.0F, /*refined=*/true, 10));
  window.push_back(makeEntry(2.0, 0.5F, true, 11));
  window.push_back(makeEntry(3.0, 1.0F, true, 12));

  std::vector<graph_slam::OptimizedPoseUpdate> updates;
  for (int id = 10; id <= 12; ++id) {
    graph_slam::OptimizedPoseUpdate u;
    u.stamp_s = 0.0;  // batch entries never match by stamp, only by id
    u.keyframe_id = id;
    u.pose = Eigen::Matrix4f::Identity();
    // 0.5 m jump on each entry -- above the default 0.2 m rebuild threshold,
    // and also above L5-19e's 0.2 m collapse threshold (would have collapsed
    // the window if applied through applyOptimizedPose N times instead).
    u.pose(0, 3) = static_cast<float>(id - 10) * 0.5F + 0.5F;
    updates.push_back(u);
  }

  const auto result = graph_slam::applyOptimizedPoseBatch(window, updates, 0.2, 0.1);
  EXPECT_EQ(result.matched_count, 3U);
  EXPECT_TRUE(result.should_rebuild);
  EXPECT_NEAR(result.max_shift_m, 0.5, 1e-6);
  ASSERT_EQ(window.size(), 3U);  // never collapsed
  EXPECT_TRUE(window[0].refined);
  EXPECT_TRUE(window[1].refined);
  EXPECT_TRUE(window[2].refined);
  EXPECT_FLOAT_EQ(window[0].pose(0, 3), 0.5F);
  EXPECT_FLOAT_EQ(window[1].pose(0, 3), 1.0F);
  EXPECT_FLOAT_EQ(window[2].pose(0, 3), 1.5F);
}

// L5-10 "no thrashing": a sub-ε shift updates poses but must NOT ask the
// caller to rebuild the target grid.
TEST(SubmapPosePolicyTest, BatchBelowThresholdUpdatesPosesButSkipsRebuild) {
  std::deque<graph_slam::SubmapKeyframe> window;
  window.push_back(makeEntry(1.0, 0.0F, true, 10));
  window.push_back(makeEntry(2.0, 0.5F, true, 11));

  std::vector<graph_slam::OptimizedPoseUpdate> updates;
  for (int id = 10; id <= 11; ++id) {
    graph_slam::OptimizedPoseUpdate u;
    u.keyframe_id = id;
    u.pose = Eigen::Matrix4f::Identity();
    u.pose(0, 3) = static_cast<float>(id == 10 ? 0.0 : 0.5) + 0.01F;  // 1 cm nudge
    updates.push_back(u);
  }

  const auto result = graph_slam::applyOptimizedPoseBatch(window, updates, 0.2, 0.1);
  EXPECT_EQ(result.matched_count, 2U);
  EXPECT_FALSE(result.should_rebuild);
  EXPECT_LT(result.max_shift_m, 0.2);
  // Poses still applied even though no rebuild was requested.
  EXPECT_FLOAT_EQ(window[0].pose(0, 3), 0.01F);
  EXPECT_FLOAT_EQ(window[1].pose(0, 3), 0.51F);
}

// L5-10: a closure whose affected range does not overlap the active submap
// window (e.g. an old closure while the window has already slid past it) is
// a cheap no-op -- no match, no rebuild request, window untouched.
TEST(SubmapPosePolicyTest, BatchOutsideWindowIsNoOp) {
  std::deque<graph_slam::SubmapKeyframe> window;
  window.push_back(makeEntry(1.0, 0.0F, true, 50));
  window.push_back(makeEntry(2.0, 0.5F, true, 51));

  std::vector<graph_slam::OptimizedPoseUpdate> updates;
  for (int id = 0; id <= 5; ++id) {  // far outside {50, 51}
    graph_slam::OptimizedPoseUpdate u;
    u.keyframe_id = id;
    u.pose = Eigen::Matrix4f::Identity();
    u.pose(0, 3) = 99.0F;
    updates.push_back(u);
  }

  const auto result = graph_slam::applyOptimizedPoseBatch(window, updates, 0.2, 0.1);
  EXPECT_EQ(result.matched_count, 0U);
  EXPECT_FALSE(result.should_rebuild);
  ASSERT_EQ(window.size(), 2U);
  EXPECT_FLOAT_EQ(window[0].pose(0, 3), 0.0F);
  EXPECT_FLOAT_EQ(window[1].pose(0, 3), 0.5F);
}

// Updates with keyframe_id == -1 (never assigned, e.g. a still-bootstrap
// entry elsewhere in the graph) must never match -- id is the ONLY key the
// batch path uses.
TEST(SubmapPosePolicyTest, BatchIgnoresNegativeKeyframeId) {
  std::deque<graph_slam::SubmapKeyframe> window;
  window.push_back(makeEntry(1.0, 0.0F, false, -1));

  std::vector<graph_slam::OptimizedPoseUpdate> updates;
  graph_slam::OptimizedPoseUpdate u;
  u.keyframe_id = -1;
  u.pose = Eigen::Matrix4f::Identity();
  u.pose(0, 3) = 42.0F;
  updates.push_back(u);

  const auto result = graph_slam::applyOptimizedPoseBatch(window, updates);
  EXPECT_EQ(result.matched_count, 0U);
  EXPECT_FALSE(window[0].refined);
  EXPECT_FLOAT_EQ(window[0].pose(0, 3), 0.0F);
}
