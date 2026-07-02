#include <gtest/gtest.h>

#include <cstdint>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include "graph_slam/loop/loop_closure_candidate_finder.hpp"

namespace {

using graph_slam::LoopClosureCandidate;
using graph_slam::LoopClosureCandidateFinder;

gtsam::Pose3 poseAt(double x, double y, double z) {
  return {gtsam::Rot3(), gtsam::Point3(x, y, z)};
}

gtsam::Key xkey(std::uint64_t i) { return gtsam::Symbol('x', i); }

// Case A — frames too young: 5 keyframes in a line, query x4, age gap < 10.
TEST(LoopCandidateFinderTest, NoCandidateFramesTooYoung) {
  gtsam::Values est;
  for (std::uint64_t i = 0; i < 5; ++i) {
    est.insert(xkey(i), poseAt(0.1 * static_cast<double>(i), 0.0, 0.0));
  }
  LoopClosureCandidateFinder finder({/*min_distance_m=*/0.5, /*min_age_difference=*/10});
  const auto candidates = finder.findCandidates(est, xkey(4));
  EXPECT_TRUE(candidates.empty());
}

// Case B — too far: 15 keyframes from origin to (100,0,0); revisit threshold 2 m.
TEST(LoopCandidateFinderTest, NoCandidateTooFar) {
  gtsam::Values est;
  for (std::uint64_t i = 0; i < 15; ++i) {
    est.insert(xkey(i), poseAt((100.0 / 14.0) * static_cast<double>(i), 0.0, 0.0));
  }
  LoopClosureCandidateFinder finder({/*min_distance_m=*/2.0, /*min_age_difference=*/10});
  const auto candidates = finder.findCandidates(est, xkey(14));
  EXPECT_TRUE(candidates.empty());
}

// Case C — valid candidate: drone excurses far then returns near x0 at x14.
TEST(LoopCandidateFinderTest, ValidCandidateDetected) {
  gtsam::Values est;
  est.insert(xkey(0), poseAt(0.0, 0.0, 0.0));
  for (std::uint64_t i = 1; i < 14; ++i) {
    est.insert(xkey(i), poseAt(10.0 * static_cast<double>(i), 0.0, 0.0));  // far excursion
  }
  est.insert(xkey(14), poseAt(0.3, 0.0, 0.0));  // returned near origin
  LoopClosureCandidateFinder finder({/*min_distance_m=*/0.5, /*min_age_difference=*/10});
  const auto candidates = finder.findCandidates(est, xkey(14));
  ASSERT_FALSE(candidates.empty());
  EXPECT_EQ(candidates.front().query_key, xkey(14));
  EXPECT_EQ(candidates.front().match_key, xkey(0));
  EXPECT_NEAR(candidates.front().distance_m, 0.3, 1e-6);
}

}  // namespace
