#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>

#include "graph_slam/graph/keyframe_policy.hpp"

namespace {

using graph_slam::graph::KeyframePolicy;
using graph_slam::graph::KeyframePolicyConfig;
using graph_slam::graph::Pose;

// Pose with a pure translation along x (no rotation).
Pose translated(double x) {
  Pose p = Pose::Identity();
  p.translation() = Eigen::Vector3d(x, 0.0, 0.0);
  return p;
}

// Pose with a pure rotation of `angle_rad` about an arbitrary (normalised) axis.
Pose rotated(double angle_rad) {
  Pose p = Pose::Identity();
  const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
  p.linear() = Eigen::AngleAxisd(angle_rad, axis).toRotationMatrix();
  return p;
}

// Geodesic angle the policy will actually see for a pose. Building an angle into
// a rotation matrix and reading it back is not bit-exact, so boundary (==) tests
// must compare against this measured value, not the constructor's input.
double geodesicAngle(const Pose& p) { return Eigen::AngleAxisd(p.linear()).angle(); }

// --- translation criterion: fires at threshold (>=), not just below ----------
TEST(KeyframePolicyTest, TranslationFiresAtThresholdNotBelow) {
  const KeyframePolicy policy{KeyframePolicyConfig{/*t=*/1.0, /*r=*/0.0, /*time=*/0.0}};
  EXPECT_TRUE(policy.is_keyframe_due(translated(1.0), 0.0));     // exactly at
  EXPECT_TRUE(policy.is_keyframe_due(translated(1.5), 0.0));     // above
  EXPECT_FALSE(policy.is_keyframe_due(translated(0.999), 0.0));  // just below
}

// --- rotation criterion: fires at threshold (>=), not just below -------------
TEST(KeyframePolicyTest, RotationFiresAtThresholdNotBelow) {
  const Pose p = rotated(0.5);
  const double measured = geodesicAngle(p);  // the angle the policy actually sees
  const KeyframePolicy at{KeyframePolicyConfig{0.0, measured, 0.0}};
  EXPECT_TRUE(at.is_keyframe_due(p, 0.0));             // exactly at (== boundary)
  EXPECT_TRUE(at.is_keyframe_due(rotated(0.6), 0.0));  // above
  const KeyframePolicy above{KeyframePolicyConfig{0.0, measured + 0.05, 0.0}};
  EXPECT_FALSE(above.is_keyframe_due(p, 0.0));  // pose just below threshold
}

// --- time criterion: fires at threshold (>=), not just below -----------------
TEST(KeyframePolicyTest, TimeFiresAtThresholdNotBelow) {
  const KeyframePolicy policy{KeyframePolicyConfig{0.0, 0.0, /*time=*/2.0}};
  EXPECT_TRUE(policy.is_keyframe_due(Pose::Identity(), 2.0));     // exactly at
  EXPECT_TRUE(policy.is_keyframe_due(Pose::Identity(), 3.0));     // above
  EXPECT_FALSE(policy.is_keyframe_due(Pose::Identity(), 1.999));  // just below
}

// --- pure translation fires on distance only (no rotation present) -----------
TEST(KeyframePolicyTest, PureTranslationFiresOnDistanceOnly) {
  const KeyframePolicy policy{KeyframePolicyConfig{1.0, 0.5, 0.0}};
  // Big translation, zero rotation -> due via translation; rotation stays quiet.
  EXPECT_TRUE(policy.is_keyframe_due(translated(2.0), 0.0));
}

// --- pure rotation fires on angle only (no translation present) --------------
TEST(KeyframePolicyTest, PureRotationFiresOnAngleOnly) {
  const KeyframePolicy policy{KeyframePolicyConfig{1.0, 0.5, 0.0}};
  // Big rotation, zero translation -> due via rotation; translation stays quiet.
  EXPECT_TRUE(policy.is_keyframe_due(rotated(1.0), 0.0));
}

// --- time-only trigger: zero motion but elapsed >= time_s -> true ------------
TEST(KeyframePolicyTest, TimeOnlyTriggerWithZeroMotion) {
  const KeyframePolicy policy{KeyframePolicyConfig{1.0, 0.5, 2.0}};
  EXPECT_TRUE(policy.is_keyframe_due(Pose::Identity(), 5.0));
}

// --- zero motion + zero elapsed -> false -------------------------------------
TEST(KeyframePolicyTest, ZeroMotionZeroElapsedIsNotDue) {
  const KeyframePolicy policy{KeyframePolicyConfig{1.0, 0.5, 2.0}};
  EXPECT_FALSE(policy.is_keyframe_due(Pose::Identity(), 0.0));
}

// --- OR semantics: any single enabled criterion fires independently ----------
TEST(KeyframePolicyTest, OrSemanticsAnySingleCriterionFires) {
  const KeyframePolicy policy{KeyframePolicyConfig{1.0, 0.5, 2.0}};
  EXPECT_TRUE(policy.is_keyframe_due(translated(1.0), 0.0));   // translation only
  EXPECT_TRUE(policy.is_keyframe_due(rotated(0.6), 0.0));      // rotation only
  EXPECT_TRUE(policy.is_keyframe_due(Pose::Identity(), 2.0));  // time only
}

// --- disabled criteria (non-positive thresholds) never fire ------------------
TEST(KeyframePolicyTest, DisabledCriteriaNeverFire) {
  // All criteria disabled -> never due, regardless of how large the inputs are.
  const KeyframePolicy off{KeyframePolicyConfig{0.0, 0.0, 0.0}};
  EXPECT_FALSE(off.is_keyframe_due(translated(1000.0), 1e6));
  EXPECT_FALSE(off.is_keyframe_due(rotated(3.0), 1e6));

  // Negative thresholds disable too.
  const KeyframePolicy neg{KeyframePolicyConfig{-1.0, -1.0, -1.0}};
  EXPECT_FALSE(neg.is_keyframe_due(translated(1000.0), 1e6));

  // Only translation enabled: huge rotation + huge elapsed stay quiet.
  const KeyframePolicy trans_only{KeyframePolicyConfig{1.0, 0.0, 0.0}};
  EXPECT_FALSE(trans_only.is_keyframe_due(rotated(3.0), 1e6));
}

// --- geodesic-angle correctness: known 30 deg about an arbitrary axis --------
TEST(KeyframePolicyTest, GeodesicAngleKnownThirtyDegrees) {
  const double deg30 = 30.0 * M_PI / 180.0;
  // Threshold just below 30 deg fires; just above does not (delta is exactly 30).
  const KeyframePolicy below{KeyframePolicyConfig{0.0, deg30 - 1e-3, 0.0}};
  const KeyframePolicy above{KeyframePolicyConfig{0.0, deg30 + 1e-3, 0.0}};
  EXPECT_TRUE(below.is_keyframe_due(rotated(deg30), 0.0));
  EXPECT_FALSE(above.is_keyframe_due(rotated(deg30), 0.0));
}

// --- large rotation (~170 deg) handled correctly: no Euler wraparound bug ----
TEST(KeyframePolicyTest, GeodesicAngleLargeRotationNoWraparound) {
  const double deg170 = 170.0 * M_PI / 180.0;
  // A naive per-axis Euler reading could fold a ~170 deg rotation into a small
  // angle; the geodesic angle must report it as large and trip a modest gate.
  const KeyframePolicy policy{KeyframePolicyConfig{0.0, /*r=*/2.0, 0.0}};  // 2 rad ~= 114.6 deg
  EXPECT_TRUE(policy.is_keyframe_due(rotated(deg170), 0.0));
  // And a 90 deg threshold (1.5708 rad) is comfortably exceeded by 170 deg.
  const KeyframePolicy ninety{KeyframePolicyConfig{0.0, M_PI / 2.0, 0.0}};
  EXPECT_TRUE(ninety.is_keyframe_due(rotated(deg170), 0.0));
}

}  // namespace
