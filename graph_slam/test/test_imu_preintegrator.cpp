// L5-01d: closed-form unit tests for ImuPreintegrator.
//
// The GTSAM preintegration model integrates a constant-acceleration measurement
// stream *exactly* (deltaP += deltaV·dt + ½·acc·dt² sums to ½·a·T²), so these
// analytic checks hold to ~machine precision, not just <1e-3.
#include "graph_slam/imu/imu_preintegrator.hpp"

#include <cmath>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>

namespace {

using graph_slam::imu::ImuPreintegrator;
using graph_slam::imu::PreintegrationConfig;

constexpr double kG = 9.81;

// A config with the real noise σ but a clean gravity magnitude for the analytic
// checks (so "measured accel = a_world + (0,0,g)" cancels exactly).
PreintegrationConfig makeConfig() {
  PreintegrationConfig c;  // LIO-SAM σ defaults
  c.gravity = kG;
  return c;
}

// Case A: zero angular rate + constant world acceleration. With identity start
// attitude the accelerometer measures specific force f = a_world - g_nav =
// a_world + (0,0,g). The exact motion is p(T)=v0·T+½·a·T², v(T)=v0+a·T, R=I.
TEST(ImuPreintegrator, ConstAccelMatchesClosedForm) {
  const PreintegrationConfig config = makeConfig();
  ImuPreintegrator preint(config);

  const Eigen::Vector3d a_world(0.3, -0.2, 0.5);  // desired world acceleration
  const Eigen::Vector3d v0(1.0, 0.5, -0.25);      // start velocity (nav frame)
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const Eigen::Vector3d measured_accel = a_world + Eigen::Vector3d(0, 0, kG);

  const double total_T = 1.0;
  const int steps = 1000;
  const double dt = total_T / steps;
  for (int k = 0; k < steps; ++k) {
    EXPECT_TRUE(preint.integrate(measured_accel, gyro, dt));
  }
  EXPECT_EQ(preint.count(), static_cast<std::size_t>(steps));
  EXPECT_NEAR(preint.deltaTij(), total_T, 1e-9);

  const gtsam::NavState state_i(gtsam::Rot3(), gtsam::Point3(0, 0, 0), v0);
  const gtsam::NavState pred = preint.predict(state_i, gtsam::imuBias::ConstantBias());

  const Eigen::Vector3d expected_p = v0 * total_T + 0.5 * a_world * total_T * total_T;
  const Eigen::Vector3d expected_v = v0 + a_world * total_T;

  EXPECT_LT((pred.position() - gtsam::Point3(expected_p)).norm(), 1e-6);
  EXPECT_LT((pred.velocity() - expected_v).norm(), 1e-6);
  // Zero gyro → no rotation.
  EXPECT_LT(gtsam::Rot3::Logmap(pred.attitude()).norm(), 1e-9);
}

// Case B: constant yaw rate + gravity-cancelling accel (pure yaw keeps body-z
// aligned with world-z, so (0,0,g) keeps world acceleration zero). The exact
// motion is ΔR=Rz(ω·T), Δp≈0, Δv≈0 (from a stationary start).
TEST(ImuPreintegrator, ConstGyroMatchesClosedForm) {
  const PreintegrationConfig config = makeConfig();
  ImuPreintegrator preint(config);

  const double omega = 0.4;  // rad/s about +z (yaw)
  const Eigen::Vector3d gyro(0.0, 0.0, omega);
  const Eigen::Vector3d measured_accel(0.0, 0.0, kG);  // cancels ENU gravity

  const double total_T = 1.0;
  const int steps = 2000;
  const double dt = total_T / steps;
  for (int k = 0; k < steps; ++k) {
    preint.integrate(measured_accel, gyro, dt);
  }

  const gtsam::NavState state_i(gtsam::Rot3(), gtsam::Point3(0, 0, 0),
                                Eigen::Vector3d::Zero());
  const gtsam::NavState pred = preint.predict(state_i, gtsam::imuBias::ConstantBias());

  const gtsam::Rot3 expected_R = gtsam::Rot3::Rz(omega * total_T);
  const double rot_err = gtsam::Rot3::Logmap(pred.attitude().between(expected_R)).norm();
  EXPECT_LT(rot_err, 1e-4);
  // Stationary start with cancelled gravity → negligible translation drift.
  EXPECT_LT(pred.position().norm(), 1e-3);
  EXPECT_LT(pred.velocity().norm(), 1e-3);
}

// Covariance must stay symmetric positive-definite and its trace must grow
// monotonically as measurements accumulate (Forster 2017 native propagation).
TEST(ImuPreintegrator, CovarianceIsSpdAndMonotone) {
  const PreintegrationConfig config = makeConfig();
  ImuPreintegrator preint(config);

  const Eigen::Vector3d accel(0.1, 0.0, kG);
  const Eigen::Vector3d gyro(0.0, 0.0, 0.05);
  const double dt = 0.004;  // 250 Hz

  double prev_trace = -1.0;
  for (int k = 0; k < 250; ++k) {  // ~1 s
    preint.integrate(accel, gyro, dt);
    const Eigen::Matrix<double, 9, 9> cov = preint.covariance();
    // Symmetric.
    EXPECT_LT((cov - cov.transpose()).norm(), 1e-9);
    // Positive-definite (smallest eigenvalue > 0).
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solver(cov);
    EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
    // Monotone-growing trace.
    const double trace = cov.trace();
    EXPECT_GT(trace, prev_trace);
    prev_trace = trace;
  }
}

// A non-positive or non-finite dt is rejected and does not advance the state.
TEST(ImuPreintegrator, RejectsNonPositiveDt) {
  const PreintegrationConfig config = makeConfig();
  ImuPreintegrator preint(config);

  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();

  EXPECT_FALSE(preint.integrate(accel, gyro, 0.0));
  EXPECT_FALSE(preint.integrate(accel, gyro, -0.01));
  EXPECT_FALSE(preint.integrate(accel, gyro, std::nan("")));
  EXPECT_EQ(preint.count(), 0u);
  EXPECT_EQ(preint.deltaTij(), 0.0);

  EXPECT_TRUE(preint.integrate(accel, gyro, 0.004));
  EXPECT_EQ(preint.count(), 1u);
}

// reset() clears the interval and zeroes the sample count.
TEST(ImuPreintegrator, ResetClearsInterval) {
  const PreintegrationConfig config = makeConfig();
  ImuPreintegrator preint(config);

  const Eigen::Vector3d accel(0.2, 0.0, kG);
  const Eigen::Vector3d gyro(0.0, 0.0, 0.1);
  for (int k = 0; k < 100; ++k) {
    preint.integrate(accel, gyro, 0.004);
  }
  ASSERT_GT(preint.deltaTij(), 0.0);

  preint.reset(gtsam::imuBias::ConstantBias());
  EXPECT_EQ(preint.count(), 0u);
  EXPECT_EQ(preint.deltaTij(), 0.0);
  EXPECT_LT(preint.deltaPij().norm(), 1e-12);
  EXPECT_LT(preint.deltaVij().norm(), 1e-12);
}

}  // namespace
