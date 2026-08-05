// L5-01: ROS-free IMU preintegration wrapper (GTSAM, Forster 2017).
//
// Compresses the hundreds of raw IMU samples between two keyframes into one
// relative-motion estimate (Δrotation, Δposition, Δvelocity) with a 9x9
// covariance, ready to become a gtsam::ImuFactor in L5-03. Pure algorithm class:
// no ROS, no PX4 — unit-testable without a simulator (see test_imu_preintegrator).
#pragma once

#include <cstddef>
#include <memory>

#include <Eigen/Core>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationParams.h>

namespace graph_slam::imu {

// One IMU reading already expressed in the body frame the SLAM stack uses
// (FLU base_link, right-handed, consistent with the ENU world — ARCHITECTURE §5).
// `stamp_s` is the sample time in seconds *in the SLAM clock domain* (sim time on
// bag replay). Post-L5-17i the bridge already publishes /imu/data in that domain
// (node clock under use_sim_time; OS-time pass-through on real hardware), so the
// backend keys the window buffer by the message HEADER stamp directly. `accel`
// is the measured specific force [m/s^2]; `gyro` is the measured angular rate
// [rad/s].
struct ImuSample {
  double stamp_s{0.0};
  Eigen::Vector3d accel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gyro{Eigen::Vector3d::Zero()};
};

// Continuous-time IMU noise densities + gravity for GTSAM preintegration
// (Forster 2017). Values are standard deviations (not variances); toGtsamParams()
// squares them onto the covariance diagonals. Defaults reuse the LIO-SAM
// params_drone_x500.yaml numbers (tuned for this exact x500 sim bag → keeps the
// ours-vs-LIO-SAM comparison fair).
struct PreintegrationConfig {
  double accel_noise_sigma{3.9939570888238808e-03};    // m/s^2/sqrt(Hz)
  double gyro_noise_sigma{1.5636343949698187e-03};     // rad/s/sqrt(Hz)
  double accel_bias_rw_sigma{6.4356659353532566e-05};  // m/s^3/sqrt(Hz)
  double gyro_bias_rw_sigma{3.5640318696367613e-05};   // rad/s^2/sqrt(Hz)
  double integration_sigma{1.0e-03};                   // integration uncertainty
  double gravity{9.8};                                 // |g| [m/s^2]; ENU → (0,0,-g)

  // Build the GTSAM base params for a plain ImuFactor: ENU up-gravity via
  // MakeSharedU, and accel/gyro/integration covariances = sigma^2·I. The bias
  // random-walk σ are deliberately NOT set here — the base PreintegrationParams
  // (LIO-SAM style, not the Combined* variant) carries no bias covariance; those
  // σ feed L5-03/L5-04's separate BetweenFactor<imuBias> instead.
  std::shared_ptr<gtsam::PreintegrationParams> toGtsamParams() const;
};

// Thin RAII wrapper over gtsam::PreintegratedImuMeasurements. Adds a dt>0 guard
// (bag replay yields ~0.4 % duplicate sim-time stamps → dt≤0, which aborts GTSAM)
// and a small typed API for the SLAM nodes. Rule of zero: the compiler-generated
// special members are correct (pim_ is copyable/movable).
class ImuPreintegrator {
 public:
  explicit ImuPreintegrator(const PreintegrationConfig& config,
                            const gtsam::imuBias::ConstantBias& bias = {});

  // Integrate one measurement over dt seconds. dt≤0 (or non-finite) is ignored
  // and returns false, so a duplicate/backwards stamp cannot abort integration.
  bool integrate(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double dt);

  // Predict the NavState at the end of the integrated interval from the state and
  // bias at the start (Forster 2017). Does not modify the integrator (const).
  gtsam::NavState predict(const gtsam::NavState& state_i,
                          const gtsam::imuBias::ConstantBias& bias) const;

  // Clear the integrated interval and set the linearization bias for the next
  // one (call at each keyframe with the newest optimized bias).
  void reset(const gtsam::imuBias::ConstantBias& bias);

  // The finalized accumulated measurement — feed this to gtsam::ImuFactor (L5-03).
  // The PIM is usable at any point; finish() is the read-side name of the API and
  // does not mutate the integrator.
  const gtsam::PreintegratedImuMeasurements& finish() const { return pim_; }

  std::size_t count() const { return count_; }
  double deltaTij() const { return pim_.deltaTij(); }
  Eigen::Vector3d deltaPij() const { return pim_.deltaPij(); }
  Eigen::Vector3d deltaVij() const { return pim_.deltaVij(); }
  gtsam::Rot3 deltaRij() const { return pim_.deltaRij(); }
  // 9x9 preintegration covariance, GTSAM order [rotation, position, velocity].
  Eigen::Matrix<double, 9, 9> covariance() const { return pim_.preintMeasCov(); }

 private:
  gtsam::PreintegratedImuMeasurements pim_;
  std::size_t count_{0};
};

}  // namespace graph_slam::imu
