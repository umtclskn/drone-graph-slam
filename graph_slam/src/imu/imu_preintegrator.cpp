// L5-01: ImuPreintegrator implementation (see imu_preintegrator.hpp).
#include "graph_slam/imu/imu_preintegrator.hpp"

#include <cmath>

#include <gtsam/base/Matrix.h>

namespace graph_slam::imu {

std::shared_ptr<gtsam::PreintegrationParams> PreintegrationConfig::toGtsamParams() const {
  // ENU / z-up navigation frame → n_gravity = (0, 0, -gravity).
  auto params = gtsam::PreintegrationParams::MakeSharedU(gravity);
  params->setAccelerometerCovariance(gtsam::I_3x3 * (accel_noise_sigma * accel_noise_sigma));
  params->setGyroscopeCovariance(gtsam::I_3x3 * (gyro_noise_sigma * gyro_noise_sigma));
  params->setIntegrationCovariance(gtsam::I_3x3 * (integration_sigma * integration_sigma));
  return params;
}

ImuPreintegrator::ImuPreintegrator(const PreintegrationConfig& config,
                                   const gtsam::imuBias::ConstantBias& bias)
    : pim_(config.toGtsamParams(), bias) {}

bool ImuPreintegrator::integrate(const Eigen::Vector3d& accel,
                                 const Eigen::Vector3d& gyro, double dt) {
  if (!std::isfinite(dt) || dt <= 0.0) {
    return false;  // duplicate/backwards/NaN stamp — GTSAM would assert on dt<=0.
  }
  pim_.integrateMeasurement(accel, gyro, dt);
  ++count_;
  return true;
}

gtsam::NavState ImuPreintegrator::predict(
    const gtsam::NavState& state_i, const gtsam::imuBias::ConstantBias& bias) const {
  return pim_.predict(state_i, bias);
}

void ImuPreintegrator::reset(const gtsam::imuBias::ConstantBias& bias) {
  pim_.resetIntegrationAndSetBias(bias);
  count_ = 0;
}

}  // namespace graph_slam::imu
