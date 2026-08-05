#include "graph_slam/registration_gate.hpp"

#include "graph_slam/initial_guess.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace graph_slam {
namespace {

/// Rotation angle (rad) encoded by the 3x3 block of an SE(3) matrix.
double rotationAngle(const Eigen::Matrix4f& m) {
  const double trace = m.block<3, 3>(0, 0).cast<double>().trace();
  return std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
}

}  // namespace

const char* toString(RegistrationStatus status) {
  switch (status) {
    case RegistrationStatus::Reliable:
      return "Reliable";
    case RegistrationStatus::NotConverged:
      return "NotConverged";
    case RegistrationStatus::PoorFit:
      return "PoorFit";
    case RegistrationStatus::Degenerate:
      return "Degenerate";
    case RegistrationStatus::LowSupport:
      return "LowSupport";
    case RegistrationStatus::PriorInconsistent:
      return "PriorInconsistent";
  }
  return "?";
}

RegistrationStatus evaluateRegistration(const RegistrationResult& result,
                                        const RegistrationGateConfig& config) {
  // 1. The optimiser must have converged (also catches empty / too-few-points
  //    inputs: NdtRegistrar returns converged=false when nothing scores).
  if (!result.converged) {
    return RegistrationStatus::NotConverged;
  }

  // 2. The fit must be good enough. fitness_score is "lower = better" (see
  //    RegistrationGateConfig); reject when it is not negative enough.
  if (result.fitness_score > config.max_fitness_score) {
    return RegistrationStatus::PoorFit;
  }

  // 3. The pose must be well-constrained in all six directions. A symmetric /
  //    ambiguous scene leaves a near-zero Hessian eigenvalue even when the fit is
  //    good, so this is the only check that catches it.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(result.hessian,
                                                                    Eigen::EigenvaluesOnly);
  const double min_eig = solver.eigenvalues()(0);  // ascending
  const double max_eig = solver.eigenvalues()(5);
  if (min_eig < config.min_hessian_eigenvalue) {
    return RegistrationStatus::Degenerate;
  }
  if (max_eig / min_eig > config.max_condition_number) {  // min_eig > 0 here
    return RegistrationStatus::Degenerate;
  }

  // 4. L5-20: enough of the scan must have scored. A fit on a thin strip can look
  //    good in fitness/Hessian while the rest of the cloud missed the target.
  if (result.total_points == 0) {
    return RegistrationStatus::LowSupport;
  }
  const double support_fraction =
      static_cast<double>(result.scored_points) / static_cast<double>(result.total_points);
  if (support_fraction < config.min_scored_fraction) {
    return RegistrationStatus::LowSupport;
  }

  // 5. L5-20: NDT must stay inside the prior's error envelope — but only when a
  //    real prior seeded the guess. Identity-as-fallback means "no information"
  //    (node could not look up EKF2), not "predict zero motion"; applying the
  //    envelope there wrongly rejects first large corrections (measured: one
  //    early ~11° yaw with guess t=0 raised XY ATE 0.08 → 0.22 m vs L5-19).
  if (result.have_prior_guess) {
    const Eigen::Matrix4f delta =
        relativePoseGuess(result.initial_guess, result.transform);
    const double delta_t = delta.block<3, 1>(0, 3).cast<double>().norm();
    const double delta_r = rotationAngle(delta);
    if (delta_t > config.max_guess_delta_t || delta_r > config.max_guess_delta_rot) {
      return RegistrationStatus::PriorInconsistent;
    }
  }

  return RegistrationStatus::Reliable;
}

}  // namespace graph_slam
