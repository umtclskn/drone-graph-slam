#include "graph_slam/registration_gate.hpp"

#include <Eigen/Eigenvalues>

namespace graph_slam {

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

  return RegistrationStatus::Reliable;
}

}  // namespace graph_slam
