#include "graph_slam/loop/loop_closure_verifier.hpp"

#include <utility>

#include <Eigen/Eigenvalues>

#include "graph_slam/registration_result.hpp"

namespace graph_slam {

LoopClosureVerifier::LoopClosureVerifier() : LoopClosureVerifier(Params{}) {}

LoopClosureVerifier::LoopClosureVerifier(Params p)
    : params_(std::move(p)), registrar_(params_.ndt_config) {}

VerificationResult LoopClosureVerifier::verify(
    const CloudPtr& query_cloud, const CloudPtr& match_cloud,
    const Eigen::Isometry3d& initial_guess) const {
  VerificationResult out;

  // Build the NDT target from the (older) match keyframe, then align the (newer)
  // query cloud onto it. The recovered transform maps query -> match, i.e.
  // T_match_query (ARCHITECTURE §8).
  NdtVoxelGrid grid(params_.grid_config);
  grid.build(match_cloud);

  const Eigen::Matrix4f guess = initial_guess.matrix().cast<float>();
  const RegistrationResult r = registrar_.align(grid, query_cloud, guess);
  out.ndt_score = r.fitness_score;

  // Gate 1 — convergence. A non-converged align (also: empty clouds / empty grid)
  // is never a loop closure.
  if (!r.converged) {
    out.reason = "not_converged";
    return out;
  }

  // Gate 2 — fit strength. fitness_score is "lower = better"; reject when it is
  // not negative enough (the two clouds do not actually overlap). Stricter than
  // the front-end gate: a loop edge must be a confident match.
  if (r.fitness_score > params_.min_ndt_score) {
    out.reason = "weak_ndt_score";
    return out;
  }

  // Gate 3 — geometric conditioning. Sigma_meas must be well-conditioned
  // (ARCHITECTURE §8). The signal is the cost Hessian (= Sigma_meas^-1, the
  // information matrix): a corridor / plane leaves a near-zero eigenvalue along
  // the unconstrained direction and a huge condition number. This is the same
  // conditioning test the NDT-11 registration gate applies — a false closure that
  // shrinks Sigma_post while increasing error (ARCHITECTURE §6) is exactly what
  // it stops.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(
      r.hessian, Eigen::EigenvaluesOnly);
  const double min_eig = solver.eigenvalues()(0);  // ascending
  const double max_eig = solver.eigenvalues()(5);
  out.sigma_min_eigenvalue = min_eig;

  if (min_eig < params_.min_sigma_eigenvalue) {
    out.reason = "degenerate_geometry";  // a pose direction is unconstrained
    return out;
  }
  if (max_eig / min_eig > params_.max_condition_number) {  // min_eig > 0 here
    out.reason = "ill_conditioned";  // one direction far softer than the rest
    return out;
  }

  // Accepted: fill the relative pose and the GTSAM noise model from Sigma_meas
  // (already stored in Pose3 tangent order [rx,ry,rz,x,y,z], NDT-12).
  out.accepted = true;
  out.relative_pose = Eigen::Isometry3d(r.transform.cast<double>());
  out.noise_model = gtsam::noiseModel::Gaussian::Covariance(r.sigma_meas.cast<double>());
  return out;
}

}  // namespace graph_slam
