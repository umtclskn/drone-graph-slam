#pragma once

#include "graph_slam/registration_result.hpp"

namespace graph_slam {

/// Verdict of the registration gate (NDT-11 + L5-20): is this measurement
/// trustworthy enough to become a graph edge? Anything other than `Reliable` is
/// a rejection.
enum class RegistrationStatus {
  Reliable,
  NotConverged,
  PoorFit,
  Degenerate,
  LowSupport,         // L5-20: too few scan points scored in the target grid
  PriorInconsistent,  // L5-20: NDT solution far from the initial guess
};

/// Human-readable name of a verdict, for node logs. Centralised here so the ROS
/// nodes (eval02/eval03/ndt_frontend) share one spelling instead of each copying
/// a local switch.
const char* toString(RegistrationStatus status);

/// Plain configuration for the registration gate. No ROS, no hidden defaults;
/// thresholds are setup-dependent (resolution / outlier_ratio) and meant to be
/// tuned via INFRA-02 — the defaults are only sane starting points.
struct RegistrationGateConfig {
  /// fitness_score convention: LOWER (more negative) = better fit. This MATCHES
  /// NdtRegistrar, whose fitness_score is the mean per-point Magnusson score
  /// (~d1 < 0 for a good fit, ~0 for a poor one). A measurement is rejected as
  /// PoorFit when its fitness_score is ABOVE this threshold (i.e. not negative
  /// enough). Inverting this would accept bad fits and reject good ones.
  double max_fitness_score = -0.5;

  /// Smallest allowed eigenvalue of the cost Hessian. Below this, a pose direction
  /// is (near-)unconstrained -> Degenerate.
  double min_hessian_eigenvalue = 1e-3;

  /// Largest allowed Hessian condition number (max/min eigenvalue). Above this,
  /// one direction is far softer than the others (ambiguous scene) -> Degenerate.
  double max_condition_number = 1e4;

  /// L5-20: minimum fraction of source points that must score in a non-empty
  /// target voxel. Below this -> LowSupport (match may be an artefact on a
  /// narrow structure). Conservative starting point; retune from bag stats.
  double min_scored_fraction = 0.3;

  /// L5-20: max translation of ξ = Log(T_guess^-1 · T_ndt) [m]. Rough 5σ of the
  /// EKF2 keyframe→scan envelope under the current keyframe policy. Above ->
  /// PriorInconsistent.
  double max_guess_delta_t = 0.5;

  /// L5-20: max rotation of ξ [rad] (~10°). Same envelope rationale as δ_t.
  double max_guess_delta_rot = 0.17;
};

/// Decide whether a RegistrationResult is reliable enough to enter the graph.
/// Pure function, ROS-free. Checks, in order: convergence -> fitness -> Hessian
/// conditioning -> support fraction -> guess consistency. The Hessian check
/// catches symmetric/ambiguous scenes; the L5-20 checks catch narrow-structure
/// artefacts and NDT wander far from the (independent) initial guess.
RegistrationStatus evaluateRegistration(const RegistrationResult& result,
                                        const RegistrationGateConfig& config);

}  // namespace graph_slam
