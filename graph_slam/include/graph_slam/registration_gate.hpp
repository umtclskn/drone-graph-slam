#pragma once

#include "graph_slam/registration_result.hpp"

namespace graph_slam {

/// Verdict of the registration gate (NDT-11): is this measurement trustworthy
/// enough to become a graph edge? Anything other than `Reliable` is a rejection.
enum class RegistrationStatus { Reliable, NotConverged, PoorFit, Degenerate };

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
};

/// Decide whether a RegistrationResult is reliable enough to enter the graph.
/// Pure function, ROS-free. Checks, in order: convergence -> fitness -> Hessian
/// conditioning. The Hessian check catches symmetric/ambiguous scenes where the
/// fit looks good but the pose is under-determined along some direction — the
/// only signal for that is the degenerate direction of `result.hessian`.
RegistrationStatus evaluateRegistration(const RegistrationResult& result,
                                        const RegistrationGateConfig& config);

}  // namespace graph_slam
