#include "graph_slam/ndt_registrar.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>

namespace graph_slam {

namespace {

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix36d = Eigen::Matrix<double, 3, 6>;

// Skew-symmetric generators of rotation about x, y, z: d/dphi Rx = Gx * Rx, etc.
const Eigen::Matrix3d kGx = (Eigen::Matrix3d() << 0, 0, 0, 0, 0, -1, 0, 1, 0).finished();
const Eigen::Matrix3d kGy = (Eigen::Matrix3d() << 0, 0, 1, 0, 0, 0, -1, 0, 0).finished();
const Eigen::Matrix3d kGz = (Eigen::Matrix3d() << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished();

/// Magnusson (2009) d1/d2 score constants of the Gaussian/uniform mixture. Same
/// closed form PCL uses: c1 = 10(1-rho), c2 = rho / res^3, d3 = -ln c2,
/// d1 = -ln(c1+c2) - d3 (note: d1 < 0), d2 = -2 ln[(-ln(c1 e^{-1/2}+c2) - d3)/d1].
struct GaussConstants {
  double d1 = 0.0;
  double d2 = 0.0;
};

GaussConstants gaussConstants(double resolution, double outlier_ratio) {
  const double c1 = 10.0 * (1.0 - outlier_ratio);
  const double c2 = outlier_ratio / (resolution * resolution * resolution);
  const double d3 = -std::log(c2);
  GaussConstants gc;
  gc.d1 = -std::log(c1 + c2) - d3;
  gc.d2 = -2.0 * std::log((-std::log(c1 * std::exp(-0.5) + c2) - d3) / gc.d1);
  return gc;
}

// p = [tx, ty, tz, roll(about x), pitch(about y), yaw(about z)];
// R = Rz(yaw) Ry(pitch) Rx(roll).
Eigen::Matrix3d rotation(double roll, double pitch, double yaw) {
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

Vector6d poseFromMatrix(const Eigen::Matrix4d& m) {
  const Eigen::Matrix3d R = m.block<3, 3>(0, 0);
  Vector6d p;
  p.head<3>() = m.block<3, 1>(0, 3);
  p(3) = std::atan2(R(2, 1), R(2, 2));                        // roll
  p(4) = std::atan2(-R(2, 0), std::hypot(R(0, 0), R(1, 0)));  // pitch
  p(5) = std::atan2(R(1, 0), R(0, 0));                        // yaw
  return p;
}

Eigen::Matrix4d matrixFromPose(const Vector6d& p) {
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  m.block<3, 3>(0, 0) = rotation(p(3), p(4), p(5));
  m.block<3, 1>(0, 3) = p.head<3>();
  return m;
}

/// Accumulated NDT objective, gradient and Hessian at one pose. The objective
/// minimised is f(p) = sum_i d1 * exp(-d2/2 * q^T Sigma^-1 q) with d1 < 0, so
/// lower f = points pulled onto the cell Gaussians.
struct Derivatives {
  double f = 0.0;
  Vector6d g = Vector6d::Zero();
  Matrix6d H = Matrix6d::Zero();
  std::size_t scored = 0;
};

Derivatives evaluate(const NdtVoxelGrid& target, const Cloud& source, const Vector6d& p,
                     const GaussConstants& gc) {
  const Eigen::Matrix3d A = Eigen::AngleAxisd(p(5), Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Matrix3d B = Eigen::AngleAxisd(p(4), Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Matrix3d C = Eigen::AngleAxisd(p(3), Eigen::Vector3d::UnitX()).toRotationMatrix();
  const Eigen::Matrix3d R = A * B * C;
  const Eigen::Vector3d t = p.head<3>();

  // First derivatives of R wrt (roll, pitch, yaw).
  const Eigen::Matrix3d dR_dr = A * B * (kGx * C);
  const Eigen::Matrix3d dR_dp = A * (kGy * B) * C;
  const Eigen::Matrix3d dR_dy = (kGz * A) * B * C;
  // Second derivatives (only angle-angle pairs are nonzero).
  const Eigen::Matrix3d d2_rr = A * B * (kGx * kGx * C);
  const Eigen::Matrix3d d2_pp = A * (kGy * kGy * B) * C;
  const Eigen::Matrix3d d2_yy = (kGz * kGz * A) * B * C;
  const Eigen::Matrix3d d2_rp = A * (kGy * B) * (kGx * C);
  const Eigen::Matrix3d d2_ry = (kGz * A) * B * (kGx * C);
  const Eigen::Matrix3d d2_py = (kGz * A) * (kGy * B) * C;

  Derivatives out;
  for (const auto& pcl_pt : source.points) {
    const Eigen::Vector3d x(pcl_pt.x, pcl_pt.y, pcl_pt.z);
    const Eigen::Vector3d xw = R * x + t;
    const VoxelCell* cell = target.cell(target.voxelIndex(xw));
    if (cell == nullptr) {
      continue;  // empty cell -> no contribution
    }

    const Eigen::Vector3d q = xw - cell->mean;
    const Eigen::Matrix3d& Ci = cell->cov_inv;
    const Eigen::Vector3d Cq = Ci * q;
    const double E = std::exp(-gc.d2 * 0.5 * q.dot(Cq));
    const double w = -gc.d1 * gc.d2 * E;  // > 0 (d1 < 0): common gradient/Hessian weight

    // Point Jacobian J (3x6): translation cols = I, rotation cols = dR/dangle * x.
    Matrix36d J;
    J.col(0) = Eigen::Vector3d::UnitX();
    J.col(1) = Eigen::Vector3d::UnitY();
    J.col(2) = Eigen::Vector3d::UnitZ();
    J.col(3) = dR_dr * x;
    J.col(4) = dR_dp * x;
    J.col(5) = dR_dy * x;
    const Matrix36d CiJ = Ci * J;
    const Eigen::Matrix<double, 1, 6> CqJ = Cq.transpose() * J;  // q^T Sigma^-1 J_j

    // Second derivatives of the transformed point (angle-angle blocks only).
    const Eigen::Vector3d s33 = d2_rr * x;
    const Eigen::Vector3d s44 = d2_pp * x;
    const Eigen::Vector3d s55 = d2_yy * x;
    const Eigen::Vector3d s34 = d2_rp * x;
    const Eigen::Vector3d s35 = d2_ry * x;
    const Eigen::Vector3d s45 = d2_py * x;
    const auto pointSecond = [&](int j, int k) -> Eigen::Vector3d {
      int a = std::min(j, k);
      int b = std::max(j, k);
      if (a < 3) return Eigen::Vector3d::Zero();  // any translation index -> 0
      if (a == 3 && b == 3) return s33;
      if (a == 4 && b == 4) return s44;
      if (a == 5 && b == 5) return s55;
      if (a == 3 && b == 4) return s34;
      if (a == 3 && b == 5) return s35;
      return s45;  // (4,5)
    };

    out.f += gc.d1 * E;
    out.g += w * CqJ.transpose();
    for (int j = 0; j < 6; ++j) {
      for (int k = j; k < 6; ++k) {
        const double term =
            -gc.d2 * CqJ(j) * CqJ(k) + J.col(k).dot(CiJ.col(j)) + Cq.dot(pointSecond(j, k));
        const double val = w * term;
        out.H(j, k) += val;
        if (k != j) out.H(k, j) += val;
      }
    }
    ++out.scored;
  }
  return out;
}

}  // namespace

MeasurementCovariance measurementCovariance(const Matrix6d& H, const NdtConfig& config) {
  MeasurementCovariance out;

  // Symmetric eigendecomposition: lets us both test conditioning and rebuild a
  // clean, symmetric-PD inverse (H^-1 = V diag(1/lambda) V^T) in one shot.
  Eigen::SelfAdjointEigenSolver<Matrix6d> es(H);
  const double min_eig = es.eigenvalues()(0);  // ascending order
  const double max_eig = es.eigenvalues()(5);
  const bool degenerate =
      min_eig <= 0.0 || (max_eig / min_eig) > config.sigma_max_condition;

  if (!degenerate) {
    const Matrix6d Hinv =
        es.eigenvectors() * es.eigenvalues().cwiseInverse().asDiagonal() *
        es.eigenvectors().transpose();
    const Matrix6d cov = config.cov_scale_factor * config.hessian_lambda * Hinv;
    // Reorder [tx,ty,tz, rx,ry,rz] -> [rx,ry,rz, tx,ty,tz]: swap the two diagonal
    // 3x3 blocks and transpose-swap the off-diagonal coupling blocks. A symmetric
    // permutation preserves positive-definiteness.
    Matrix6d reordered;
    reordered.block<3, 3>(0, 0) = cov.block<3, 3>(3, 3);  // rot-rot
    reordered.block<3, 3>(3, 3) = cov.block<3, 3>(0, 0);  // trans-trans
    reordered.block<3, 3>(0, 3) = cov.block<3, 3>(3, 0);  // rot-trans
    reordered.block<3, 3>(3, 0) = cov.block<3, 3>(0, 3);  // trans-rot
    out.sigma = reordered.cast<float>();
    out.from_hessian = true;
    return out;
  }

  // Fallback: geometry-aware diagonal, already in Pose3 order (rot x3, trans x3).
  const auto sr2 = static_cast<float>(config.sigma_rot_fallback * config.sigma_rot_fallback);
  const auto st2 = static_cast<float>(config.sigma_t_fallback * config.sigma_t_fallback);
  out.sigma.diagonal() << sr2, sr2, sr2, st2, st2, st2;
  out.from_hessian = false;
  return out;
}

RegistrationResult NdtRegistrar::align(const NdtVoxelGrid& target, const CloudConstPtr& source,
                                       const Eigen::Matrix4f& init_guess) const {
  RegistrationResult result;  // converged=false, identity transform, zero Hessian
  if (!source || source->empty() || target.numVoxels() == 0) {
    return result;
  }

  const GaussConstants gc = gaussConstants(target.resolution(), config_.outlier_ratio);
  Vector6d p = poseFromMatrix(init_guess.cast<double>());

  Derivatives d = evaluate(target, *source, p, gc);
  for (int iter = 1; iter <= config_.max_iterations; ++iter) {
    result.iterations = iter;
    if (d.scored == 0) {
      break;  // nothing landed in the grid -> cannot improve
    }

    // Newton direction -H^-1 g, with a positive-definite guard on H so it is a
    // descent direction even far from the optimum.
    // The raw NDT Hessian is indefinite far from the optimum (the planar cells
    // give a hugely anisotropic Sigma^-1). Floor the eigenvalues *relative* to the
    // largest one so Hr is positive-definite AND well-conditioned — an absolute
    // floor would leave Hr near-singular and the Newton step would explode.
    Matrix6d Hr = d.H;
    Eigen::SelfAdjointEigenSolver<Matrix6d> es(Hr, Eigen::EigenvaluesOnly);
    const double min_eig = es.eigenvalues()(0);
    const double max_eig = es.eigenvalues()(5);
    const double floor = std::max(config_.regularization, 1e-3 * max_eig);
    if (min_eig < floor) {
      Hr.diagonal().array() += (floor - min_eig);
    }
    Vector6d direction = Hr.ldlt().solve(d.g);
    // Trust radius: cap the per-iteration pose change. The NDT cost is stiff and
    // uses hard voxel assignment, so a large step ejects most points from their
    // cells; keeping the step small (relative to the cell size) keeps the
    // quadratic model valid. Near the optimum the Newton step is already small,
    // so this only bites far away.
    const double kTrust = 0.05;
    if (direction.norm() > kTrust) {
      direction *= kTrust / direction.norm();
    }

    // Backtracking line search: shrink the step until the objective decreases.
    double alpha = 1.0;
    Derivatives next;
    bool improved = false;
    for (int ls = 0; ls < 12; ++ls) {
      next = evaluate(target, *source, p - alpha * direction, gc);
      if (next.scored > 0 && next.f < d.f) {
        improved = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!improved) {
      result.converged = true;  // no decreasing step found -> at a (local) minimum
      break;
    }

    const Vector6d step = alpha * direction;
    p -= step;
    d = next;
    if (step.norm() < config_.step_epsilon) {
      result.converged = true;
      break;
    }
  }

  // d is evaluated at the final pose, so its Hessian/score are consistent with
  // the returned transform (NDT-12 reads result.hessian).
  result.transform = matrixFromPose(p).cast<float>();
  result.hessian = d.H;
  result.fitness_score = d.scored > 0 ? d.f / static_cast<double>(d.scored) : 0.0;

  // NDT-12: measurement covariance from the same Hessian (Sigma = lambda * H^-1,
  // reordered to Pose3 order), with the geometry-aware diagonal fallback when H is
  // degenerate. Downstream wiring into GTSAM is Sprint 3, not here.
  const MeasurementCovariance cov = measurementCovariance(d.H, config_);
  result.sigma_meas = cov.sigma;
  result.sigma_from_hessian = cov.from_hessian;
  return result;
}

}  // namespace graph_slam
