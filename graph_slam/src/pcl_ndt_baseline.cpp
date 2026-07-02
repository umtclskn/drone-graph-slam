#include "graph_slam/pcl_ndt_baseline.hpp"

#include <pcl/registration/ndt.h>

namespace graph_slam {

RegistrationResult PclNdtBaseline::align(const CloudConstPtr& source, const CloudConstPtr& target,
                                         const Eigen::Matrix4f& init_guess) const {
  RegistrationResult result;  // converged=false, identity transform by default
  if (!source || !target || source->empty() || target->empty()) {
    return result;
  }

  pcl::NormalDistributionsTransform<PointT, PointT> ndt;
  ndt.setResolution(static_cast<float>(config_.resolution));
  ndt.setStepSize(config_.step_size);
  ndt.setTransformationEpsilon(config_.transformation_epsilon);
  ndt.setMaximumIterations(config_.max_iterations);
  ndt.setInputSource(source);
  ndt.setInputTarget(target);

  Cloud aligned;
  ndt.align(aligned, init_guess);

  result.transform = ndt.getFinalTransformation();
  result.converged = ndt.hasConverged();
  result.iterations = ndt.getFinalNumIteration();
  result.fitness_score = ndt.getFitnessScore();
  return result;
}

}  // namespace graph_slam
