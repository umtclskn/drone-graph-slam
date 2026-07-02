// NDT-13: thin entry point. The node lives in ndt_frontend_node.cpp behind the
// createNdtFrontendNode() factory so the same translation unit links into both
// this executable and the gtest smoke test (a .cpp with main() cannot).

#include <rclcpp/rclcpp.hpp>

#include "ndt_frontend_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(graph_slam::createNdtFrontendNode());
  rclcpp::shutdown();
  return 0;
}
