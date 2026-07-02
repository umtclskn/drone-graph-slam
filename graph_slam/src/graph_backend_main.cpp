#include <rclcpp/rclcpp.hpp>

#include "graph_backend_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(graph_slam::createGraphBackendNode());
  rclcpp::shutdown();
  return 0;
}
