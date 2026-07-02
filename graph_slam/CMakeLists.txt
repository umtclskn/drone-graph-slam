cmake_minimum_required(VERSION 3.8)
project(graph_slam)

# --- Language standard ------------------------------------------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Dependencies -----------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
# common: pcl::PointCloud (NDT-01). filters: CropBox/VoxelGrid/removeNaN (NDT-03
# Preprocessor). registration: pcl::NormalDistributionsTransform (NDT-09).
find_package(PCL REQUIRED COMPONENTS common filters registration)
find_package(GTSAM REQUIRED)

# ROS deps are used ONLY by the EVAL-03 overlay node executable (below); the core
# graph_slam library stays ROS-free.
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(ament_index_cpp REQUIRED)

# --- Core library -----------------------------------------------------------
# graph_slam carries compiled algorithm classes (NDT-03 Preprocessor, NDT-05/06
# NdtVoxelGrid, NDT-09 hand-rolled NdtRegistrar + PclNdtBaseline cross-check) on
# top of the point_types.hpp aliases.
add_library(${PROJECT_NAME}
  src/preprocessor.cpp
  src/quality_checker.cpp
  src/ndt_voxel_grid.cpp
  src/ndt_registrar.cpp
  src/pcl_ndt_baseline.cpp
  src/registration_gate.cpp
  src/relative_error.cpp
  src/trajectory_metrics.cpp
  src/graph/graph_bootstrap.cpp
  src/graph/odometry_accumulator.cpp
  src/graph/odometry_factor_builder.cpp
  src/graph/graph_optimizer.cpp
  src/loop/loop_closure_candidate_finder.cpp
  src/loop/loop_closure_verifier.cpp
  src/eval/graph_visualizer.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  ${PCL_INCLUDE_DIRS})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${PCL_DEFINITIONS})
# PUBLIC: the public headers expose PCL point types (pcl_common via
# point_types.hpp) and Eigen (crop bounds in PreprocessConfig), so consumers
# need them. PRIVATE: pcl_filters is an implementation detail of preprocessor.cpp
# and never appears in a header.
target_link_libraries(${PROJECT_NAME}
  PUBLIC ${PCL_COMMON_LIBRARIES} Eigen3::Eigen gtsam
  PRIVATE ${PCL_FILTERS_LIBRARIES} ${PCL_REGISTRATION_LIBRARIES})

# --- EVAL-03 overlay node (ROS executable; core lib stays ROS-free) ---------
add_executable(eval03_overlay_node src/eval03_overlay_node.cpp)
target_link_libraries(eval03_overlay_node ${PROJECT_NAME})
ament_target_dependencies(eval03_overlay_node
  rclcpp sensor_msgs std_srvs geometry_msgs tf2_ros pcl_conversions ament_index_cpp)

# --- EVAL-02 per-pair relative-error node (ROS executable; lib stays ROS-free) -
add_executable(eval02_relative_error_node src/eval02_relative_error_node.cpp)
target_link_libraries(eval02_relative_error_node ${PROJECT_NAME})
ament_target_dependencies(eval02_relative_error_node
  rclcpp sensor_msgs geometry_msgs nav_msgs pcl_conversions)

# --- NDT-13 full front-end node (ROS executable; lib stays ROS-free) ----------
# The node class + factory live in ndt_frontend_node.cpp (no main, so the gtest
# can link it); ndt_frontend_main.cpp is the thin entry point. The private header
# ndt_frontend_node.hpp is in src/ and not installed.
add_executable(ndt_frontend_node src/ndt_frontend_node.cpp src/ndt_frontend_main.cpp)
target_include_directories(ndt_frontend_node PRIVATE src)
target_link_libraries(ndt_frontend_node ${PROJECT_NAME})
ament_target_dependencies(ndt_frontend_node
  rclcpp sensor_msgs nav_msgs geometry_msgs pcl_conversions tf2_ros)

# --- GRAPH-VIS-01 graph visualization node (ROS executable) -----------------
add_executable(graph_visualizer_node src/graph_visualizer_node.cpp)
target_link_libraries(graph_visualizer_node ${PROJECT_NAME})
ament_target_dependencies(graph_visualizer_node
  rclcpp nav_msgs geometry_msgs visualization_msgs tf2_ros)

# --- GRAPH-VIS-01 graph back-end node (ROS executable) ----------------------
# Subscribes to /ndt_frontend/ndt_odom; drives KeyframePolicy + OdometryAccumulator
# + GraphOptimizer (iSAM2); publishes /slam/* + map→odom TF.
add_executable(graph_backend_node src/graph_backend_node.cpp src/graph_backend_main.cpp)
target_include_directories(graph_backend_node PRIVATE src)
target_link_libraries(graph_backend_node ${PROJECT_NAME})
ament_target_dependencies(graph_backend_node
  rclcpp nav_msgs geometry_msgs visualization_msgs tf2_ros std_msgs
  sensor_msgs pcl_conversions)

# --- Install ----------------------------------------------------------------
install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(TARGETS eval03_overlay_node DESTINATION lib/${PROJECT_NAME})
install(TARGETS eval02_relative_error_node DESTINATION lib/${PROJECT_NAME})
install(TARGETS ndt_frontend_node DESTINATION lib/${PROJECT_NAME})
install(TARGETS graph_visualizer_node DESTINATION lib/${PROJECT_NAME})
install(TARGETS graph_backend_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch rviz scripts config DESTINATION share/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(PCL Eigen3 GTSAM)

# --- Tests ------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_preprocessor test/test_preprocessor.cpp)
  target_link_libraries(test_preprocessor ${PROJECT_NAME})
  ament_add_gtest(test_quality_checker test/test_quality_checker.cpp)
  target_link_libraries(test_quality_checker ${PROJECT_NAME})
  ament_add_gtest(test_ndt_voxel_grid test/test_ndt_voxel_grid.cpp)
  target_link_libraries(test_ndt_voxel_grid ${PROJECT_NAME})
  ament_add_gtest(test_ndt_registrar test/test_ndt_registrar.cpp)
  target_link_libraries(test_ndt_registrar ${PROJECT_NAME})
  ament_add_gtest(test_registration_gate test/test_registration_gate.cpp)
  target_link_libraries(test_registration_gate ${PROJECT_NAME})
  ament_add_gtest(test_ndt_covariance test/test_ndt_covariance.cpp)
  target_link_libraries(test_ndt_covariance ${PROJECT_NAME})
  ament_add_gtest(test_eval03_overlay test/test_eval03_overlay.cpp)
  target_link_libraries(test_eval03_overlay ${PROJECT_NAME})
  ament_add_gtest(test_relative_error test/test_relative_error.cpp)
  target_link_libraries(test_relative_error ${PROJECT_NAME})
  ament_add_gtest(test_trajectory_metrics test/test_trajectory_metrics.cpp)
  target_link_libraries(test_trajectory_metrics ${PROJECT_NAME})
  # SLAM-01: keyframe policy is header-only (graph/keyframe_policy.hpp); the test
  # only needs the package include dir + Eigen, not the compiled library.
  ament_add_gtest(test_keyframe_policy test/test_keyframe_policy.cpp)
  target_include_directories(test_keyframe_policy PRIVATE include)
  target_link_libraries(test_keyframe_policy Eigen3::Eigen)
  ament_add_gtest(test_graph_bootstrap test/test_graph_bootstrap.cpp)
  target_link_libraries(test_graph_bootstrap ${PROJECT_NAME})
  ament_add_gtest(test_odometry_accumulator test/test_odometry_accumulator.cpp)
  target_link_libraries(test_odometry_accumulator ${PROJECT_NAME})
  ament_add_gtest(test_odometry_factor_builder test/test_odometry_factor_builder.cpp)
  target_link_libraries(test_odometry_factor_builder ${PROJECT_NAME})
  ament_add_gtest(test_graph_optimizer test/test_graph_optimizer.cpp)
  target_link_libraries(test_graph_optimizer ${PROJECT_NAME})
  ament_add_gtest(test_graph_visualizer test/test_graph_visualizer.cpp)
  target_link_libraries(test_graph_visualizer ${PROJECT_NAME})
  # SLAM-09: loop-closure candidate finder (ROS-free; links the compiled library).
  ament_add_gtest(test_loop_candidate_finder test/test_loop_closure_candidate_finder.cpp)
  target_link_libraries(test_loop_candidate_finder ${PROJECT_NAME})
  # GRAPH-VIS-01: back-end node smoke test — links the node TU directly (no main).
  ament_add_gtest(test_graph_backend_node
    test/test_graph_backend_node.cpp src/graph_backend_node.cpp)
  target_include_directories(test_graph_backend_node PRIVATE src)
  target_link_libraries(test_graph_backend_node ${PROJECT_NAME})
  ament_target_dependencies(test_graph_backend_node
    rclcpp nav_msgs geometry_msgs visualization_msgs tf2_ros std_msgs
    sensor_msgs pcl_conversions)
  # SLAM-10: loop-closure verifier (ROS-free; links the compiled library).
  ament_add_gtest(test_loop_closure_verifier test/test_loop_closure_verifier.cpp)
  target_link_libraries(test_loop_closure_verifier ${PROJECT_NAME})
  # NDT-13 node smoke test: links the node TU (factory, no main) directly so it can
  # construct the node; needs the src/ private header and the node's ROS deps.
  ament_add_gtest(test_ndt_frontend_node
    test/test_ndt_frontend_node.cpp src/ndt_frontend_node.cpp)
  target_include_directories(test_ndt_frontend_node PRIVATE src)
  target_link_libraries(test_ndt_frontend_node ${PROJECT_NAME})
  ament_target_dependencies(test_ndt_frontend_node
    rclcpp sensor_msgs nav_msgs geometry_msgs pcl_conversions tf2_ros)

  # INFRA-01: TF chain smoke test (map->odom->base_link lookup after first scan).
  ament_add_gtest(test_tf_tree test/test_tf_tree.cpp src/ndt_frontend_node.cpp)
  target_include_directories(test_tf_tree PRIVATE src)
  target_link_libraries(test_tf_tree ${PROJECT_NAME})
  ament_target_dependencies(test_tf_tree
    rclcpp sensor_msgs nav_msgs geometry_msgs pcl_conversions tf2_ros)

  # NDT-14 replay-harness logic: ROS-free pytest of the baseline YAML parse and
  # the pass/fail evaluation (the harness lazy-imports rclpy, so this stays
  # simulator-free). The live bag replay is run by hand via the script.
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_ndt14_harness test/test_ndt14_harness.py)
endif()

ament_package()
