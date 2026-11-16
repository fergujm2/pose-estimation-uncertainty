cmake_minimum_required(VERSION 3.8)
project(charuco_pose_estimation)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(Eigen3 REQUIRED)
# Pin the /usr/local GTSAM 4.3 install, as pinpoint does. NO_DEFAULT_PATH because the sourced
# workspace's CMAKE_PREFIX_PATH (searched before HINTS) contains ves_smoother's GTSAMConfig.cmake,
# whose installed headers lack NoiseModelFactorN.
find_package(GTSAM 4.3 REQUIRED PATHS /usr/local/lib/cmake/GTSAM NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED)

# Interface library for executables
add_library(external_libs INTERFACE)
target_include_directories(external_libs INTERFACE ${GTSAM_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(external_libs INTERFACE ${OpenCV_LIBS} gtsam)

# Charuco tracker executable
add_executable(pose_estimator
  src/main.cpp
  src/pose_estimator_node.cpp
  src/corner_factor.cpp
)

target_compile_features(pose_estimator PUBLIC c_std_99 cxx_std_17)
target_include_directories(pose_estimator PRIVATE src)
target_link_libraries(pose_estimator external_libs)

ament_target_dependencies(pose_estimator
    rclcpp
    geometry_msgs
    sensor_msgs
    std_msgs
    cv_bridge
    visualization_msgs
)

install(
  TARGETS pose_estimator
  DESTINATION lib/${PROJECT_NAME}
)

# Test image/camera info publisher
add_executable(test_data_publisher
  src/test_data_publisher.cpp
)

target_compile_features(test_data_publisher PUBLIC c_std_99 cxx_std_17)
target_include_directories(test_data_publisher PRIVATE src)
target_link_libraries(test_data_publisher external_libs)

ament_target_dependencies(test_data_publisher
    rclcpp
    sensor_msgs
    cv_bridge
    ament_index_cpp
)

install(
  TARGETS test_data_publisher
  DESTINATION lib/${PROJECT_NAME}
)

# Other directory installs
install(
  DIRECTORY test_data
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# ves_smoother puts its own libgtsam.so.4 on LD_LIBRARY_PATH. DT_RPATH (unlike RUNPATH) is searched
# before it, so these always load /usr/local's GTSAM, matching the headers built against.
foreach(target pose_estimator test_data_publisher)
  set_target_properties(${target} PROPERTIES BUILD_RPATH /usr/local/lib INSTALL_RPATH /usr/local/lib)
  target_link_options(${target} PRIVATE -Wl,--disable-new-dtags)
endforeach()

ament_package()