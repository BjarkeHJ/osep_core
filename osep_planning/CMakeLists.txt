cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)
project(osep_planning)

# Set compile constraints
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROS 2 + interface tooling
find_package(ament_cmake REQUIRED)

# Other deps for your nodes
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.12 REQUIRED)
find_package(osep_skeleton_decomp REQUIRED)

set(dependencies
  rclcpp
  pcl_conversions
  tf2_ros
  tf2_sensor_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  osep_skeleton_decomp
)

# Define executable
set(EXEC_PLAN osep_planner)
add_executable(${EXEC_PLAN}
                src/planner_node.cpp
                src/viewpoint_manager.cpp
                src/planner.cpp)


target_include_directories(${EXEC_PLAN} PUBLIC
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>
                        ${PCL_INCLUDE_DIRS})

target_link_libraries(${EXEC_PLAN} 
                        ${PCL_LIBRARIES} 
                        Eigen3::Eigen)

ament_target_dependencies(${EXEC_PLAN} 
                        ${dependencies})

target_compile_features(${EXEC_PLAN} 
                        PUBLIC cxx_std_17)

install(TARGETS 
        ${EXEC_PLAN}
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
        DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()