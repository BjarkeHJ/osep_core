cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)
project(osep_skeleton_decomp)

# Set compile constraints
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROS2 dependencies (msg)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
# ROS2 dependencies (run-time)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
# Other libraries
find_package(Eigen3 REQUIRED)
find_package(PCL 1.12 REQUIRED)

# Generata message interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
                        "msg/Vertex.msg"
                        "msg/GlobalSkeleton.msg"
                        DEPENDENCIES std_msgs geometry_msgs builtin_interfaces)
ament_export_dependencies(rosidl_default_runtime)

# Define executable
set(EXEC_ROSA osep_rosa)
set(EXEC_GSKEL osep_gskel)
add_executable(${EXEC_ROSA}
                src/rosa_node.cpp
                src/rosa.cpp)

add_executable(${EXEC_GSKEL}
                src/gskel_node.cpp
                src/gskel.cpp)

# Dependencies list
set(dependencies
    rclcpp
    pcl_conversions
    tf2_ros
    tf2_sensor_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    std_msgs)


foreach(tgt IN ITEMS ${EXEC_ROSA} ${EXEC_GSKEL})

        target_include_directories(${tgt} PUBLIC
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>
                        ${PCL_INCLUDE_DIRS})

        target_link_libraries(${tgt} 
                        ${PCL_LIBRARIES} 
                        Eigen3::Eigen)

        ament_target_dependencies(${tgt} 
                        ${dependencies})

        target_compile_features(${tgt} 
                        PUBLIC cxx_std_17)

        rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
        target_link_libraries(${tgt} ${cpp_typesupport_target})

endforeach()

install(TARGETS 
        ${EXEC_ROSA}
        ${EXEC_GSKEL} 
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