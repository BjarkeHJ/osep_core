cmake_minimum_required(VERSION 3.8)
project(osep_planner)

# Core
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ROS deps used at runtime
find_package(rclpy REQUIRED)
# find_package(sensor_msgs REQUIRED)
# find_package(geometry_msgs REQUIRED)

# Install the Python package directory "osep_planner/"
# (this makes `import osep_planner` work)
ament_python_install_package(${PROJECT_NAME})

# Install ROS executable scripts from scripts/
# (ensure the files are +x)
install(
    PROGRAMS scripts/exec_planner
    DESTINATION lib/${PROJECT_NAME}
)

# Install launch files (optional)
install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
