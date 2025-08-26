#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <eigen3/Eigen/Dense>
#include <tf2/utils.h>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <unordered_map>

// Define a struct for A* nodes
struct PlannerNode {
    int x, y;
    float cost;
    bool operator>(const PlannerNode &other) const {
        return cost > other.cost;
    }
};

class PathInterpolator : public rclcpp::Node {
public:
    // --- Constructor ---
    PathInterpolator();

private:
    // --- ROS2 Subscriptions ---
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr viewpoints_sub_;

    // --- ROS2 Publishers ---
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr viewpoints_adjusted_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr raw_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr smoothed_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr ground_truth_trajectory_pub_;

    // --- Timers ---
    rclcpp::TimerBase::SharedPtr ground_truth_timer_;

    // --- TF2 ---
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // --- Map and Path Data ---
    nav_msgs::msg::OccupancyGrid::SharedPtr costmap_;
    nav_msgs::msg::Path adjusted_viewpoints_;
    nav_msgs::msg::Path ground_truth_trajectory_;

    // --- Parameters and State ---
    std::string frame_id_;
    static constexpr int obstacle_threshold_ = 75;
    double interpolation_distance_;
    bool path_invalid_flag_ = false;
    double safety_distance_;
    double extra_safety_distance_;

    // --- Callbacks ---
    void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void viewpointsCallback(const nav_msgs::msg::Path::SharedPtr msg);
    void updateGroundTruthTrajectory();
    void planAndPublishPath();
    bool checkPathPreconditions();
    geometry_msgs::msg::PoseStamped getValidCurrentPosition();
    nav_msgs::msg::Path buildInitPath(const geometry_msgs::msg::PoseStamped& current_position);
    bool planSegments(const nav_msgs::msg::Path& init_path, nav_msgs::msg::Path& raw_path, int& fail_idx);
    void handlePathFailure(const geometry_msgs::msg::PoseStamped& current_position);
    void publishPaths(const nav_msgs::msg::Path& raw_path);
    
    // Helper: Interpolate and adjust intermediate points between start and goal
    std::vector<geometry_msgs::msg::PoseStamped> interpolateAndAdjust(
        const geometry_msgs::msg::PoseStamped &start,
        const geometry_msgs::msg::PoseStamped &goal,
        bool &invalid_flag);

    // Helper: A* search for 2D grid
    bool aStarSearch(
        int start_x, int start_y, int goal_x, int goal_y,
        std::unordered_map<int, int> &came_from,
        std::unordered_map<int, float> &cost_so_far,
        std::function<int(int, int)> toIndex,
        std::function<float(int, int, int, int)> heuristic);

    // Helper: Reconstruct path from A* result
    std::vector<geometry_msgs::msg::PoseStamped> reconstructPath(
        int goal_x, int goal_y, int width, float resolution,
        const geometry_msgs::msg::PoseStamped &start,
        const geometry_msgs::msg::PoseStamped &goal,
        std::unordered_map<int, int> &came_from,
        std::unordered_map<int, float> &cost_so_far,
        std::function<int(int, int)> toIndex);

    // Helper: Adjust path for collisions and downsample
    std::vector<geometry_msgs::msg::PoseStamped> adjustAndDownsamplePath(
        const std::vector<geometry_msgs::msg::PoseStamped> &path);

    // --- Utility Methods ---
    geometry_msgs::msg::PoseStamped getCurrentPosition();
    std::pair<geometry_msgs::msg::PoseStamped, bool> adjustViewpointForCollision(
        const geometry_msgs::msg::PoseStamped &viewpoint, float distance, float resolution, int max_attempts);
    tf2::Quaternion interpolateYaw(
        const geometry_msgs::msg::Pose &start_pose,
        const geometry_msgs::msg::Pose &goal_pose,
        float t);
    std::vector<geometry_msgs::msg::PoseStamped> planPath(
        const geometry_msgs::msg::PoseStamped &start,
        const geometry_msgs::msg::PoseStamped &goal);
    std::vector<geometry_msgs::msg::PoseStamped> downsamplePath(
        const std::vector<geometry_msgs::msg::PoseStamped> &path, double min_distance);
    std::vector<geometry_msgs::msg::PoseStamped> smoothPath(
        const std::vector<geometry_msgs::msg::PoseStamped> &path, double interpolation_distance);

    // --- Smoothing helpers ---
    std::vector<double> cubicSplineInterp(const std::vector<double> &t, const std::vector<double> &values, const std::vector<double> &t_new, bool is_yaw);
    std::optional<geometry_msgs::msg::PoseStamped> findAdjustedViewpoint(const geometry_msgs::msg::PoseStamped &pose, double interpolation_distance);

    // Moving average filter for path smoothing (window size, including yaw)
    std::vector<geometry_msgs::msg::PoseStamped> movingAverageFilter(const std::vector<geometry_msgs::msg::PoseStamped>& path, int window);
};
