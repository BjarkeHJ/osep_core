
#include "path_interpolator.hpp"
#include <chrono>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <unordered_map>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

PathInterpolator::PathInterpolator() : Node("planner") {
	this->declare_parameter<std::string>("frame_id", "base_link");
	this->declare_parameter<double>("interpolation_distance", 2.0);
	this->declare_parameter<std::string>("costmap_topic", "/local_costmap/costmap");
	this->declare_parameter<std::string>("viewpoints_topic", "/viewpoints");
	this->declare_parameter<std::string>("viewpoints_adjusted_topic", "/viewpoints_adjusted");
	this->declare_parameter<std::string>("path_topic", "/path");
	this->declare_parameter<std::string>("ground_truth_topic", "/ground_truth");
	this->declare_parameter<int>("ground_truth_update_interval", 2000);
	this->declare_parameter<double>("safety_distance", 10.0);

	frame_id_ = this->get_parameter("frame_id").as_string();
	interpolation_distance_ = this->get_parameter("interpolation_distance").as_double();
	std::string costmap_topic = this->get_parameter("costmap_topic").as_string();
	std::string viewpoints_topic = this->get_parameter("viewpoints_topic").as_string();
	std::string viewpoints_adjusted_topic = this->get_parameter("viewpoints_adjusted_topic").as_string();
	std::string path_topic = this->get_parameter("path_topic").as_string();
	std::string ground_truth_topic = this->get_parameter("ground_truth_topic").as_string();
	int ground_truth_update_interval = this->get_parameter("ground_truth_update_interval").as_int();
	safety_distance_ = this->get_parameter("safety_distance").as_double();
	extra_safety_distance_ = 0.1 * safety_distance_;

	tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
	tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

	ground_truth_trajectory_.header.frame_id = frame_id_;

	ground_truth_timer_ = this->create_wall_timer(
		std::chrono::milliseconds(ground_truth_update_interval),
		std::bind(&PathInterpolator::updateGroundTruthTrajectory, this));

	rclcpp::QoS qos_profile(rclcpp::KeepLast(1));
	qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
	qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

	costmap_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
		costmap_topic, 10,
		std::bind(&PathInterpolator::costmapCallback, this, std::placeholders::_1));

	viewpoints_sub_ = this->create_subscription<nav_msgs::msg::Path>(
		viewpoints_topic, qos_profile,
		std::bind(&PathInterpolator::viewpointsCallback, this, std::placeholders::_1));

	viewpoints_adjusted_pub_ = this->create_publisher<nav_msgs::msg::Path>(viewpoints_adjusted_topic, 10);
	raw_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(path_topic + "/raw_path", 10);
	smoothed_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(path_topic, 10);
	ground_truth_trajectory_pub_ = this->create_publisher<nav_msgs::msg::Path>(ground_truth_topic, 10);
}



geometry_msgs::msg::PoseStamped PathInterpolator::getCurrentPosition() {
	geometry_msgs::msg::PoseStamped current_position;
	try {
		geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
			costmap_->header.frame_id, frame_id_, tf2::TimePointZero);
		current_position.pose.position.x = transform.transform.translation.x;
		current_position.pose.position.y = transform.transform.translation.y;
		current_position.pose.position.z = transform.transform.translation.z;
		current_position.pose.orientation = transform.transform.rotation;
		current_position.header.frame_id = costmap_->header.frame_id;
	} catch (const tf2::TransformException &ex) {
		RCLCPP_ERROR(this->get_logger(), "Failed to get transform: %s", ex.what());
		current_position.header.frame_id = "";
	}
	return current_position;
}

void PathInterpolator::updateGroundTruthTrajectory() {
	if (!costmap_) {
		RCLCPP_WARN(this->get_logger(), "Costmap is not initialized, skipping update");
		return;
	}
	geometry_msgs::msg::PoseStamped current_position = getCurrentPosition();
	if (current_position.header.frame_id.empty()) {
		RCLCPP_WARN(this->get_logger(), "Invalid current position, skipping update");
		return;
	}
	ground_truth_trajectory_.header.frame_id = costmap_->header.frame_id;
	ground_truth_trajectory_.header.stamp = this->now();
	ground_truth_trajectory_.poses.push_back(current_position);
	ground_truth_trajectory_pub_->publish(ground_truth_trajectory_);
}

std::pair<geometry_msgs::msg::PoseStamped, bool> PathInterpolator::adjustviewpointForCollision(
	const geometry_msgs::msg::PoseStamped &viewpoint, float distance, float resolution, int max_attempts) {
	geometry_msgs::msg::PoseStamped adjusted_viewpoint = viewpoint;
	bool was_adjusted = false;
	if (!costmap_) {
		RCLCPP_ERROR(this->get_logger(), "Costmap is null");
		adjusted_viewpoint.header.frame_id = "";
		return {adjusted_viewpoint, was_adjusted};
	}
	tf2::Quaternion quat;
	tf2::fromMsg(adjusted_viewpoint.pose.orientation, quat);
	double yaw = tf2::getYaw(quat);
	for (int attempt = 0; attempt < max_attempts; ++attempt) {
		int x_index = static_cast<int>((adjusted_viewpoint.pose.position.x - costmap_->info.origin.position.x) / resolution);
		int y_index = static_cast<int>((adjusted_viewpoint.pose.position.y - costmap_->info.origin.position.y) / resolution);
		if (x_index >= 0 && x_index < static_cast<int>(costmap_->info.width) &&
			y_index >= 0 && y_index < static_cast<int>(costmap_->info.height)) {
			int index = y_index * costmap_->info.width + x_index;
			if (costmap_->data[index] <= obstacle_threshold_) {
				geometry_msgs::msg::PoseStamped forward_viewpoint = adjusted_viewpoint;
				forward_viewpoint.pose.position.x += distance * std::cos(yaw);
				forward_viewpoint.pose.position.y += distance * std::sin(yaw);
				int forward_x_index = static_cast<int>((forward_viewpoint.pose.position.x - costmap_->info.origin.position.x) / resolution);
				int forward_y_index = static_cast<int>((forward_viewpoint.pose.position.y - costmap_->info.origin.position.y) / resolution);
				if (forward_x_index >= 0 && forward_x_index < static_cast<int>(costmap_->info.width) &&
					forward_y_index >= 0 && forward_y_index < static_cast<int>(costmap_->info.height)) {
					int forward_index = forward_y_index * costmap_->info.width + forward_x_index;
					if (costmap_->data[forward_index] <= obstacle_threshold_) {
						return {adjusted_viewpoint, was_adjusted};
					}
				}
			}
		}
		adjusted_viewpoint.pose.position.x -= distance * std::cos(yaw);
		adjusted_viewpoint.pose.position.y -= distance * std::sin(yaw);
		was_adjusted = true;
	}
	adjusted_viewpoint.header.frame_id = "";
	return {adjusted_viewpoint, was_adjusted};
}

tf2::Quaternion PathInterpolator::interpolateYaw(
	const geometry_msgs::msg::Pose &start_pose,
	const geometry_msgs::msg::Pose &goal_pose,
	float t) {
	tf2::Quaternion start_quat, goal_quat;
	tf2::fromMsg(start_pose.orientation, start_quat);
	tf2::fromMsg(goal_pose.orientation, goal_quat);
	double start_yaw = tf2::getYaw(start_quat);
	double goal_yaw = tf2::getYaw(goal_quat);
	double delta_yaw = goal_yaw - start_yaw;
	if (delta_yaw > M_PI) {
		delta_yaw -= 2 * M_PI;
	} else if (delta_yaw < -M_PI) {
		delta_yaw += 2 * M_PI;
	}
	double interpolated_yaw = start_yaw + t * delta_yaw;
	tf2::Quaternion interpolated_quat;
	interpolated_quat.setRPY(0, 0, interpolated_yaw);
	return interpolated_quat;
}


std::vector<geometry_msgs::msg::PoseStamped> PathInterpolator::planPath(
	const geometry_msgs::msg::PoseStamped &start,
	const geometry_msgs::msg::PoseStamped &goal) {
	if (!costmap_) {
		RCLCPP_ERROR(this->get_logger(), "No costmap available");
		return {};
	}
	bool invalid_flag = false;
	int width = costmap_->info.width;
	int height = costmap_->info.height;
	float resolution = costmap_->info.resolution;
	auto toIndex = [&](int x, int y) { return y * width + x; };
	auto heuristic = [&](int x1, int y1, int x2, int y2) {
		return std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
	};
	std::vector<geometry_msgs::msg::PoseStamped> viewpoints = {start};
	float distance = std::sqrt(
		std::pow(goal.pose.position.x - start.pose.position.x, 2) +
		std::pow(goal.pose.position.y - start.pose.position.y, 2) +
		std::pow(goal.pose.position.z - start.pose.position.z, 2));
	if (distance > interpolation_distance_) {
		int num_intermediate_points = static_cast<int>(std::ceil(distance / interpolation_distance_));
		for (int i = 1; i <= num_intermediate_points; ++i) {
			float t = static_cast<float>(i) / (num_intermediate_points + 1);
			geometry_msgs::msg::PoseStamped intermediate;
			intermediate.header.frame_id = costmap_->header.frame_id;
			intermediate.pose.position.x = start.pose.position.x + t * (goal.pose.position.x - start.pose.position.x);
			intermediate.pose.position.y = start.pose.position.y + t * (goal.pose.position.y - start.pose.position.y);
			intermediate.pose.position.z = start.pose.position.z + t * (goal.pose.position.z - start.pose.position.z);
			tf2::Quaternion quaternion = interpolateYaw(start.pose, goal.pose, t);
			intermediate.pose.orientation = tf2::toMsg(quaternion);
			auto [adjusted_intermediate, was_adjusted] = adjustviewpointForCollision(intermediate, extra_safety_distance_, resolution, 10);
			if (adjusted_intermediate.header.frame_id.empty()) {
				invalid_flag = true;
				break;
			} else {
				if (was_adjusted) {
					viewpoints.push_back(adjusted_intermediate);
				}
			}
		}
	}
	viewpoints.push_back(goal);
	if (!invalid_flag) {
		return viewpoints;
	}
	std::vector<geometry_msgs::msg::PoseStamped> full_path;
	if (viewpoints.size() < 2) {
		RCLCPP_ERROR(this->get_logger(), "Not enough viewpoints to plan a path");
		return {};
	}
	int start_x = static_cast<int>((start.pose.position.x - costmap_->info.origin.position.x) / resolution);
	int start_y = static_cast<int>((start.pose.position.y - costmap_->info.origin.position.y) / resolution);
	int goal_x = static_cast<int>((goal.pose.position.x - costmap_->info.origin.position.x) / resolution);
	int goal_y = static_cast<int>((goal.pose.position.y - costmap_->info.origin.position.y) / resolution);
	std::priority_queue<PlannerNode, std::vector<PlannerNode>, std::greater<PlannerNode>> open_list;
	std::unordered_map<int, int> came_from;
	std::unordered_map<int, float> cost_so_far;
	open_list.push({start_x, start_y, 0});
	cost_so_far[toIndex(start_x, start_y)] = 0;
	std::vector<int> dx = {1, -1, 0, 0};
	std::vector<int> dy = {0, 0, 1, -1};
	while (!open_list.empty()) {
		PlannerNode current = open_list.top();
		open_list.pop();
		if (current.x == goal_x && current.y == goal_y) {
			break;
		}
		for (size_t j = 0; j < dx.size(); ++j) {
			int next_x = current.x + dx[j];
			int next_y = current.y + dy[j];
			if (next_x < 0 || next_y < 0 || next_x >= width || next_y >= height) {
				continue;
			}
			int index = toIndex(next_x, next_y);
			if (costmap_->data[index] > obstacle_threshold_) {
				continue;
			}
			float new_cost = cost_so_far[toIndex(current.x, current.y)] + 1;
			if (!cost_so_far.count(index) || new_cost < cost_so_far[index]) {
				cost_so_far[index] = new_cost;
				float priority = new_cost + heuristic(next_x, next_y, goal_x, goal_y);
				open_list.push({next_x, next_y, priority});
				came_from[index] = toIndex(current.x, current.y);
			}
		}
	}
	int current_index = toIndex(goal_x, goal_y);
	float total_distance_2d = cost_so_far[toIndex(goal_x, goal_y)];
	if (total_distance_2d <= 0.0f) {
		RCLCPP_ERROR(this->get_logger(), "Failed to calculate a valid path distance.");
		return {};
	}
	while (came_from.count(current_index)) {
		int x = current_index % width;
		int y = current_index / width;
		geometry_msgs::msg::PoseStamped pose;
		pose.header.frame_id = costmap_->header.frame_id;
		pose.pose.position.x = x * resolution + costmap_->info.origin.position.x;
		pose.pose.position.y = y * resolution + costmap_->info.origin.position.y;
		float dz = goal.pose.position.z - start.pose.position.z;
		float distance_to_start_2d = cost_so_far[current_index];
		if (total_distance_2d > 0.0) {
			float t = std::clamp(distance_to_start_2d / total_distance_2d, 0.0f, 1.0f);
			pose.pose.position.z = start.pose.position.z + t * dz;
			tf2::Quaternion quaternion = interpolateYaw(start.pose, goal.pose, t);
			pose.pose.orientation = tf2::toMsg(quaternion);
		} else {
			pose.pose.position.z = start.pose.position.z + dz;
			pose.pose.orientation = goal.pose.orientation;
		}
		full_path.push_back(pose);
		current_index = came_from[current_index];
	}
	full_path.push_back(start);
	std::reverse(full_path.begin(), full_path.end());
	if (full_path.size() <= 2) {
		RCLCPP_ERROR(this->get_logger(), "A* failed to find a valid path. Only start and end points are available.");
		return {};
	}
	std::vector<geometry_msgs::msg::PoseStamped> adjusted_full_path;
	for (const auto &pose : full_path) {
		auto [adjusted_pose, was_adjusted] = adjustviewpointForCollision(pose, extra_safety_distance_, costmap_->info.resolution, 5);
		adjusted_full_path.push_back(adjusted_pose);
	}
	adjusted_full_path.erase(
		std::remove_if(
			adjusted_full_path.begin(),
			adjusted_full_path.end(),
			[](const geometry_msgs::msg::PoseStamped &pose) {
				return pose.header.frame_id.empty();
			}),
		adjusted_full_path.end());
	full_path = std::move(adjusted_full_path);
	full_path = downsamplePath(full_path, interpolation_distance_);
	return full_path;
}

std::vector<geometry_msgs::msg::PoseStamped> PathInterpolator::downsamplePath(
	const std::vector<geometry_msgs::msg::PoseStamped> &path, double min_distance) {
	if (path.size() < 2) {
		return path;
	}
	std::vector<geometry_msgs::msg::PoseStamped> downsampled_path;
	downsampled_path.push_back(path.front());
	for (size_t i = 1; i < path.size(); ++i) {
		const auto &last_point = downsampled_path.back().pose.position;
		const auto &current_point = path[i].pose.position;
		double distance = std::sqrt(
			std::pow(current_point.x - last_point.x, 2) +
			std::pow(current_point.y - last_point.y, 2) +
			std::pow(current_point.z - last_point.z, 2));
		if (distance >= min_distance) {
			downsampled_path.push_back(path[i]);
		}
	}
	return downsampled_path;
}

std::vector<geometry_msgs::msg::PoseStamped> PathInterpolator::smoothPath(
	const std::vector<geometry_msgs::msg::PoseStamped> &path, double interpolation_distance) {
	if (path.size() < 2) {
		return path;
	}
	std::vector<double> x, y, z, yaw;
	for (const auto &pose : path) {
		x.push_back(pose.pose.position.x);
		y.push_back(pose.pose.position.y);
		z.push_back(pose.pose.position.z);
		tf2::Quaternion quat;
		tf2::fromMsg(pose.pose.orientation, quat);
		yaw.push_back(tf2::getYaw(quat));
	}
	std::vector<double> t(x.size(), 0.0);
	for (size_t i = 1; i < x.size(); ++i) {
		t[i] = t[i - 1] + std::sqrt(
			std::pow(x[i] - x[i - 1], 2) +
			std::pow(y[i] - y[i - 1], 2) +
			std::pow(z[i] - z[i - 1], 2));
	}
	double total_length = t.back();
	int num_points = static_cast<int>(std::ceil(total_length / interpolation_distance));
	std::vector<double> t_new(num_points);
	for (int i = 0; i < num_points; ++i) {
		t_new[i] = i * interpolation_distance;
	}
	auto cubicSpline = [](const std::vector<double> &t, const std::vector<double> &values, const std::vector<double> &t_new, bool is_yaw = false) {
		std::vector<double> result(t_new.size());
		for (size_t i = 0; i < t_new.size(); ++i) {
			auto it = std::lower_bound(t.begin(), t.end(), t_new[i]);
			size_t idx = std::distance(t.begin(), it);
			if (idx == 0) {
				result[i] = values[0];
			} else if (idx >= t.size()) {
				result[i] = values.back();
			} else {
				double t1 = t[idx - 1], t2 = t[idx];
				double v1 = values[idx - 1], v2 = values[idx];
				if (is_yaw) {
					double delta_yaw = v2 - v1;
					if (delta_yaw > M_PI) {
						delta_yaw -= 2 * M_PI;
					} else if (delta_yaw < -M_PI) {
						delta_yaw += 2 * M_PI;
					}
					double interpolated_yaw = v1 + (delta_yaw * (t_new[i] - t1) / (t2 - t1));
					result[i] = std::fmod(interpolated_yaw + M_PI, 2 * M_PI) - M_PI;
				} else {
					result[i] = v1 + (v2 - v1) * (t_new[i] - t1) / (t2 - t1);
				}
			}
		}
		return result;
	};
	std::vector<double> x_smooth = cubicSpline(t, x, t_new);
	std::vector<double> y_smooth = cubicSpline(t, y, t_new);
	std::vector<double> z_smooth = cubicSpline(t, z, t_new);
	std::vector<double> yaw_smooth = cubicSpline(t, yaw, t_new, true);
	std::vector<geometry_msgs::msg::PoseStamped> smoothed_path;
	smoothed_path.push_back(path.front());
	for (size_t i = 1; i < t_new.size(); ++i) {
		geometry_msgs::msg::PoseStamped pose;
		pose.header = path.front().header;
		pose.pose.position.x = x_smooth[i];
		pose.pose.position.y = y_smooth[i];
		pose.pose.position.z = z_smooth[i];
		tf2::Quaternion quaternion;
		quaternion.setRPY(0, 0, yaw_smooth[i]);
		pose.pose.orientation = tf2::toMsg(quaternion);
		auto adjusted_it = std::find_if(
			adjusted_viewpoints_.poses.begin(), adjusted_viewpoints_.poses.end(),
			[&pose, interpolation_distance](const geometry_msgs::msg::PoseStamped &adjusted_pose) {
				return std::sqrt(
					std::pow(pose.pose.position.x - adjusted_pose.pose.position.x, 2) +
					std::pow(pose.pose.position.y - adjusted_pose.pose.position.y, 2) +
					std::pow(pose.pose.position.z - adjusted_pose.pose.position.z, 2)) <= interpolation_distance;
			});
		if (adjusted_it != adjusted_viewpoints_.poses.end()) {
			smoothed_path.push_back(*adjusted_it);
		} else {
			smoothed_path.push_back(pose);
		}
	}
	return smoothed_path;
}

// Callbacks starts here

void PathInterpolator::planAndPublishPath() {
	if (!costmap_ || adjusted_viewpoints_.poses.empty()) {
		RCLCPP_ERROR(this->get_logger(), "Cannot plan path: costmap or trajectory path is missing");
		return;
	}
	geometry_msgs::msg::PoseStamped current_position = getCurrentPosition();
	if (current_position.header.frame_id.empty()) {
		RCLCPP_ERROR(this->get_logger(), "Failed to retrieve current position");
		return;
	}

	nav_msgs::msg::Path init_path;
	init_path.header.stamp = this->now();
	init_path.header.frame_id = costmap_->header.frame_id;
	init_path.poses.push_back(current_position);
	for (const auto &viewpoint : adjusted_viewpoints_.poses) {
		init_path.poses.push_back(viewpoint);
	}
	nav_msgs::msg::Path raw_path;
	raw_path.header.stamp = this->now();
	raw_path.header.frame_id = costmap_->header.frame_id;
	nav_msgs::msg::Path smoothed_path;
	int idx = -1;
	for (size_t i = 0; i < init_path.poses.size() - 1 && i < 4; ++i) {
		const auto &start = init_path.poses[i];
		const auto &goal = init_path.poses[i + 1];
		auto segment_path = planPath(start, goal);
		if (segment_path.empty()) {
			RCLCPP_ERROR(this->get_logger(), "Failed to plan a valid path segment between viewpoints %zu and %zu.", i, i + 1);
			path_invalid_flag_ = true;
			idx = i;
			break;
		}
		path_invalid_flag_ = false;
		raw_path.poses.insert(raw_path.poses.end(), segment_path.begin(), segment_path.end());
	}
	if (path_invalid_flag_ && idx == 0) {
		RCLCPP_ERROR(this->get_logger(), "Path planning failed. Marking the path as invalid and aborting.");
		smoothed_path.header.stamp = this->now();
		smoothed_path.header.frame_id = costmap_->header.frame_id;
		geometry_msgs::msg::PoseStamped current_position_adjusted = adjustviewpointForCollision(current_position, extra_safety_distance_, costmap_->info.resolution, 10).first;
		if (current_position_adjusted.header.frame_id.empty()) {
			RCLCPP_ERROR(this->get_logger(), "Failed to adjust current position for collision-free zone");
			nav_msgs::msg::Path empty_path;
			empty_path.header.stamp = this->now();
			empty_path.header.frame_id = "map";
			smoothed_path_pub_->publish(empty_path);
			return;
		}
		smoothed_path.poses.push_back(current_position_adjusted);
		smoothed_path_pub_->publish(smoothed_path);
		return;
	}
	raw_path_pub_->publish(raw_path);
	std::vector<geometry_msgs::msg::PoseStamped> smoothed_poses = smoothPath(raw_path.poses, interpolation_distance_);
	smoothed_path.header.stamp = this->now();
	smoothed_path.header.frame_id = costmap_->header.frame_id;
	smoothed_path.poses = smoothed_poses;
	smoothed_path_pub_->publish(smoothed_path);
}

void PathInterpolator::costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
	costmap_ = msg;
	if (!adjusted_viewpoints_.poses.empty()) {
		planAndPublishPath();
	}
}

void PathInterpolator::viewpointsCallback(const nav_msgs::msg::Path::SharedPtr msg) {
	if (!costmap_) {
		RCLCPP_ERROR(this->get_logger(), "Costmap is null");
		return;
	}
	if (msg->poses.empty()) {
		RCLCPP_WARN(this->get_logger(), "Received empty Path message");
		return;
	}
	nav_msgs::msg::Path all_adjusted_viewpoints;
	all_adjusted_viewpoints.header = msg->header;

	adjusted_viewpoints_.header = msg->header;
	adjusted_viewpoints_.poses.clear();

	for (const auto &pose : msg->poses) {
		auto [adjusted_pose, _] = adjustviewpointForCollision(pose, extra_safety_distance_, costmap_->info.resolution, 5);
		all_adjusted_viewpoints.poses.push_back(adjusted_pose);
		if (!adjusted_pose.header.frame_id.empty()) {
			adjusted_viewpoints_.poses.push_back(adjusted_pose);
		}
	}
	viewpoints_adjusted_pub_->publish(all_adjusted_viewpoints);
}

int main(int argc, char **argv) {
	rclcpp::init(argc, argv);
	auto node = std::make_shared<PathInterpolator>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
