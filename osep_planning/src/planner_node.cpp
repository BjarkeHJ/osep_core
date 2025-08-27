/* 

ROS2 Node for Path Planning

*/

#include "planner.hpp"
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>


class PlannerNode : public rclcpp::Node {
public:
    PlannerNode();
    
private:
    /* Functions */
    void viewpoints_callback(geometry_msgs::msg::PoseArray::ConstSharedPtr vpts_msg);
    void process_tick();

    /* ROS2 */
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr vpts_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

    rclcpp::TimerBase::SharedPtr tick_timer_;

    /* Params */
    std::string viewpoints_topic_;
    std::string path_topic_;
    int tick_ms_;

    /* Utils */
    std::mutex latest_mutex_;
    std::unique_ptr<PathPlanner> planner_;

    /* Data */
    PlannerConfig planner_cfg;
    geometry_msgs::msg::PoseArray::ConstSharedPtr latest_msg_;

};

PlannerNode::PlannerNode() : Node("PlannerNode") {
    /* LAUNCH FILE PARAMETER DECLARATIONS */
    // MISC
    viewpoints_topic_ = declare_parameter<std::string>("viewpoints_topic", "/osep/viewpoint_manager/viewpoints");
    path_topic_ = declare_parameter<std::string>("path_topic", "/osep/path_planner/path");
    // PATHPLANNER

    /* OBJECT INITIALIZATION */
    planner_ = std::make_unique<PathPlanner>(planner_cfg);

    /* ROS2 */
    auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(1));
    sub_qos.reliable().transient_local();
    vpts_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(viewpoints_topic_,
                                                                         sub_qos,
                                                                         std::bind(&PlannerNode::viewpoints_callback, this, std::placeholders::_1));
    
    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(1));
    pub_qos.reliable().transient_local();
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(path_topic_, pub_qos);

    tick_timer_ = create_wall_timer(std::chrono::milliseconds(tick_ms_), std::bind(&PlannerNode::process_tick, this));
    
    
}

void PlannerNode::viewpoints_callback(geometry_msgs::msg::PoseArray::ConstSharedPtr vpts_msg) {
    
}

void PlannerNode::process_tick() {
    geometry_msgs::msg::PoseArray::ConstSharedPtr msg;
    {
        std::scoped_lock lk(latest_mutex_);
        msg = latest_msg_;
        latest_msg_.reset();
    }   

    if (planner_->planner_run()) {
        // publish path...
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}