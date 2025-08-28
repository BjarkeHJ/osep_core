/* 

ROS2 Node for Path Planning

*/

#include "planner.hpp"
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>

#include "osep_planning/msg/viewpoint.hpp"
#include "osep_planning/msg/viewpoint_set.hpp"
using MsgViewpoint = osep_planning::msg::Viewpoint;
using MsgViewpointSet = osep_planning::msg::ViewpointSet;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode();
    
private:
    /* Functions */
    void viewpoints_callback(MsgViewpointSet::ConstSharedPtr vpts_msg);
    void process_tick();

    /* ROS2 */
    rclcpp::Subscription<MsgViewpointSet>::SharedPtr vpts_sub_;
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
    MsgViewpointSet::ConstSharedPtr latest_vpt_msg_;
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
    vpts_sub_ = this->create_subscription<MsgViewpointSet>(viewpoints_topic_,
                                                                         sub_qos,
                                                                         std::bind(&PlannerNode::viewpoints_callback, this, std::placeholders::_1));
    
    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(1));
    pub_qos.reliable().transient_local();
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(path_topic_, pub_qos);

    tick_timer_ = create_wall_timer(std::chrono::milliseconds(tick_ms_), std::bind(&PlannerNode::process_tick, this));
}

void PlannerNode::viewpoints_callback(MsgViewpointSet::ConstSharedPtr vpts_msg) {
    if (!vpts_msg) return;
    std::scoped_lock lk(latest_mutex_);
    latest_vpt_msg_ = std::move(vpts_msg);
    return;
}

void PlannerNode::process_tick() {
    MsgViewpointSet::ConstSharedPtr msg;
    {
        std::scoped_lock lk(latest_mutex_);
        msg = latest_vpt_msg_;
        latest_vpt_msg_.reset();
    }   

    if (!msg) return;

    auto& vpts_updt = planner_->input_viewpoints();
    vpts_updt.clear();
    vpts_updt.reserve(msg->viewpoints.size());
    for (const auto& vpm : msg->viewpoints) {
        Viewpoint vp;
        vp.target_vid = vpm.vid;
        vp.target_vp_pos = vpm.vp_pos_id;
        vp.position.x() = vpm.position.x;
        vp.position.y() = vpm.position.y;
        vp.position.z() = vpm.position.z;
        vp.yaw = vpm.yaw;
        vp.updated = true;
        vpts_updt.push_back(vp);
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