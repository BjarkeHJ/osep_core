/* 

ROS2 Node for Path Planning

*/

#include "viewpoint_manager.hpp"
#include "planner.hpp"

#include <mutex>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "osep_skeleton_decomp/msg/vertex.hpp"
#include "osep_skeleton_decomp/msg/global_skeleton.hpp"
using MsgVertex = osep_skeleton_decomp::msg::Vertex;
using MsgSkeleton = osep_skeleton_decomp::msg::GlobalSkeleton;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode();
    
private:
    /* Functions */
    void skeleton_callback(MsgSkeleton::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        latest_skel_ = std::move(msg);
    }
    void map_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        latest_map_ = std::move(msg);
    }
    void process_tick();
    
    /* ROS2 */
    rclcpp::Subscription<MsgSkeleton>::SharedPtr skel_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::TimerBase::SharedPtr tick_timer_;

    /* Params */
    ViewpointConfig vpman_cfg;
    PlannerConfig planner_cfg;
    std::string skeleton_topic_;
    std::string map_topic_;
    std::string path_topic_;
    std::string viewpoint_topic_;
    int tick_ms_;

    /* Data */
    MsgSkeleton::ConstSharedPtr latest_skel_;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_map_;

    /* Utils */
    std::mutex mtx_;
    std::unique_ptr<ViewpointManager> vpman_;
    std::unique_ptr<PathPlanner> planner_;

    /* Visualization */
    void publish_viewpoints();
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr vpt_pub_; // vis
    rclcpp::TimerBase::SharedPtr vpt_timer_;
    std::vector<Vertex> current_vers;
    std_msgs::msg::Header current_header;
};

PlannerNode::PlannerNode() : Node("PlannerNode") {
    /* LAUNCH FILE PARAMETER DECLARATIONS */
    tick_ms_ = declare_parameter<int>("tick_ms", 200);

    // TOPICS
    skeleton_topic_ = declare_parameter<std::string>("skeleton_topic", "/osep/gskel/global_skeleton_vertices");
    map_topic_ = declare_parameter<std::string>("map_topic", "/osep/lidar_map/global_map");
    path_topic_ = declare_parameter<std::string>("path_topic", "/osep/planner/path");
    viewpoint_topic_ = declare_parameter<std::string>("viewpoints_topic", "/osep/planner/viewpoints"); // vis

    // VIEWPOINTMANAGER
    vpman_cfg.map_voxel_size = declare_parameter<float>("map_voxel_size", 1.0f);
    vpman_cfg.vpt_disp_dist = declare_parameter<float>("vpt_displacement_dist", 12.0f);

    // PATHPLANNER
    

    /* OBJECT INITIALIZATION */
    vpman_ = std::make_unique<ViewpointManager>(vpman_cfg);
    planner_ = std::make_unique<PathPlanner>(planner_cfg);

    /* ROS2 */
    auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
    skel_sub_ = this->create_subscription<MsgSkeleton>(skeleton_topic_,
                sub_qos,
                std::bind(&PlannerNode::skeleton_callback, this, std::placeholders::_1));
    map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(map_topic_,
                sub_qos,
                std::bind(&PlannerNode::map_callback, this, std::placeholders::_1));


    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(path_topic_, pub_qos);
    vpt_pub_  = this->create_publisher<geometry_msgs::msg::PoseArray>(viewpoint_topic_, pub_qos);

    tick_timer_ = this->create_wall_timer(std::chrono::milliseconds(tick_ms_), std::bind(&PlannerNode::process_tick, this));

    vpt_timer_ = this->create_wall_timer(std::chrono::milliseconds(200), std::bind(&PlannerNode::publish_viewpoints, this));
}

void PlannerNode::publish_viewpoints() {
    if (current_vers.empty()) return;
    geometry_msgs::msg::PoseArray vpts_msg;
    vpts_msg.header = current_header;
    
    for (const auto& v : current_vers) {
        for (const auto& vp : v.vpts) {
            if (vp.invalid) continue;
            geometry_msgs::msg::Pose p;
            p.position.x = vp.position.x();
            p.position.y = vp.position.y();
            p.position.z = vp.position.z();
            p.orientation.x = vp.orientation.x();
            p.orientation.y = vp.orientation.y();
            p.orientation.z = vp.orientation.z();
            p.orientation.w = vp.orientation.w();
            vpts_msg.poses.emplace_back(std::move(p));
        }
    }
    vpt_pub_->publish(vpts_msg);
}

void PlannerNode::process_tick() {
    MsgSkeleton::ConstSharedPtr skel_msg;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr map_msg;
    {
        std::scoped_lock lk(mtx_);
        skel_msg = latest_skel_;
        map_msg = latest_map_;
        latest_skel_.reset();
        latest_map_.reset();
    }   

    if (!skel_msg || !map_msg) return;
    
    // Set current map
    auto& map = vpman_->input_map(); // should be update map and change in place???
    pcl::fromROSMsg(*map_msg, map);

    // Fill from skeleton message...
    std::vector<Vertex> skel_inc;
    skel_inc.reserve(skel_msg->vertices.size());
    for (const auto& mv : skel_msg->vertices) {
        Vertex v;
        v.vid = mv.id;
        v.position.x = mv.position.x;
        v.position.y = mv.position.y;
        v.position.z = mv.position.z;
        v.type = mv.type;
        v.pos_update = mv.pos_update;
        v.type_update = mv.type_update;

        for (auto nb : mv.adj) {
            v.nb_ids.push_back(static_cast<int>(nb));
        }
        skel_inc.push_back(v);
    }

    vpman_->update_skeleton(skel_inc); // updates the skeleton in place - preserves or changes the current state

    if (vpman_->viewpoint_run()) {
        const auto& vers_w_vpts = vpman_->output_skeleton();

        // Overwrite for vis publishing
        // current_vers = vers_w_vpts;
        // current_header = skel_msg->header;
        // ------------------------------

        // std::cout << vers_w_vpts.size() << std::endl;

        planner_->update_skeleton(vers_w_vpts);
        if (planner_->planner_run()) {
            const auto& test = planner_->output_skeleton();
            // std::cout << test.size() << std::endl;
            current_vers = test;
            current_header = skel_msg->header;
        }
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}