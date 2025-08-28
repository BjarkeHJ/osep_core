/* 

ROS2 Node for Viewpoint managing

*/

#include "viewpoint_manager.hpp"
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "osep_skeleton_decomp/msg/vertex.hpp"
#include "osep_skeleton_decomp/msg/global_skeleton.hpp"

using MsgVertex = osep_skeleton_decomp::msg::Vertex;
using MsgSkeleton = osep_skeleton_decomp::msg::GlobalSkeleton;

class ViewpointNode : public rclcpp::Node {
public:
    ViewpointNode();
    
private:
    /* Functions */
    void skeleton_callback(MsgSkeleton::ConstSharedPtr vtx_msg);
    void map_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg);
    void process_tick();
    void publish_viewpoints(const std::vector<Vertex>& vertices, const std_msgs::msg::Header& src_header);

    // Visualization
    void publish_all_viewpoints(const std::vector<Vertex>& vertices, const std_msgs::msg::Header& src_header); // For visualization...
    
    /* ROS2 */
    rclcpp::Subscription<MsgSkeleton>::SharedPtr skel_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr vp_pub_; // publish viewpoint message
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr all_vp_pub_; // vis
    rclcpp::TimerBase::SharedPtr tick_timer_;

    /* Params */
    std::string skel_topic_;
    std::string vp_topic_;
    std::string all_vp_topic_;
    std::string map_topic_;
    std::string costmap_topic_;
    std::string global_frame_id_;
    int tick_ms_;

    /* Utils */
    std::mutex latest_mutex_;
    std::unique_ptr<ViewpointManager> vpman_;

    /* Data */
    ViewpointConfig vpman_cfg;
    MsgSkeleton::ConstSharedPtr latest_msg_;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_map_msg_;
};

ViewpointNode::ViewpointNode() : Node("ViewpointManagerNode") {
    /* LAUNCH FILE PARAMETER DECLARATIONS */
    // MISC
    skel_topic_ = declare_parameter<std::string>("skeleton_topic", "/osep/gskel/global_skeleton_vertices");

    costmap_topic_ = declare_parameter<std::string>("costmap_topic", "/osep/local_costmap/costmap");
    vp_topic_ = declare_parameter<std::string>("viewpoint_topic", "/osep/viewpoint_manager/viewpoints");
    all_vp_topic_ = declare_parameter<std::string>("viewpoint_topic", "/osep/viewpoint_manager/all_viewpoints");
    map_topic_ = declare_parameter<std::string>("map_topic", "/osep/lidar_map/global_map");
    tick_ms_ = declare_parameter<int>("tick_ms", 50);
    global_frame_id_ = declare_parameter<std::string>("global_frame_id", "odom");
    // VIEWPOINT MANAGER

    vpman_cfg.map_voxel_size = declare_parameter<float>("map_voxel_size", 1.0f);
    vpman_cfg.vpt_disp_dist = declare_parameter<float>("vpt_displacement_dist", 12.0f);

    /* OBJECT INITIALIZATION */
    vpman_ = std::make_unique<ViewpointManager>(vpman_cfg);

    /* ROS2 */
    auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    // sub_qos.reliable().transient_local();
    skel_sub_ = this->create_subscription<MsgSkeleton>(skel_topic_,
                                                       sub_qos,
                                                       std::bind(&ViewpointNode::skeleton_callback, this, std::placeholders::_1));
    map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(map_topic_,
                                                                        sub_qos,
                                                                        std::bind(&ViewpointNode::map_callback, this, std::placeholders::_1));
    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(1));
    pub_qos.reliable().transient_local();

    vp_pub_  = this->create_publisher<geometry_msgs::msg::PoseArray>(vp_topic_, pub_qos);
    all_vp_pub_  = this->create_publisher<geometry_msgs::msg::PoseArray>(all_vp_topic_, pub_qos);
    
    tick_timer_ = create_wall_timer(std::chrono::milliseconds(tick_ms_), std::bind(&ViewpointNode::process_tick, this));
    /* INITIALIZE DATA STRUCTURES */
}

void ViewpointNode::skeleton_callback(MsgSkeleton::ConstSharedPtr vtx_msg) {
    if (!vtx_msg) return;
    std::scoped_lock lk(latest_mutex_);
    latest_msg_ = std::move(vtx_msg);
    return;
}

void ViewpointNode::map_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg) {
    if (!pcd_msg) return;
    std::scoped_lock lk(latest_mutex_);
    latest_map_msg_ = std::move(pcd_msg);
    return;
}

void ViewpointNode::publish_all_viewpoints(const std::vector<Vertex>& vertices, const std_msgs::msg::Header& src_header) {
    if (vertices.empty()) return;
    geometry_msgs::msg::PoseArray vpts_msg;
    vpts_msg.header = src_header;
    
    for (const auto& v : vertices) {
        for (const auto& vp : v.vpts) {
            if (vp.invalid) continue;
            geometry_msgs::msg::Pose p;
            p.position.x = vp.position.x();
            p.position.y = vp.position.y();
            p.position.z = vp.position.z();
            // fill orientation if you have it:
            p.orientation.x = vp.orientation.x();
            p.orientation.y = vp.orientation.y();
            p.orientation.z = vp.orientation.z();
            p.orientation.w = vp.orientation.w();
            vpts_msg.poses.emplace_back(std::move(p));
        }
    }
    
    all_vp_pub_->publish(vpts_msg);
}


void ViewpointNode::process_tick() {
    MsgSkeleton::ConstSharedPtr msg;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr map_msg;
    {
        std::scoped_lock lk(latest_mutex_);
        msg = latest_msg_;
        latest_msg_.reset();
        map_msg = latest_map_msg_;
        latest_map_msg_.reset();
    }

    if (!msg || !map_msg) return;

    auto& skel = vpman_->input_skeleton();
    skel.clear();
    skel.reserve(msg->vertices.size());
    for (const auto& mv : msg->vertices) {
        Vertex v;
        v.vid = mv.id;
        v.position.x = mv.position.x;
        v.position.y = mv.position.y;
        v.position.z = mv.position.z;
        v.type = mv.type;
        v.pos_update = mv.pos_update;
        v.type_update = mv.type_update;
        
        v.nb_ids.clear();
        for (auto nb : mv.adj) {
            v.nb_ids.push_back((int)nb);
        }
        skel.push_back(v);
    }

    auto& map = vpman_->input_map();
    pcl::fromROSMsg(*map_msg, map);
    

    if (vpman_->viewpoint_run()) {
        // auto& viewpoints = vpman_->output_vpts(); // fetch generated viewpoints
        // publish_viewpoints(viewpoints, map_msg->header);
        auto& vertices = vpman_->output_skeleton(); // fetch generated viewpoints
        publish_all_viewpoints(vertices, map_msg->header);
    }
}



int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ViewpointNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}