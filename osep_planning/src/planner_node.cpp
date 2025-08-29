/* 

ROS2 Node for Path Planning

*/

#include "viewpoint_manager.hpp"
#include "planner.hpp"

#include <mutex>
#include <rclcpp/rclcpp.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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

    // void update_skeleton();
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
    std::string viewpoint_topic_; // vis
    std::string graph_topic_; // vis
    int tick_ms_;
    float map_voxel_size_;

    /* Data */
    MsgSkeleton::ConstSharedPtr latest_skel_;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_map_;

    // std::vector<Vertex> skeleton_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
    std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> map_octree_;

    /* Utils */
    std::mutex mtx_;
    std::unique_ptr<ViewpointManager> vpman_;
    std::unique_ptr<PathPlanner> planner_;

    /* Visualization */
    void publish_viewpoints();
    void publish_graph();
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr vpt_pub_; // viewpoint vis
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_pub_;
    rclcpp::TimerBase::SharedPtr vpt_timer_;
    std_msgs::msg::Header current_header;
};

PlannerNode::PlannerNode() : Node("PlannerNode") {
    /* LAUNCH FILE PARAMETER DECLARATIONS */
    tick_ms_ = declare_parameter<int>("tick_ms", 200);
    map_voxel_size_ = declare_parameter<float>("map_voxel_size", 1.0f);

    // TOPICS
    skeleton_topic_ = declare_parameter<std::string>("skeleton_topic", "/osep/gskel/global_skeleton_vertices");
    map_topic_ = declare_parameter<std::string>("map_topic", "/osep/lidar_map/global_map");
    path_topic_ = declare_parameter<std::string>("path_topic", "/osep/planner/path");
    viewpoint_topic_ = declare_parameter<std::string>("viewpoints_topic", "/osep/planner/viewpoints"); // vis
    graph_topic_= declare_parameter<std::string>("graph_topic", "/osep/planner/graph"); // vis

    // VIEWPOINTMANAGER
    vpman_cfg.vpt_safe_dist = declare_parameter<float>("vpt_safe_dist", 12.0f);
    vpman_cfg.map_voxel_size = map_voxel_size_;

    // PATHPLANNER
    planner_cfg.map_voxel_size = map_voxel_size_;
    
    /* OBJECT INITIALIZATION */
    vpman_ = std::make_unique<ViewpointManager>(vpman_cfg);
    planner_ = std::make_unique<PathPlanner>(planner_cfg);

    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    map_octree_ = std::make_shared<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>>(map_voxel_size_); // initialize octree

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
    graph_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(graph_topic_, pub_qos);

    tick_timer_ = this->create_wall_timer(std::chrono::milliseconds(tick_ms_), std::bind(&PlannerNode::process_tick, this));

    vpt_timer_ = this->create_wall_timer(std::chrono::milliseconds(200), 
        [this]{ publish_viewpoints(); publish_graph(); });
}

void PlannerNode::publish_viewpoints() {
    if (!planner_) return;
    const auto& current_vers = planner_->output_skeleton();
    const int N = static_cast<int>(current_vers.size());
    if (N == 0) return;

    geometry_msgs::msg::PoseArray vpts_msg;
    vpts_msg.header = current_header;
    
    for (const auto& v : current_vers) {
        for (const auto& vp : v.vpts) {
            // if (vp.invalid) continue;
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

void PlannerNode::publish_graph() {
    if (!planner_) return;
    const auto& G = planner_->output_graph();
    const int N = static_cast<int>(G.nodes.size());
    if (N == 0) return;

    visualization_msgs::msg::MarkerArray arr;
    {
        visualization_msgs::msg::Marker clr;
        clr.header = current_header; // frame + stamp; falls back to empty if not set
        clr.ns = "graph_edges";
        clr.id = 0;
        clr.action = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(clr);
    }

    auto make_line_list = [&](const std::string& ns, int id,
                              float r, float g, float b, float a,
                              double thickness_m)
        -> visualization_msgs::msg::Marker {
        visualization_msgs::msg::Marker m;
        m.header = current_header;               // use your skeleton/map frame
        m.ns = ns;
        m.id = id;
        m.type = visualization_msgs::msg::Marker::LINE_LIST;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = thickness_m;                 // line width (meters)
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
        m.pose.orientation.w = 1.0;              // identity
        m.lifetime = rclcpp::Duration::from_seconds(0.0); // persistent until next update
        return m;
    };

    // Two markers: topo (e.g., red), geom (e.g., cyan)
    auto topo = make_line_list("graph_topological", 1, 1.0f, 0.1f, 0.1f, 1.0f, 0.05);
    auto geom = make_line_list("graph_geometric",   2, 0.1f, 0.9f, 1.0f, 1.0f, 0.03);

    // Avoid drawing each undirected edge twice: only add (i,j) with i < j.
    for (int i = 0; i < N; ++i) {
        const auto& ni = G.nodes[i];
        for (size_t k = 0; k < G.adj[i].size(); ++k) {
            int j = G.adj[i][k];
            if (j <= i) continue; // draw once
            const auto& flags = G.adjf[i][k];

            geometry_msgs::msg::Point pa, pb;
            pa.x = ni.pos.x(); pa.y = ni.pos.y(); pa.z = ni.pos.z();
            const auto& nj = G.nodes[j];
            pb.x = nj.pos.x(); pb.y = nj.pos.y(); pb.z = nj.pos.z();

            if (flags.is_topological) {
                topo.points.push_back(pa);
                topo.points.push_back(pb);
            }
            if (flags.is_geometric) {
                geom.points.push_back(pa);
                geom.points.push_back(pb);
            }
        }
    }

    // Push only if non-empty, to avoid spamming RViz with empty markers
    if (!topo.points.empty()) arr.markers.push_back(std::move(topo));
    if (!geom.points.empty()) arr.markers.push_back(std::move(geom));

    graph_pub_->publish(arr);
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
    current_header = skel_msg->header;

    // Set current map (node-owned) - Temp swap to avoid member realloc
    {
        pcl::PointCloud<pcl::PointXYZ> tmp;
        pcl::fromROSMsg(*map_msg, tmp);
        {
            std::scoped_lock lk(mtx_);
            map_cloud_->swap(tmp);
        }
    }

    map_octree_->deleteTree();
    map_octree_->setInputCloud(map_cloud_);
    map_octree_->addPointsFromInputCloud();

    // Set map in vpman_ and planner_
    vpman_->set_map(map_cloud_, map_octree_);
    planner_->set_map(map_cloud_, map_octree_);

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

        planner_->update_skeleton(vers_w_vpts);
        if (planner_->planner_run()) {  
            // do something
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