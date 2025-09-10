/* 

ROS2 Node for Path Planning

*/

#include "types.hpp"
#include "viewpoint_manager.hpp"
#include "planner.hpp"

#include <mutex>
#include <unordered_set>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> 

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

struct PendingTarget {uint64_t id; int vid; int k; };

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode();
    
private:
    /* Functions */
    void skeleton_callback(MsgSkeleton::ConstSharedPtr msg) {
        if (!msg) return;
        std::lock_guard<std::mutex> lk(state_mtx_);
        latest_skel_ = std::move(msg);
    }
    void map_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        if (!msg) return;
        std::lock_guard<std::mutex> lk(state_mtx_);
        latest_map_ = std::move(msg);
    }
    void adjusted_callback(nav_msgs::msg::Path::ConstSharedPtr msg);

    void publish_path();
    void publish_init_path();

    std::optional<geometry_msgs::msg::PoseStamped> get_drone_pose(const std::string& target_frame, const std::string drone_frame, const rclcpp::Duration& timeout = rclcpp::Duration::from_seconds(0.5));
    void update_skeleton(const std::vector<Vertex>& skel_in);

    void process_tick(); // handles planning (slower)
    void control_tick(); // handles target checking (faster)
    
    /* ROS2 */
    rclcpp::Subscription<MsgSkeleton>::SharedPtr skel_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr adjusted_sub_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    // rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr target_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::TimerBase::SharedPtr tick_timer_;
    rclcpp::TimerBase::SharedPtr ctl_timer_;

    /* Params */
    ViewpointConfig vpman_cfg;
    PlannerConfig planner_cfg;

    std::string skeleton_topic_;
    std::string map_topic_;
    std::string target_topic_;
    std::string adjusted_topic_;
    
    std::string path_topic_; // vis
    std::string viewpoint_topic_; // vis
    std::string graph_topic_; // vis

    std::string drone_frame_;
    std::string global_frame_;

    int tick_ms_;
    int ctl_ms_;
    float map_voxel_size_;
    float safe_dist_;
    float reached_dist_th_;

    float prev_dist_to_target = 1e6f;


    /* Data */
    MsgSkeleton::ConstSharedPtr latest_skel_;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_map_;
    
    std::vector<Vertex> skeleton_;
    std::unordered_map<int, int> vid2idx_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
    std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> map_octree_;
    
    bool bootstrap_mode_;
    size_t bootstrap_idx_ = 0;
    std::vector<geometry_msgs::msg::PoseStamped> bootstrap_waypoints_;

    std::vector<PendingTarget> last_pub_;
    rclcpp::Time last_pub_time_;
    int adjust_timeout_ms_;
    bool adjusted_ = true;

    /* Utils */
    std::mutex state_mtx_; // protects node/state data
    std::mutex planner_mtx_; // protexts planner api call and planner internal data

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
    bootstrap_mode_ = declare_parameter<bool>("bootstrap_mode", true);
    tick_ms_ = declare_parameter<int>("tick_ms", 200);
    ctl_ms_ = declare_parameter<int>("control_ms", 50);
    map_voxel_size_ = declare_parameter<float>("map_voxel_size", 1.0f);
    reached_dist_th_ = declare_parameter<float>("reached_dist_th", 3.0f);
    safe_dist_ = declare_parameter<float>("safe_dist", 12.0f);
    adjust_timeout_ms_ = declare_parameter<int>("adjuste_timeout_ms", 250);

    // TOPICS
    skeleton_topic_ = declare_parameter<std::string>("skeleton_topic", "/osep/gskel/global_skeleton_vertices"); // Global skeleton
    map_topic_ = declare_parameter<std::string>("map_topic", "/osep/tsdf/static_pointcloud"); // Global map
    target_topic_ = declare_parameter<std::string>("target_topic", "/osep/viewpoints"); // Target path
    adjusted_topic_ = declare_parameter<std::string>("adjusted_topic", "/osep/viewpoints_adjusted"); // Recieved adjustments

    path_topic_ = declare_parameter<std::string>("path_topic", "/osep/planner/path"); // vis
    viewpoint_topic_ = declare_parameter<std::string>("viewpoints_topic", "/osep/planner/viewpoints"); // vis
    graph_topic_= declare_parameter<std::string>("graph_topic", "/osep/planner/graph"); // vis

    drone_frame_ = declare_parameter<std::string>("frame_id", "base_link");
    global_frame_ = declare_parameter<std::string>("global_frame_id", "odom");

    // VIEWPOINTMANAGER
    vpman_cfg.vpt_safe_dist = safe_dist_;
    vpman_cfg.map_voxel_size = map_voxel_size_;
    vpman_cfg.cam_hfov_rad = 60.0f * static_cast<float>(M_PI) / 180.0f;
    vpman_cfg.cam_vfov_rad = 40.0f * static_cast<float>(M_PI) / 180.0f;
    vpman_cfg.cam_Nx = 40;
    vpman_cfg.cam_Ny = 30;
    vpman_cfg.cam_max_range = 20.0f;

    // PATHPLANNER
    planner_cfg.graph_radius = declare_parameter<float>("graph_radius", 20.0f);
    planner_cfg.safe_dist = safe_dist_;
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
    adjusted_sub_ = this->create_subscription<nav_msgs::msg::Path>(adjusted_topic_,
                sub_qos,
                std::bind(&PlannerNode::adjusted_callback, this, std::placeholders::_1));

    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
    // target_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(target_topic_, pub_qos);
    target_pub_ = this->create_publisher<nav_msgs::msg::Path>(target_topic_, pub_qos);

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(path_topic_, pub_qos); // vis
    vpt_pub_  = this->create_publisher<geometry_msgs::msg::PoseArray>(viewpoint_topic_, pub_qos); // vis
    graph_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(graph_topic_, pub_qos); // vis

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    tick_timer_ = this->create_wall_timer(std::chrono::milliseconds(tick_ms_), std::bind(&PlannerNode::process_tick, this));
    ctl_timer_ = this->create_wall_timer(std::chrono::milliseconds(ctl_ms_), std::bind(&PlannerNode::control_tick, this));

    vpt_timer_ = this->create_wall_timer(std::chrono::milliseconds(200), 
        [this]{ publish_viewpoints(); publish_graph(); });

    /* Bootstrap Path - Hardcoded */
    bootstrap_waypoints_.resize(2);
    for (int i=0; i<2; ++i) {
        bootstrap_waypoints_[i].pose.orientation.w = 1.0; // identity
    }
    bootstrap_waypoints_[0].pose.position.x = 50.0;
    bootstrap_waypoints_[0].pose.position.y = 0.0;
    bootstrap_waypoints_[0].pose.position.z = 100.0;

    bootstrap_waypoints_[1].pose.position.x = 180.0;
    bootstrap_waypoints_[1].pose.position.y = 0.0;
    bootstrap_waypoints_[1].pose.position.z = 120.0;
}

void PlannerNode::publish_viewpoints() {
    std::lock_guard<std::mutex> lk(state_mtx_);

    if (skeleton_.empty()) return;

    const int N = static_cast<int>(skeleton_.size());
    if (N == 0) return;

    geometry_msgs::msg::PoseArray vpts_msg;
    vpts_msg.header = current_header;
    
    for (const auto& v : skeleton_) {
        if (v.vpts.empty()) continue;
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

void PlannerNode::publish_graph() {
    if (!planner_) return;

    // copy header under state lock
    std_msgs::msg::Header hdr;
    {
        std::lock_guard<std::mutex> lk(state_mtx_);
        hdr = current_header;
    }

    // read graph under planner lock
    Graph g;
    {
        std::lock_guard<std::mutex> lk(planner_mtx_);
        g = planner_->get_graph();
    }

    const int N = g.nodes.size();
    if (N == 0 || static_cast<int>(g.adj.size()) != N) return;

    visualization_msgs::msg::MarkerArray arr;
    // Clear previous markers
    {
        visualization_msgs::msg::Marker clr;
        clr.header = hdr;
        clr.ns = "graph";
        clr.id = 0;
        clr.action = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(clr);
    }

    auto eigenToPoint = [](const Eigen::Vector3f& p) {
        geometry_msgs::msg::Point q;
        q.x = p.x(); q.y = p.y(); q.z = p.z();
        return q;
    };

    auto weightToColor = [](float w) {
        // map weight → color, here blue(low)→red(high)
        std_msgs::msg::ColorRGBA c;
        float t = std::clamp(w / 10.0f, 0.0f, 1.0f); // scale weights (adjust divisor!)
        c.r = t;
        c.g = 0.2f * (1.0f - t);
        c.b = 1.0f - t;
        c.a = 0.9f;
        return c;
    };

    // ---------- Edges (LINE_LIST) ----------
    {
        visualization_msgs::msg::Marker m;
        m.header = hdr;
        m.ns = "graph_edges";
        m.id = 2;
        m.type = visualization_msgs::msg::Marker::LINE_LIST;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.lifetime = rclcpp::Duration(0,0); // forever
        m.pose.orientation.w = 1.0;
        m.scale.x = 0.03;  // line width
        m.color.r = 1.0f;
        m.color.g = 1.0f;
        m.color.b = 1.0f;
        m.color.a = 0.85f;

        // We’ll only add (u,v) with u < v to avoid duplicates
        size_t edge_pairs = 0;
        for (int u = 0; u < N; ++u) edge_pairs += g.adj[u].size();
        edge_pairs /= 2; // rough upper bound for reserving
        m.points.reserve(edge_pairs * 2);

        for (int u = 0; u < N; ++u) {
            const auto& Nu = g.adj[u];
            const auto& pu = g.nodes[u].p;
            for (const auto& e : Nu) {
                const int v = e.to;
                if (v <= u) continue; // dedup undirected
                const auto& pv = g.nodes[v].p;
                std_msgs::msg::ColorRGBA c = weightToColor(e.w);
                m.points.push_back(eigenToPoint(pu));
                m.colors.push_back(c);
                m.points.push_back(eigenToPoint(pv));
                m.colors.push_back(c);
            }
        }
        arr.markers.push_back(std::move(m));
    }

    // ---------- (Optional) Edge Direction Arrows ----------
    // If you later add directed edges, create a Marker with type=ARROW per edge.

    graph_pub_->publish(arr);

}

void PlannerNode::publish_path() {

    std_msgs::msg::Header hdr;
    {
        std::lock_guard<std::mutex> lk(state_mtx_);
        hdr = current_header;
    }

    std::vector<Viewpoint> path_vpts;
    {
        std::lock_guard<std::mutex> lk(planner_mtx_);
        path_vpts = planner_->current_path();
    }

    // const auto& path_vpts = planner_->current_path();
    nav_msgs::msg::Path path; 
    path.header = hdr;
    path.poses.reserve(path_vpts.size());
    for (const auto& vp : path_vpts) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = hdr;
        ps.pose.position.x = vp.position.x();
        ps.pose.position.y = vp.position.y();
        ps.pose.position.z = vp.position.z();
        ps.pose.orientation.x = vp.orientation.x();
        ps.pose.orientation.y = vp.orientation.y();
        ps.pose.orientation.z = vp.orientation.z();
        ps.pose.orientation.w = vp.orientation.w();
        path.poses.emplace_back(std::move(ps));
    }

    path_pub_->publish(path);
}

void PlannerNode::publish_init_path() {
    nav_msgs::msg::Path bs_path;
    bs_path.header.frame_id = global_frame_;
    bs_path.header.stamp = this->get_clock()->now();

    for (auto& wp : bootstrap_waypoints_) {
        wp.header.frame_id = global_frame_;
        wp.header.stamp = this->get_clock()->now();
        bs_path.poses.push_back(wp);
    }
    path_pub_->publish(bs_path);
}

void PlannerNode::adjusted_callback(nav_msgs::msg::Path::ConstSharedPtr msg) {
    if (!msg || msg->poses.empty()) return;

    size_t applied = 0;
    size_t expected = 0;

    {
        std::scoped_lock lk(state_mtx_);

        const size_t n = std::min(msg->poses.size(), last_pub_.size());
        expected = last_pub_.size();

        for (size_t i=0; i<n; ++i) {
            const auto& P = msg->poses[i].pose;
            const auto& tag = last_pub_[i];

            auto it = vid2idx_.find(tag.vid);
            if (it == vid2idx_.end()) continue;

            Vertex& v = skeleton_[it->second];
            Viewpoint* vp_ptr = nullptr;

            if (tag.k >= 0 && tag.k < static_cast<int>(v.vpts.size()) && v.vpts[tag.k].vptid == tag.id) {
                vp_ptr = &v.vpts[tag.k];
            }
            else {
                for (auto& cand : v.vpts) {
                    if (cand.vptid == tag.id) {
                        vp_ptr = &cand;
                        break;
                    }
                }
            }

            if (!vp_ptr) continue;

            vp_ptr->position = Eigen::Vector3f(P.position.x, P.position.y, P.position.z);
            Eigen::Quaternionf q(P.orientation.w, P.orientation.x, P.orientation.y, P.orientation.z);
            q.normalize();
            vp_ptr->orientation = q;
            const float ys = 2.0f * (q.w()*q.z() + q.x()*q.y());
            const float yc = 1.0f - 2.0f * (q.y()*q.y() + q.z()*q.z());
            vp_ptr->yaw = std::atan2(ys, yc);
            vp_ptr->updated = true;

            if (msg->poses[i].header.frame_id.empty()) {
                vp_ptr->invalid = true;
            }

            applied++;
        }
    }

    if (applied == expected) {
        adjusted_ = true;
    }
    else {
        RCLCPP_WARN(get_logger(), "Only applied %zu/%zu adjusted viewpoints; keeping gate closed.", applied, expected);
    }
}

void PlannerNode::update_skeleton(const std::vector<Vertex>& skel_in) { 
    // Update or insert incoming vertices
    for (const auto& vin : skel_in) {
        auto it = vid2idx_.find(vin.vid); // pointer to the index of vid2idx_ if vin.vid exists
        if (it == vid2idx_.end()) {
            // vin.vid not found -> new vertex
            Vertex vnew = vin; // copy new vertex
            vid2idx_[vnew.vid] = static_cast<int>(skeleton_.size());
            vnew.spawn_vpts = true;
            skeleton_.push_back(std::move(vnew));
            continue;
        }

        // Vertex already exists -> update in place to preserve information
        const int idx = it->second;
        Vertex& vcur = skeleton_[idx]; // by ref (mutate)
        vcur.pos_update = vin.pos_update;
        vcur.type_update = vin.type_update;
        vcur.position = vin.position;
        vcur.type = vin.type;
        vcur.nb_ids = vin.nb_ids;
    }

    // if vid is in planner skeleton but not in incoming skeleton -> remove here (has been deleted)
    std::unordered_set<int> incoming;
    incoming.reserve(skel_in.size() * 2);
    for (const auto& v : skel_in) {
        incoming.insert(v.vid);
    }

    // look from end to start to preserve valid indexing
    for (int i = static_cast<int>(skeleton_.size()) - 1; i >= 0; --i) {
        if (incoming.count(skeleton_[i].vid)) continue; // still valid
        
        int last = static_cast<int>(skeleton_.size()) - 1;
        // swap and pop back
        if (i != last) {
            std::swap(skeleton_[i], skeleton_[last]);
        }
        skeleton_.pop_back();
    }

    // rebuild vid2idx_
    vid2idx_.clear();
    vid2idx_.reserve(skeleton_.size());
    for (int i=0; i<static_cast<int>(skeleton_.size()); ++i) {
        vid2idx_[skeleton_[i].vid] = i;
    }
}

std::optional<geometry_msgs::msg::PoseStamped> PlannerNode::get_drone_pose(const std::string& target_frame, const std::string drone_frame, const rclcpp::Duration& timeout) {
    const auto tf_timeout = tf2::durationFromSec(timeout.seconds());
    if (!tf_buffer_->canTransform(target_frame, drone_frame, tf2::TimePointZero, tf_timeout)) {
        RCLCPP_WARN(this->get_logger(), "TF not available: %s -> %s", drone_frame.c_str(), target_frame.c_str());
        return std::nullopt;
    }

    try {
        // auto T = tf_buffer_->lookupTransform(target_frame, drone_frame, tf2::TimePointZero);
        geometry_msgs::msg::PoseStamped drone_pose_in, drone_pose_out;
        drone_pose_in.header.frame_id = drone_frame;
        drone_pose_in.header.stamp = rclcpp::Time(0, 0, get_clock()->get_clock_type());
        drone_pose_in.pose.position.x = 0.0;
        drone_pose_in.pose.position.y = 0.0;
        drone_pose_in.pose.position.z = 0.0;
        drone_pose_in.pose.orientation.w = 1.0; // identity

        // Convert pose into target frame (uses the same latest transform)
        drone_pose_out = tf_buffer_->transform(drone_pose_in, target_frame, tf_timeout);
        
        return drone_pose_out;
    }
    catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "TF expection: %s", ex.what());
        return std::nullopt;
    }
}

void PlannerNode::process_tick() {
    MsgSkeleton::ConstSharedPtr skel_msg;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr map_msg;
    {
        std::scoped_lock lk(state_mtx_);
        skel_msg = latest_skel_;
        map_msg = latest_map_;
        latest_skel_.reset();
        latest_map_.reset();
    }
    
    if (!skel_msg || !map_msg) return;
    current_header = skel_msg->header;

    // Build map_cloud / octree (node owned) - Protected
    {
        // std::lock_guard<std::mutex> lk(state_mtx_);
        std::scoped_lock lk(state_mtx_);
        current_header = skel_msg->header;
        
        // map
        pcl::PointCloud<pcl::PointXYZ> tmp;
        pcl::fromROSMsg(*map_msg, tmp);
        map_cloud_->swap(tmp);

        // octree
        Eigen::Vector4f minpt, maxpt;
        pcl::getMinMax3D(*map_cloud_, minpt, maxpt);
        const float pad = 10.0f * map_voxel_size_;
        map_octree_->deleteTree();
        map_octree_->setInputCloud(map_cloud_);
        map_octree_->defineBoundingBox(
            minpt.x() - pad, minpt.y() - pad, minpt.z() - pad,
            maxpt.x() + pad, maxpt.y() + pad, maxpt.z() + pad
        );
        map_octree_->addPointsFromInputCloud();
        
        // update skeleton_ and vid2idx
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
        update_skeleton(skel_inc);
    
        // below should maybe also be protected by planner_mtx_???
        vpman_->set_map(map_cloud_, map_octree_);
        vpman_->set_vid2idx(vid2idx_);
        (void)vpman_->update_viewpoints(skeleton_);
    }

    bool planned = false;
    auto pose_opt = get_drone_pose(global_frame_, drone_frame_);
    if (!pose_opt) return;
    Eigen::Vector3f drone_pos(pose_opt->pose.position.x, pose_opt->pose.position.y, pose_opt->pose.position.z);
    Eigen::Quaternionf drone_ori;
    drone_ori.w() = pose_opt->pose.orientation.w;
    drone_ori.x() = pose_opt->pose.orientation.x;
    drone_ori.y() = pose_opt->pose.orientation.y;
    drone_ori.z() = pose_opt->pose.orientation.z;

    // Calling planner -> needs both protections
    {
        std::scoped_lock lock_all(state_mtx_, planner_mtx_); // Important to lock in that order
        if (bootstrap_mode_) return;

        planner_->set_map(map_cloud_, map_octree_);
        planner_->set_vid2idx(vid2idx_);
        planner_->set_drone_pose(drone_pos, drone_ori);
        planned = planner_->plan(skeleton_);
    }

    if (planned) {
        publish_path();
    }
}

void PlannerNode::control_tick() {
    auto pose_opt = get_drone_pose(global_frame_, drone_frame_);
    if (!pose_opt) return;

    Eigen::Vector3f drone_pos(pose_opt->pose.position.x,
                              pose_opt->pose.position.y,
                              pose_opt->pose.position.z);
    
    /* BOOTSTRAP MODE - INITIAL HARDCODED FLIGHT*/
    if (bootstrap_mode_) {
        std::lock_guard<std::mutex> lk(state_mtx_);
        if (!skeleton_.empty()) {
            RCLCPP_INFO(this->get_logger(), "Skeleton detected - Planner Takeover!");
            bootstrap_mode_ = false;
            return;
        }

        if (bootstrap_idx_ < bootstrap_waypoints_.size()) {
            const auto& tgt = bootstrap_waypoints_[bootstrap_idx_].pose;
            Eigen::Vector3f tgt_pos(tgt.position.x, tgt.position.y, tgt.position.z);
            float dist = (tgt_pos - drone_pos).norm();
            if (dist <= reached_dist_th_) {
                bootstrap_idx_++;
            }
        }
        else {
            bootstrap_mode_ = false;
        }

        if (bootstrap_mode_ && bootstrap_idx_ < bootstrap_waypoints_.size()) {
            nav_msgs::msg::Path tgt_msg;
            tgt_msg.header.frame_id = global_frame_;
            bootstrap_waypoints_[bootstrap_idx_].header.frame_id = global_frame_;
            bootstrap_waypoints_[bootstrap_idx_].header.stamp = this->get_clock()->now(); 
            tgt_msg.header.stamp = this->get_clock()->now();
            tgt_msg.poses.push_back(bootstrap_waypoints_[bootstrap_idx_]);
            target_pub_->publish(tgt_msg);
        }
        return; // don't run planner yet
    }

    std::scoped_lock lock_all(state_mtx_, planner_mtx_);

    // track reached viewpoints 
    Viewpoint current;
    float dist_to_target;
    if (planner_->get_next_target(current)) {
        dist_to_target = (current.position - drone_pos).norm();
        
        if (dist_to_target > prev_dist_to_target) {
            std::cout << "Reached target" << std::endl;
            planner_->notify_reached(skeleton_);
            vpman_->commit_coverage(current); // mutates skeleton_ coverage flags
            prev_dist_to_target = 1e6f;
        }

        else if (dist_to_target <= reached_dist_th_) {
            std::cout << "Dist to target: " << dist_to_target << std::endl;
            prev_dist_to_target = dist_to_target;
        }


    }

    if (!adjusted_) {
        const auto dt = this->get_clock()->now() - last_pub_time_;
        if (dt < rclcpp::Duration(std::chrono::milliseconds(adjust_timeout_ms_))) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500, "Awaiting adjusted viewpoints!");
            return;
        }
        adjusted_ = true;
    }

    std::vector<Viewpoint> targets;
    int k = 20;
    if (!planner_->get_next_k_targets(targets, k)) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Could not get path - Exiting!");
        return;
    }

    // Record last_pub_ and build message (state-owned)
    last_pub_.clear();
    last_pub_.reserve(targets.size());

    nav_msgs::msg::Path msg;
    msg.header.frame_id = global_frame_;
    msg.header.stamp = this->get_clock()->now();

    for (const auto& t : targets) {
        last_pub_.push_back({t.vptid, t.target_vid, t.target_vp_pos});
        geometry_msgs::msg::PoseStamped ps; ps.header = msg.header;
        ps.pose.position.x = t.position.x();
        ps.pose.position.y = t.position.y();
        ps.pose.position.z = t.position.z();
        ps.pose.orientation.x = t.orientation.x();
        ps.pose.orientation.y = t.orientation.y();
        ps.pose.orientation.z = t.orientation.z();
        ps.pose.orientation.w = t.orientation.w();
        msg.poses.push_back(ps);
    }

    target_pub_->publish(msg);
    last_pub_time_ = this->get_clock()->now();
    adjusted_ = false;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


