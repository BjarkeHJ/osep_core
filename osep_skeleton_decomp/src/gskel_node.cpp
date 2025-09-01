/* 

ROS 2 Node: Global Skeleton

TODO: Publish edges (markers)
TODO: Publish branch segments ??

*/

#include "gskel.hpp"
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "osep_skeleton_decomp/msg/vertex.hpp"
#include "osep_skeleton_decomp/msg/global_skeleton.hpp"

using MsgVertex = osep_skeleton_decomp::msg::Vertex;
using MsgSkeleton = osep_skeleton_decomp::msg::GlobalSkeleton;

class GSkelNode : public rclcpp::Node {
public:
    GSkelNode();

private:
    /* Functions */
    void vertex_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg);
    void process_tick();
    void publish_gskel(pcl::PointCloud<pcl::PointXYZ>::Ptr& gskel_out, const std_msgs::msg::Header& src_header);
    void publish_gvert(const std::vector<Vertex>& vertices, const std_msgs::msg::Header& src_header);
    void publish_edges(const std::vector<Vertex>& vertices, const std_msgs::msg::Header& src_header);
    rclcpp::Time tf_safe_stamp(const std::string& target, const std::string& source);

    /* ROS2 */
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr vers_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr gskel_pub_;
    rclcpp::Publisher<MsgSkeleton>::SharedPtr gvert_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr conn_mk_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr tick_timer_;
    
    /* Params */
    std::string rosa_topic_;
    std::string gskel_topic_;
    std::string gvert_topic_;
    std::string edge_topic_;
    std::string global_frame_id_;
    int tick_ms_;
    
    /* Utils */
    std::mutex latest_mutex_;
    std::unique_ptr<GSkel> gskel_;
    
    /* Data */
    GSkelConfig gskel_cfg;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_msg_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_skeleton;
};

GSkelNode::GSkelNode() : Node("GSkelNode") {
    /* LAUNCH FILE PARAMETER DECLARATIONS */
    // MISC
    rosa_topic_ = declare_parameter<std::string>("rosa_topic", "/osep/rosa/local_rosa_points");
    gskel_topic_ = declare_parameter<std::string>("gskel_topic", "/osep/gskel/global_skeleton_cloud");
    gvert_topic_ = declare_parameter<std::string>("gvert_topic", "/osep/gskel/global_skeleton_vertices");
    edge_topic_ = declare_parameter<std::string>("edge_topic", "/osep/gskel/global_skeleton_edges");
    tick_ms_ = declare_parameter<int>("tick_ms", 50);
    global_frame_id_ = declare_parameter<std::string>("global_frame_id", "odom");
    // GSKEL
    gskel_cfg.gnd_th = declare_parameter<float>("gskel_gnd_th", 60.0f);
    gskel_cfg.fuse_dist_th = declare_parameter<float>("gskel_fuse_dist_th", 3.0f);
    gskel_cfg.fuse_conf_th = declare_parameter<float>("gskel_fuse_conf_th", 0.05f);
    gskel_cfg.lkf_pn = declare_parameter<float>("gskel_lkf_pn", 0.0001f);
    gskel_cfg.lkf_mn = declare_parameter<float>("gskel_lkf_mn", 0.1f);
    gskel_cfg.max_obs_wo_conf = declare_parameter<int>("gskel_max_obs_wo_conf", 5);
    gskel_cfg.niter_smooth_vertex = declare_parameter<int>("gskel_niter_vertex_smooth", 3);
    gskel_cfg.vertex_smooth_coef = declare_parameter<float>("gskel_vertex_smooth_coef", 0.3f);
    gskel_cfg.min_branch_length = declare_parameter<int>("gskel_min_branch_length", 5);

    /* OBJECT INITIALIZATION */
    gskel_ = std::make_unique<GSkel>(gskel_cfg);

    /* ROS2 */
    auto sub_qos = rclcpp::SensorDataQoS().keep_last(1);
    vers_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(rosa_topic_,
                                                                         sub_qos,
                                                                         std::bind(&GSkelNode::vertex_callback, this, std::placeholders::_1));
    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(1));
    pub_qos.reliable().transient_local();
    gskel_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(gskel_topic_, pub_qos);
    gvert_pub_ = this->create_publisher<MsgSkeleton>(gvert_topic_, pub_qos);
    conn_mk_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(edge_topic_, pub_qos);

    tick_timer_ = create_wall_timer(std::chrono::milliseconds(tick_ms_), std::bind(&GSkelNode::process_tick, this));

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
                                                                            
    /* INITIALIZE DATA STRUCTURES */
    global_skeleton.reset(new pcl::PointCloud<pcl::PointXYZ>);
    global_skeleton->points.reserve(1000);
    RCLCPP_INFO(this->get_logger(), "GSkelNode Initialized");
}

void GSkelNode::vertex_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr vertex_msg) {
    if (!vertex_msg) return;
    std::scoped_lock lk(latest_mutex_);
    latest_msg_ = std::move(vertex_msg);
}

void GSkelNode::publish_gskel(pcl::PointCloud<pcl::PointXYZ>::Ptr& gskel_out, const std_msgs::msg::Header& src_header) {
    if (!gskel_out) return;

    gskel_out->width = gskel_out->points.size();
    gskel_out->height = 1;
    gskel_out->is_dense = true;

    if (gskel_out->points.size() != static_cast<int>(gskel_out->width) * gskel_out->height) {
        RCLCPP_ERROR(get_logger(), "[PUBLISH GSKEL] Error: Bad Cloud Dims!");
        return;
    }

    sensor_msgs::msg::PointCloud2 gskel_cloud_msg;
    pcl::toROSMsg(*gskel_out, gskel_cloud_msg);
    gskel_cloud_msg.header = src_header;
    if (gskel_cloud_msg.header.frame_id.empty()) gskel_cloud_msg.header.frame_id = global_frame_id_;
    if (gskel_cloud_msg.header.stamp.sec == 0 && gskel_cloud_msg.header.stamp.nanosec == 0) {
        gskel_cloud_msg.header.stamp = this->get_clock()->now();
    }
    gskel_pub_->publish(gskel_cloud_msg);
}

void GSkelNode::publish_gvert(const std::vector<Vertex>& vertices, const std_msgs::msg::Header& src_header) {
    MsgSkeleton gskel_msg;
    gskel_msg.header = src_header;
    if (gskel_msg.header.frame_id.empty()) gskel_msg.header.frame_id = global_frame_id_;
    if (gskel_msg.header.stamp.sec == 0 && gskel_msg.header.stamp.nanosec == 0) {
        gskel_msg.header.stamp = this->get_clock()->now();
    } 

    gskel_msg.vertices.reserve(vertices.size());
    gskel_msg.vertices.resize(vertices.size());
    for (size_t i=0; i<vertices.size(); ++i) {
        const auto& v = vertices[i];
        MsgVertex v_msg;
        v_msg.id = v.vid;
        v_msg.position.x = v.position.x;
        v_msg.position.y = v.position.y;
        v_msg.position.z = v.position.z;
        v_msg.type = v.type;
        v_msg.pos_update = v.pos_update;
        v_msg.type_update = v.type_update;
        for (int nid : v.nb_ids) {
            v_msg.adj.push_back(nid);
        }
        if (v.pos_update || v.type_update) {
            v_msg.last_update = this->get_clock()->now();
        }
        gskel_msg.vertices[i] = v_msg;
    }
    gvert_pub_->publish(std::move(gskel_msg));
}

void GSkelNode::publish_edges(const std::vector<Vertex>& vertices, const std_msgs::msg::Header& src_header) {
    visualization_msgs::msg::Marker edges_msg;
    edges_msg.header = src_header;
    if (edges_msg.header.frame_id.empty()) edges_msg.header.frame_id = global_frame_id_;
    if (edges_msg.header.stamp.sec == 0 && edges_msg.header.stamp.nanosec == 0) {
        edges_msg.header.stamp = this->get_clock()->now();
    }
    edges_msg.type = visualization_msgs::msg::Marker::LINE_LIST;
    edges_msg.action = visualization_msgs::msg::Marker::ADD;
    edges_msg.pose.orientation.w = 1.0;
    edges_msg.scale.x = 0.05;
    edges_msg.color.r = 0.0;
    edges_msg.color.g = 0.0;
    edges_msg.color.b = 0.1;
    edges_msg.color.a = 1.0;

    geometry_msgs::msg::Point p1, p2;
    for (const auto& v : vertices) {
        const auto& nbs = v.nb_ids;
        for (int nb : nbs) {
            const auto& nb_v = vertices[nb];
            if (v.vid < nb_v.vid) {
                p1.x = v.position.x;
                p1.y = v.position.y;
                p1.z = v.position.z;
                p2.x = nb_v.position.x;
                p2.y = nb_v.position.y;
                p2.z = nb_v.position.z;
                edges_msg.points.push_back(p1);
                edges_msg.points.push_back(p2);
            }
        }
    }

    conn_mk_pub_->publish(edges_msg);
}

void GSkelNode::process_tick() {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr msg;
    {
        std::scoped_lock lk(latest_mutex_);
        msg = latest_msg_;
        latest_msg_.reset();
    }
    
    bool updated = false;
    rclcpp::Time meas_stamp;

    if (msg) {
        // get transform and transform vertices
        geometry_msgs::msg::TransformStamped T;
        try {
            const bool zero_stamp = (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0);
            if (zero_stamp) {
                T = tf_buffer_->lookupTransform(global_frame_id_,
                                                msg->header.frame_id,
                                                tf2::TimePointZero,
                                                tf2::durationFromSec(0.1));
            }
            else {
                T = tf_buffer_->lookupTransform(global_frame_id_,
                                                msg->header.frame_id,
                                                rclcpp::Time(msg->header.stamp),
                                                rclcpp::Duration::from_seconds(0.1));
            }
        }
        catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(), "TF Lookup failed: %s", ex.what());
            msg.reset();
        }

        if (msg) {
            sensor_msgs::msg::PointCloud2 gmsg;
            tf2::doTransform(*msg, gmsg, T);
            pcl::fromROSMsg(gmsg, gskel_->input_vertices());

            // Run GSkel and update global skeleton 
            if (gskel_->gskel_run()) {
                auto out = gskel_->output_cloud();
                if (out) *global_skeleton = *out;
                updated = true;
                meas_stamp = rclcpp::Time(msg->header.stamp);
            }
        }
    }

    // Publish the global skeleton (even without update)
    std_msgs::msg::Header hdr;
    hdr.frame_id = global_frame_id_;
    if (updated && meas_stamp.nanoseconds() > 0) {
        hdr.stamp = meas_stamp;
    }
    else {
        hdr.stamp = tf_safe_stamp("lidar", global_frame_id_);
    }
    publish_gskel(global_skeleton, hdr);
    publish_gvert(gskel_->output_vertices(), hdr);
    publish_edges(gskel_->output_vertices(), hdr);
}

rclcpp::Time GSkelNode::tf_safe_stamp(const std::string& target, const std::string& source) {
    const auto now = this->get_clock()->now();
    // If TF can transform at 'now', use it (best for live viz)
    if (tf_buffer_->canTransform(target, source, now, tf2::durationFromSec(0.0))) {
        return now;
    }
    // Otherwise fall back to the latest available TF time
    try {
        auto T = tf_buffer_->lookupTransform(target, source, tf2::TimePointZero);
        return rclcpp::Time(T.header.stamp);
    } catch (...) {
        // Last resort: slightly behind now to avoid leading-edge TF misses
        return now - rclcpp::Duration::from_seconds(0.02); // 20 ms
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GSkelNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}