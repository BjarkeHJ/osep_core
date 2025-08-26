/*

LiDAR mapping Node — EMA voxel grid + noise suppression

*/

#include <chrono>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class LidarMapNode : public rclcpp::Node {
public:
    LidarMapNode();

private:
    struct VoxelIndex {
        int x, y, z;
        bool operator==(const VoxelIndex& o) const noexcept {
            return x==o.x && y==o.y && z==o.z;
        }
    };
    struct VoxelHash {
        std::size_t operator()(const VoxelIndex& v) const noexcept {
            // Fast 3D int hash (works fine with negatives)
            // (x * p1) ^ (y * p2) ^ (z * p3)
            const uint64_t ux = static_cast<uint64_t>(static_cast<int64_t>(v.x));
            const uint64_t uy = static_cast<uint64_t>(static_cast<int64_t>(v.y));
            const uint64_t uz = static_cast<uint64_t>(static_cast<int64_t>(v.z));
            return (ux * 73856093ULL) ^ (uy * 19349663ULL) ^ (uz * 83492791ULL);
        }
    };
    struct VoxelStat {
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        Eigen::Vector3f var  = Eigen::Vector3f::Zero(); // optional; EMA of squared error
        int count = 0; // hit count
        bool initialized = false;
    };

    // ---- ROS I/O
    void pointcloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg);
    void publish_map();

    // ---- Helpers
    inline VoxelIndex toIndex(const Eigen::Vector3f& p) const {
        return VoxelIndex{
            static_cast<int>(std::floor(p.x() / voxel_size_)),
            static_cast<int>(std::floor(p.y() / voxel_size_)),
            static_cast<int>(std::floor(p.z() / voxel_size_))
        };
    }
    inline Eigen::Vector3f centerOf(const VoxelIndex& v) const {
        return Eigen::Vector3f(
            (v.x + 0.5f) * voxel_size_,
            (v.y + 0.5f) * voxel_size_,
            (v.z + 0.5f) * voxel_size_
        );
    }

    // ---- ROS
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;

    rclcpp::TimerBase::SharedPtr pub_timer_;
    rclcpp::TimerBase::SharedPtr maint_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ---- Params
    std::string pcd_topic_;
    std::string map_topic_;
    std::string global_frame_;

    float voxel_size_;          // grid resolution [m]
    float ema_alpha_;           // [0..1] position EMA gain (higher -> faster adapt)
    float max_range_;           // m (<=0 disables)
    float min_range_;
    float ground_min_z_;        // m (filter below this global Z)
    int publish_ms_;          // publish period

    // Noise suppression (publish-time gates)
    int min_count_;         
    int min_neighbors_;       // need this many neighbors in 26-neighborhood
    float neighbor_count_min_; // neighbors must have at least this count

    // Pruning bounds
    float prune_count_;        // drop voxels below this weight during maintenance
    int max_voxels_;          // soft cap; prune lightest if exceeded (optional)

    // ---- Data
    std::unordered_map<VoxelIndex, VoxelStat, VoxelHash> grid_;
    std::mutex grid_mutex_;
};

LidarMapNode::LidarMapNode() : Node("LidarMapNode")
{
    // Parameters (tuned defaults)
    pcd_topic_ = declare_parameter<std::string>("lidar_topic", "/isaac/lidar/raw/pointcloud");
    map_topic_ = declare_parameter<std::string>("lidar_map_topic", "/osep/lidar_map/global_map");
    global_frame_ = declare_parameter<std::string>("global_frame", "odom");

    voxel_size_ = declare_parameter<float>("voxel_size", 1.0f); // m
    ema_alpha_ = declare_parameter<float>("ema_alpha", 0.3f);   // 0.1–0.4 common

    min_range_ = declare_parameter<float>("min_range", 0.5);
    max_range_ = declare_parameter<float>("max_range", 120.0f);
    ground_min_z_ = declare_parameter<float>("ground_min_z", 60.0f);

    publish_ms_ = declare_parameter<int>("publish_ms", 100);

    min_count_ = declare_parameter<int>("min_weight", 10);
    min_neighbors_ = declare_parameter<int>("min_neighbors", 2);
    neighbor_count_min_= declare_parameter<float>("neighbor_weight_min", 2);

    prune_count_ = declare_parameter<float>("prune_weight", 1);
    max_voxels_ = declare_parameter<int>("max_voxels", 800000);  // soft cap

    // ROS I/O
    pcd_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pcd_topic_, rclcpp::SensorDataQoS(),
        std::bind(&LidarMapNode::pointcloud_callback, this, std::placeholders::_1));

    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(map_topic_, rclcpp::SystemDefaultsQoS());    

    pub_timer_ = create_wall_timer(
        std::chrono::milliseconds(publish_ms_), std::bind(&LidarMapNode::publish_map, this));

    // TF
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

void LidarMapNode::pointcloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg)
{
    if (!pcd_msg || pcd_msg->data.empty()) return;

    // TF to global
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_->lookupTransform(
            global_frame_, pcd_msg->header.frame_id, pcd_msg->header.stamp, tf2::durationFromSec(0.1));
    } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "TF lookup failed (%s -> %s): %s",
            pcd_msg->header.frame_id.c_str(), global_frame_.c_str(), e.what());
        return;
    }
    const Eigen::Isometry3d T_map_from_cloud = tf2::transformToEigen(tf);
    Eigen::Matrix4f T = T_map_from_cloud.matrix().cast<float>();

    // Convert incoming cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    try {
        pcl::fromROSMsg(*pcd_msg, *cloud);
    } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "fromROSMsg failed: %s", e.what());
        return;
    }

    // Clean NaNs
    std::vector<int> index_map;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, index_map);
    if (cloud->empty()) return;

    // Range crop (sensor frame)
    if (min_range_ > 0.0) {
        auto filtered = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        filtered->reserve(cloud->size());
        const float min_r2 = min_range_ * min_range_;
        for (const auto& p : cloud->points) {
            const float r2 = p.x*p.x + p.y*p.y + p.z*p.z;
            if (r2 >= min_r2) filtered->push_back(p);
        }
        cloud.swap(filtered);
        if (cloud->empty()) return;
    }
    if (max_range_ > 0.0) {
        const float r2max = max_range_ * max_range_;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cropped(new pcl::PointCloud<pcl::PointXYZ>);
        cropped->reserve(cloud->size());
        for (const auto &p : cloud->points) {
            const float r2 = p.x*p.x + p.y*p.y + p.z*p.z;
            if (r2 <= r2max) cropped->push_back(p);
        }
        cloud.swap(cropped);
        if (cloud->empty()) return;
    }

    // Transform to global frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_map->reserve(cloud->size());
    pcl::transformPointCloud(*cloud, *cloud_map, T);
    if (cloud_map->empty()) return;

    // Update EMA voxels
    std::scoped_lock lk(grid_mutex_);
    for (const auto &p : cloud_map->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        if (p.z < ground_min_z_) continue;

        Eigen::Vector3f pf(p.x, p.y, p.z); // Actual point position
        VoxelIndex vidx = toIndex(pf);
        auto &v = grid_[vidx]; // Center of voxel

        // If the voxel has not seen any points
        if (!v.initialized) {
          v.mean = pf;
          v.var  = Eigen::Vector3f::Zero();
          v.count = 1;
          v.initialized = true;
          continue;
        }
        else {
          // EMA mean
          const Eigen::Vector3f err  = pf - v.mean;
          v.mean += ema_alpha_ * err;

          // EMA variance (per-axis) around the *updated* mean
          const Eigen::Vector3f err2 = (pf - v.mean).cwiseAbs2();
          v.var  = (1.f - ema_alpha_) * v.var + ema_alpha_* err2;

          // Plain hit count
          v.count += 1;
        }
    }

    // Soft cap: if exceeding max_voxels_, prune some light voxels right away
    if (max_voxels_ > 0 && static_cast<int>(grid_.size()) > max_voxels_) {
        // Quick linear pass (keep heaviest)
        std::vector<VoxelIndex> to_drop;
        to_drop.reserve(grid_.size()/10);
        const size_t target = grid_.size() - max_voxels_;
        for (const auto& kv : grid_) {
            if (kv.second.count < prune_count_) {
                to_drop.push_back(kv.first);
                if (to_drop.size() >= target) break;
            }
        }
        for (const auto& k : to_drop) grid_.erase(k);
    }
}

void LidarMapNode::publish_map() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    out->reserve(grid_.size());

    {
        std::scoped_lock lk(grid_mutex_);
        if (grid_.empty()) return;

        // For each voxel that’s “confident”, require neighbor support
        for (const auto& kv : grid_) {
            const VoxelIndex& c = kv.first;
            const VoxelStat&  v = kv.second;
            if (!v.initialized) continue;
            if (v.count < min_count_) continue;

            int neighbors = 0;
            for (int dx=-1; dx<=1; ++dx) {
                for (int dy=-1; dy<=1; ++dy) {
                    for (int dz=-1; dz<=1; ++dz) {
                        if (dx==0 && dy==0 && dz==0) continue;
                        VoxelIndex n{c.x+dx, c.y+dy, c.z+dz};
                        auto itn = grid_.find(n);
                        if (itn != grid_.end() && itn->second.initialized &&
                            itn->second.count >= neighbor_count_min_) {
                            ++neighbors;
                        }
                    }
                }
            }
            if (neighbors < min_neighbors_) continue;

            // std gate drop
            const Eigen::Vector3f stddev = v.var.cwiseMax(1e-12f).cwiseSqrt();
            const float max_std = 0.6f * voxel_size_;
            if (stddev.maxCoeff() > max_std) continue;
            
            const Eigen::Vector3f c3 = centerOf(c);
            out->points.emplace_back(c3.x(), c3.y(), c3.z());
            // out->points.emplace_back(v.mean.x(), v.mean.y(), v.mean.z());
        }
    }

    if (out->empty()) return;

    out->width = static_cast<uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = true;

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*out, msg);
    msg.header.stamp = now();
    msg.header.frame_id = global_frame_;
    map_pub_->publish(msg);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarMapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
