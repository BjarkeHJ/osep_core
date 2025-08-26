/*

LiDAR mapping Node — EMA voxel grid with temporal decay + noise suppression

- Insert/update by voxel index (hash map) => O(1)
- Each voxel stores:
    mean (EMA centroid), weight (decayed hit count), variance (optional), last_seen
- Maintenance timer:
    * decays weights of stale voxels
    * prunes very light/old voxels
- Publish:
    * outputs centroids for voxels that pass:
        weight >= min_weight AND neighbor_support >= min_neighbors

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
        Eigen::Vector3f var  = Eigen::Vector3f::Zero();  // optional; EMA of squared error
        float weight = 0.f;                               // decayed hit count
        rclcpp::Time last_seen;
        bool initialized = false;
    };

    // ---- ROS I/O
    void pointcloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg);
    void publish_map();
    void maintenance_tick();

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
            (v.x + 0.5f) * static_cast<float>(voxel_size_),
            (v.y + 0.5f) * static_cast<float>(voxel_size_),
            (v.z + 0.5f) * static_cast<float>(voxel_size_)
        );
    }
    inline float decayFactor(double dt_sec) const {
        if (decay_lambda_ <= 0.0) return 1.0f;
        return static_cast<float>(std::exp(-decay_lambda_ * dt_sec));
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

    double voxel_size_;          // m (grid resolution)
    double ema_alpha_;           // [0..1] position EMA gain (higher -> faster adapt)
    double half_life_sec_;       // seconds for weight to half (<=0 disables decay)
    double decay_lambda_;        // computed from half_life_sec_: ln(2)/half_life
    double max_range_;           // m (<=0 disables)
    double ground_min_z_;        // m (filter below this global Z)
    int    publish_ms_;          // publish period
    int    maintenance_ms_;      // maintenance (decay/prune) period

    // Noise suppression (publish-time gates)
    double min_weight_;          // output only voxels with >= this decayed count
    int    min_neighbors_;       // need this many neighbors in 26-neighborhood
    double neighbor_weight_min_; // neighbors must have at least this weight

    // Pruning bounds
    double prune_weight_;        // drop voxels below this weight during maintenance
    double max_age_sec_;         // hard age cap (<=0 disables)
    size_t max_voxels_;          // soft cap; prune lightest if exceeded (optional)

    // ---- Data
    std::unordered_map<VoxelIndex, VoxelStat, VoxelHash> grid_;
    std::mutex grid_mutex_;
};

LidarMapNode::LidarMapNode() : Node("LidarMapNode")
{
    // Parameters (tuned defaults)
    pcd_topic_      = declare_parameter<std::string>("lidar_topic", "/isaac/lidar/raw/pointcloud");
    map_topic_      = declare_parameter<std::string>("lidar_map_topic", "/osep/lidar_map/global_map");
    global_frame_   = declare_parameter<std::string>("global_frame", "odom");

    voxel_size_     = declare_parameter<double>("voxel_size", 0.25); // m
    ema_alpha_      = declare_parameter<double>("ema_alpha", 0.3);   // 0.1–0.4 common
    half_life_sec_  = declare_parameter<double>("half_life_sec", 0.0);
    decay_lambda_   = (half_life_sec_ > 0.0) ? std::log(2.0)/half_life_sec_ : 0.0;

    max_range_      = declare_parameter<double>("max_range", 120.0);
    ground_min_z_   = declare_parameter<double>("ground_min_z", -1e9);

    publish_ms_     = declare_parameter<int>("publish_ms", 100);
    maintenance_ms_ = declare_parameter<int>("maintenance_ms", 500);

    min_weight_         = declare_parameter<double>("min_weight", 3.0);
    min_neighbors_      = declare_parameter<int>("min_neighbors", 2);
    neighbor_weight_min_= declare_parameter<double>("neighbor_weight_min", 2.0);

    prune_weight_   = declare_parameter<double>("prune_weight", 0.5);
    max_age_sec_    = declare_parameter<double>("max_age_sec", 0.0);    // 0 = disable
    max_voxels_     = declare_parameter<int>("max_voxels", 800000);  // soft cap

    // ROS I/O
    pcd_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pcd_topic_, rclcpp::SensorDataQoS(),
        std::bind(&LidarMapNode::pointcloud_callback, this, std::placeholders::_1));

    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(map_topic_, rclcpp::SystemDefaultsQoS());

    pub_timer_ = create_wall_timer(
        std::chrono::milliseconds(publish_ms_), std::bind(&LidarMapNode::publish_map, this));

    maint_timer_ = create_wall_timer(
        std::chrono::milliseconds(maintenance_ms_), std::bind(&LidarMapNode::maintenance_tick, this));

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
    pcl::PointCloud<PointT>::Ptr cloud(new CloudT());
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

    // Optional range crop (sensor frame)
    if (max_range_ > 0.0) {
        const float r2max = static_cast<float>(max_range_ * max_range_);
        CloudT::Ptr cropped(new CloudT());
        cropped->reserve(cloud->size());
        for (const auto &p : cloud->points) {
            const float r2 = p.x*p.x + p.y*p.y + p.z*p.z;
            if (r2 <= r2max) cropped->push_back(p);
        }
        cloud.swap(cropped);
        if (cloud->empty()) return;
    }

    // Transform to global frame
    CloudT::Ptr cloud_map(new CloudT());
    cloud_map->reserve(cloud->size());
    pcl::transformPointCloud(*cloud, *cloud_map, T);
    if (cloud_map->empty()) return;

    const rclcpp::Time now_t = now();

    // Update EMA voxels
    std::scoped_lock lk(grid_mutex_);
    for (const auto &p : cloud_map->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        if (p.z < static_cast<float>(ground_min_z_)) continue;

        Eigen::Vector3f pf(p.x, p.y, p.z);
        VoxelIndex vidx = toIndex(pf);
        auto &v = grid_[vidx];

        if (!v.initialized) {
            v.mean = pf;
            v.var  = Eigen::Vector3f::Zero();
            v.weight = 1.f;
            v.last_seen = now_t;
            v.initialized = true;
            continue;
        }

        // Temporal decay since last_seen
        const double dt = (now_t - v.last_seen).seconds();
        const float  d  = decayFactor(dt);
        v.weight *= d;
        v.var    *= d; // decay variance too (simple, effective)

        // EMA update
        const Eigen::Vector3f err = pf - v.mean;
        v.mean += static_cast<float>(ema_alpha_) * err;
        // EMA variance (per-axis)
        const Eigen::Vector3f err2 = (pf - v.mean).cwiseAbs2();
        v.var  = (1.f - static_cast<float>(ema_alpha_)) * v.var + static_cast<float>(ema_alpha_) * err2;

        // Increment weight for this observation
        v.weight += 1.f;
        v.last_seen = now_t;
    }

    // Soft cap: if exceeding max_voxels_, prune some light voxels right away
    if (max_voxels_ > 0 && grid_.size() > max_voxels_) {
        // Quick linear pass (keep heaviest)
        std::vector<VoxelIndex> to_drop;
        to_drop.reserve(grid_.size()/10);
        const size_t target = grid_.size() - max_voxels_;
        for (const auto& kv : grid_) {
            if (kv.second.weight < prune_weight_) {
                to_drop.push_back(kv.first);
                if (to_drop.size() >= target) break;
            }
        }
        for (const auto& k : to_drop) grid_.erase(k);
    }
}

void LidarMapNode::maintenance_tick()
{
    std::scoped_lock lk(grid_mutex_);
    if (grid_.empty()) return;

    const rclcpp::Time tnow = now();
    for (auto it = grid_.begin(); it != grid_.end(); ) {
        VoxelStat &v = it->second;

        const double age = (tnow - v.last_seen).seconds();
        const float  d   = decayFactor(age);

        // Lazy apply decay (brings weight close to real-time)
        float decayed_w = v.weight * d;

        bool old = (max_age_sec_ > 0.0) && (age > max_age_sec_);
        if (old || decayed_w < prune_weight_) {
            it = grid_.erase(it);
        } else {
            // Optionally clamp stored weight to decayed value to prevent unbounded growth
            v.weight = decayed_w;
            ++it;
        }
    }
}

void LidarMapNode::publish_map()
{
    pcl::PointCloud<PointT>::Ptr out(new CloudT());
    out->reserve(grid_.size());

    {
        std::scoped_lock lk(grid_mutex_);
        if (grid_.empty()) return;

        // For each voxel that’s “confident”, require neighbor support
        for (const auto& kv : grid_) {
            const VoxelIndex& c = kv.first;
            const VoxelStat&  v = kv.second;
            if (!v.initialized) continue;
            if (v.weight < static_cast<float>(min_weight_)) continue;

            int neighbors = 0;
            for (int dx=-1; dx<=1; ++dx) {
                for (int dy=-1; dy<=1; ++dy) {
                    for (int dz=-1; dz<=1; ++dz) {
                        if (dx==0 && dy==0 && dz==0) continue;
                        VoxelIndex n{c.x+dx, c.y+dy, c.z+dz};
                        auto itn = grid_.find(n);
                        if (itn != grid_.end() && itn->second.initialized &&
                            itn->second.weight >= static_cast<float>(neighbor_weight_min_)) {
                            ++neighbors;
                        }
                    }
                }
            }
            if (neighbors < min_neighbors_) continue;

            // Optional: variance gate to drop flicker (tune e.g., max stddev 0.5*voxel)
            const Eigen::Vector3f stddev = v.var.cwiseMax(1e-12f).cwiseSqrt();
            const float max_std = 0.6f * static_cast<float>(voxel_size_);
            if (stddev.maxCoeff() > max_std) continue;

            out->points.emplace_back(v.mean.x(), v.mean.y(), v.mean.z());
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
