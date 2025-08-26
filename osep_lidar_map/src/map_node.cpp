/* 

LiDAR mapping Node

*/

#include <chrono>
#include <mutex>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/octree/octree_search.h>

class LidarMapNode : public rclcpp::Node {
public:
    LidarMapNode();

private:
    void pointcloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg);
    void publish_map();
    void maybe_rebuild();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
    rclcpp::TimerBase::SharedPtr pub_timer_;
    rclcpp::TimerBase::SharedPtr rebuild_timer_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::string pcd_topic_;
    std::string lidar_map_topic_;
    std::string global_frame_;
    float voxel_leaf_size_; // resolution
    float insert_radius_;
    float max_range_;
    float gnd_min_z_; // remove points below this

    int pub_ms_;
    int rebuild_ms_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr global_map_;
    std::unique_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> octree_;
    std::mutex map_mutex_;
    bool need_rebuild_{false};
};

LidarMapNode::LidarMapNode() : Node("LidarMapNode") {

    /* PARAMS */
    pcd_topic_ = declare_parameter<std::string>("lidar_topic", "/isaac/lidar/raw/pointcloud");
    lidar_map_topic_ = declare_parameter<std::string>("lidar_map_topic_", "/osep/lidar_map/global_map");
    global_frame_ = declare_parameter<std::string>("global_frame", "odom");
    voxel_leaf_size_ = declare_parameter<float>("voxel_leaf_size", 1.0f);
    insert_radius_ = declare_parameter<float>("insert_radius", 0.8f);
    max_range_ = declare_parameter<float>("max_range", 400.0f);
    gnd_min_z_ = declare_parameter<float>("gnd_min_z", 60.0f);
    pub_ms_ = declare_parameter<int>("publish_ms", 100);
    rebuild_ms_ = declare_parameter<int>("rebuild_ms", 2000);

    if (insert_radius_ <= 0.0 || insert_radius_ > voxel_leaf_size_) {
      insert_radius_ = 0.6 * voxel_leaf_size_;
    }

    /* ROS I/O */
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                                        pcd_topic_, rclcpp::SensorDataQoS(),
                                        std::bind(&LidarMapNode::pointcloud_callback, this, std::placeholders::_1));
        
    map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(lidar_map_topic_, rclcpp::SystemDefaultsQoS());
    
    pub_timer_ = create_wall_timer(std::chrono::milliseconds(pub_ms_),
                                    std::bind(&LidarMapNode::publish_map, this));
    rebuild_timer_ = create_wall_timer(std::chrono::milliseconds(rebuild_ms_),
                                    std::bind(&LidarMapNode::maybe_rebuild, this));

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    /* Data Storage */
    global_map_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    global_map_->reserve(200000);

    octree_ = std::make_unique<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>>(static_cast<float>(voxel_leaf_size_));
    octree_->setInputCloud(global_map_);
}

void LidarMapNode::pointcloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg) {
    if (pcd_msg->data.empty()) return;

    // Lookup transform
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_->lookupTransform(global_frame_, pcd_msg->header.frame_id, pcd_msg->header.stamp, tf2::durationFromSec(0.1));
    } 
    catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF lookup failed (%s -> %s): %s",
                           pcd_msg->header.frame_id.c_str(), global_frame_.c_str(), e.what());
      return;
    }
    const Eigen::Isometry3d T_map_from_cloud = tf2::transformToEigen(tf);

    // Convert to cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    try {
      pcl::fromROSMsg(*pcd_msg, *cloud);
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "fromROSMsg failed: %s", e.what());
      return;
    }

    // Clean NaNs
    std::vector<int> index_map;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, index_map);
    if (cloud->empty()) return;

    // Range crop
    if (max_range_ > 0.0) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cropped(new pcl::PointCloud<pcl::PointXYZ>());
      cropped->reserve(cloud->size());
      const float r2max = static_cast<float>(max_range_ * max_range_);
      for (const auto &p : cloud->points) {
        const float r2 = p.x*p.x + p.y*p.y + p.z*p.z;
        if (r2 <= r2max) cropped->push_back(p);
      }
      cloud.swap(cropped);
    }
    if (cloud->empty()) return;

    // Transform to global frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map(new pcl::PointCloud<pcl::PointXYZ>());
    cloud_map->reserve(cloud->size());
    Eigen::Matrix4f T = T_map_from_cloud.matrix().cast<float>();
    pcl::transformPointCloud(*cloud, *cloud_map, T);

    // Append to global map
    {
        std::scoped_lock lk(map_mutex_);
        
        std::vector<int> idxs;
        std::vector<float> d2s;

        for (const auto &p : cloud_map->points) {
          if (!std::isfinite(p.x) || !std::isfinite(p.z) || !std::isfinite(p.z)) continue;
          if (p.z < gnd_min_z_) continue;
          
          idxs.clear();
          d2s.clear();
          
          const bool empty = (octree_->radiusSearch(p, static_cast<float>(insert_radius_), idxs, d2s) == 0);
        
          if (empty) {
            global_map_->points.push_back(p);
            octree_->addPointToCloud(p, global_map_);
          }
          else {
            need_rebuild_ = true;
          }
        }

        // Maintain organized metadata
        global_map_->width  = static_cast<uint32_t>(global_map_->points.size());
        global_map_->height = 1;
        global_map_->is_dense = true;
    }  
}


void LidarMapNode::maybe_rebuild()
{
    std::scoped_lock lk(map_mutex_);
    if (!need_rebuild_ || global_map_->empty()) return;

    // Compress map to centroids with VoxelGrid (leaf = voxel_leaf_size_)
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setLeafSize(static_cast<float>(voxel_leaf_size_),
                   static_cast<float>(voxel_leaf_size_),
                   static_cast<float>(voxel_leaf_size_));
    vg.setInputCloud(global_map_);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.filter(*filtered);

    // Swap and rebuild octree
    global_map_.swap(filtered);
    octree_.reset(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(static_cast<float>(voxel_leaf_size_)));
    octree_->setInputCloud(global_map_);
    octree_->addPointsFromInputCloud();

    need_rebuild_ = false;
}

void LidarMapNode::publish_map() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_copy(new pcl::PointCloud<pcl::PointXYZ>());
    
    {
      std::scoped_lock lk(map_mutex_);
      if (global_map_->empty()) return;
      *map_copy = *global_map_; // shallow-ish copy of points
    }

    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*map_copy, out);
    out.header.stamp = this->now();
    out.header.frame_id = global_frame_;
    map_pub_->publish(out);
}


int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LidarMapNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}