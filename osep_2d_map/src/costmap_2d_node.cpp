#include "costmap_2d.hpp"
#include <queue>
#include <limits>
#include <cmath>

ESDF2dCostMapNode::ESDF2dCostMapNode()
: Node("costmap_2d_node"),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
  // Declare and retrieve parameters
  this->declare_parameter("resolution", 1.0);     // Cell size in meters
  this->declare_parameter("free_center_radius", 10.0); // Radius of the free center area
  this->declare_parameter("local_map_size", 200.0);    // Local map size (200 m x 200 m)
  this->declare_parameter("global_map_size", 1500.0);  // Global map size (1500 m x 1500 m)
  this->declare_parameter("frame_id", "base_link");    // Map centered at base_link
  this->declare_parameter("safety_distance", 10.0);  // Central safety distance
  this->declare_parameter("costmap_topic", "/local_costmap/costmap");

  resolution_ = this->get_parameter("resolution").as_double();
  free_center_radius_ = this->get_parameter("free_center_radius").as_double();
  local_map_size_ = this->get_parameter("local_map_size").as_double();
  global_map_size_ = this->get_parameter("global_map_size").as_double();
  frame_id_ = this->get_parameter("frame_id").as_string();
  safety_distance_ = this->get_parameter("safety_distance").as_double();
  costmap_topic_ = this->get_parameter("costmap_topic").as_string();
  safety_distance_min_ = safety_distance_ - 0.2 * safety_distance_;
  safety_distance_max_ = safety_distance_ + 0.2 * safety_distance_;


  // Compute grid dimensions
  local_grid_size_ = static_cast<int>(local_map_size_ / resolution_);
  global_grid_size_ = static_cast<int>(global_map_size_ / resolution_);
  local_half_size_ = local_map_size_ / 2.0;
  global_half_size_ = global_map_size_ / 2.0;

  // Initialize the global map
  global_map_.info.resolution = resolution_;
  global_map_.info.width = global_grid_size_;
  global_map_.info.height = global_grid_size_;
  global_map_.info.origin.position.x = -global_half_size_;
  global_map_.info.origin.position.y = -global_half_size_;
  global_map_.info.origin.position.z = 0.0;
  global_map_.info.origin.orientation.w = 1.0;
  global_map_.data.resize(global_grid_size_ * global_grid_size_, -1); // Initialize as unknown

  // Subscribe to the ESDF point cloud topic
  esdf_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/nvblox_node/static_esdf_pointcloud", 10,
    std::bind(&ESDF2dCostMapNode::esdf_callback, this, std::placeholders::_1)
  );

  // Publisher for the global cost map
  global_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("osep/global_costmap/costmap", 10);

  // Publisher for the local cost map
  local_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(costmap_topic_, 10);

  esdf_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("osep/esdf_costmap", 10);

  RCLCPP_INFO(this->get_logger(), "ESDF cost map node initialized.");
}

bool ESDF2dCostMapNode::validate_pointcloud2_fields(const sensor_msgs::msg::PointCloud2& msg) {
    return msg.fields.size() >= 4 &&
           msg.fields[0].name == "x" &&
           msg.fields[1].name == "y" &&
           msg.fields[2].name == "z" &&
           msg.fields[3].name == "intensity";
}

void ESDF2dCostMapNode::convert_cloud_to_esdf_grid(
    const sensor_msgs::msg::PointCloud2& msg,
    std::vector<float>& esdf_grid,
    std::vector<bool>& esdf_mask,
    int grid_width,
    int grid_height,
    float origin_x,
    float origin_y,
    float resolution
) {
    // Convert to PCL cloud
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::fromROSMsg(msg, cloud);

    // Initialize grid and mask
    esdf_grid.assign(grid_width * grid_height, NAN);
    esdf_mask.assign(grid_width * grid_height, false);

    // Fill grid and mask from cloud
    for (const auto& point : cloud.points) {
        int grid_x = static_cast<int>((point.x - origin_x) / resolution);
        int grid_y = static_cast<int>((point.y - origin_y) / resolution);

        if (grid_x >= 0 && grid_x < grid_width && grid_y >= 0 && grid_y < grid_height) {
            int idx = grid_y * grid_width + grid_x;
            esdf_grid[idx] = point.intensity;
            esdf_mask[idx] = true;
        }
    }
}

void ESDF2dCostMapNode::fill_esdf_holes_wavefront(
    std::vector<float>& esdf_grid,
    std::vector<bool>& esdf_mask,
    int width,
    int height,
    float safety_distance,
    float resolution)
{
    struct Cell { int x, y; float value; };
    std::queue<Cell> q;

    // Initialize queue with all valid cells < safety_distance
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (esdf_mask[idx] && esdf_grid[idx] < safety_distance) {
                q.push({x, y, esdf_grid[idx]});
            }
        }
    }

    // 8-connected neighbors (including diagonals)
    int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};

    while (!q.empty()) {
        Cell cell = q.front(); q.pop();
        for (int k = 0; k < 8; ++k) {
            int nx = cell.x + dx[k];
            int ny = cell.y + dy[k];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = ny * width + nx;
                // Use sqrt(2)*resolution for diagonal neighbors
                float step = (k < 4) ? resolution : static_cast<float>(std::sqrt(2.0)) * resolution;
                float new_val = std::min(cell.value + step, safety_distance);
                if (!esdf_mask[nidx] || (esdf_grid[nidx] > new_val)) {
                    esdf_grid[nidx] = new_val;
                    esdf_mask[nidx] = true;
                    if (new_val < safety_distance) {
                        q.push({nx, ny, new_val});
                    }
                }
            }
        }
    }
}

std::optional<geometry_msgs::msg::TransformStamped> ESDF2dCostMapNode::get_transform_to_odom() {
    try {
        return tf_buffer_.lookupTransform("odom", frame_id_, rclcpp::Time(0));
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "Could not transform %s to odom: %s", frame_id_.c_str(), ex.what());
        return std::nullopt;
    }
}

void ESDF2dCostMapNode::extract_local_from_global(nav_msgs::msg::OccupancyGrid& local_map) {
  for (int y = 0; y < local_grid_size_; ++y) {
    for (int x = 0; x < local_grid_size_; ++x) {
      int global_x = static_cast<int>((local_map.info.origin.position.x + x * resolution_ - global_map_.info.origin.position.x) / resolution_);
      int global_y = static_cast<int>((local_map.info.origin.position.y + y * resolution_ - global_map_.info.origin.position.y) / resolution_);
      int global_index = global_y * global_grid_size_ + global_x;
      int local_index = y * local_grid_size_ + x;
      if (global_x >= 0 && global_x < global_grid_size_ && global_y >= 0 && global_y < global_grid_size_) {
          local_map.data[local_index] = global_map_.data[global_index];
      }
    }
  }
}
// Erosion filter: eats away the outer layer of nonzero cells
void ESDF2dCostMapNode::erosion_filter(nav_msgs::msg::OccupancyGrid& local_map) {
    std::vector<int8_t> filtered = local_map.data;
    for (int y = 1; y < local_grid_size_ - 1; ++y) {
        for (int x = 1; x < local_grid_size_ - 1; ++x) {
            int idx = y * local_grid_size_ + x;
            if (local_map.data[idx] > 0) {
                int zero_count = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nidx = (y + dy) * local_grid_size_ + (x + dx);
                        if (local_map.data[nidx] == 0) {
                            zero_count++;
                        }
                    }
                }
                if (zero_count >= 5) {
                    filtered[idx] = 0;
                } else if (zero_count >= 3) {
                    if (local_map.data[idx] == 100) {
                        filtered[idx] = 50;
                    } else {
                        filtered[idx] = 0;
                    }
                }
            }
        }
    }
    local_map.data = std::move(filtered);
}

void ESDF2dCostMapNode::overwrite_local_with_esdf(
    nav_msgs::msg::OccupancyGrid& local_map,
    const std::vector<float>& esdf_grid,
    const std::vector<bool>& esdf_mask
) {
    for (int y = 0; y < local_grid_size_; ++y) {
        for (int x = 0; x < local_grid_size_; ++x) {
            int idx = y * local_grid_size_ + x;
            if (!esdf_mask[idx]) continue; // skip if not set by ESDF

            float distance = esdf_grid[idx];
            int cost;
            if (distance < 0.0f) {
                cost = 100;
            } else if (distance <= safety_distance_min_) {
                cost = 100;
            } else if (distance <= safety_distance_max_) {
                cost = static_cast<int>(100.0f * (1.0f - (distance - safety_distance_min_) /
                                                  (safety_distance_max_ - safety_distance_min_)));
            } else {
                cost = 0;
            }
            cost = std::clamp(cost, 0, 100);
            local_map.data[idx] = cost;
        }
    }
}

nav_msgs::msg::OccupancyGrid ESDF2dCostMapNode::create_local_map(const geometry_msgs::msg::TransformStamped& transform) {
  nav_msgs::msg::OccupancyGrid local_map;
  local_map.info.resolution = resolution_;
  local_map.info.width = local_grid_size_;
  local_map.info.height = local_grid_size_;
  local_map.info.origin.position.x = std::floor((transform.transform.translation.x - local_half_size_ - global_map_.info.origin.position.x) / resolution_) * resolution_ + global_map_.info.origin.position.x;
  local_map.info.origin.position.y = std::floor((transform.transform.translation.y - local_half_size_ - global_map_.info.origin.position.y) / resolution_) * resolution_ + global_map_.info.origin.position.y;
  local_map.info.origin.position.z = 0.0;
  local_map.info.origin.orientation.w = 1.0;
  local_map.data.resize(local_grid_size_ * local_grid_size_, -1); // Initialize as unknown
  return local_map;
}

void ESDF2dCostMapNode::clear_local_center(nav_msgs::msg::OccupancyGrid& local_map) {
  double clear_radius = free_center_radius_; // Radius in meters
  int clear_radius_cells = static_cast<int>(clear_radius / resolution_);

  // Calculate the center of the global map in the local map's coordinate system
  int global_center_x = static_cast<int>((global_map_.info.origin.position.x + global_half_size_ - local_map.info.origin.position.x) / resolution_);
  int global_center_y = static_cast<int>((global_map_.info.origin.position.y + global_half_size_ - local_map.info.origin.position.y) / resolution_);

  for (int y = 0; y < local_grid_size_; ++y) {
    for (int x = 0; x < local_grid_size_; ++x) {
      int dx = x - global_center_x;
      int dy = y - global_center_y;
      if (dx * dx + dy * dy <= clear_radius_cells * clear_radius_cells) {
          local_map.data[y * local_grid_size_ + x] = 0; // Clear cell
      }
    }
  }
}

void ESDF2dCostMapNode::merge_local_to_global(const nav_msgs::msg::OccupancyGrid& local_map) {
  for (int y = 0; y < local_grid_size_; ++y) {
    for (int x = 0; x < local_grid_size_; ++x) {
      int local_index = y * local_grid_size_ + x;
      int global_x = static_cast<int>((local_map.info.origin.position.x + x * resolution_ - global_map_.info.origin.position.x) / resolution_);
      int global_y = static_cast<int>((local_map.info.origin.position.y + y * resolution_ - global_map_.info.origin.position.y) / resolution_);
      int global_index = global_y * global_grid_size_ + global_x;

    if (global_x >= 0 && global_x < global_grid_size_ && global_y >= 0 && global_y < global_grid_size_) {
        if (local_map.data[local_index] != -1) { // If the local cell is observed
            global_map_.data[global_index] = local_map.data[local_index];
        }
      }
    }
  }
}

void ESDF2dCostMapNode::publish_maps(const nav_msgs::msg::OccupancyGrid& local_map) {
    nav_msgs::msg::OccupancyGrid local_map_copy = local_map;
    local_map_copy.header.stamp = this->now();
    local_map_copy.header.frame_id = "odom";
    local_map_pub_->publish(local_map_copy);

    global_map_.header.stamp = this->now();
    global_map_.header.frame_id = "odom";
    global_map_pub_->publish(global_map_);
}

void ESDF2dCostMapNode::publish_esdf_grid_meters(
    const std::vector<float>& esdf_grid,
    const std::vector<bool>& esdf_mask,
    const nav_msgs::msg::OccupancyGrid& local_map)
{
    nav_msgs::msg::OccupancyGrid esdf_msg = local_map;
    esdf_msg.header.stamp = this->now();
    esdf_msg.header.frame_id = "odom";
    for (size_t i = 0; i < esdf_grid.size(); ++i) {
        if (!esdf_mask[i] || std::isnan(esdf_grid[i])) {
            esdf_msg.data[i] = -1; // unknown
        } else {
            float d = esdf_grid[i];
            int val;
            if (d < 0.0f) {
                val = 100;
            } else if (d <= 5.0f) {
                val = 100;
            } else if (d <= 10.0f) {
                val = 50;
            } else if (d <= 15.0f) {
                val = 25;
            } else {
                val = 0;
            }
            esdf_msg.data[i] = val;
        }
    }
    esdf_grid_pub_->publish(esdf_msg);
}

void ESDF2dCostMapNode::esdf_callback(const sensor_msgs::msg::PointCloud2::SharedPtr esdf_msg)
{

    // Check for required fields before converting
    if (!validate_pointcloud2_fields(*esdf_msg)) {
        RCLCPP_WARN(this->get_logger(), "PointCloud2 does not have expected fields!");
        return;
    }

    auto transform = get_transform_to_odom();
    if (!transform) return;

    nav_msgs::msg::OccupancyGrid local_map = create_local_map(*transform);
    extract_local_from_global(local_map);
    erosion_filter(local_map);


    std::vector<float> esdf_grid;
    std::vector<bool> esdf_mask;
    convert_cloud_to_esdf_grid(
        *esdf_msg,
        esdf_grid,
        esdf_mask,
        local_grid_size_,
        local_grid_size_,
        local_map.info.origin.position.x,
        local_map.info.origin.position.y,
        resolution_
    );

    // Only fill ESDF holes if there are more than 1000 points in the ESDF message
    if (esdf_msg->width * esdf_msg->height > 1000) {
        fill_esdf_holes_wavefront(esdf_grid, esdf_mask, local_grid_size_, local_grid_size_, safety_distance_, resolution_);
    }
    if (esdf_msg->width * esdf_msg->height > 0) {
        overwrite_local_with_esdf(local_map, esdf_grid, esdf_mask);
    } else {
        RCLCPP_WARN(this->get_logger(), "No valid ESDF data available for local map update.");
    }
    
    clear_local_center(local_map);
    merge_local_to_global(local_map);

    publish_maps(local_map);
    publish_esdf_grid_meters(esdf_grid, esdf_mask, local_map);
}

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ESDF2dCostMapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
