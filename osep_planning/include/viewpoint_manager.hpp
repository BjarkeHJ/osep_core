#ifndef VIEWPOINT_MANAGER_HPP_
#define VIEWPOINT_MANAGER_HPP_

#include "types.hpp"

#include <iostream>
#include <chrono>
#include <unordered_set>
#include <pcl/octree/octree.h>

#define RUN_STEP(fn) \
    do { \
        bool ok = (fn)(); \
        running = ok; \
        if (!ok) return false; \
    } while (0)

struct ViewpointConfig {
    float map_voxel_size;
    float vpt_safe_dist;
};

class ViewpointManager {
public:
    ViewpointManager(const ViewpointConfig &cfg);
    
    void set_map(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> oct = nullptr) {
        gmap = std::move(cloud);
        octree_ = std::move(oct); 
    }

    bool update_viewpoints(std::vector<Vertex>& gskel);
    void set_vid2idx(std::unordered_map<int,int>& vid2idx) { gskel_vid2idx = vid2idx; } // Copy current unique vid mapping

private:
    /* Functions */
    bool sample_viewpoints(std::vector<Vertex>& gskel);
    bool filter_viewpoints(std::vector<Vertex>& gskel);
    
    bool viewpoint_scoring(); // ??

    /* Helpers */
    std::vector<Viewpoint> generate_viewpoints(std::vector<Vertex>& gskel, Vertex& v);
    void build_local_frame(std::vector<Vertex>& gskel, Vertex& v, Eigen::Vector3f& that, Eigen::Vector3f& n1hat, Eigen::Vector3f& n2hat);
    float distance_to_free_space(const Eigen::Vector3f& p_in, const Eigen::Vector3f dir_in);

    /* Inlines */
    inline float deg2rad(float d) { return d * float(M_PI) / 180.0f; }
    inline float wrapPi(float a) {
        float twoPi = 2.0f * float(M_PI);
        float x = std::fmod(a + float(M_PI), twoPi);
        if (x < 0) x += twoPi;
        return x-float(M_PI);
    }
    inline float yaw_to_face(const Eigen::Vector3f& c, const Eigen::Vector3f& p) {
        // yaw to face target p from camera c
        Eigen::Vector2f d = (p.head<2>() - c.head<2>());
        return std::atan2(d.y(), d.x());
    }
    inline Eigen::Quaternionf yaw_to_quat(float yaw) { return Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ())); }
    inline bool is_not_safe_dist(const Eigen::Vector3f& p) {
        std::vector<int> ids; std::vector<float> d2s; pcl::PointXYZ qp(p.x(), p.y(), p.z());
        return octree_->radiusSearch(qp, cfg_.vpt_safe_dist, ids, d2s) > 0;
    }

    /* DATA */
    bool running; 
    ViewpointConfig cfg_;
    std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> octree_;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr gmap;
    std::unordered_map<int,int> gskel_vid2idx;
    
    // std::vector<int> walk_branch(int start_idx, int nb_idx, const std::vector<char>& allowed, std::unordered_set<std::pair<int,int>, PairHash>& visited_edges);
};

#endif