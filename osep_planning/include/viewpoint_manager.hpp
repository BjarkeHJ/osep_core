#ifndef VIEWPOINT_MANAGER_HPP_
#define VIEWPOINT_MANAGER_HPP_

#include "types.hpp"
#include "raycast.hpp"

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

    float cam_hfov_rad;
    float cam_vfov_rad;
    float cam_Nx;
    float cam_Ny;
    float cam_max_range;
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
    void commit_coverage(const Viewpoint& vp);

private:
    /* Functions */
    bool sample_viewpoints(std::vector<Vertex>& gskel);
    bool filter_viewpoints(std::vector<Vertex>& gskel);
    bool score_viewpoints(std::vector<Vertex>& gskel); // ??

    /* Helpers */
    std::vector<Viewpoint> generate_viewpoints(std::vector<Vertex>& gskel, Vertex& v);
    void build_local_frame(std::vector<Vertex>& gskel, Vertex& v, Eigen::Vector3f& that, Eigen::Vector3f& n1hat, Eigen::Vector3f& n2hat);
    float distance_to_free_space(const Eigen::Vector3f& p_in, const Eigen::Vector3f dir_in);
    
    GainStats estimate_viewpoint_coverage(const Viewpoint& vp);
    void build_rayset(float hfov_rad, float vfov_rad, int Nx, int Ny, float maxR);
    bool is_occ_voxel(const Eigen::Vector3f& p);


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

    /* For giving viewpoints unique handle ids for downs stream synchronization - packs vertex ID and unique viewpoint id into key*/
    static inline uint64_t make_vpt_handle(int vid, uint32_t seq) {
        return ( (uint64_t(uint32_t(vid)) << 32) | uint64_t(seq));
    }

    /* For map key hashing */
    inline uint64_t grid_key(const Eigen::Vector3f& p, float S) {
        int ix = static_cast<int>(std::floor(p.x() / S));
        int iy = static_cast<int>(std::floor(p.y() / S));
        int iz = static_cast<int>(std::floor(p.z() / S));
        // Simple interleaving (keep it fast)
        return ( (uint64_t(uint32_t(ix)) & 0x1FFFFF) << 42 )
             | ( (uint64_t(uint32_t(iy)) & 0x1FFFFF) << 21 )
             | (  uint64_t(uint32_t(iz)) & 0x1FFFFF );
    }
    
    
    std::unordered_map<int, uint32_t> per_vertex_seg; // vid -> seq counter

    /* DATA */
    bool running; 
    ViewpointConfig cfg_;
    std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> octree_;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr gmap;
    std::unordered_map<int,int> gskel_vid2idx;

    CoverageState cov_; // Global coverage state
    RaySet rayset_; // pixel grid for ray casting
};

#endif