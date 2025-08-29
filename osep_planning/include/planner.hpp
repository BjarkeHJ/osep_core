#ifndef PLANNER_HPP_
#define PLANNER_HPP_

#include "types.hpp"
#include <iostream>
#include <chrono>
#include <unordered_set>
#include <pcl/octree/octree.h>
#include <pcl/kdtree/kdtree_flann.h>

#define RUN_STEP(fn) \
    do { \
        bool ok = (fn)(); \
        running = ok; \
        if (!ok) return false; \
    } while (0)


struct PlannerConfig {
    float map_voxel_size;
    float geometric_radius = 10.0f;
    int geom_kmax = 6;
    float topo_radius = 100.0f;
    int topo_kmax = 100;
};

struct PlannerData {
    Graph graph;

    Viewpoint start;
    Viewpoint end;
};


class PathPlanner {
public:
    PathPlanner(const PlannerConfig& cfg);

    void update_skeleton(const std::vector<Vertex>& verts);
    void set_map(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> oct = nullptr) {
       gmap = std::move(cloud);
       octree_ = std::move(oct); 
    }
    void set_vid2idx(std::unordered_map<int,int>& vid2idx) { gskel_vid2idx = vid2idx; } // Copy current unique vid mapping

    bool plan_path(std::vector<Vertex>& gskel);

private:
    std::unordered_map<int,int> gskel_vid2idx;

    /* Functions */
    bool build_graph(std::vector<Vertex>& gskel);
    bool generate_path(std::vector<Vertex>& gskel);

    /* Helper */
    bool line_of_sight(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

    /* Params & Data */
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr gmap;
    std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> octree_;

    pcl::KdTreeFLANN<pcl::PointXYZ> vpt_kdtree; // remove again?
    pcl::PointCloud<pcl::PointXYZ>::Ptr vpt_cloud; // remove again?

    PlannerConfig cfg_;
    bool running;

    PlannerData PD;

};


#endif