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
    
    float budget = 60.0f;          // max travel cost for this horizon (meters or edge cost units)
    float lambda = 1.0f;           // travel-vs-reward trade-off
    float subgraph_radius = 40.0f; // BFS radius from current node (cost sum)
    int   subgraph_max_nodes = 200;
    float hysteresis = 0.15f;      // replan only if new_score > old*(1+hysteresis)
    int   warm_steps = 1;          // how many first steps to commit before re-planning
    float geometric_bias = 0.0f;   // optional: add cost penalty for geometric edges
    float topo_bonus = 0.0f;       // optional: subtract cost on topological edges
};

struct DronePose {
    Eigen::Vector3f pos;
    Eigen::Quaternionf ori;
};

struct PlannerData {
    Graph graph;
    DronePose drone_pose;

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
    void set_drone_pose(const Eigen::Vector3f& pos, const Eigen::Quaternionf ori) {
        if (!pos.allFinite()) return;
        if (!ori.coeffs().allFinite() || ori.norm() < 1e-6f) return;
        PD.drone_pose.pos = pos;
        PD.drone_pose.ori = ori;
        plan = true; // recieved valid drone pose
    }

    bool plan_path(std::vector<Vertex>& gskel);
    const Graph& get_graph() { return PD.graph; }

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
    PlannerData PD;
    bool running;
    bool plan = false;

};


#endif