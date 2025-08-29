#ifndef PLANNER_HPP_
#define PLANNER_HPP_

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


struct PlannerConfig {
    float map_voxel_size;
    float geometric_radius = 10.0f;
    int geom_kmax = 6;
    float topo_radius = 100.0f;
    int topo_kmax = 100;
};

struct PlannerData {
    size_t gskel_size;
    std::vector<Vertex> gskel;
    std::unordered_map<int,int> gskel_vid2idx;

    Viewpoint start;
    Viewpoint end;

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr gmap; // owned by node - should not be mutated

    Graph G;
};


class PathPlanner {
public:
    PathPlanner(const PlannerConfig& cfg);
    bool planner_run();
    void update_skeleton(const std::vector<Vertex>& verts);

    void set_map(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> oct = nullptr) {
       PD.gmap = std::move(cloud);
       octree_ = std::move(oct); 
    }

    // pcl::PointCloud<pcl::PointXYZ>& input_map() { return *PD.gmap; }
    std::vector<Vertex>& output_skeleton() { return PD.gskel; }
    const Graph& output_graph() { return PD.G; }

private:
    /* Functions */
    

    bool generate_path();

    /* Graph construction steps */
    void rebuild_viewpoint_index();
    void build_topological_edges();
    void build_geometric_edges();

    /* Helper */
    void merge_viewpoints(Vertex& vcur, const Vertex& vin);

    static inline bool approx_eq(float a, float b, float eps=1e-6f){ return std::fabs(a-b) <= eps; }
    static inline bool approx_eq3(const Eigen::Vector3f& a,const Eigen::Vector3f& b,float eps2=1e-8f){ return (a - b).squaredNorm() <= eps2; }

    bool line_of_sight(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

    /* Params & Data */
    PlannerConfig cfg_;
    bool running;
    PlannerData PD;

    std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> octree_;

};


#endif