#ifndef GSKEL_HPP_
#define GSKEL_HPP_

#include <lkf_vertex_fuse.hpp>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <pcl/common/common.h>
#include <pcl/common/point_tests.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Core>

#define RUN_STEP(fn) \
    do { \
        bool ok = (fn)(); \
        running = ok; \
        if (!ok) return false; \
    } while (0)

   /* std::cout << (ok ? "[SUCCESS] " : "[FAILED] ") << #fn << std::endl; \ */

struct GSkelConfig {
    float gnd_th;
    float fuse_dist_th;
    float fuse_conf_th;
    float lkf_pn;
    float lkf_mn;
    int max_obs_wo_conf;
    int niter_smooth_vertex;
    float vertex_smooth_coef;
    int min_branch_length;
};

struct Edge {
    int u, v; // vertex idxs of the edge
    float w; // weight of the edge (length)
    bool operator<(const Edge &other) const {
        return w < other.w;
    }
};

struct UnionFind {
    std::vector<int> parent; // parent[i] is parent of node i

    UnionFind(int n) : parent(n) {
        // Constructor: Initially, every node is its own parent
        for (int i=0; i<n; ++i) parent[i] = i;
    }

    int find(int x) {
        // Find the root of the component that x belongs to
        // Recursively follow the "chain" until x is its own parent...
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    bool unite(int x, int y) {
        // Merge the sets that x and y belong to
        int rx = find(x); 
        int ry = find(y);
        if (rx == ry) return false;
        parent[ry] = rx; // Union: Make on root the parent of the other (merge the chains into one)
        return true;
    }
};


struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int vid = -1;
    std::vector<int> nb_ids;
    pcl::PointXYZ position;
    VertexLKF kf;

    int obs_count = 0;
    int unconf_check = 0;
    int type = 0;
    int smooth_iters;
    bool just_approved = false;
    bool frozen = false;
    bool conf_check = false;
    bool marked_for_deletion = false;
    bool pos_update = false;
    bool type_update = false;
};

struct GSkelData {
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_cands;
    
    std::vector<Vertex> prelim_vers;
    std::vector<Vertex> global_vers;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_vers_cloud;
    std::vector<int> new_vers_indxs;
    
    std::vector<int> joints;
    std::vector<int> leafs;
    std::vector<std::vector<int>> branches;
    std::vector<std::vector<int>> global_adj;
    std::vector<Edge> edges;

    int next_vid = 0;
    size_t gskel_size;
};

class GSkel {
public:
    /* Public methods */
    explicit GSkel(const GSkelConfig& cfg);
    bool gskel_run();
    pcl::PointCloud<pcl::PointXYZ>& input_vertices() { return *GD.new_cands; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud() { return GD.global_vers_cloud; }
    std::vector<Vertex>& output_vertices() { return GD.global_vers; }

private:
    /* Functions */
    bool increment_skeleton();
    bool graph_adj();
    bool mst();
    bool vertex_merge();
    bool prune();
    bool smooth_vertex_positions();
    bool vid_manager();    
    
    /* Helper */
    void build_cloud_from_vertices();
    void graph_decomp();
    void merge_into(int keep, int del);
    bool size_assert();

    GSkelConfig cfg_;
    bool running;

    /* Data */
    GSkelData GD;
    const Eigen::Matrix3f Q = Eigen::Matrix3f::Identity() * cfg_.lkf_pn; // lkf process noise cov
    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity() * cfg_.lkf_mn; // lkf meaurement noise cov

    /* Utils */
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree_{new pcl::search::KdTree<pcl::PointXYZ>};


};




#endif // GSKEL_HPP_