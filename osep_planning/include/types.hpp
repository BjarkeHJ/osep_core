#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <unordered_set>
#include <Eigen/Core>
#include <pcl/common/common.h>


struct Viewpoint {
    Eigen::Vector3f position;
    float yaw;
    Eigen::Quaternionf orientation;
    
    int target_vid = -1; // corresponding vertex id
    int target_vp_pos = -1; // index of corresponding vertex vpts vector
    
    float score = 0.0f;
    
    bool updated = false;
    bool invalid = false;
    
    bool in_path = false;
    bool visited = false;
};

struct Vertex {
    int vid = -1;
    std::vector<int> nb_ids;
    pcl::PointXYZ position;
    int type = 0;
    bool pos_update = false;
    bool type_update = false;

    std::vector<Viewpoint> vpts; // Vertex viewpoints
};

struct ViewpointHandle {
    int vid = -1; // skeleton vertex id
    int k = -1; // index inside the vertex' vpts vector
    bool operator==(const ViewpointHandle& o) const noexcept { return vid==o.vid && k==o.k; }
};

struct ViewpointNode {
    int gid = -1; // global viewpoint id in flat array
    ViewpointHandle h; // (vid, k)
    Eigen::Vector3f pos;
    float yaw;
    bool valid = true;
};

struct EdgeFlag {
    bool is_topological; // Connected viewpoints vertices are adjacent OR viewpoints correspond to the same viewpoint?
    bool is_geometric; // Connected viewpoints are close? (LOS check?)
};

struct Graph {
    std::vector<ViewpointNode> nodes;
    std::vector<std::vector<int>> adj;
    std::vector<std::vector<EdgeFlag>> adjf;

    std::unordered_map<long long, int> handle2gid; // key(vid, k) -> gid 
};

// Hash key builder: Takes two ints (32 bit) and stores them as a single key long int (64 bit)
// Unsigned important (signed k would corrupt the msb)
static inline long long hk(int vid, int k) {
    return ( (static_cast<long long>(vid) << 32) ^ static_cast<unsigned long long>(k) );
}

// Undirected edge key
static inline long long ek(int a, int b) {
    if (a > b) std::swap(a, b);
    return ( (static_cast<long long>(a) << 32) ^ static_cast<unsigned long long>(b) );
}

// Push edge (undirected) only once; attach flags
struct EdgeBuilder {
    Graph& G;
    std::unordered_set<long long> seen; // clear per rebuild

    EdgeBuilder(Graph& g) : G(g) { seen.reserve(1 << 16); }

    void add(int i, int j, bool is_topo, bool is_geom) {
        if (i == j) return;
        long long key = ek(i, j);
        if (!seen.insert(key).second) {
            // already added; optionally OR-in flags if you keep a separate edge store
            return;
        }
        G.adj[i].push_back(j);
        G.adjf[i].push_back({is_topo, is_geom});
        G.adj[j].push_back(i);
        G.adjf[j].push_back({is_topo, is_geom});
    }
};



#endif