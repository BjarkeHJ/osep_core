/* 

Main paht planning algorithm

*/

#include "planner.hpp"

PathPlanner::PathPlanner(const PlannerConfig& cfg) : cfg_(cfg) {
    gmap.reset(new pcl::PointCloud<pcl::PointXYZ>);
    vpt_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    running = 1;
}


bool PathPlanner::plan_path(std::vector<Vertex>& gskel) {
    /* Main public function - Runs the path planning pipeline */

    build_graph(gskel);
    generate_path(gskel);
    return 1;
}

bool PathPlanner::build_graph(std::vector<Vertex>& gskel) {
    PD.graph.nodes.clear();
    PD.graph.adj.clear();

    Graph G;
    std::unordered_map<uint64_t, int> handle2gid;
    handle2gid.reserve(2048);

    std::unordered_map<int, std::vector<int>> vids_to_gid;
    vids_to_gid.reserve(gskel.size() * 2);
    
    // Create nodes
    int gid = 0;
    for (const Vertex& v : gskel) {
        for (int k=0; k<static_cast<int>(v.vpts.size()); ++k) {
            const Viewpoint& vp = v.vpts[k];
            // if (vp.invalid) continue;

            GraphNode n;
            n.gid = gid;
            n.vid = v.vid;
            n.k = k;
            n.p = vp.position;
            n.yaw = vp.yaw;
            n.score = vp.score;

            G.nodes.push_back(n);
            vids_to_gid[v.vid].push_back(gid);
            handle2gid[ hk(v.vid, k) ] = gid;

            ++gid;
        }
    }

    G.adj.resize(G.nodes.size());
    if (G.nodes.empty()) {
        PD.graph = G;
        return 0;
    }
    
    //  std::unordered_set<uint64_t> edge_set; edge_set.reserve(G.nodes.size()*4);

    // // ----- 2A) Intra-vertex links (each VP to its nearest sibling) -----
    // if (connect_intra) {
    //     for (const auto& kv : vids_to_gids) {
    //         const auto& glist = kv.second;        // gids for this vid
    //         const int m = (int)glist.size();
    //         if (m <= 1) continue;

    //         // For each node, connect to nearest same-vid node (simple sparse linking)
    //         for (int ii = 0; ii < m; ++ii) {
    //             int gi = glist[ii];
    //             const auto& pi = G.nodes[gi].p;

    //             float best_d2 = std::numeric_limits<float>::infinity();
    //             int best_gj = -1;

    //             for (int jj = 0; jj < m; ++jj) {
    //                 if (ii == jj) continue;
    //                 int gj = glist[jj];
    //                 float dd2 = d2(pi, G.nodes[gj].p);
    //                 if (dd2 < best_d2) {
    //                     best_d2 = dd2;
    //                     best_gj = gj;
    //                 }
    //             }

    //             if (best_gj >= 0) {
    //                 const auto& pj = G.nodes[best_gj].p;
    //                 if (!line_of_sight || line_of_sight(pi, pj)) {
    //                     add_edge_ud(G, gi, best_gj, std::sqrt(best_d2), edge_set);
    //                 }
    //             }
    //         }

    //         // (Optional) If you prefer a strictly connected set with m-1 edges,
    //         // you could build a tiny MST per vid instead of "each to nearest".
    //     }
    // }

    // // ----- 2B) Inter-vertex links (across skeleton adjacency) -----
    // if (connect_inter) {
    //     for (const auto& u : gskel) {
    //         // gids that belong to vertex u
    //         auto itU = vids_to_gids.find(u.vid);
    //         if (itU == vids_to_gids.end() || itU->second.empty()) continue;

    //         for (int vvid : u.nb_ids) {
    //             // guard: only handle each undirected skel edge once (u.vid < vvid)
    //             if (u.vid >= vvid) continue;

    //             auto itV = vids_to_gids.find(vvid);
    //             if (itV == vids_to_gids.end() || itV->second.empty()) continue;

    //             const auto& gids_u = itU->second;
    //             const auto& gids_v = itV->second;

    //             // For each u-viewpoint, connect to K nearest in v (small sets -> brute force OK)
    //             for (int gi : gids_u) {
    //                 struct Cand { int gj; float d2; };
    //                 std::vector<Cand> cands; cands.reserve((size_t)std::min(inter_K, (int)gids_v.size()));

    //                 // collect nearest K
    //                 for (int gj : gids_v) {
    //                     float dd2 = d2(G.nodes[gi].p, G.nodes[gj].p);
    //                     // keep a small unsorted list up to K
    //                     if ((int)cands.size() < inter_K) {
    //                         cands.push_back({gj, dd2});
    //                     } else {
    //                         // replace worst if better
    //                         int worst = 0; float worst_d2 = cands[0].d2;
    //                         for (int t=1; t<(int)cands.size(); ++t) {
    //                             if (cands[t].d2 > worst_d2) { worst = t; worst_d2 = cands[t].d2; }
    //                         }
    //                         if (dd2 < worst_d2) cands[worst] = {gj, dd2};
    //                     }
    //                 }

    //                 // add edges (LOS-gated)
    //                 for (const auto& c : cands) {
    //                     if (!line_of_sight || line_of_sight(G.nodes[gi].p, G.nodes[c.gj].p)) {
    //                         add_edge_ud(G, gi, c.gj, std::sqrt(c.d2), edge_set);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }



    return 1;
}



bool PathPlanner::generate_path(std::vector<Vertex>& gskel) {

    return 1;
}


/* HELPERS */

bool PathPlanner::line_of_sight(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    if (!octree_) {
        return 0; // nothing to do...
    }

    Eigen::Vector3f d = b - a;
    float L = d.norm();
    if (L <= 1e-6f) return 1;
    Eigen::Vector3f u = d / L;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::AlignedPointTVector centers;
    octree_->getIntersectedVoxelCenters(a, u, centers);

    const float tol = 0.6f * cfg_.map_voxel_size;
    const float tol2 = tol * tol;
    for (const auto& c : centers) {
        Eigen::Vector3f cv = c.getVector3fMap();
        if ((cv - a).squaredNorm() <= tol2) continue;
        if ((cv - b).squaredNorm() <= tol2) continue;
        return 0;
    }
    return 1;
}



/*
IDEAS:
- Start: Drone position 

- Some path cost proportional to the steps needed along the skeleton adjacency
- Some path reward

*/








