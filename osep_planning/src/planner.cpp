/* 

Main paht planning algorithm

*/

#include "planner.hpp"

PathPlanner::PathPlanner(const PlannerConfig& cfg) : cfg_(cfg) {
    PD.gmap.reset(new pcl::PointCloud<pcl::PointXYZ>);
    // PD.gmap->points.reserve(50000);
    // octree_ = std::make_shared<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>>(cfg_.map_voxel_size);
    running = 1;
}

void PathPlanner::update_skeleton(const std::vector<Vertex>& verts) {
    /* Updates skeleton in place */

    // Rebuild if out-of-synch
    if (PD.gskel_vid2idx.size() != PD.gskel.size()) {
        PD.gskel_vid2idx.clear();
        for (int i=0; i<(int)PD.gskel.size(); ++i) {
            PD.gskel_vid2idx[PD.gskel[i].vid] = i;
        }
    }

    std::unordered_set<int> incoming_vids; 
    incoming_vids.reserve(verts.size()*2);

    for (const auto& vin : verts) {
        incoming_vids.insert(vin.vid);

        auto it = PD.gskel_vid2idx.find(vin.vid);
        if (it == PD.gskel_vid2idx.end()) {
            // New vertex
            Vertex vnew;
            vnew.vid = vin.vid;
            vnew.nb_ids = vin.nb_ids;
            vnew.position = vin.position;
            vnew.type = vin.type;
            
            PD.gskel_vid2idx[vnew.vid] = static_cast<int>(PD.gskel.size());
            PD.gskel.push_back(std::move(vnew));
            
            merge_viewpoints(PD.gskel.back(), vin); // insert vin's vpts in the new vertex
            continue;
        }
    
        // Existing (Preserves viewpoint specific changes)
        const int idx = it->second;
        Vertex& vcur = PD.gskel[idx]; // by ref (mutate)
        vcur.nb_ids = vin.nb_ids;
        vcur.position = vin.position;
        vcur.type = vin.type;
        merge_viewpoints(vcur, vin);
    }

    // If VID is missing from incoming set -> Prune it (and vpts) from the skeleton
    for (int i=(int)PD.gskel.size()-1; i>=0; --i) {
        int vid = PD.gskel[i].vid;
        if (incoming_vids.count(vid)) continue;

        int last = (int)PD.gskel.size()-1;
        if (i != last) {
            std::swap(PD.gskel[i], PD.gskel[last]);
            PD.gskel_vid2idx[PD.gskel[i].vid] = i;
        }
        PD.gskel.pop_back();
        PD.gskel_vid2idx.erase(vid);
    }

    PD.gskel_size = PD.gskel.size();
}

bool PathPlanner::planner_run() {
    RUN_STEP(generate_path);

    return running;
}

bool PathPlanner::generate_path() {

    // Flatten viewpoints (skips invalid / visited)
    rebuild_viewpoint_index();

    // Build edges
    build_topological_edges();
    build_geometric_edges();

    return 1;
}


/* HELPER */

void PathPlanner::rebuild_viewpoint_index() {
    PD.G.nodes.clear();
    PD.G.handle2gid.clear();

    int gid = 0;
    // PD.G.nodes.reserve()

    for (const auto& v : PD.gskel) {
        if (v.vid < 0) continue;
        for (int k=0; k<static_cast<int>(v.vpts.size()); ++k) {
            const auto& vp = v.vpts[k];
            // if (vp.invalid) continue;
            if (vp.visited) continue; // Maybe not exclude visited here?
            ViewpointNode n;
            n.gid = gid;
            n.h = {v.vid, k};
            n.pos = vp.position;
            n.yaw = vp.yaw;
            n.valid = true;
            PD.G.nodes.push_back(std::move(n));
            PD.G.handle2gid[hk(v.vid, k)] = gid;
            ++gid;
        }
    }

    // resize adjacencies to size of nodes
    const int N = static_cast<int>(PD.G.nodes.size());
    PD.G.adj.assign(N, {});
    PD.G.adjf.assign(N, {});
}

void PathPlanner::build_topological_edges() {
    const int N = static_cast<int>(PD.G.nodes.size());
    if (N == 0) return;

    // vid -> gids lookup
    std::unordered_map<int, std::vector<int>> vids_to_gids;
    vids_to_gids.reserve(PD.gskel.size()*2);
    for (const auto& n : PD.G.nodes) vids_to_gids[n.h.vid].push_back(n.gid);

    EdgeBuilder eb(PD.G);
    const float r2 = (cfg_.topo_radius > 0.0f) ? cfg_.topo_radius * cfg_.topo_radius : std::numeric_limits<float>::infinity();

    auto connect_side = [&](const std::vector<int>& A, const std::vector<int>& B, int kmax){
        for (int ga : A) {
            const auto& pa = PD.G.nodes[ga].pos;
            std::vector<std::pair<float,int>> cand; cand.reserve(B.size());
            for (int gb : B) {
                const auto& pb = PD.G.nodes[gb].pos;
                float d2 = (pa - pb).squaredNorm();
                if (d2 > r2) continue;
                if (!line_of_sight(pa, pb)) continue;
                cand.emplace_back(d2, gb);
            }
            std::sort(cand.begin(), cand.end(),
                      [](auto& x, auto& y){ return x.first < y.first; });
            int cnt = 0;
            for (auto& pr : cand) {
                eb.add(ga, pr.second, /*topo=*/true, /*geom=*/false);
                if (++cnt >= kmax) break;
            }
        }
    };

    // Iterate each skeleton edge (u, v) once; connect both directions
    for (const auto& u : PD.gskel) {
        auto itU = vids_to_gids.find(u.vid);
        if (itU == vids_to_gids.end()) continue;
        const auto& gids_u = itU->second;

        for (int vvid : u.nb_ids) {
            if (u.vid >= vvid) continue; // undirected: handle once
            auto itV = vids_to_gids.find(vvid);
            if (itV == vids_to_gids.end()) continue;
            const auto& gids_v = itV->second;

            // u -> v
            connect_side(gids_u, gids_v, cfg_.topo_kmax);
            // v -> u (symmetry to ensure every viewpoint on v can latch onto u)
            connect_side(gids_v, gids_u, cfg_.topo_kmax);

            // Optional: ensure at least one connection per side
            // (useful if LOS/radius prunes everything except one)
            if (!gids_u.empty() && !gids_v.empty()) {
                // Find absolute nearest pair (u,v) with LOS; add if not present
                float best = std::numeric_limits<float>::infinity();
                int bu=-1, bv=-1;
                for (int ga : gids_u) {
                    const auto& pa = PD.G.nodes[ga].pos;
                    for (int gb : gids_v) {
                        const auto& pb = PD.G.nodes[gb].pos;
                        float d2 = (pa - pb).squaredNorm();
                        if (d2 < best && d2 <= r2 && line_of_sight(pa, pb)) { best = d2; bu = ga; bv = gb; }
                    }
                }
                if (bu!=-1) eb.add(bu, bv, /*topo=*/true, /*geom=*/false);
            }
        }
    }
}

void PathPlanner::build_geometric_edges() {
    const int N = static_cast<int>(PD.G.nodes.size());
    if (N == 0) return;

    const float r2 = cfg_.geometric_radius * cfg_.geometric_radius;

    // Brute-force KNN - Swap to KDtree later
    for (int i=0; i<N; ++i) {
        std::vector<std::pair<int,int>> cand;
        const auto& ni = PD.G.nodes[i];
        for (int j=0; j<N; ++j) {
            if (j == i) continue;
            const auto& nj = PD.G.nodes[j];
            // skip if same vertex and identical k (same viewpoint)
            if (ni.h.vid == nj.h.vid && ni.h.k == nj.h.k) continue;
            float d2 = (ni.pos - nj.pos).squaredNorm();
            if (d2 > r2) continue;
            if (!line_of_sight(ni.pos, nj.pos)) continue;
            cand.emplace_back(d2, j);
        }
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b){ return a.first < b.first; });
        int cnt = 0;
        for (auto& pr : cand) {
            int j = pr.second;
            PD.G.adj[i].push_back(j);
            PD.G.adjf[i].push_back({ /*is_topological=*/false, /*is_geometric=*/true });
            PD.G.adj[j].push_back(i);
            PD.G.adjf[j].push_back({ /*is_topological=*/false, /*is_geometric=*/true });
            if (++cnt >= cfg_.geom_kmax) break;
        }
    }
}

bool PathPlanner::line_of_sight(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    if (!octree_ || !PD.gmap || PD.gmap->empty()) {
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

void PathPlanner::merge_viewpoints(Vertex& vcur, const Vertex& vin) {
    auto is_full_set_replacement = [&](const Vertex& was, const Vertex& now) -> bool {
        if (was.vpts.size() != now.vpts.size()) return true;
        if (was.type != now.type) return true;

        std::unordered_set<int> keys_was;
        keys_was.reserve(was.vpts.size() * 2);
        for (const auto& vp : was.vpts) keys_was.insert(vp.target_vp_pos);
        for (const auto& vp : now.vpts) {
            if (!keys_was.count(vp.target_vp_pos)) return true;
        }
        return false;
    };

    // If full viewpoint set replacement
    if (is_full_set_replacement(vcur, vin)) {
        // optional: check if in other data 
        vcur.vpts.clear();
        vcur.vpts = vin.vpts; // copy aligned
        // optional: restore other data with new info
        return;
    }    

    // If only moved:
    std::unordered_map<int,int> key2idx;
    key2idx.reserve(vcur.vpts.size());
    for (int i=0; i<(int)vcur.vpts.size(); ++i) {
        key2idx[vcur.vpts[i].target_vp_pos] = i;
    }

    std::unordered_set<int> incoming; incoming.reserve(vin.vpts.size()*2);

    // Add or update present ones
    for (const auto& vp_in : vin.vpts) {
        incoming.insert(vp_in.target_vp_pos);
        auto it = key2idx.find(vp_in.target_vp_pos);

        if (it == key2idx.end()) {
            // new vp appeared -- should not happen?
            vcur.vpts.push_back(vp_in);
            continue;
        }

        // Existing: only apply if changed
        auto& vp_cur = vcur.vpts[it->second];

        // Your manager sets vp.updated when it moves/gets adjusted in filtering stage.
        // We still verify geometry to be safe and to suppress tiny jitter updates.
        bool moved = !approx_eq3(vp_cur.position, vp_in.position);
        bool yawed = !approx_eq(vp_cur.yaw, vp_in.yaw);
        bool scor  = !approx_eq(vp_cur.score, vp_in.score);

        if (vp_in.updated || moved || yawed || scor) {
            if (moved) vp_cur.position = vp_in.position;
            if (yawed) { vp_cur.yaw = vp_in.yaw; vp_cur.orientation = vp_in.orientation; }
            if (scor)  vp_cur.score = vp_in.score;
        }
    }

    // Remove any old vp that vanished from the manager’s list (manager deleted it)
    for (int i=(int)vcur.vpts.size()-1; i>=0; --i) {
        int key = vcur.vpts[i].target_vp_pos;
        if (incoming.count(key)) continue;
        // if (is_on_path(vcur.vid, key)) mark_path_dirty("vp deleted");
        // graph_remove_viewpoint(vcur.vid, key);
        int last = (int)vcur.vpts.size()-1;
        if (i != last) std::swap(vcur.vpts[i], vcur.vpts[last]);
        vcur.vpts.pop_back();
    }  
}




/*
IDEAS:
- Start: Drone position 

- Some path cost proportional to the steps needed along the skeleton adjacency
- Some path reward

*/








// void PathPlanner::build_topological_edges() {
//     const int N = static_cast<int>(PD.G.nodes.size());
//     if (N == 0) return;

//     auto add_undir_topo = [&](int a, int b) {
//         PD.G.adj[a].push_back(b);
//         PD.G.adjf[a].push_back({true, false});
//         PD.G.adj[b].push_back(a);
//         PD.G.adjf[b].push_back({true, false});
//     };

//     std::unordered_map<int, std::vector<int>> vids_to_gids;
//     vids_to_gids.reserve(PD.gskel.size() * 2);
//     for (const auto& n : PD.G.nodes) {
//         vids_to_gids[n.h.vid].push_back(n.gid);
//     }

//     // for each skeleton edge (u <-> v), connect K nearest pair
//     for (const auto& u : PD.gskel) {
//         const auto& Nu = u.nb_ids; // nbs to vertex u
//         auto itU = vids_to_gids.find(u.vid); // find graph index to 
//         if (itU == vids_to_gids.end()) continue;
//         const auto& gids_u = itU->second;

//         for (int vvid : Nu) {
//             if (u.vid >= vvid) continue; // only wire edge once (undirected)

//             auto itV = vids_to_gids.find(vvid);
//             if (itV == vids_to_gids.end()) continue;
//             const auto& gids_v = itV->second;

//             // for each node on u, pick the kmax nearest nodes on v within radius
//             for (int gu : gids_u) {
//                 std::vector<std::pair<int,int>> cand;
//                 cand.reserve(gids_v.size());
//                 const auto& nu = PD.G.nodes[gu];
//                 for (int gv : gids_v) {
//                     const auto& nv = PD.G.nodes[gv];
//                     float d2 = (nu.pos - nv.pos).squaredNorm();
//                     // if (cfg_.topo_radius > 0.0f && d2 > r2) continue;

//                     if (!line_of_sight(nu.pos, nv.pos)) continue; 
//                     cand.emplace_back(d2, gv);
//                 }
//                 std::sort(cand.begin(), cand.end(), [](auto& a, auto&b){ return a.first < b.first; });
//                 int cnt = 0;
//                 for (auto& pr : cand) {
//                     int j = pr.second;
//                     add_undir_topo(gu, j);
                    
//                     if (++cnt >= cfg_.topo_kmax) break;
//                 }
//             }
//         }
//     }
// }