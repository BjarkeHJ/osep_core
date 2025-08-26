/* 

Main algorithm for the OSEP viewpoint manager

TODO: Incorporate 2d_local_costmap into the viewpoint sampling process.

*/

#include "viewpoint_manager.hpp"

ViewpointManager::ViewpointManager(const ViewpointConfig& cfg) : cfg_(cfg) {
    VD.global_skel.reserve(1000);
    VD.global_vpts.reserve(5000);
    VD.updated_vertices.reserve(100);
    running = 1;
}


bool ViewpointManager::viewpoint_run() {
    VD.gskel_size = VD.global_skel.size();
    RUN_STEP(fetch_updated_vertices);
    RUN_STEP(branch_extract);
    RUN_STEP(viewpoint_sampling);

    return running;
}

bool ViewpointManager::fetch_updated_vertices() {
    if (VD.gskel_size == 0) return 0;
    VD.updated_vertices.clear();
    for (const auto& v : VD.global_skel) {
        if (v.pos_update || v.type_update) {
            VD.updated_vertices.push_back(v.vid);
        }
    }

    // Rebuild graph cache
    const int N = VD.gskel_size;
    vid2idx_.clear();
    vid2idx_.reserve(N*2);
    idx2vid_.clear();
    idx2vid_.assign(N, -1);
    
    for (int i=0; i<N; ++i) {
        vid2idx_[VD.global_skel[i].vid] = i;
        idx2vid_[i] = VD.global_skel[i].vid;
    }
    
    VD.global_adj.assign(N, {});
    degree_.assign(N,0);
    is_endpoint_.assign(N,0);

    for (int i=0; i<N; ++i) {
        const auto& nbs = VD.global_skel[i].nb_ids;
        auto& out = VD.global_adj[i];
        out.reserve(nbs.size());
        for (int nb_vid : nbs) {
            auto it = vid2idx_.find(nb_vid);
            if (it != vid2idx_.end()) {
                out.push_back(it->second);
            }
        }
        degree_[i] = static_cast<int>(out.size());
    }
    for (int i=0; i<N; ++i) {
        int d = degree_[i];
        is_endpoint_[i] = (d == 1 || d > 2) ? 1 : 0;
    }
    return 1;
}

bool ViewpointManager::branch_extract() {
    const int N = VD.gskel_size;
    if (N == 0) return 0;
    const int branch_min_vers_ = 5;

    std::vector<int> updated_idxs;
    updated_idxs.reserve(VD.updated_vertices.size());
    for (int vid : VD.updated_vertices) {
        auto it = vid2idx_.find(vid);
        if (it != vid2idx_.end()) {
            updated_idxs.push_back(it->second);
        }
    }

    // const bool full_rebuild = VD.branches.empty() || updated_idxs.empty();
    const bool full_rebuild = VD.branches.empty();
    // const bool full_rebuild = true;

    if (full_rebuild) {
        VD.branches.clear();
        std::vector<char> allowed(N,1);
        std::unordered_set<std::pair<int,int>, PairHash> visited;
        visited.reserve(std::max(64, 2*N));

        for (int i=0; i<N; ++i) {
            if (!is_endpoint_[i]) continue; // start at endpoint
            for (int nb_i : VD.global_adj[i]) {
                auto br = walk_branch(i, nb_i, allowed, visited);
                if (!br.empty()) {
                    const bool enough_vers = (static_cast<int>(br.size()) - 1) >= branch_min_vers_;
                    if (enough_vers) VD.branches.emplace_back(std::move(br));
                }
            }
        }
        return 1;
    }

    // Incremental build
    std::vector<char> in_region(N,0);
    std::vector<int> stack;
    stack.reserve(256);

    auto push_idx = [&](int i) {
        if (i < 0 || i >= N) return;
        if (!in_region[i]) {
            in_region[i] = 1; // mark index as in region with updated vertex
            stack.push_back(i);
        }
    };

    // Seed the region with updated vertices and their neighbors
    for (int ui : updated_idxs) {
        push_idx(ui);
        for (int nb_i : VD.global_adj[ui]) {
            push_idx(nb_i);
        }
    }

    // DFS flood-fill the region and stopping at endpoints
    while (!stack.empty()) {
        int i = stack.back();
        stack.pop_back();
        if (is_endpoint_[i]) continue; // stop expanding the region at an endpoint (reached depth in search - trace back)
        for (int nb_i : VD.global_adj[i]) {
            push_idx(nb_i);
        }
    }

    bool any_in = false;
    for (char c : in_region) {
        if (c) {
            any_in = true;
            break;
        }
    }

    if (!any_in) return 1; // nothing to do...

    auto edge_key= [](int u, int v) {
        if (u > v) std::swap(u,v);
        return std::make_pair(u,v);
    };
    
    // Build the region edges
    std::unordered_set<std::pair<int,int>, PairHash> region_edges;
    region_edges.reserve(8 * (int)updated_idxs.size() + 64);
    for (int i=0; i<N; ++i) {
        if (!in_region[i]) continue; // focus on the region (skip non-update areas)
        for (int j : VD.global_adj[i]) {
            if (!in_region[j]) continue;
            if (j <= i) continue; // avoid duplicate edges (undirected...)
            region_edges.insert(edge_key(i,j));
        }
    }

    auto branch_hits_region = [&](const std::vector<int>& br_vids) -> bool {
        if (br_vids.size() < 2) return false;
        for (size_t k=1; k<br_vids.size(); ++k) {
            auto ita = vid2idx_.find(br_vids[k-1]);
            auto itb = vid2idx_.find(br_vids[k]);
            if (ita == vid2idx_.end() || itb == vid2idx_.end()) continue;
            if (region_edges.count(edge_key(ita->second, itb->second))) {
                return true;
            }
        }
        return false;
    };

    // Remove any previously stored branches that crosses region edges
    std::vector<std::vector<int>> kept;
    kept.reserve(VD.branches.size());
    for (auto& br : VD.branches) {
        if (!branch_hits_region(br)) {
            kept.emplace_back(std::move(br));
        }
    }
    VD.branches.swap(kept);

    // Reseed new branch walks from endpoints inside the update region
    std::unordered_set<std::pair<int,int>, PairHash> visited_local;
    visited_local.reserve(region_edges.size());
    for (int i=0; i<N; ++i) {
        if (!in_region[i]) continue;
        if (!is_endpoint_[i]) continue;
        for (int nb_i : VD.global_adj[i]) {
            // if (!in_region[nb_i]) continue;
            auto br = walk_branch(i, nb_i, in_region, visited_local);
            if (!br.empty()) {
                const bool enough_vers = (static_cast<int>(br.size()) - 1) >= branch_min_vers_;
                if (enough_vers) VD.branches.emplace_back(std::move(br));
            }
        }
    }
    return 1;
}

bool ViewpointManager::viewpoint_sampling() {

    for (size_t i=0; i<VD.gskel_size; ++i) {
        auto& vertex = VD.global_skel[i];
        if (vertex.type_update || vertex.pos_update) vertex.vpts.clear();
        if (!vertex.vpts.empty()) continue; // already has viewpoints and is not cleared

        std::vector<Viewpoint> new_vpts = generate_viewpoint(vertex.vid);
        
        
    }

    return 1;
}

/* Helpers */

std::vector<Viewpoint> ViewpointManager::generate_viewpoint(int id) {
    std::vector<Viewpoint> vpts_out;
    auto& v = VD.global_skel[id];
    const auto& type = v.type;

    if (type == 1) {
        // leaf
        int id_adj = VD.global_adj[v.vid][0];
        auto& v_nb = VD.global_skel[id_adj];
        const Eigen::Vector3f p1 = v.position.getVector3fMap();
        const Eigen::Vector3f p2 = v_nb.position.getVector3fMap();
        Eigen::Vector3f dir = p2 - p1;
        if (dir.norm() < 1e-2f) return vpts_out;
        dir.normalize();
        return vpts_out;
    }
    else if (type == 2) {
        // branch 

        return vpts_out;
    }
    else if (type == 3) {
        // joint

        return vpts_out;
    }
    else {
        vpts_out.clear();
        return vpts_out;
    }
}

std::vector<int> ViewpointManager::walk_branch(int start_idx, int nb_idx, const std::vector<char>& allowed, std::unordered_set<std::pair<int,int>, PairHash>& visited_edges) {
    auto edge_key = [](int u, int v) {
        if (u > v) std::swap(u,v);
        return std::make_pair(u,v);
    };

    auto edge_seen = [&](int a, int b) {
        return visited_edges.count(edge_key(a,b)) != 0;
    };
    
    auto mark_edge = [&](int a, int b) {
        visited_edges.insert(edge_key(a,b));
    };

    std::vector<int> out_vids;
    out_vids.reserve(32);

    if (start_idx < 0 || nb_idx < 0) return out_vids; // invalid
    if (!allowed[start_idx] || !allowed[nb_idx]) return out_vids; // not allowed
    if (edge_seen(start_idx, nb_idx)) return out_vids; // already seen

    int prev = start_idx;
    int curr = nb_idx;

    out_vids.push_back(idx2vid_[start_idx]);

    while (true) {
        out_vids.push_back(idx2vid_[curr]);
        mark_edge(prev, curr);

        if (is_endpoint_[curr] && curr != start_idx) break; // found endpoint
        int next_idx = -1;

        for (int nb_i : VD.global_adj[curr]) {
            if (nb_i == prev) continue;
            if (!allowed[nb_i]) continue;
            if (!edge_seen(curr, nb_i)) {
                next_idx = nb_i;
                break;
            }
        }

        // fallback: any nb neq prev
        if (next_idx == -1) {
            for (int nb_i : VD.global_adj[curr]) {
                if (nb_i != prev) {
                    next_idx = nb_i;
                    break;
                }
            }
        }

        if (next_idx == -1) break; // no next found!

        prev = curr;
        curr = next_idx;
    }
    
    if (out_vids.size() < 2) {
        out_vids.clear();
    }

    return out_vids;
}
