/* 

Main path planning algorithm

*/

#include "planner.hpp"

PathPlanner::PathPlanner(const PlannerConfig& cfg) : cfg_(cfg) {
    gmap.reset(new pcl::PointCloud<pcl::PointXYZ>);
    vpt_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    running = 1;
}


bool PathPlanner::plan(std::vector<Vertex>& gskel) {
    /* Main public function - Runs the path planning pipeline */
    running = build_graph(gskel);

    if (!plan_path) return 1; // exit without executing path planning
    running = rh_plan_tick();
    running = set_path(gskel);

    std::cout << "Planned Path - Path Lenght: " << PD.path_out.size() << std::endl;

    return running;
}

bool PathPlanner::build_graph(std::vector<Vertex>& gskel) {
    /* CHANGE THIS TO MY ORIGINAL WORK AND CLASSIFY EDGES BASED ON VERTEX ADJACENCY */

    PD.graph.nodes.clear();
    PD.graph.adj.clear();
    Graph G;

    std::unordered_map<int, std::vector<int>> vids_to_gid;
    vids_to_gid.reserve(gskel.size() * 2);
    
    // Create nodes
    for (const Vertex& v : gskel) {
        for (int k=0; k<static_cast<int>(v.vpts.size()); ++k) {
            const Viewpoint& vp = v.vpts[k];

            if (vp.invalid) continue;
            if (vp.visited) continue;

            GraphNode n;
            n.vptid = vp.vptid; // set unique handle id
            n.gid = static_cast<int>(G.nodes.size());
            n.vid = v.vid;
            n.k = k;
            n.p = vp.position;
            n.yaw = vp.yaw;
            n.score = vp.score;
            vids_to_gid[v.vid].push_back(n.gid);
            G.nodes.push_back(std::move(n));
        }
    }

    G.adj.resize(G.nodes.size());
    if (G.nodes.empty()) {
        PD.graph = std::move(G);
        return 0;
    }
    
    std::unordered_set<uint64_t> edge_set;
    edge_set.reserve(G.nodes.size() * 4);
    for (const auto& u : gskel) {
        auto itU = vids_to_gid.find(u.vid);
        if (itU == vids_to_gid.end() || itU->second.empty()) continue;
        
        if (u.type == 1) {
            // connect in order
            const auto& gids = itU->second;
            const int m = static_cast<int>(gids.size());
            if (m < 2) continue;

            for (int i=0; i+1<m; ++i) {
                int g0 = gids[i];
                int g1 = gids[i+1];
                const auto& p0 = G.nodes[g0].p;
                const auto& p1 = G.nodes[g1].p;
                const float dd2 = d2(p0, p1);
                if (line_of_sight(p0, p1)) {
                    const float w = std::sqrt(dd2);
                    add_edge(G, g0, g1, w, edge_set, true); // add topological graph
                }
            }
            continue;
        }
        else {
            for (int nb_vid : u.nb_ids) {
                auto itnb = gskel_vid2idx.find(nb_vid);
                if (itnb == gskel_vid2idx.end()) continue;
                const int nb_idx = itnb->second;
                const auto& v = gskel[nb_idx];
                auto itV = vids_to_gid.find(v.vid);
                if (itV == vids_to_gid.end() || itV->second.empty()) continue;

                const auto& u_gids = itU->second;
                const auto& v_gids = itV->second;

                for (int gu : u_gids) {
                    const auto& pu = G.nodes[gu].p;
                    int best_gv = -1;
                    float best_d2 = std::numeric_limits<float>::infinity();

                    for (int gv : v_gids) {
                        const auto& pv = G.nodes[gv].p;
                        float dd2 = d2(pu, pv);
                        if (dd2 < best_d2 && line_of_sight(pu, pv)) {
                            best_d2 = dd2;
                            best_gv = gv;
                        }
                    }

                    if (best_gv >= 0) {
                        const float w = std::sqrt(best_d2);
                        add_edge(G, gu, best_gv, w, edge_set, true); // add topological graph
                    }
                }
            }
        }
    }

    // handle unique viewpoint ids
    PD.h2g.clear();
    PD.g2h.assign(G.nodes.size(), 0ull); // 0 (as unsigned long long (64bit))
    // PD.g2h.clear();
    // PD.g2h.resize(G.nodes.size());
    for (auto& n : G.nodes) {
        if (n.vptid != 0ull) {
            PD.h2g[n.vptid] = n.gid;
            PD.g2h[n.gid] = n.vptid;
        }
    }

    PD.graph = std::move(G);
    return 1;
}

bool PathPlanner::rh_plan_tick() {
    if (PD.graph.nodes.empty()) {
        PD.rhs.exec_path_ids.clear();
        PD.rhs.next_target_id = 0ull;
        return 0;
    }

    if (PD.rhs.start_id == 0ull) { // default 0 unsigned long long
        std::cout << "[RH PLANNER] Setting new starting point!" << std::endl;
        int start_gid = pick_start_gid_near_drone(); // get graph index of starting point
        if (start_gid < 0) return 0;
        PD.rhs.start_id = PD.g2h[start_gid]; // set start id as corresponding unique id
        if (PD.rhs.start_id == 0ull) return 0; // still default (should not happen if vptid > 0)
    }

    // Determine starting id
    int start_gid = -1;
    auto it = PD.h2g.find(PD.rhs.start_id);
    if (it != PD.h2g.end()) {
        // found match
        start_gid = it->second;
    }
    else {
        // id missing -> pick nearest and update handle
        int sg = pick_start_gid_near_drone();
        if (sg < 0) return 0;
        PD.rhs.start_id = PD.g2h[sg];
        if (PD.rhs.start_id == 0ull) return 0;
        start_gid = sg;
    }
    
    std::vector<int> prev_exec_gids;
    prev_exec_gids.reserve(PD.rhs.exec_path_ids.size());
    for (uint64_t hid : PD.rhs.exec_path_ids) {
        int gid = -1;
        auto it = PD.h2g.find(hid);
        if (it != PD.h2g.end()) {
            gid = it->second;
        }
        if (gid >= 0) {
            prev_exec_gids.push_back(gid);
        }
    }
    bool had_prev_plan = !prev_exec_gids.empty();
    had_prev_plan = false;
    
    // Extract subgraph around start
    auto cand = build_subgraph(start_gid); // subgraph gids
    if (cand.empty()) return 0;

    // remove already visisted from consideration (but keep start)
    auto is_visited_gid = [&](int g) -> bool {
        if (g == start_gid) return false;
        if (g < 0 || g >= static_cast<int>(PD.g2h.size())) return true;
        uint64_t hid = PD.g2h[g];
        if (hid == 0ull) return true; // invalid / missing handle -> drop from candidates
        return PD.rhs.visited.count(hid) > 0; // drop if already visited this handle!
    };
    cand.erase(std::remove_if(cand.begin(), cand.end(), is_visited_gid), cand.end());

    // distances on subgraph
    std::vector<std::vector<float>> D; // distances between graph nodes (edge weights)
    std::vector<std::vector<int>> parent; // gid predecessors from source (cand[i])
    compute_apsp(cand, D, parent);

    // greedy orienteering
    auto order = greedy_orienteering(cand, start_gid, D);

    // Score for hysteresis: sum reward - lambda*cost
    float rew = 0.0f;
    float cost = 0.0f;

    for (size_t i=0; i<order.size(); ++i) {
        rew += node_reward(PD.graph.nodes[order[i]]);
    }

    // subgraph -> global graph mapping
    std::unordered_map<int,int> loc;
    loc.reserve(cand.size() * 2);
    for (int i=0; i<static_cast<int>(cand.size()); ++i) {
        loc[cand[i]] = i;
    }

    // Compute path cost
    for (size_t i=0; i+1<order.size(); ++i) {
        auto ita = loc.find(order[i]);
        auto itb = loc.find(order[i+1]);
        if (ita == loc.end() || itb == loc.end()) continue;
        float dab = D[ita->second][itb->second];
        if (std::isfinite(dab)) {
            cost += dab;
        }
    }

    float score = rew - cfg_.lambda * cost;
    bool accept = (PD.rhs.last_plan_score < 0.0f) || 
                  (score > PD.rhs.last_plan_score * (1.f + cfg_.hysteresis)) ||
                  !had_prev_plan;
                  //   (PD.rhs.exec_path_ids.empty());

    if (!accept) return 0;

    auto exec_gids = expand_to_graph_path(order, cand, parent);
    if (exec_gids.empty()) return 0;

    PD.rhs.exec_path_ids.clear();
    PD.rhs.exec_path_ids.reserve(exec_gids.size());
    for (int g : exec_gids) {
        uint64_t vptid = PD.g2h[g];
        if (vptid == 0ull) continue;
        // if (vptid == PD.rhs.start_id) continue;
        if (PD.rhs.visited.count(vptid)) continue;
        PD.rhs.exec_path_ids.push_back(vptid);
    }

    PD.rhs.last_plan_score = score;

    // choose next planning target
    // PD.rhs.next_target_id = (PD.rhs.exec_path_ids.size() >= 2)
    //                          ? PD.rhs.exec_path_ids[1]
    //                          : PD.rhs.exec_path_ids.front();
    
    PD.rhs.next_target_id = PD.rhs.exec_path_ids.front(); // make first element the target for velocity controller

    return 1;
}

bool PathPlanner::set_path(std::vector<Vertex>& gskel) {
    PD.path_out.clear();
    PD.path_out.reserve(PD.rhs.exec_path_ids.size());

    for (uint64_t hid : PD.rhs.exec_path_ids) {
        auto itG = PD.h2g.find(hid);
        if (itG == PD.h2g.end()) continue; // viewpoint not present
        int gid = itG->second;
        if (gid < 0 || gid >= static_cast<int>(PD.graph.nodes.size())) continue;

        const GraphNode& gn = PD.graph.nodes[gid];

        // vid -> vertex index in gskel
        auto itV = gskel_vid2idx.find(gn.vid);
        if (itV == gskel_vid2idx.end()) continue;
        const int vidx = itV->second;
        if (vidx < 0 || vidx >= static_cast<int>(gskel.size())) continue;

        Vertex& vx = gskel[vidx];
        int kk = gn.k;
        if (kk < 0 || kk >= static_cast<int>(vx.vpts.size()) || vx.vpts[kk].vptid != hid) {
            kk = -1;
            for (int i=0; i<static_cast<int>(vx.vpts.size()); ++i) {
                if (vx.vpts[i].vptid == hid) {
                    kk = i;
                    break;
                }
            }
            if (kk < 0) continue; // still not found somehow
        }

        Viewpoint vp = vx.vpts[kk];
        vp.target_vid = gn.vid;
        vp.target_vp_pos = kk;
        vp.in_path = true;
        PD.path_out.push_back(std::move(vp));
    }

    return !PD.path_out.empty();
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

int PathPlanner::pick_start_gid_near_drone() {
    if (PD.graph.nodes.empty()) return -1;
    float best = std::numeric_limits<float>::infinity();
    int best_id = -1;
    const Eigen::Vector3f& dpos = PD.drone_pose.pos;
    for (const auto& n : PD.graph.nodes) {
        float d2 = (n.p - dpos).squaredNorm();
        if (d2 < best) {
            best = d2;
            best_id = n.gid;
        }
    }
    return best_id;
}

std::vector<int> PathPlanner::build_subgraph(int start_gid) {
    const int N = static_cast<int>(PD.graph.nodes.size());
    std::vector<float> dist(N, std::numeric_limits<float>::infinity());
    std::vector<char> in(N, 0);
    using P = std::pair<float,int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    dist[start_gid] = 0.0f;
    pq.emplace(0.0f, start_gid);

    while (!pq.empty()) {
        auto [d,u] = pq.top();
        pq.pop();
        if (d != dist[u]) continue;
        // if (d > cfg_.subgraph_radius) break;
        if (d > cfg_.subgraph_radius) continue;

        in[u] = 1;
        for (const auto& e : PD.graph.adj[u]) {
            float w = edge_cost(e);
            int v = e.to;
            if (dist[v] > d + w) {
                dist[v] = d + w;
                pq.emplace(dist[v], v);
            }
        }
    }

    std::vector<int> cand;
    cand.reserve(cfg_.subgraph_max_nodes);
    for (int i=0; i<N && static_cast<int>(cand.size())<cfg_.subgraph_max_nodes; ++i) {
        if (in[i]) {
            cand.push_back(i);
        }
    }
    return cand; // subgraph of gids
}

float PathPlanner::edge_cost(const GraphEdge& e) {
    float w = e.w;
    if (cfg_.geometric_bias != 0.0f && !e.topo) w += cfg_.geometric_bias;
    if (cfg_.topo_bonus != 0.0f && e.topo) w = std::max(0.0f, w - cfg_.topo_bonus);
    return w;
}

float PathPlanner::node_reward(const GraphNode& n) {
    return n.score;
}

void PathPlanner::compute_apsp(const std::vector<int>& cand, std::vector<std::vector<float>>& D, std::vector<std::vector<int>>& parent) {
    /* All-pairs shortest path algorithm */
    const int M = static_cast<int>(cand.size());
    D.assign(M, std::vector<float>(M, std::numeric_limits<float>::infinity()));
    parent.assign(M, std::vector<int>(PD.graph.nodes.size(), -1));

    // Only compute for subgraph
    std::vector<char> allow(PD.graph.nodes.size(), 0);
    for (int gid: cand) {
        allow[gid] = 1;
    }

    // For each candidate in subgraph -> compute shortest distance to any other node in subgraph
    std::vector<float> dist;
    std::vector<int> par;
    for (int i=0; i<M; ++i) {
        int src = cand[i];
        dijkstra(allow, src, dist, par);

        for (int j=0; j<M; ++j) {
            D[i][j] = dist[cand[j]];
        }
        parent[i] = par;
    }
}

void PathPlanner::dijkstra(const std::vector<char>& allow, int s, std::vector<float>& dist, std::vector<int>& parent) {
    const int N = static_cast<int>(PD.graph.nodes.size());
    dist.assign(N, std::numeric_limits<float>::infinity());
    parent.assign(N, -1);
    using P = std::pair<float,int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    dist[s] = 0.0f;
    pq.emplace(0.0f, s);
    
    while (!pq.empty()) {
        auto [d,u] = pq.top();
        pq.pop();
        if (d != dist[u]) continue;
        for (const auto& e : PD.graph.adj[u]) {
            int v = e.to;
            if (!allow[v]) continue; // not in subgraph (not allowed)
            float w = edge_cost(e);
            if (dist[v] > d + w) {
                dist[v] = d + w;
                parent[v] = u;
                pq.emplace(dist[v], v);
            }
        }
    }
}

std::vector<int> PathPlanner::greedy_orienteering(const std::vector<int>& cand, int start_gid, const std::vector<std::vector<float>>& D) {
    // map gid -> local index
    std::unordered_map<int,int> loc;
    loc.reserve(cand.size() * 2);
    for (int i=0; i<static_cast<int>(cand.size()); ++i) {
        loc[cand[i]] = i;
    }

    if (!loc.count(start_gid)) {
        return {start_gid};
    }

    // initial path is [start]
    std::vector<int> order;
    order.push_back(start_gid);
    float used = 0.0f;

    auto insert_gain = [&](int k_gid, int pos) -> std::pair<float, float> {
        // delta cost if insert k between pos-1 and pos
        int a_gid = order[pos-1];
        int b_gid = (pos < static_cast<int>(order.size()) ? order[pos] : -1);

        float dc = 0.0f;
        int ia = loc[a_gid];
        int ik = loc[k_gid];
        float dak = D[ia][ik];

        if (!std::isfinite(dak)) {
            return {-1e18f, std::numeric_limits<float>::infinity()};
        }
        dc += dak;
        if (b_gid != -1) {
            int ib = loc[b_gid];
            float dkb = D[ik][ib];
            if (!std::isfinite(dkb)) {
                return {-1e18f, std::numeric_limits<float>::infinity()};
            }
            float dab = D[ia][ib];
            if (!std::isfinite(dab)) {
                return {-1e18f, std::numeric_limits<float>::infinity()};
            }
            dc += dkb - dab;
        }
        float gain = node_reward(PD.graph.nodes[k_gid]) - cfg_.lambda * dc;
        return {gain, dc};
    };

    std::unordered_set<int> in_path(order.begin(), order.end());
    while (true) {
        float best_gain = -1e18f;
        float best_dc = 0.0f;
        int best_k = -1;
        int best_pos = -1;

        for (int k_gid : cand) {
            if (in_path.count(k_gid)) continue;

            // try all insertion positions 1...|order|
            for (int pos=1; pos<= static_cast<int>(order.size()); ++pos) {
                auto [g, dc] = insert_gain(k_gid, pos);
                if (g > best_gain && std::isfinite(dc) && (used + dc) <= cfg_.budget) {
                    best_gain = g;
                    best_dc = dc;
                    best_k = k_gid;
                    best_pos = pos;
                }
            }
        }

        if (best_k == -1 || best_gain <= 0.0f) break;
        
        order.insert(order.begin() + best_pos, best_k);

        in_path.insert(best_k);
        used += best_dc; 
    }

    two_opt_improve(order, D, loc);
    return order;
}

void PathPlanner::two_opt_improve(std::vector<int>& order, const std::vector<std::vector<float>>& D, const std::unordered_map<int,int>& loc) {
    if (order.size() < 4) return;

    auto idx = [&](int gid) -> int {
        auto it = loc.find(gid);
        return (it == loc.end()) ? -1 : it->second; // index in the APSP cand
    };

    bool improved = true;
    while (improved) {
        improved = false;
        for (int a=0; a+2<static_cast<int>(order.size()); ++a) {
            for (int b=a+1; b+1<static_cast<int>(order.size()); ++b) {
                int i = order[a];
                int i2 = order[a+1];
                int j = order[b];
                int j2 = order[b+1];

                // Mapped to global index
                int li = idx(i);
                int li2 = idx(i2);
                int lj = idx(j);
                int lj2 = idx(j2);

                if (li<0 || li2<0 || lj<0 || lj2<0) continue;

                float oldc = D[li][li2] + D[lj][lj2];
                float newc = D[li][lj] + D[li2][lj2];

                if (std::isfinite(newc) && newc + 1e-6 < oldc) {
                    std::reverse(order.begin()+a+1, order.begin()+b+1);
                    improved = true;
                }
            }
        }
    }
}

std::vector<int> PathPlanner::expand_to_graph_path(const std::vector<int>& order, const std::vector<int>& cand, const std::vector<std::vector<int>>& parent) {
    // cand[i] is gid; Parent[i][g] gives predecessor of g in shortest path from cand[i]

    std::vector<int> exec;
    if (order.empty()) return exec; // return empty path...
    
    std::unordered_map<int,int> loc;
    loc.reserve(cand.size() * 2);
    for (int i=0; i<static_cast<int>(cand.size()); ++i) {
        loc[cand[i]] = i;
    }

    auto has_edge = [&](int u, int v) -> bool {
        for (const auto& e : PD.graph.adj[u]) {
            if (e.to == v) return true;
        }
        return false;
    };

    auto append_subgraph_path = [&](int src, int dst) -> bool {
        if (has_edge(src, dst)) {
            exec.push_back(dst);
            return true;
        }

        auto itS = loc.find(src);
        if (itS == loc.end()) return false;
        int is = itS->second;

        std::vector<int> rev;
        int cur = dst;
        while (cur != -1) {
            rev.push_back(cur);
            if (cur == src) break;
            cur = parent[is][cur]; // predecessor on path from src to cur
        }
        if (rev.empty() || rev.back() != src) return false; // unreachable

        for (int i=static_cast<int>(rev.size() - 2); i>=0; --i) {
            exec.push_back(rev[i]);
        }
        return true;
    };


    // Start with first node
    exec.push_back(order.front());
    for (size_t t=1; t<order.size(); ++t) {
        int src = order[t-1];
        int dst = order[t];

        if (!append_subgraph_path(src, dst)) {
            // fallback
            std::vector<float> dist;
            std::vector<int> par;
            std::vector<char> allow(PD.graph.nodes.size(), 1); // allow all
            dijkstra(allow, src, dist, par);

            std::vector<int> rev;
            int cur = dst;
            while (cur != -1) {
                rev.push_back(cur);
                if (cur == src) break;
                cur = par[cur];
            }
            if (rev.empty() || rev.back() != src) {
                return {}; // give up
            }
            for (int i=static_cast<int>(rev.size() - 2); i>=0; --i) {
                exec.push_back(rev[i]);
            }
        }
    }

    // final safety
    for (size_t i=1; i<exec.size(); ++i) {
        if (!has_edge(exec[i-1], exec[i])) {
            return {}; // should not happen... forces replan
        }
    }

    return exec;
}



/* Public methods for Control updates */
bool PathPlanner::get_next_target(Viewpoint& out) {
    if (PD.rhs.next_target_id == 0ull) return false;

    auto it = PD.h2g.find(PD.rhs.next_target_id);
    if (it == PD.h2g.end()) return false;
    int g = it->second;
    if (g < 0 || g >= static_cast<int>(PD.graph.nodes.size())) return false;

    const GraphNode& gn = PD.graph.nodes[g];
    out.position = gn.p;
    out.yaw = gn.yaw;
    out.orientation = yaw_to_quat(gn.yaw);
    out.target_vid = gn.vid;
    out.target_vp_pos = gn.k;
    out.in_path = true;
    return true;
}

bool PathPlanner::get_start(Viewpoint& out) {
    if (PD.rhs.start_id == 0ull) return false;
    auto it = PD.h2g.find(PD.rhs.start_id);
    if (it == PD.h2g.end()) return false;
    int g = it->second;
    if (g < 0 || g >= static_cast<int>(PD.graph.nodes.size())) return false;
    const GraphNode& gn = PD.graph.nodes[g];
    out.position = gn.p;
    out.yaw = gn.yaw;
    out.orientation = yaw_to_quat(gn.yaw);
    out.target_vid = gn.vid;
    out.target_vp_pos = gn.k;
    out.in_path = true;
    return true;
}

bool PathPlanner::notify_reached(std::vector<Vertex>& gskel) {
    if (PD.rhs.next_target_id == 0ull) return false;
    
    (void)mark_visited_in_skeleton(PD.rhs.next_target_id, gskel);

    PD.rhs.visited.insert(PD.rhs.next_target_id);
    PD.rhs.start_id = PD.rhs.next_target_id;


    uint64_t new_next = 0ull;
    bool past_anchor = false;
    for (uint64_t h : PD.rhs.exec_path_ids) {
        if (!past_anchor) {
            if (h == PD.rhs.start_id) {
                past_anchor = true;
                continue;
            }
        }
        if (!h) continue;
        if (PD.rhs.visited.count(h) == 0) {
            new_next = h;
            break;
        }
    }
    PD.rhs.next_target_id = new_next;
    return true;
}

bool PathPlanner::mark_visited_in_skeleton(uint64_t hid, std::vector<Vertex>& gskel) {
    if (!hid) return false;

    int gid = -1;
    auto it = PD.h2g.find(hid);
    if (it == PD.h2g.end()) return false;
    gid = it->second;

    int vid = -1;
    int k = -1;

    if (gid >= 0 && gid < static_cast<int>(PD.graph.nodes.size())) {
        const auto& gn = PD.graph.nodes[gid];
        vid = gn.vid;
        k = gn.k;
    }
    else {
        // fallback: try to derive from handle
        vid = static_cast<int>(hid >> 32);
        k = -1;
    }

    auto itV = gskel_vid2idx.find(vid);
    if (itV == gskel_vid2idx.end()) return false;
    Vertex& vx = gskel[itV->second];

    if (k >= 0 && k < static_cast<int>(vx.vpts.size()) && vx.vpts[k].vptid == hid) {
        vx.vpts[k].visited = true;
        // vx.vpts[k].in_path = false;
        return true;
    }

    for (auto& vp : vx.vpts) {
        if (vp.vptid == hid) {
            vp.visited = true;
            // vp.in_path = false;
            return true;
        }
    }

    std::cout << "[MARK VISITED] Viewpoint Not Found!" << std::endl;
    return false; // not found this tick
}





/* 

TODO:
- Actually delte viewpoints (Set invalid = delete / Fine for now)
- Make LOS a corridor (0.5 safe dist?)
- Change graph build to geometric and implement edge weigthing bonus for topological graphs (same vertex or adjacent vertex)
- 

- Incorporate with controller -> run true sim (Done!)

- viewpoint manager: delete viewpoints accordingly
- Tune weightings for the planner (understand it)

- Multi vpt path for better path executing (set k of path - still only commit to 1 between plans)

- color seen voxels in real time? 

*/