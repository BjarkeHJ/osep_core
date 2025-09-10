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
    rehydrate_state(gskel);

    if (!plan_path) return 1; // exit without executing path planning
    running = rh_plan_tick();
    running = set_path(gskel);

    // std::cout << "Planned Path - Path Lenght: " << PD.path_out.size() << std::endl;

    return running;
}

bool PathPlanner::build_graph(std::vector<Vertex>& gskel) {
    PD.graph.nodes.clear();
    PD.graph.adj.clear();
    Graph G;

    std::unordered_map<int, std::vector<int>> vids_to_gid;
    vids_to_gid.reserve(gskel.size() * 2);
    
    // Create nodes for each valid viewpoint!
    for (const Vertex& v : gskel) {
        for (int k=0; k<static_cast<int>(v.vpts.size()); ++k) {
            const Viewpoint& vp = v.vpts[k];

            if (vp.invalid || vp.vptid == 0ull) continue;
            // if (vp.visited) continue; // excluded visited from graph

            if (vp.vptid == 0ull) {
                std::cerr << "[GRAPH] Warning: viewpoint with zero handle id!" << std::endl;
                // continue;
            }

            GraphNode n;
            n.vptid = vp.vptid; // set unique handle id
            n.gid = static_cast<int>(G.nodes.size());
            n.vid = v.vid;
            n.k = k;
            n.p = vp.position;
            n.yaw = vp.yaw;
            n.score = vp.score; // Score obtained from viewpoint scoring (vpman)
            vids_to_gid[v.vid].push_back(n.gid);
            G.nodes.push_back(std::move(n));
        }
    }

    G.adj.resize(G.nodes.size());
    if (G.nodes.empty()) {
        PD.graph = std::move(G);
        return 0;
    }

    // Geometric adjacency with Gabriel testing and Corridor collision check
    vpt_cloud->points.clear();
    vpt_cloud->resize(G.nodes.size());
    for (size_t i=0; i<G.nodes.size(); ++i) {
        vpt_cloud->points[i].x = G.nodes[i].p.x();
        vpt_cloud->points[i].y = G.nodes[i].p.y();
        vpt_cloud->points[i].z = G.nodes[i].p.z();
    }
    vpt_kdtree.setInputCloud(vpt_cloud);

    std::unordered_set<uint64_t> edge_set; // will contain the set of undirected edges in the graph
    edge_set.reserve(G.nodes.size() * 4);

    std::vector<int> ids;
    std::vector<float> d2s;
    ids.reserve(32);
    d2s.reserve(32);

    for (int u=0; u < static_cast<int>(G.nodes.size()); ++u) {
        const auto& np = vpt_cloud->points[u];
        
        ids.clear();
        d2s.clear();
        int found = vpt_kdtree.radiusSearch(np, cfg_.graph_radius, ids, d2s);
        if (found <= 1) continue; //radiusSearch include np itself

        struct Cand { int v; float d2; };
        std::vector<Cand> cands;
        cands.reserve(found - 1);
        for (int j=1; j<found; ++j) {
            const int v = ids[j];
            if (v == u || v < u) continue; // undirected edges
            cands.push_back( {v, d2s[j] } );
        }
        
        // std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){ return a.d2 < b.d2; });
        
        // Find vertex of viewpoint u
        int uvid = G.nodes[u].vid;
        auto itU = gskel_vid2idx.find(uvid);
        if (itU == gskel_vid2idx.end()) continue;
        int idx = itU->second;
        const Vertex& gu = gskel[idx];
        const Eigen::Vector3f& p1 = np.getVector3fMap();

        for (const auto& c : cands) {
            const int j = c.v;
            const Eigen::Vector3f& p2 = vpt_cloud->points[j].getVector3fMap();
            if (!line_of_sight(p1, p2)) continue;
            
            // gabriel graph test
            Eigen::Vector3f mid = 0.5 * (p1 + p2);
            const float r2 = c.d2 * 0.25; // (d/2)² 
            bool keep_edge = true;

            for (int kk=0; kk<found; ++kk) {
                int k = ids[kk];
                if (k == u || k == j) continue;
                const Eigen::Vector3f& p3 = G.nodes[k].p;
                if ( (p3 - mid).squaredNorm() < r2) {
                    keep_edge = false;
                    break;
                }
            }

            if (!keep_edge) continue;

            const auto& vvid = G.nodes[j].vid;
            const bool topo = (uvid == vvid || std::find(gu.nb_ids.begin(), gu.nb_ids.end(), vvid) != gu.nb_ids.end());

            // if (!topo) continue; // only topological graph edges

            float w = std::sqrt(c.d2);
            
            // min edge length - maybe include yaw difference?
            const float min_edge_len = 0.0f;
            if (w < min_edge_len) continue;

            add_edge(G, u, c.v, w, edge_set, topo);
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

    // Check if current path is still valid -> If so: append new points

    // Extract subgraph around start
    std::vector<char> allow_transit; // all nodes within subgraph radius
    auto cand = build_subgraph(start_gid, allow_transit); // subgraph gids (number of nodes bounded in cand)

    std::cout << "[GRAPH] Number of graph nodes: " << PD.graph.nodes.size() << std::endl;
    if (cand.empty()) return 0;

    // Run planner on current horizon (sub-graph)
    std::vector<int> dfs_order;
    float best_score = -1e9f;
    bounded_dfs_plan(start_gid, cand, allow_transit, dfs_order, best_score);

    std::cout << "[PLAN] Number of dfs viewpoints: " << dfs_order.size() << std::endl;

    if (dfs_order.empty()) {
        return 0;
    }

    std::vector<uint64_t> new_path;
    new_path.reserve(dfs_order.size());
    for (int g : dfs_order) {
        if (g < 0 || g >= static_cast<int>(PD.g2h.size())) continue;
        uint64_t hid = PD.g2h[g];
        if (hid == 0ull) continue; // invalid handle
        new_path.push_back(hid);
    }
    
    std::cout << "[PATH] Number of executable viewpoints: " << new_path.size() << " with score " << best_score << std::endl;
    if (new_path.empty()) return 0;

    float old_score = PD.rhs.last_plan_score;
    bool accept = false;

    if (PD.rhs.exec_path_ids.empty() || old_score < 0.0f) {
        accept = true; // no previous path
    }
    else {
        float thresh = old_score * (1.0f + cfg_.hysteresis);
        std::cout << "Old Score: " << old_score << " New Score: " << best_score << " Thresh: " << thresh << std::endl;
        accept = (best_score > thresh);
    }

    if (accept) {
        std::cout << "Path Accepted!" << std::endl;
        PD.rhs.exec_path_ids = std::move(new_path);
        PD.rhs.last_plan_score = best_score;

        PD.rhs.next_target_id = 0ull;
        for (size_t i = 0; i<PD.rhs.exec_path_ids.size(); ++i) {
            uint64_t h = PD.rhs.exec_path_ids[i];
            if (!PD.rhs.visited.count(h)) {
                if (i == 0 && h == PD.rhs.start_id) continue; // already visited start
                PD.rhs.next_target_id = h;
                break;
            }
        }

        if (PD.rhs.next_target_id == 0ull) {
            PD.rhs.next_target_id = (PD.rhs.exec_path_ids.size() > 1) ? PD.rhs.exec_path_ids[1] : PD.rhs.exec_path_ids.front();
        }
    }

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
void PathPlanner::rehydrate_state(std::vector<Vertex>& gskel) {
    // remove visited that no longer exists
    {
        std::vector<uint64_t> to_erase;
        to_erase.reserve(PD.rhs.visited.size());
        for (auto h : PD.rhs.visited) {
            if (!PD.h2g.count(h)) to_erase.push_back(h);
        }

        for (auto h : to_erase) PD.rhs.visited.erase(h);
    }

    // remap start_id to current graph
    if (PD.rhs.start_id != 0ull && !PD.h2g.count(PD.rhs.start_id)) {
        PD.rhs.start_id = 0ull;
    }
    if (PD.rhs.start_id == 0ull) {
        int sg = pick_start_gid_near_drone();
        if (sg >= 0 && sg < static_cast<int>(PD.g2h.size()) && PD.g2h[sg] != 0ull) {
            PD.rhs.start_id = PD.g2h[sg];
        }
    }

    // clear exec_path_ids against current graph and visisted set
    {
        std::vector<uint64_t> cleaned;
        cleaned.reserve(PD.rhs.exec_path_ids.size());
        for (auto h : PD.rhs.exec_path_ids) {
            if (h == 0ull) continue;
            if (!PD.h2g.count(h)) continue; // not in current graph
            if (PD.rhs.visited.count(h)) continue; // already visited
            if (cleaned.empty() || cleaned.back() != h) {
                cleaned.push_back(h);
            }
        }
        PD.rhs.exec_path_ids = std::move(cleaned);
    }

    // PD.rhs.next_target_id = 0ull;
    // for (uint64_t h : PD.rhs.exec_path_ids) {
    //     if (PD.h2g.count(h) && !PD.rhs.visited.count(h)) {
    //         PD.rhs.next_target_id = h;
    //         break;
    //     }
    // }

    retarget_head();
    if (PD.rhs.next_target_id == 0ull && PD.rhs.start_id != 0ull && PD.h2g.count(PD.rhs.start_id)) {
        PD.rhs.next_target_id = PD.rhs.start_id;
    }
}

int PathPlanner::pick_start_gid_near_drone() {
    if (PD.graph.nodes.empty()) return -1;
    float best = std::numeric_limits<float>::infinity();
    int best_id = -1;
    const Eigen::Vector3f& dpos = PD.drone_pose.pos;
    for (const auto& n : PD.graph.nodes) {
        uint64_t hid = PD.g2h[n.gid]; // viewpoint id handle
        if (hid == 0ull) continue; // invalid handle
        if (!PD.h2g.count(hid)) continue; // stale handle: skip
        if (PD.rhs.visited.count(hid)) continue; // if visited -> skip

        float d2 = (n.p - dpos).squaredNorm();
        if (d2 < best) {
            best = d2;
            best_id = n.gid;
        }
    }
    return best_id;
}

std::vector<int> PathPlanner::build_subgraph(int start_gid, std::vector<char>& allow_transit) {
    const int N = static_cast<int>(PD.graph.nodes.size());
    if (N == 0 || start_gid < 0 || start_gid >= N) {
        allow_transit.clear();
        return {};
    }

    // run a radius bounded dijkstra on the graph nodes
    std::vector<char> allow(N, 1); // allow all nodes
    std::vector<float> dist;
    std::vector<int> parent;
    dijkstra(allow, start_gid, dist, parent, cfg_.subgraph_radius);

    allow_transit.assign(N, 0);
    struct Cand { int gid; float d; };
    std::vector<Cand> cands;
    cands.reserve(N);
    for (int i=0; i<N; ++i) {
        if (allow[i] && dist[i] <= cfg_.subgraph_radius) {
            allow_transit[i] = 1;
            cands.push_back( {i, dist[i]} );
        }
    }

    // sort candidates by distance to fetch the shortest distances if capped by max nodes
    std::sort(cands.begin(), cands.end(), 
            [](const Cand& a, const Cand& b) { 
                if (a.d != b.d) return a.d < b.d;
                return a.gid < b.gid;
            });

    // cap to max nodes
    const int K = std::min<int>(cfg_.subgraph_max_nodes, static_cast<int>(cands.size()));
    std::vector<int> out;
    out.reserve(K);
    for (int i=0; i<K; ++i) {
        out.push_back(cands[i].gid);
    }
    return out;
}

void PathPlanner::bounded_dfs_plan(int start_gid, const std::vector<int>& subgraph, const std::vector<char> allow_transit, std::vector<int>& path, float& best_score) {
    path.clear();
    if (subgraph.empty() || start_gid < 0 || start_gid >= static_cast<int>(PD.graph.nodes.size())) return;

    std::unordered_set<int> sub_set(subgraph.begin(), subgraph.end());

    float max_node_rew = 0.0f;
    for (int gid : subgraph) {
        if (gid < 0 || gid >= static_cast<int>(PD.graph.nodes.size())) continue;
        max_node_rew = std::max(max_node_rew, node_reward(PD.graph.nodes[gid]));
    }

    std::vector<int> cur_path;
    cur_path.reserve(cfg_.dfs_max_depth + 1);
    std::vector<bool> on_path(PD.graph.nodes.size(), false);

    auto push_node = [&](int u) { cur_path.push_back(u); on_path[u] = true; };
    auto pop_node = [&]() { on_path[cur_path.back()] = false; cur_path.pop_back(); };

    std::function<void(int, int, float, float)> dfs = [&](int u, int depth, float cur_rew, float cur_cost) {
        {
            float cur_score = cur_rew - cfg_.lambda*cur_cost;
            if (cur_score > best_score) {
                best_score = cur_score;
                path = cur_path;
            }
        }

        if (depth >= cfg_.dfs_max_depth) return;

        float optimistic = cur_rew + (cfg_.dfs_max_depth - depth) * max_node_rew;
        if (optimistic - cfg_.lambda * cur_cost < best_score) return; // prune
        
        struct Cand { int v; float key; float edge_w; };
        std::vector<Cand> cands;
        cands.reserve(PD.graph.adj[u].size());

        for (const auto& e : PD.graph.adj[u]) {
            int v = e.to;
            if (v < 0 || v >= static_cast<int>(PD.graph.nodes.size())) continue; // invalid
            if (!allow_transit[v]) continue; // not allowed
            if (!is_in_subgraph(v, sub_set)) continue; // not in subgraph
            if (on_path[v]) continue; // already on path
            if (PD.rhs.visited.count(PD.g2h[v])) continue; // already visited

            float w = edge_cost(e);
            float next_cost  = cur_cost + w;
            if (next_cost > cfg_.budget) continue; // over budget

            float step_gain = node_reward(PD.graph.nodes[v]) - cfg_.lambda * w;
            float key = step_gain + 0.01f * node_reward(PD.graph.nodes[v]); // slight preference to high reward nodes
            cands.push_back( {v, key, w} );
        }

        if (cands.empty()) return;

        std::partial_sort(cands.begin(), cands.begin() + std::min<int>(cfg_.dfs_beam_width, static_cast<int>(cands.size())), cands.end(),
            [](const Cand& a, const Cand& b) { return a.key > b.key; } );

        int K = std::min<int>(cfg_.dfs_beam_width, static_cast<int>(cands.size()));
        for (int i=0; i<K; ++i) {
            int v = cands[i].v;
            float w = cands[i].edge_w;
            float next_rew = cur_rew + node_reward(PD.graph.nodes[v]);
            float next_cost = cur_cost + w;
            
            push_node(v);
            dfs(v, depth + 1, next_rew, next_cost);
            pop_node();
        }
    };

    // initialize with starting node
    if (!is_in_subgraph(start_gid, sub_set) || !allow_transit[start_gid]) return;
    push_node(start_gid);
    dfs(start_gid, 0, node_reward(PD.graph.nodes[start_gid]), 0.0f);
    pop_node();
}

void PathPlanner::dijkstra(const std::vector<char>& allow, int src, std::vector<float>& dist, std::vector<int>& parent, const float radius) {
    const int N = static_cast<int>(PD.graph.nodes.size());
    dist.assign(N, std::numeric_limits<float>::infinity()); // vector of distances from starting node
    parent.assign(N, -1); // vector of predecessors of a node 

    using P = std::pair<float,int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;

    if (src < 0 || src >= N || !allow[src]) return; // invalid source node

    dist[src] = 0.0f;
    pq.emplace(0.0f, src);
    
    while (!pq.empty()) {
        auto [d,u] = pq.top();
        pq.pop();

        if (d != dist[u]) continue;
        if (!allow[u]) continue; // u not allowed (not in subgraph)
        if (d > radius) continue; // beyond radius cap (radius defaults to infinity)

        for (auto& e : PD.graph.adj[u]) {
            int v = e.to;
            if (v < 0 || v >= N) continue; // invalid
            if (!allow[v]) continue; // v not allowed (not in subgraph)

            float w = edge_cost(e);
            const float nd = d + w; // new distance

            // if the new distance is smaller than the previous to node v -> improved
            if (nd < dist[v]) {
                dist[v] = nd;
                parent[v] = u; // update predecessor
                pq.emplace(nd, v);
            }
        }
    }
}

bool PathPlanner::line_of_sight(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    /* Corridor check */
    if (!octree_) {
        return 0; // nothing to do...
    }

    Eigen::Vector3f d = b - a;
    float L = d.norm();
    if (L <= 1e-6f) return 1;
    Eigen::Vector3f u = d / L;
    
    // build local frame
    Eigen::Vector3f tmp = (std::abs(u.z()) < 0.9f) ? Eigen::Vector3f::UnitZ() : Eigen::Vector3f::UnitY();
    Eigen::Vector3f n1 = (tmp.cross(u)).normalized(); // Local Y
    Eigen::Vector3f n2 = (u.cross(n1)).normalized(); // local Z

    const float hx = 0.5f * L;
    const float hy = 0.25f * cfg_.safe_dist;
    const float hz = hy;

    Eigen::Matrix3f R;
    R.col(0) = u;
    R.col(1) = n1;
    R.col(2) = n2;
    Eigen::Vector3f half_ext = Eigen::Vector3f(hx, hy, hz);
    Eigen::Vector3f aabb_half = (R.cwiseAbs2() * half_ext).eval();

    Eigen::Vector3f center = 0.5f * (a + b);
    Eigen::Vector3f bbmin = center - aabb_half;
    Eigen::Vector3f bbmax = center + aabb_half;

    // Slight inflation
    const float vx = cfg_.map_voxel_size;
    bbmin.array() -= 0.5f * vx;
    bbmax.array() += 0.5f * vx;

    std::vector<int> ids;
    octree_->boxSearch(bbmin, bbmax, ids);

    const auto& cloud = octree_->getInputCloud();
    for (int idx : ids) {
        const auto& c = cloud->points[idx];
        Eigen::Vector3f p(c.x, c.y, c.z);
        Eigen::Vector3f pl = R.transpose() * (p - center);

        if (std::abs(pl.x()) <= hx && std::abs(pl.y()) <= hy && std::abs(pl.z()) <= hz) {
            return 0;
        }
    }   
    return 1;
}

float PathPlanner::edge_cost(const GraphEdge& e) {
    float w = e.w;
    if (cfg_.geometric_bias != 0.0f && !e.topo) w += cfg_.geometric_bias;
    if (cfg_.topo_bonus != 0.0f && e.topo) w = std::max(0.0f, w - cfg_.topo_bonus);

    if (w < 0.0f) w = 0.0f; // clamp edge cost to zero
    
    return w;
}

float PathPlanner::node_reward(const GraphNode& n) {

    return n.score;
}

float PathPlanner::path_travel_cost(const std::vector<int>& gids) {
    if (gids.size() < 2) return 0.0f;
    float c = 0.0f;

    for (size_t i=1; i<gids.size(); ++i) {
        bool found = false;
        for (auto& e : PD.graph.adj[gids[i-1]]) {
            if (e.to == gids[i]) {
                c += edge_cost(e);
                found = true;
                break;
            }
        }

        if (!found) {
            std::cout << "PathTravelCost: Edge did not exist..." << std::endl;
            return std::numeric_limits<float>::infinity();
        }

    }

    return c;
}

uint64_t PathPlanner::retarget_head() {
    for (uint64_t h : PD.rhs.exec_path_ids) {
        if (h && PD.h2g.count(h) && !PD.rhs.visited.count(h)) {
            PD.rhs.start_id = h;
            PD.rhs.next_target_id = h;
            return h;
        }
    }
    PD.rhs.next_target_id = 0ull;
    return 0ull;
}

/* Public methods for Control updates */
bool PathPlanner::get_next_target(Viewpoint& out) {
    if (PD.rhs.next_target_id == 0ull || 
        !PD.h2g.count(PD.rhs.next_target_id) || 
        PD.rhs.visited.count(PD.rhs.next_target_id)) {
        
        if (retarget_head() == 0ull) {
            return false;
        }
        return false;
    }

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

bool PathPlanner::get_next_k_targets(std::vector<Viewpoint>& out_k, int k) {
    out_k.clear();
    if (k <= 0) return false;

    const auto& seq = PD.rhs.exec_path_ids;

    if (seq.empty()) {
        std::cout << "[GET PATH] No point in path!" << std::endl;
        return false;
    }

    size_t start_i = 0;
    if (PD.rhs.next_target_id != 0ull) {
        auto it = std::find(seq.begin(), seq.end(), PD.rhs.next_target_id);
        if (it != seq.end()) {
            start_i = std::distance(seq.begin(), it);
        }
    }

    int produced = 0;
    for (size_t i = start_i; i < seq.size() && produced < k; ++i) {
        const uint64_t id = seq[i];

        // Map handle -> current graph gid
        auto it = PD.h2g.find(id);
        if (it == PD.h2g.end()) continue; // stale handle: skip
        const int g = it->second;
        if (g < 0 || g >= static_cast<int>(PD.graph.nodes.size())) continue; // stale gid: skip

        const GraphNode& gn = PD.graph.nodes[g];

        Viewpoint vp;
        vp.position = gn.p;
        vp.yaw = gn.yaw;
        vp.orientation = yaw_to_quat(gn.yaw);
        vp.target_vid = gn.vid;
        vp.target_vp_pos = gn.k;
        vp.vptid = gn.vptid;
        vp.in_path = true;

        out_k.push_back(std::move(vp));
        ++produced;
    }

    return produced > 0;
}

bool PathPlanner::notify_reached(std::vector<Vertex>& gskel) {
    if (PD.rhs.next_target_id == 0ull) return false;
    
    if (!PD.h2g.count(PD.rhs.next_target_id)) {
        if (retarget_head() == 0ull) {
            std::cout << "[NOTIFY] No mapped target available!\n";
            return false;
        }
    }

    (void)mark_visited_in_skeleton(PD.rhs.next_target_id, gskel);

    PD.rhs.visited.insert(PD.rhs.next_target_id);
    PD.rhs.start_id = PD.rhs.next_target_id;

    // Erase prefix up to (and including) the reached handle
    auto it = std::find(PD.rhs.exec_path_ids.begin(), PD.rhs.exec_path_ids.end(), PD.rhs.start_id);
    if (it != PD.rhs.exec_path_ids.end()) {
        PD.rhs.exec_path_ids.erase(PD.rhs.exec_path_ids.begin(), std::next(it));
    }

    

    // Drop any stale/visisted handles
    PD.rhs.exec_path_ids.erase(
        std::remove_if(PD.rhs.exec_path_ids.begin(), PD.rhs.exec_path_ids.end(),
                       [&](uint64_t h){ return PD.rhs.visited.count(h) || !PD.h2g.count(h); }),
        PD.rhs.exec_path_ids.end());
    
    PD.rhs.next_target_id = PD.rhs.exec_path_ids.empty() ? 0ull : PD.rhs.exec_path_ids.front();

    retarget_head();
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

std::vector<int> PathPlanner::handles_to_gids(const std::vector<uint64_t>& h) {
    std::vector<int> out;
    out.reserve(h.size());
    for (auto id : h) {
        auto it = PD.h2g.find(id);
        if (it == PD.h2g.end()) continue;
        out.push_back(it->second);
    }
    return out;
}

float PathPlanner::path_score_from_gids(const std::vector<int>& gids) {
    if (gids.empty()) return -1e9f;
    float rew = 0.0f;
    for (int g : gids) {
        rew += node_reward(PD.graph.nodes[g]);
    }
    float cost = path_travel_cost(gids);
    if (!std::isfinite(cost)) return -1e9f;
    return rew - cfg_.lambda * cost;
}

int PathPlanner::common_prefix_len(const std::vector<int>& A, const std::vector<int>& B) {
    int L = std::min<int>(A.size(), B.size());
    int i = 0;
    for (; i<L; ++i) {
        if (A[i] != B[i]) break;
    }
    return i;
}


/* 
TODO:
- Actually delte viewpoints (Treats invalid = delete / Fine for now)
- Make LOS a corridor (0.5 safe dist?) (Done)
- Incorporate with controller -> run true sim (Done!)
- Multi vpt path for better path executing (set k of path - still only commit to 1 between plans) (Done)

- In exec path: Optimize path order for minimum uturns etc (start at closest to drone -> visit all)

- Implement adjusted viewpoint 

- Implement edge weigthing bonus for topological graphs (same vertex or adjacent vertex)

- Tune weightings for the planner (understand it)

- color seen voxels in real time? 

- End-of-Mission Criteria: No more valid viewpoints with sufficient score! -> Return to start?
*/
