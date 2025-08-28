/* 

Main paht planning algorithm

*/

#include "planner.hpp"

PathPlanner::PathPlanner(const PlannerConfig& cfg) : cfg_(cfg) {
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
    

    return 1;
}


/* HELPER */

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