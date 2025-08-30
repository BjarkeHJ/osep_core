#ifndef RHO_HPP_
#define RHO_HPP_

#include <vector>
#include <unordered_set>

struct RHState {
        int start_gid = -1;
        std::vector<int> exec_path_gids;  // expanded sequence of gids to execute (graph-connected)
        std::vector<int> coarse_order;    // chosen viewpoints in order (before expansion)
        float last_plan_score = -1.0f;
        int   next_target_gid = -1;
        std::unordered_set<int> visited;  // viewpoints already visited (or set by caller)
    };

#endif