#ifndef RHO_HPP_
#define RHO_HPP_

#include <vector>
#include <unordered_set>

struct RHState {
    // All ids here are unique viewpoint identifiers packed as 64 bit (vid, vpt counter (unique value for each generated viewpoint))
    uint64_t start_id = 0ull; // unique vpt identifier
    uint64_t next_target_id = 0ull;
    std::vector<uint64_t> exec_path_ids;  // expanded sequence of vptid handles to execute (graph-connected)
    std::unordered_set<uint64_t> visited;  // viewpoints already visited
    float last_plan_score = -1.0f;
};

#endif