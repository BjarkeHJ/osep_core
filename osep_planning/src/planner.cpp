/* 

Main paht planning algorithm

*/

#include "planner.hpp"

PathPlanner::PathPlanner(const PlannerConfig& cfg) : cfg_(cfg) {
    running = 1;
}

bool PathPlanner::planner_run() {
    RUN_STEP(generate_path);

    std::cout << "Planner: Updated viewpoints size: " << PD.updated_viewpoints.size() << std::endl;

    return running;
}

bool PathPlanner::handle_viewpoints() {
    for (const auto& vp : PD.updated_viewpoints) {
        //
    }

    return 1;
}


bool PathPlanner::generate_path() {
    

    return 1;
}