#ifndef PLANNER_HPP_
#define PLANNER_HPP_

#include <iostream>
#include <chrono>
#include <pcl/common/common.h>
#include <Eigen/Core>

#define RUN_STEP(fn) \
    do { \
        bool ok = (fn)(); \
        running = ok; \
        if (!ok) return false; \
    } while (0)


struct PlannerConfig {
    int test;
};

struct Viewpoint {
    Eigen::Vector3f position;
    float yaw;
    // Eigen::Quaternionf orientation;
    int target_vid = -1;
    int target_vp_pos = -1;

    float score = 0.0f;
    bool updated;
    bool in_path = false;
    bool visited = false;
};

struct PlannerData {
    std::vector<Viewpoint> updated_viewpoints;
    Viewpoint start;
    Viewpoint end;
    
};


class PathPlanner {
public:
    PathPlanner(const PlannerConfig& cfg);
    bool planner_run();
    std::vector<Viewpoint>& input_viewpoints() { return PD.updated_viewpoints; }

private:
    /* Functions */
    bool handle_viewpoints();


    bool generate_path();

    /* Helper */


    /* Params */
    PlannerConfig cfg_;
    bool running;

    /* Data */
    PlannerData PD;
};


#endif