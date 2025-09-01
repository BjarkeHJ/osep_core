# Unique stable vertex ID
uint32 id

# Vertex position (x,y,z in global frame)
geometry_msgs/Point32 position

# Adjacent vertices ids
uint32[] adj

# Vertex type (leaf=1, branch=2, joint=3)
uint8 type

# Vertex position update flag
bool pos_update

# Vertex type update flag
bool type_update

# Time since last update
builtin_interfaces/Time last_update