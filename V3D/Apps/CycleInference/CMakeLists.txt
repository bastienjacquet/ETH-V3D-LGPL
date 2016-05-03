project(CycleInference)

include_directories(${V3D_INCLUDE_DIRS} ${EXTRA_INC_DIRS} ${V3D_DIR}/Apps/SfM)

include (${V3D_DIR}/Config/v3d_macros.cmake)
include (${V3D_DIR}/Config/local_config.cmake)

set (CMAKE_BUILD_TYPE Release)
#set (CMAKE_BUILD_TYPE Debug)

add_library (V3D-ETH STATIC inference_common.cpp cycle_inference.cpp)

add_v3d_executable (visualize_matches visualize_matches.cpp)
add_v3d_executable (verify_EG_edges verify_EG_edges.cpp)
add_v3d_executable (verify_EG_stage2 verify_EG_stage2.cpp)

target_link_libraries(visualize_matches V3D sqlite3 glut Cg CgGL GLEW)
target_link_libraries(verify_EG_edges V3D-ETH V3D dai lpsolve55 dl sqlite3)
target_link_libraries(verify_EG_stage2 reconstruction V3D-ETH V3D dai lpsolve55 dl sqlite3)
