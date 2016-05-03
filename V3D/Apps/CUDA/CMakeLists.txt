include_directories(${V3D_INCLUDE_DIRS} ${EXTRA_INC_DIRS})
link_directories(${V3D_DIR} ${EXTRA_LIB_DIRS} ${CUDA_DIR}/lib)
link_libraries (V3D ${EXTRA_LIBRARIES} cudart)

#set (CMAKE_BUILD_TYPE Debug)

#set (NVCC nvcc)
#set (NVCC_ARGS --compiler-bindir=/home/chzach/tmp/gcc-4.3 -I${CMAKE_CURRENT_SOURCE_DIR} -DV3DLIB_ENABLE_CUDA -c --use_fast_math --maxrregcount 16 -O2)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_denoise_tvl1_kernel.${OBJEXT}
#                   COMMAND ${NVCC} ARGS -c -keep --use_fast_math --maxrregcount 16 -O2 ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_tvl1_kernel.cu
                   COMMAND ${NVCC} ARGS ${NVCC_ARGS} ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_tvl1_kernel.cu
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_tvl1_kernel.cu)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_denoise_tvl1_kernel_tex1d.${OBJEXT}
#                   COMMAND ${NVCC} ARGS -c -keep --use_fast_math --maxrregcount 16 -O2 ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_tvl1_kernel_tex1d.cu
                   COMMAND ${NVCC} ARGS ${NVCC_ARGS} ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_tvl1_kernel_tex1d.cu
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_tvl1_kernel_tex1d.cu)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_denoise_tvl2_kernel.${OBJEXT}
#                   COMMAND ${NVCC} ARGS -c -keep --maxrregcount 16 -O3 ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_tvl2_kernel.cu
                   COMMAND ${NVCC} ARGS ${NVCC_ARGS} ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_tvl2_kernel.cu
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_tvl2_kernel.cu)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_denoise_maxflow_kernel.${OBJEXT}
#                   COMMAND ${NVCC} ARGS -c -keep --maxrregcount 16 -O3 ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_maxflow_kernel.cu
                   COMMAND ${NVCC} ARGS ${NVCC_ARGS} ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_maxflow_kernel.cu
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_denoise_maxflow_kernel.cu)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_tvl1_flow_kernel.${OBJEXT}
#                   COMMAND ${NVCC} ARGS -c -keep --maxrregcount 16 -O3 ${CMAKE_CURRENT_SOURCE_DIR}/cuda_tvl1_flow_kernel.cu
                   COMMAND ${NVCC} ARGS ${NVCC_ARGS} ${CMAKE_CURRENT_SOURCE_DIR}/cuda_tvl1_flow_kernel.cu
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_tvl1_flow_kernel.cu)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_tvl1_flow_bregman_kernel.${OBJEXT}
#                   COMMAND ${NVCC} ARGS -c -keep --maxrregcount 16 -O3 ${CMAKE_CURRENT_SOURCE_DIR}/cuda_tvl1_flow_bregman_kernel.cu
                   COMMAND ${NVCC} ARGS ${NVCC_ARGS} ${CMAKE_CURRENT_SOURCE_DIR}/cuda_tvl1_flow_bregman_kernel.cu
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_tvl1_flow_bregman_kernel.cu)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_cv_segmentation_kernel.${OBJEXT}
                   COMMAND ${NVCC} ARGS ${NVCC_ARGS} ${CMAKE_CURRENT_SOURCE_DIR}/cuda_cv_segmentation_kernel.cu
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_cv_segmentation_kernel.cu)

add_v3d_executable (cuda_denoise_tvl1 cuda_denoise_tvl1.cpp ${CMAKE_CURRENT_BINARY_DIR}/cuda_denoise_tvl1_kernel.${OBJEXT})
add_v3d_executable (cuda_denoise_tvl1_tex1d cuda_denoise_tvl1_tex1d.cpp ${CMAKE_CURRENT_BINARY_DIR}/cuda_denoise_tvl1_kernel_tex1d.${OBJEXT})
add_v3d_executable (cuda_denoise_tvl2 cuda_denoise_tvl2.cpp ${CMAKE_CURRENT_BINARY_DIR}/cuda_denoise_tvl2_kernel.${OBJEXT})

#add_v3d_executable (cuda_denoise_maxflow cuda_denoise_maxflow.cpp ${CMAKE_CURRENT_BINARY_DIR}/cuda_denoise_maxflow_kernel.${OBJEXT})
add_v3d_executable (cuda_tvl1_flow cuda_tvl1_flow.cpp ${CMAKE_CURRENT_BINARY_DIR}/cuda_tvl1_flow_kernel.${OBJEXT})

add_v3d_executable (cuda_tvl1_flow_bregman cuda_tvl1_flow_bregman.cpp ${CMAKE_CURRENT_BINARY_DIR}/cuda_tvl1_flow_bregman_kernel.${OBJEXT})

add_v3d_executable (cuda_cv_segmentation cuda_cv_segmentation.cpp ${CMAKE_CURRENT_BINARY_DIR}/cuda_cv_segmentation_kernel.${OBJEXT})
#add_v3d_executable (cuda_cvl1_segmentation cuda_cvl1_segmentation.cpp ${CMAKE_CURRENT_BINARY_DIR}/cuda_cv_segmentation_kernel.${OBJEXT})
