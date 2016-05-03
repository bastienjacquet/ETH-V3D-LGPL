cmake_minimum_required(VERSION 2.6)

project(V3D)

set (V3D_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set (V3D_INCLUDE_DIRS ${V3D_DIR}/.)

include (Config/v3d_macros.cmake)

include (Config/local_config.cmake)

include_directories(${V3D_INCLUDE_DIRS} ${EXTRA_INC_DIRS})


source_group("Base" REGULAR_EXPRESSION Base/.*cpp|Base.*h)
set (BASE_SRC
    Base/v3d_config.h
    Base/v3d_config.cpp
    Base/v3d_database.h
    Base/v3d_database.cpp
    Base/v3d_exception.h
    Base/v3d_image.h
    Base/v3d_image.cpp
    Base/v3d_imageprocessing.h
    Base/v3d_storage.h
    Base/v3d_storage.cpp
    Base/v3d_timer.h
    Base/v3d_exifreader.h
    Base/v3d_exifreader.cpp
    Base/v3d_utilities.h
    Base/v3d_utilities.cpp
    Base/v3d_videoio.h
    Base/v3d_videoio.cpp
)
if (V3DLIB_ENABLE_SOCKETS)
    set (BASE_SRC ${BASE_SRC} Base/v3d_socket.h)
endif (V3DLIB_ENABLE_SOCKETS)

source_group("Math" REGULAR_EXPRESSION Math/.*cpp|Math.*h)
set (MATH_SRC
    Math/v3d_linear.h
    Math/v3d_linearbase.h
    Math/v3d_mathutilities.h
    Math/v3d_mathutilities.cpp
    Math/v3d_nonlinlsq.h
    Math/v3d_nonlinlsq.cpp
    Math/v3d_optimization.h
    Math/v3d_optimization.cpp
    Math/v3d_optimization_lp.cpp
    Math/v3d_optimization_lm.cpp
    Math/v3d_sparseeig.h
    Math/v3d_sparseeig.cpp
    Math/v3d_sparsechol.h
    Math/v3d_sparsechol.cpp
)

source_group("Geometry" REGULAR_EXPRESSION Geometry/.*cpp|Geometry.*h)
set (GEOMETRY_SRC
    Geometry/v3d_cameramatrix.h
    Geometry/v3d_homography.h
    Geometry/v3d_homography.cpp
    Geometry/v3d_metricbundle.h
    Geometry/v3d_metricbundle.cpp
    Geometry/v3d_mviewutilities.h
    Geometry/v3d_mviewutilities.cpp
    Geometry/v3d_mviewutilities_rotations.cpp
    Geometry/v3d_poseutilities.h
    Geometry/v3d_poseutilities.cpp
#    Geometry/v3d_poseutilities_3plus1pt.cpp
    Geometry/v3d_poseutilities_2_1plus1pt.cpp
    Geometry/v3d_selfcalibration.h
    Geometry/v3d_selfcalibration.cpp
    Geometry/v3d_stereobundle.h
    Geometry/v3d_stereobundle.cpp
    Geometry/v3d_tripletutilities.h
    Geometry/v3d_tripletutilities.cpp
)
if (V3DLIB_ENABLE_ARPACK)
    set (GEOMETRY_SRC ${GEOMETRY_SRC}
        Geometry/v3d_mviewinitialization.h
        Geometry/v3d_mviewinitialization.cpp
        Geometry/v3d_mviewinitialization_huber.cpp
    )
endif (V3DLIB_ENABLE_ARPACK)

source_group("GL" REGULAR_EXPRESSION GL/.*cpp|GL/.*h)
if (V3DLIB_ENABLE_GPGPU)
set (GL_SRC
    GL/v3d_gpubase.h
    GL/v3d_gpubase.cpp
    GL/v3d_gpubinarysegmentation.h
    GL/v3d_gpubinarysegmentation.cpp
    GL/v3d_gpuconvexmrf.h
    GL/v3d_gpuconvexmrf.cpp
    GL/v3d_gpuflow.h
    GL/v3d_gpuflow.cpp
    GL/v3d_gpucolorflow.h
    GL/v3d_gpucolorflow.cpp
    GL/v3d_gpupyramid.h
    GL/v3d_gpupyramid.cpp
    GL/v3d_gpuklt.h
    GL/v3d_gpuklt.cpp
    GL/v3d_gpupottslabeling.h
    GL/v3d_gpupottslabeling.cpp
    GL/v3d_gpustereoklt.h
    GL/v3d_gpustereoklt.cpp
    GL/v3d_gpuundistort.h
    GL/v3d_gpuundistort.cpp
    GL/v3d_gpuvarstereo.h
    GL/v3d_gpuvarstereo.cpp
)
else (V3DLIB_ENABLE_GPGPU)
set (GL_SRC  )
endif (V3DLIB_ENABLE_GPGPU)

source_group("CUDA" REGULAR_EXPRESSION CUDA/.*cpp|CUDA/.*h|CUDA/.*cu)
if (V3DLIB_ENABLE_CUDA)
set (CUDA_SRC
    CUDA/v3d_cudamatching.h
    CUDA/v3d_cudamatching.cu
    CUDA/v3d_cudaflow.h
    CUDA/v3d_cudaflow.cu
    CUDA/v3d_cudasegmentation.h
    CUDA/v3d_cudasegmentation.cu
)
set (CUDA_OUTPUTS
    ${CMAKE_CURRENT_BINARY_DIR}/CUDA/v3d_cudamatching.${OBJEXT}
    ${CMAKE_CURRENT_BINARY_DIR}/CUDA/v3d_cudaflow.${OBJEXT}
    ${CMAKE_CURRENT_BINARY_DIR}/CUDA/v3d_cudasegmentation.${OBJEXT}
)
else (V3DLIB_ENABLE_CUDA)
   set (CUDA_SRC )
endif (V3DLIB_ENABLE_CUDA)

source_group("GL/Shaders" REGULAR_EXPRESSION GL/Shaders.*cg)

set (ALL_SRC
    ${BASE_SRC} ${MATH_SRC} ${GEOMETRY_SRC} ${GL_SRC} ${CUDA_SRC}
)

if (V3DLIB_ENABLE_CUDA)
   set (NVCC nvcc)
#   set (NVCC_ARGS -I${CMAKE_CURRENT_SOURCE_DIR} -DV3DLIB_ENABLE_CUDA -c --use_fast_math --maxrregcount 16 -O2)
#   set (NVCC_ARGS -I${CMAKE_CURRENT_SOURCE_DIR} -DV3DLIB_ENABLE_CUDA -c)
#   set (NVCC_ARGS --compiler-bindir=/home/chzach/tmp/gcc-4.3 -I${CMAKE_CURRENT_SOURCE_DIR} -DV3DLIB_ENABLE_CUDA -c --use_fast_math --maxrregcount 16 -O2)
   set (NVCC_ARGS -I${CMAKE_CURRENT_SOURCE_DIR} -DV3DLIB_ENABLE_CUDA -c --use_fast_math -O2)
#   set (NVCC_ARGS -I${CMAKE_CURRENT_SOURCE_DIR} -DV3DLIB_ENABLE_CUDA -c --use_fast_math --maxrregcount 16 -O2 --keep)

   add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/CUDA/v3d_cudamatching.${OBJEXT}
                      COMMAND ${NVCC} ARGS ${NVCC_ARGS} ${EXTRA_NVCC_ARGS} -o ${CMAKE_CURRENT_BINARY_DIR}/CUDA/v3d_cudamatching.${OBJEXT} ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/v3d_cudamatching.cu
                      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/v3d_cudamatching.cu)

   add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/CUDA/v3d_cudaflow.${OBJEXT}
                      COMMAND ${NVCC} ARGS ${NVCC_ARGS} ${EXTRA_NVCC_ARGS} -o ${CMAKE_CURRENT_BINARY_DIR}/CUDA/v3d_cudaflow.${OBJEXT} ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/v3d_cudaflow.cu
                      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/v3d_cudaflow.cu)

   add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/CUDA/v3d_cudasegmentation.${OBJEXT}
                      COMMAND ${NVCC} ARGS ${NVCC_ARGS} ${EXTRA_NVCC_ARGS} -o ${CMAKE_CURRENT_BINARY_DIR}/CUDA/v3d_cudasegmentation.${OBJEXT} ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/v3d_cudasegmentation.cu
                      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/v3d_cudasegmentation.cu)

   add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/CUDA/v3d_verticalfusion.${OBJEXT}
                      COMMAND ${NVCC} ARGS ${NVCC_ARGS} ${EXTRA_NVCC_ARGS} -o ${CMAKE_CURRENT_BINARY_DIR}/CUDA/v3d_verticalfusion.${OBJEXT} ${CMAKE_CURRENT_SOURCE_DIR}/Model/v3d_verticalfusion.cu
                      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Model/v3d_verticalfusion.cu)
endif (V3DLIB_ENABLE_CUDA)

file (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CUDA)

add_library(V3D ${ALL_SRC} ${CUDA_OUTPUTS})
add_subdirectory(Apps)
