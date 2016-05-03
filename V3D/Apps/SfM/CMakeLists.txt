project(Large-SfM)


include (${V3D_DIR}/Config/v3d_macros.cmake)
include (${V3D_DIR}/Config/local_config.cmake)

#link_libraries (V3D png jpeg colamd pthread dsdp lpsolve55 dl)
link_libraries (V3D png jpeg colamd pthread lpsolve55 dl)

set (CMAKE_BUILD_TYPE Release)
#set (CMAKE_BUILD_TYPE Debug)

add_library (reconstruction STATIC reconstruction_common.cpp reconstruction_common2.cpp)

add_v3d_executable (undistort_images undistort_images.cpp)
add_v3d_executable (undistort_images_nonstd undistort_images_nonstd.cpp)
add_v3d_executable (list_images_with_exif list_images_with_exif.cpp)
add_v3d_executable (generate_calibration_db generate_calibration_db.cpp)
add_v3d_executable (match_using_voctree match_using_voctree.cpp)
add_v3d_executable (match_sequence match_sequence.cpp)
add_v3d_executable (generate_triples generate_triples.cpp)
add_v3d_executable (generate_submodels generate_submodels.cpp)
add_v3d_executable (generate_submodels_L1 generate_submodels_L1.cpp)
add_v3d_executable (align_submodels align_submodels.cpp)
add_v3d_executable (reconstruct_from_submodels reconstruct_from_submodels.cpp)
add_v3d_executable (reconstruct_from_submodels_L1 reconstruct_from_submodels_L1.cpp)
add_v3d_executable (v3dmodel2wrl v3dmodel2wrl.cpp)
add_v3d_executable (reconstruct_from_EGs_Conic_L1 reconstruct_from_EGs_Conic_L1.cpp)
add_v3d_executable (reconstruct_from_EGs_L1 reconstruct_from_EGs_L1.cpp)

target_link_libraries(undistort_images V3D Cg CgGL glut GLEW)
target_link_libraries(undistort_images_nonstd V3D Cg CgGL glut GLEW)
target_link_libraries(match_using_voctree reconstruction V3D cublas png jpeg colamd sqlite3 arpack gfortran)
target_link_libraries(match_sequence reconstruction V3D cublas png jpeg colamd sqlite3 arpack gfortran)
target_link_libraries(generate_triples reconstruction V3D colamd sqlite3 arpack gfortran)
target_link_libraries(generate_submodels reconstruction V3D colamd sqlite3 arpack gfortran)
target_link_libraries(generate_submodels_L1 reconstruction V3D colamd sqlite3 arpack gfortran lpsolve55 dl)
target_link_libraries(align_submodels reconstruction V3D sqlite3)
target_link_libraries(reconstruct_from_submodels reconstruction V3D colamd sqlite3 arpack gfortran)
target_link_libraries(reconstruct_from_submodels_L1 reconstruction V3D colamd sqlite3 arpack gfortran)
target_link_libraries(reconstruct_from_EGs_Conic_L1 reconstruction V3D colamd sqlite3 arpack gfortran lpsolve55 dl)
target_link_libraries(reconstruct_from_EGs_L1 reconstruction V3D colamd sqlite3 arpack gfortran lpsolve55 dl)

if (V3DLIB_ENABLE_SIFTGPU)
  add_v3d_executable (extract_sift_gpu extract_sift_gpu.cpp)
  target_link_libraries(extract_sift_gpu siftgpu Cg CgGL glut IL siftgpu GLEW)
endif (V3DLIB_ENABLE_SIFTGPU)

if (V3DLIB_ENABLE_VLFEAT)
  add_v3d_executable_with_openmp (extract_sift_vlfeat extract_sift_vlfeat.cpp)
  target_link_libraries(extract_sift_vlfeat vl)
endif (V3DLIB_ENABLE_VLFEAT)

if(V3DLIB_ENABLE_OPENCV)
    FIND_PACKAGE( OpenCV REQUIRED )
    add_v3d_executable_with_openmp (extract_sift_opencv extract_sift_opencv.cpp)
    target_link_libraries(extract_sift_opencv ${OpenCV_LIBS})
endif(V3DLIB_ENABLE_OPENCV)

#target_link_libraries(verify_submodel_alignment reconstruction V3D sqlite3)

set_source_files_properties (reconstruct_from_EGs_Conic_L1.cpp PROPERTIES COMPILE_FLAGS -fopenmp)
set_target_properties (reconstruct_from_EGs_Conic_L1 PROPERTIES LINK_FLAGS -fopenmp)
