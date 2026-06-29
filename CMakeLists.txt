cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(PointCloudPipeline)

find_package(PCL 1.7 REQUIRED)
find_package(VTK REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pipeline src/main.cpp src/mesh_sampling.cpp src/don_segmentation.cpp)
target_link_libraries(pipeline ${PCL_LIBRARIES} ${VTK_LIBRARIES})

add_executable(voxel_grid src/voxel_grid.cpp)
target_link_libraries(voxel_grid ${PCL_LIBRARIES})
