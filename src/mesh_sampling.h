#ifndef MESH_SAMPLING_H
#define MESH_SAMPLING_H

#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Returns a downsampled point cloud by uniformly sampling a CAD model (PLY/OBJ) and voxelizing it
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr
perform_mesh_sampling(const std::string& input_file, int n_samples, float leaf_size, bool write_normals, bool write_colors);

#endif // MESH_SAMPLING_H
