# Point Cloud Segmentation and Classification

**AIM:** To classify and segment point clouds using PCL and TensorFlow.

## Description
This project involves two main steps: Segmentation and Classification.
1. **Segmentation:** Implemented by calculating Difference of Normal (DoN) features using the Point Cloud Library (PCL).
2. **Classification:** Implemented using a Convolutional Neural Network (CNN) based on the PointNet architecture (training scripts not included in this repository).

## Repository Structure
- `src/`: Contains the C++ source files for point cloud processing.
  - `main.cpp`: The unified executable that processes a CAD model through mesh sampling and DoN segmentation.
  - `mesh_sampling.cpp` / `.h`: Uniformly samples a CAD model (PLY/OBJ) to produce a dense point cloud and applies a VoxelGrid filter.
  - `don_segmentation.cpp` / `.h`: Computes Difference of Normal features, filters by magnitude, and extracts clusters.
  - `voxel_grid.cpp`: Standalone executable for VoxelGrid downsampling.
- `scripts/`: Python and shell scripts for data preparation.
  - `helper_project.py`: Helper scripts for data format conversion (e.g., PCD to HDF5) and renaming.
  - `off2numpy.py`: Script to convert OFF meshes to numpy arrays.
  - `convert2ply.sh`: Bash script to batch convert CAD models to PLY format.
- `CMakeLists.txt`: Root CMake configuration file.

## Prerequisites
To build the C++ pipeline, you need:
- [CMake](https://cmake.org/) (>= 2.8)
- [PCL](http://pointclouds.org/) (Point Cloud Library) 1.7 or higher
- [VTK](https://vtk.org/) (Required by PCL for mesh sampling)

## Building the Pipeline

From the root of the repository, run:

```bash
mkdir build
cd build
cmake ..
make
```

This will produce the `pipeline` executable.

## Usage

Run the unified segmentation pipeline on a CAD model:

```bash
./pipeline <input_model.ply/.obj> <n_samples> <leaf_size> <small_scale> <large_scale> <threshold> <segradius>
```

**Example:**
```bash
./pipeline car.ply 100000 0.01 0.1 1.0 0.1 0.2
```

### Steps Performed by `pipeline`:
1. **Mesh Sampling:** The input `car.ply` is uniformly sampled with `100,000` points and downsampled using a VoxelGrid leaf size of `0.01`.
2. **DoN Segmentation:** The sampled point cloud is processed using DoN with a small scale of `0.1` and a large scale of `1.0`. Features with magnitude less than `0.1` are filtered out, and the remaining points are clustered using Euclidean clustering with a tolerance of `0.2`.

Intermediate files like `don.pcd`, `don_filtered.pcd`, and `don_cluster_*.pcd` will be saved in your working directory.

## Step by step process:
- [x] Point cloud Segmentation
- [x] Batch convert CAD models to .ply
- [x] Downsampling point clouds using Meshsampling
- [x] Convert PCD to h5 
- [x] Input data normalization
- [x] Data Augmentation (Rotation and scaling)
- [ ] Train scripts
- [ ] Others to follow
