#include "mesh_sampling.h"
#include "don_segmentation.h"
#include <iostream>
#include <cstdlib>

using namespace std;

int main(int argc, char** argv)
{
  if (argc < 8)
  {
    cerr << "Usage: " << argv[0] << " <input.ply/.obj> <n_samples> <leaf_size> <scale1> <scale2> <threshold> <segradius>" << endl;
    cerr << "Example: " << argv[0] << " model.ply 100000 0.01 0.1 1.0 0.1 0.2" << endl;
    return -1;
  }

  string input_mesh = argv[1];
  int n_samples = atoi(argv[2]);
  float leaf_size = atof(argv[3]);
  double scale1 = atof(argv[4]);
  double scale2 = atof(argv[5]);
  double threshold = atof(argv[6]);
  double segradius = atof(argv[7]);

  cout << "=== Step 1: Mesh Sampling ===" << endl;
  auto sampled_cloud = perform_mesh_sampling(input_mesh, n_samples, leaf_size, true, true);
  if (!sampled_cloud || sampled_cloud->empty())
  {
    cerr << "Mesh sampling failed or returned empty cloud!" << endl;
    return -1;
  }
  
  cout << "Sampled cloud has " << sampled_cloud->points.size() << " points." << endl;

  cout << "=== Step 2: DoN Segmentation ===" << endl;
  perform_don_segmentation(sampled_cloud, scale1, scale2, threshold, segradius);

  cout << "Pipeline completed successfully!" << endl;

  return 0;
}
