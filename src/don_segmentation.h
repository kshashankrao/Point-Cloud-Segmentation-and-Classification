#ifndef DON_SEGMENTATION_H
#define DON_SEGMENTATION_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

void perform_don_segmentation(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud, 
                              double scale1, double scale2, double threshold, double segradius);

#endif // DON_SEGMENTATION_H
