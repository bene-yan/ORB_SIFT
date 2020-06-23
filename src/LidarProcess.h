//
// Created by bene on 22/06/2020 .
//


#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core/core.hpp>

#include <Eigen/QR>

#include "ORBTest.h"
namespace lego_loam{

	using namespace ORB_SIFT;
	const double DEG_TO_RAD = M_PI / 180.0;
	const double RAD_TO_DEG = 180.0/M_PI;
  
  	void resetParameters();
  	void getParamFromYAML(const std::string yamlFilePath);
  	void test();
  	void projectPointCloud(pcl::PointCloud<PointType>::Ptr &_laser_cloud_in,pcl::PointCloud<PointType>::Ptr &_full_cloud);
  	void groundRemoval(pcl::PointCloud<PointType>::Ptr &_full_cloud,pcl::PointCloud<PointType>::Ptr &_ground_cloud);
  }
