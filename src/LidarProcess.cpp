#include "LidarProcess.h"

namespace lego_loam{
using namespace std;

  int _vertical_scans;
  int _horizontal_scans;
  float _ang_bottom;
  float vertical_angle_top;
  float _ang_resolution_X;
  float _ang_resolution_Y;
  int _ground_scan_index;
  float _sensor_mount_angle;
  //Eigen::MatrixXf _range_mat;   // range matrix for range image
  Eigen::MatrixXi _label_mat;   // label matrix for segmentaiton marking
  Eigen::Matrix<int8_t,Eigen::Dynamic,Eigen::Dynamic> _ground_mat;  // ground matrix for ground cloud marking
  
  //pcl::PointCloud<PointType>::Ptr _full_cloud;
  //pcl::PointCloud<PointType>::Ptr _full_info_cloud;

  //pcl::PointCloud<PointType>::Ptr _ground_cloud;
  
void getParamFromYAML(const string yamlFilePath)
{
	//use OpenCV
	cv::FileStorage loam_Param(yamlFilePath, cv::FileStorage::READ);
	//_vertical_scans=loam_Param["lego_loam"]["laser"]["num_vertical_scans"];
	//_ang_bottom=loam_Param["lego_loam"]["laser"]["vertical_angle_bottom"];
	loam_Param["lego_loam.laser.num_vertical_scans"]>>_vertical_scans;
	loam_Param["lego_loam.laser.num_horizontal_scans"]>>_horizontal_scans;
	loam_Param["lego_loam.laser.vertical_angle_bottom"]>>_ang_bottom;
	loam_Param["lego_loam.laser.ground_scan_index"]>>_ground_scan_index;
	loam_Param["lego_loam.laser.sensor_mount_angle"]>>_sensor_mount_angle;
	loam_Param["lego_loam.laser.vertical_angle_top"]>>vertical_angle_top;
	
	_ang_resolution_X = (M_PI*2) / (_horizontal_scans);
	_ang_resolution_Y = DEG_TO_RAD*(vertical_angle_top - _ang_bottom) / float(_vertical_scans-1);
	_ang_bottom = -( _ang_bottom - 0.1) * DEG_TO_RAD;
	
	
}
void test()
{
	cout<<"_vertical_scans:"<<_vertical_scans<<endl;
	cout<<"_horizontal_scans:"<<_horizontal_scans<<endl;
	cout<<"_ang_bottom:"<<_ang_bottom*RAD_TO_DEG<<endl;
	cout<<"vertical_angle_top:"<<vertical_angle_top<<endl;
	cout<<"_ang_resolution_X:"<<_ang_resolution_X*RAD_TO_DEG<<endl;
	cout<<"_ang_resolution_Y:"<<_ang_resolution_Y*RAD_TO_DEG<<endl;
	cout<<"_ground_scan_index:"<<_ground_scan_index<<endl;
	cout<<"_sensor_mount_angle:"<<_sensor_mount_angle<<endl;
	//cout<<"ground_mat:"<<endl<<_ground_mat<<endl;
	
}
void resetParameters() {
  //const size_t cloud_size = _vertical_scans * _horizontal_scans;

  //_range_mat.resize(_vertical_scans, _horizontal_scans);
  _ground_mat.resize(_vertical_scans, _horizontal_scans);
  //_label_mat.resize(_vertical_scans, _horizontal_scans);

  //_range_mat.fill(FLT_MAX);
  _ground_mat.setZero();
  //_label_mat.setZero();


}

void projectPointCloud(pcl::PointCloud<PointType>::Ptr &_laser_cloud_in,pcl::PointCloud<PointType>::Ptr &_full_cloud) {
	
	const size_t cloud_size = _vertical_scans * _horizontal_scans;
  	_full_cloud->resize(cloud_size);
  // range image projection
  const size_t cloudSize = _laser_cloud_in->points.size();

  for (size_t i = 0; i < cloudSize; ++i) {
    PointType thisPoint = _laser_cloud_in->points[i];
    
    float range = sqrt(thisPoint.x * thisPoint.x +
                       thisPoint.y * thisPoint.y +
                       thisPoint.z * thisPoint.z);

    // find the row and column index in the image for this point
    float verticalAngle = std::asin(thisPoint.z / range);
        //std::atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y));

    int rowIdn = (verticalAngle + _ang_bottom) / _ang_resolution_Y;
    if (rowIdn < 0 || rowIdn >= _vertical_scans) {
      continue;
    }

    float horizonAngle = std::atan2(thisPoint.x, thisPoint.y);

    int columnIdn = -round((horizonAngle - M_PI_2) / _ang_resolution_X) + _horizontal_scans * 0.5;

    if (columnIdn >= _horizontal_scans){
      columnIdn -= _horizontal_scans;
    }

    if (columnIdn < 0 || columnIdn >= _horizontal_scans){
      continue;
    }

    if (range < 0.1){
      continue;
    }

    //_range_mat(rowIdn, columnIdn) = range;

    //thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

    size_t index = columnIdn + rowIdn * _horizontal_scans;
    _full_cloud->points[index] = thisPoint;
    // the corresponding range of a point is saved as "intensity"
    //_full_info_cloud->points[index] = thisPoint;
    //_full_info_cloud->points[index].intensity = range;
  }
}

void groundRemoval(pcl::PointCloud<PointType>::Ptr &_full_cloud,pcl::PointCloud<PointType>::Ptr &_ground_cloud) {
  // _ground_mat
  // -1, no valid info to check if ground of not
  //  0, initial value, after validation, means not ground
  //  1, ground
  _ground_cloud->clear();
  for (size_t j = 0; j < _horizontal_scans; ++j) {
    for (size_t i = 0; i < _ground_scan_index; ++i) {
      size_t lowerInd = j + (i)*_horizontal_scans;
      size_t upperInd = j + (i + 1) * _horizontal_scans;

      if (_full_cloud->points[lowerInd].intensity <= 0 ||
          _full_cloud->points[upperInd].intensity <= 0) {
        // no info to check, invalid points
        _ground_mat(i, j) = -1;
        continue;
      }
      

      float dX =
          _full_cloud->points[upperInd].x - _full_cloud->points[lowerInd].x;
      float dY =
          _full_cloud->points[upperInd].y - _full_cloud->points[lowerInd].y;
      float dZ =
          _full_cloud->points[upperInd].z - _full_cloud->points[lowerInd].z;

      float vertical_angle = std::atan2(dZ , sqrt(dX * dX + dY * dY + dZ * dZ));

      // TODO: review this change

      if ( (vertical_angle - _sensor_mount_angle) <= 5 * DEG_TO_RAD) {
        _ground_mat(i, j) = 1;
        _ground_mat(i + 1, j) = 1;
      }
    }
  }
  /*
  // extract ground cloud (_ground_mat == 1)
  // mark entry that doesn't need to label (ground and invalid point) for
  // segmentation note that ground remove is from 0~_N_scan-1, need _range_mat
  // for mark label matrix for the 16th scan
  for (size_t i = 0; i < _vertical_scans; ++i) {
    for (size_t j = 0; j < _horizontal_scans; ++j) {
      if (_ground_mat(i, j) == 1 ||
          _range_mat(i, j) == FLT_MAX) {
        _label_mat(i, j) = -1;
      }
    }
  }
  */

  for (size_t i = 0; i <= _ground_scan_index; ++i) {
    for (size_t j = 0; j < _horizontal_scans; ++j) {
      if (_ground_mat(i, j) == 1)
        _ground_cloud->push_back(_full_cloud->points[j + i * _horizontal_scans]);
    }
  }
}

}
