//
// Created by bene on 22/06/2020 .
//


#include <iostream>
#include <cmath>
#include <algorithm>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


  int _vertical_scans;
  int _horizontal_scans;
  float _ang_bottom;
  float _ang_resolution_X;
  float _ang_resolution_Y;
  float _segment_alpha_X;
  float _segment_alpha_Y;
  float _segment_theta;
  int _segment_valid_point_num;
  int _segment_valid_line_num;
  int _ground_scan_index;
  float _sensor_mount_angle;
