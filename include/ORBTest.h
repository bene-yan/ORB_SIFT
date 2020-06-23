//
// Created by bene on 2020/5/11.
//

#ifndef ORB_SIFT_ORBTEST_H
#define ORB_SIFT_ORBTEST_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vector>

#include "../include/ORBextractor.h"
#include "../include/Frame.h"
#include "../include/ORBmatcher.h"
#include "../include/HomoDecomp.h"

namespace ORB_SIFT{

typedef pcl::PointXYZI PointType;

    class ORBTest{
        typedef pair<int,int> Match;
    public:
        ORBTest();
        ORBTest(std::string strSettingPath);
        ~ORBTest(){delete mpORBextractor;}

        void GrabImage(const cv::Mat &img, const double &timestamp
                ,pcl::PointCloud<PointType>::Ptr &orgin_cloud);

        void SaveResult(const string& SaveFileName);
    protected:
        void Compute_HW_ROI();
        void WarpROI();
        void DrawMatches();
        void Shift_Keys_From_ROI_To_Origin();


        void DrawROI(cv::Mat& image);

        void ORBMatch() ;
        //void eliminateWrongMatch();
        //void GroundFeatureByLidar(pcl::PointCloud<PointType>::Ptr &ground_cloud);
        void ProjectLidarCloud(pcl::PointCloud<PointType>::Ptr &ground_cloud
                                        ,vector<cv::Point2f> &validProjectionPoints);
        //-----

        void CopyKeys() ;
    public:
        //camera parameter
        cv::Mat mK;
        cv::Mat mDistCoef;
        cv::Mat mP0; //kitti相机0的投影矩阵
        float mbf;
        bool mbRGB;
        ORBextractor* mpORBextractor;
        //velodyne to Camera0 Transform
        cv::Mat mTr;

        vector<cv::Point2f> lidarProjPts;

        //仅用于展示，不可用于计算
        std::vector<cv::KeyPoint> Last_mvKeysROI;   //1
        std::vector<cv::KeyPoint> Curr_mvKeysROI;   //2
        cv::Mat mDescriptors;

        Frame mCurrentFrame;
        Frame mLastFrame;
        vector<int> vnMatches12;
        int mMatches;

        cv::Mat mCurrentImg;
        cv::Mat mLastImg;
        cv::Mat mROI_Img;

        cv::Rect mROI;
        double ROI_middle_col,ROI_lower_row;

        bool mbFirstImg;
    protected:
        cv::Point mROIOrigin;    //兴趣区域的原点在原图中坐标
        int mImg_HEIGHT;
        int mImg_WIDTH;


    };
}

#endif //ORB_SIFT_ORBTEST_H
