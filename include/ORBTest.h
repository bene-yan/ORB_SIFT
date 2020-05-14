//
// Created by bene on 2020/5/11.
//

#ifndef ORB_SIFT_ORBTEST_H
#define ORB_SIFT_ORBTEST_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>

#include <vector>

#include"ORBextractor.h"

namespace ORB_SIFT{
    class ORBTest{
    public:
        ORBTest();
        ORBTest(std::string strSettingPath);
        ~ORBTest(){delete mpORBextractor;}
        void Extract_ORB(const cv::Mat &im);
        void Shift_Keys_From_ROI_To_Origin();
        void GetROIOrigin(cv::Rect roi);

        void DrawROI(const cv::Mat& image,
                const double lower_row,const double middle_col,
                 cv::Mat& ROIimage);  //

        //camera parameter
        cv::Mat mK;
        cv::Mat mDistCoef;
        float mbf;
        bool mbRGB;
        ORBextractor* mpORBextractor;

        std::vector<cv::KeyPoint> mvKeys;
        cv::Mat mDescriptors;

        cv::Point mROIOrigin;    //兴趣区域的原点在原图中坐标



    };
}

#endif //ORB_SIFT_ORBTEST_H
