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
        void Extract_ORB(const cv::Mat &im);
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



    };
}

#endif //ORB_SIFT_ORBTEST_H
