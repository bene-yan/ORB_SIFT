//
// Created by bene on 2020/5/13.
//

#ifndef ORB_SIFT_SIFTTEST_H
#define ORB_SIFT_SIFTTEST_H

#include <iostream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <vector>

#include "ORBTest.h"

namespace ORB_SIFT{
    class ORBTest;
    class SIFTTest: public ORBTest{
        //SIFTextractor(int nfeatures, float scaleFactor, int nlevels,int iniThFAST, int minThFAST);
    public:
        SIFTTest();
        SIFTTest(std::string strSettingPath);
        //~SIFTextractor(){delete  sift_detector;}  //这个析构会发生段错误，难道是sift_detector已经销毁了？
        void GrabImage_sift(const cv::Mat &img, const double &timestamp);
        void Extract_SIFT(const cv::Mat &im);
        void CullingZeroLevel();
        void SIFTMatch();
        void UpdateLast(const cv::Mat& img);
        void FindHomography();
        //void Shift_Keys_From_ROI_To_Origin();
        //void GetROIOrigin(cv::Rect roi);
        //void DrawFeatures();
    protected:
        cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> sift_detector;
        cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> sift_Descriptor;
        cv::Ptr<cv::BFMatcher> sift_matcher;

        cv::Mat mDescriptors_Curr;
        cv::Mat mDescriptors_Last;
        vector<cv::DMatch> mMatches;

    public:
        std::vector<cv::KeyPoint> mvKeysROI_0_Last;
        std::vector<cv::KeyPoint> mvKeysROI_0_Curr;
        std::vector<cv::KeyPoint> mSift_keys;
        cv::Point mROIOrigin;    //兴趣区域的原点在原图中坐标



    };
}

#endif //ORB_SIFT_SIFTTEST_H
