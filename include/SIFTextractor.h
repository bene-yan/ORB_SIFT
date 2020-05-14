//
// Created by bene on 2020/5/13.
//

#ifndef ORB_SIFT_SIFTEXTRACTOR_H
#define ORB_SIFT_SIFTEXTRACTOR_H

#include <iostream>
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <vector>
namespace ORB_SIFT{
    class SIFTextractor{
        //SIFTextractor(int nfeatures, float scaleFactor, int nlevels,int iniThFAST, int minThFAST);
    public:
        SIFTextractor();
        //~SIFTextractor(){delete  sift_detector;}  //这个析构会发生段错误，难道是sift_detector已经销毁了？
        void Extract_SIFT(const cv::Mat &im);
        void Shift_Keys_From_ROI_To_Origin();
        void GetROIOrigin(cv::Rect roi);
    protected:
        cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> sift_detector;
    public:
        std::vector<cv::KeyPoint> mSift_keys;
        cv::Point mROIOrigin;    //兴趣区域的原点在原图中坐标



    };
}

#endif //ORB_SIFT_SIFTEXTRACTOR_H
