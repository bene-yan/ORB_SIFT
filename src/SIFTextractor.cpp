//
// Created by bene on 2020/5/13.
//

#include "../include/SIFTextractor.h"
namespace ORB_SIFT{
    SIFTextractor::SIFTextractor(){
        sift_detector=cv::xfeatures2d::SiftFeatureDetector::create();
    }

    void SIFTextractor::Extract_SIFT(const cv::Mat &im){
        sift_detector->detect(im,mSift_keys);
        Shift_Keys_From_ROI_To_Origin();    //调用此函数前需要先调用GetROIOrigin
    }

    //调用此函数前需要先调用GetROIOrigin
    void SIFTextractor::Shift_Keys_From_ROI_To_Origin()
    {
        size_t N=mSift_keys.size();
        for(size_t ni=0;ni<N;ni++)
        {
            mSift_keys[ni].pt.x+=mROIOrigin.x;
            mSift_keys[ni].pt.y+=mROIOrigin.y;
        }
    }

    void SIFTextractor::GetROIOrigin(cv::Rect roi)
    {
        mROIOrigin=roi.tl();
    }


}