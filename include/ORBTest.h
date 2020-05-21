//
// Created by bene on 2020/5/11.
//

#ifndef ORB_SIFT_ORBTEST_H
#define ORB_SIFT_ORBTEST_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>

#include <vector>

#include "../include/ORBextractor.h"
#include "../include/Frame.h"
#include "../include/ORBmatcher.h"

namespace ORB_SIFT{
    class ORBTest{
        typedef pair<int,int> Match;
    public:
        ORBTest();
        ORBTest(std::string strSettingPath);
        ~ORBTest(){delete mpORBextractor;}

        void GrabImage(const cv::Mat &img, const double &timestamp);

        void SaveResult(const string& SaveFileName);
    protected:
        void Compute_HW_ROI();
        void WarpROI();
        void DrawMatches();
        void Shift_Keys_From_ROI_To_Origin();


        void DrawROI(cv::Mat& image);

        void ORBMatch() ;
        void findHomography();
        //-----
        void GenerateSets();
        void findHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
        cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
        float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);
        void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
        void CopyKeys() ;
    public:
        //camera parameter
        cv::Mat mK;
        cv::Mat mDistCoef;
        float mbf;
        bool mbRGB;
        ORBextractor* mpORBextractor;

        std::vector<cv::KeyPoint> Last_mvKeysROI;   //1
        std::vector<cv::KeyPoint> Curr_mvKeysROI;   //2
        cv::Mat mDescriptors;

        Frame mCurrentFrame;
        Frame mLastFrame;
        vector<int> vnMatches12;
        int mMatches;
        vector<Match> mvMatches12;  //Match=pair
        vector<bool> mvbMatched1;

        cv::Mat mCurrentImg;
        cv::Mat mLastImg;
        cv::Mat mROI_Img;

        cv::Rect mROI;

        bool mbFirstImg;
    protected:
        cv::Point mROIOrigin;    //兴趣区域的原点在原图中坐标
        int mImg_HEIGHT;
        int mImg_WIDTH;

        int mMaxIterations;
        // Ransac sets
        vector<vector<size_t> > mvSets;
        // Standard Deviation and Variance
        float mSigma, mSigma2;


    };
}

#endif //ORB_SIFT_ORBTEST_H
