//
// Created by bene on 2020/5/25.
//

#ifndef ORB_SIFT_HOMODECOMP_H
#define ORB_SIFT_HOMODECOMP_H
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>
#include <vector>

#include "../include/Frame.h"


namespace ORB_SIFT {
    class HomoDecomp{
        typedef pair<int,int> Match;

    public:
        //HomoDecomp();
        HomoDecomp(cv::Mat &K,Frame &LastFrame,Frame &CurrentFrame,vector<int> &vnMatches12,int maxIterations);
        void eliminateWrongMatch();
        void GenerateSets();
        void DecompHomography(float &score, cv::Mat &R21, cv::Mat &t21);

        cv::Mat ComputeH21(const vector <cv::Point2f> &vP1, const vector <cv::Point2f> &vP2);
    protected:
        void Normalize(const vector <cv::KeyPoint> &vKeys, vector <cv::Point2f> &vNormalizedPoints, cv::Mat &T);

        bool ReconstructH(cv::Mat &H21, cv::Mat &K, cv::Mat &R21, cv::Mat &t21, cv::Mat &n1,vector <cv::Point2f> vPn1i);
/*
        int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector <cv::KeyPoint> &vKeys1,
                    const vector <cv::KeyPoint> &vKeys2,
                    const vector <Match> &vMatches12,
                    const cv::Mat &K, vector <cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);
*/
        int CheckRT(vector <Match> vMatches12, cv::Mat R, cv::Mat N,float d);
        bool CheckVisibility(cv::Mat &H21,vector<cv::Point2f> vPn1i,float di);

        void
        Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

        void Triangulate(const cv::Point &pt1, const cv::Point &pt2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

        bool isCoplanar(vector <cv::Mat> &Points3D, cv::Mat &normal);

    public:
        //Data From Test
        Frame mLastFrame;
        Frame mCurrentFrame;
        vector<int> mvnMatches12;
        cv::Mat mK;
    protected:
        //self data
        vector<Match> mPairMatch12;
        vector<bool> mvbMatched1;

        int mMaxIterations;
        // Ransac sets
        vector<vector<size_t> > mvSets;
    };

}
#endif //ORB_SIFT_HOMODECOMP_H
