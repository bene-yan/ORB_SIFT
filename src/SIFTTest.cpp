//
// Created by bene on 2020/5/13.
//

#include "../include/SIFTTest.h"
namespace ORB_SIFT{
    SIFTTest::SIFTTest(){
        sift_detector=cv::xfeatures2d::SiftFeatureDetector::create();
    }
    SIFTTest::SIFTTest(std::string strSettingPath){
        // Load camera parameters from settings file

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        /*
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3,3,CV_32F);
        K.at<float>(0,0) = fx;
        K.at<float>(1,1) = fy;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4,1,CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if(k3!=0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if(fps==0)
            fps=30;

        // Max/Min Frames to insert keyframes and to check relocalisation
        //mMinFrames = 0;
        //mMaxFrames = fps;

        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if(DistCoef.rows==5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if(mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;
        */
        // Load SIFT parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        int nOctaveLayers=fSettings["SIFT.nOctaveLayers"];
        double contrastThreshold=fSettings["SIFT.contrastThreshold"]; //the larger the less feature
        double edgeThreshold=fSettings["SIFT.edgeThreshold"];    //the larger the more feature
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"]; //sigma
        int nLevels = fSettings["ORBextractor.nLevels"];

        //int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        //int fMinThFAST = fSettings["ORBextractor.minThFAST"];
        //TODO:tuna sift param
        sift_detector=cv::xfeatures2d::SiftFeatureDetector::create(nFeatures,
                nOctaveLayers,contrastThreshold,edgeThreshold,fScaleFactor);
    }

    void SIFTTest::Extract_SIFT(const cv::Mat &im){
        sift_detector->detect(im,mSift_keys);
        Shift_Keys_From_ROI_To_Origin();    //调用此函数前需要先调用GetROIOrigin
    }

    //调用此函数前需要先调用GetROIOrigin
    void SIFTTest::Shift_Keys_From_ROI_To_Origin()
    {
        size_t N=mSift_keys.size();
        for(size_t ni=0;ni<N;ni++)
        {
            mSift_keys[ni].pt.x+=mROIOrigin.x;
            mSift_keys[ni].pt.y+=mROIOrigin.y;
        }
    }

    void SIFTTest::GetROIOrigin(cv::Rect roi)
    {
        mROIOrigin=roi.tl();
    }

    //void SIFTTest::DrawFeatures(){}



}