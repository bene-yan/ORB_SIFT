//
// Created by bene on 2020/5/13.
//

#include "../include/SIFTTest.h"
using namespace std;
namespace ORB_SIFT{
    SIFTTest::SIFTTest(){
        sift_detector=cv::xfeatures2d::SiftFeatureDetector::create();
        sift_Descriptor=cv::xfeatures2d::SiftDescriptorExtractor::create();
        sift_matcher=cv::BFMatcher::create();
    }
    SIFTTest::SIFTTest(std::string strSettingPath){
        // Load camera parameters from settings file
        mbFirstImg=true;
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

        sift_Descriptor=cv::xfeatures2d::SiftDescriptorExtractor::create();

        sift_matcher=cv::BFMatcher::create(cv::NORM_L2,true);
    }

    void SIFTTest::GrabImage_sift(const cv::Mat &img, const double &timestamp) {

        UpdateLast(img);
        if(mbFirstImg)
        {
            Compute_HW_ROI();
            mbFirstImg=false;
        }
        WarpROI();


        Extract_SIFT(mROI_Img); //提取sift特征保存到Curr_mvKeysROI //并计算描述子

        if(Last_mvKeysROI.size()>0)
        {
            SIFTMatch();
            FindHomography();
        }




    }

    void SIFTTest::Extract_SIFT(const cv::Mat &im){
        sift_detector->detect(im,Curr_mvKeysROI);
        //CullingZeroLevel();
        sift_Descriptor->compute(mROI_Img,Curr_mvKeysROI,mDescriptors_Curr);
        //sift_Descriptor->compute(mROI_Img,mvKeysROI_0_Curr,mDescriptors_Curr);

        Shift_Keys_From_ROI_To_Origin();
    }
    void SIFTTest::CullingZeroLevel()
    {
        for(size_t i=0,N=Curr_mvKeysROI.size();i<N;i++)
        {
            int level=Curr_mvKeysROI[i].octave;
            if(level>0)
                continue;
            mvKeysROI_0_Curr.push_back(Curr_mvKeysROI[i]);
        }
    }

    void SIFTTest::SIFTMatch()
    {

        sift_matcher->match(mDescriptors_Last,mDescriptors_Curr,mMatches);

        cout<<mMatches.size()<<" sift matches."<<endl;

        cv::Mat img_matches;
        cv::drawMatches(mLastImg,Last_mvKeysROI,mCurrentImg,Curr_mvKeysROI,
                        mMatches,img_matches,
                        cv::Scalar::all(-1),cv::Scalar(0,0,255));
        //cv::drawMatches(mLastImg,mvKeysROI_0_Last,mCurrentImg,mvKeysROI_0_Curr,
        //                        mMatches,img_matches);


        cv::imshow("sift_matches",img_matches);
        cv::waitKey();
    }
    void SIFTTest::FindHomography()
    {
        vector<cv::Point2f> Keypoints_last;
        vector<cv::Point2f> Keypoints_curr;
        for(int i=0;i<(int)mMatches.size();i++)
        {
            Keypoints_last.push_back(Last_mvKeysROI[mMatches[i].queryIdx].pt);
            Keypoints_curr.push_back(Curr_mvKeysROI[mMatches[i].trainIdx].pt);
        }
        cv::Mat H_sift=cv::findHomography(Keypoints_last,Keypoints_curr,cv::RANSAC);
        cout<<cv::format(H_sift,cv::Formatter::FMT_C)<<";"<<endl;
    }

    void SIFTTest::UpdateLast(const cv::Mat& img)
    {
        mLastImg=mCurrentImg;
        mCurrentImg=img;
        Last_mvKeysROI=Curr_mvKeysROI;
        mDescriptors_Last=mDescriptors_Curr;
        mvKeysROI_0_Last=mvKeysROI_0_Curr;

    }
/*
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
*/
    //void SIFTTest::DrawFeatures(){}



}