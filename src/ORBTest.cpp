//
// Created by bene on 2020/5/11.
//

#include "ORBextractor.h"
//#include "ORBmatcher.h"

#include <iostream>
#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ORBTest.h"

using namespace std;
using namespace cv;
namespace ORB_SIFT{

    ORBTest::ORBTest(std::string strSettingPath)
    {
        // Load camera parameters from settings file

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
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

        // Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractor=new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    }

    void ORBTest::Extract_ORB(const cv::Mat &im)
    {
        //cv::Mat image_ROI;
        //DrawROI(im,0.333333,0.2,image_ROI);
        (*mpORBextractor)(im,cv::Mat(),mvKeys,mDescriptors);
        Shift_Keys_From_ROI_To_Origin();

    }
    void ORBTest::Shift_Keys_From_ROI_To_Origin()
    {
        size_t N=mvKeys.size();
        for(size_t ni=0;ni<N;ni++)
        {
            mvKeys[ni].pt.x+=mROIOrigin.x;
            mvKeys[ni].pt.y+=mROIOrigin.y;
        }
    }
    void ORBTest::GetROIOrigin(cv::Rect roi)
    {
        mROIOrigin=roi.tl();
    }

    //cv::Mat DrawROI(cv::InputArray image,const int start_col,const int start_row, const int width,const int height)
    void ORBTest::DrawROI(const cv::Mat& image,
            const double lower_row,const double middle_col,
             cv::Mat& ROIimage)
    {
        //cv::Mat im_Mat=image.getMat();
        //cout<<lower_row<<" "<<middle_col<<endl;
        //cout<<"Colums:"<<image.cols<<" Rows:"<<image.rows<<endl;
        const int x=(0.5-0.5*middle_col)*(image.cols);  //起始列
        const int y=(1-lower_row)*(image.rows); //起始行
        const int width=middle_col*(image.cols);
        const int height=lower_row*(image.rows);
        //cout<<"x:"<<x<<" y:"<<y<<" w:"<<width<<" h:"<<height<<endl;
        cv::Mat LabelROI_im=image.clone();
        rectangle(LabelROI_im,Rect(x,y,width,height),Scalar(255,0,0),2);
        cv::imshow("ROIsrc",LabelROI_im);
        ROIimage=image(Rect(x,y,width,height));//.clone(); //这样可能并没有发生复制，只是ROIimage指向了ROI区域？
        //cv::imshow("ROI",ROIimage);
        //imwrite("../ROI_image/");
        waitKey();
    }

}

