//
// Created by bene on 2020/5/11.
//

#include "ORBextractor.h"
//#include "ORBmatcher.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ORBTest.h"

using namespace std;
using namespace cv;
namespace ORB_SIFT {

    ORBTest::ORBTest(){};

    ORBTest::ORBTest(std::string strSettingPath):mbFirstImg(true) {
        // Load camera parameters from settings file

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

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
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        // Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractor = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    }

    void ORBTest::GrabImage(const cv::Mat &img, const double &timestamp) {

        mCurrentImg=img;
        if(mbFirstImg)
        {
            Compute_HW_ROI();
            mbFirstImg=false;
        }

        WarpROI();

        mLastFrame = Frame(mCurrentFrame);
        mCurrentFrame = Frame(mROI_Img, timestamp, mpORBextractor, mK, mDistCoef, mbf);
        Last_mvKeysROI=Curr_mvKeysROI;
        CopyKeys();
        if(mCurrentFrame.mnId>0)
        {
            ORBMatch();
            DrawMatches();
        }
        mLastImg=mCurrentImg;


    }

    void ORBTest::Compute_HW_ROI()
    {
        const double middle_col=0.2;
        const double lower_row=0.333333;

        mImg_WIDTH=mCurrentImg.cols;
        mImg_HEIGHT=mCurrentImg.rows;
        cout<<"Height: "<<mImg_HEIGHT<<" Width: "<<mImg_WIDTH<<endl;
        const int x=(0.5-0.5*middle_col)*(mCurrentImg.cols);  //起始列
        const int y=(1-lower_row)*(mCurrentImg.rows); //起始行
        const int width=middle_col*(mCurrentImg.cols);
        const int height=lower_row*(mCurrentImg.rows);
        cout<<"x:"<<x<<" y:"<<y<<" w:"<<width<<" h:"<<height<<endl;
        mROI=Rect(x,y,width,height);
    }

    void ORBTest::WarpROI()
    {
        mROI_Img=mCurrentImg(mROI);//.clone(); //这样可能并没有发生复制，只是ROIimage指向了ROI区域？
        //cv::imshow("ROI",mROI_Img);
        //waitKey();
    }
    void ORBTest::DrawMatches()
    {
        if(mCurrentFrame.mnId==0)
            return;

        cv::Mat LastImg_RGB=mLastImg.clone();
        cv::Mat CurrImg_RGB=mCurrentImg.clone();
        cvtColor(LastImg_RGB,LastImg_RGB,CV_GRAY2RGB);
        cvtColor(CurrImg_RGB,CurrImg_RGB,CV_GRAY2RGB);

        cv::drawKeypoints(LastImg_RGB,Last_mvKeysROI,LastImg_RGB,Scalar(0,255,0),0);
        DrawROI(LastImg_RGB);
        cv::drawKeypoints(CurrImg_RGB,Curr_mvKeysROI,CurrImg_RGB,Scalar(0,255,0),0);
        DrawROI(CurrImg_RGB);
        //cout<<"Draw Matches"<<endl;

        //int height=img1.rows;
        //int width=img1.cols;
        cv::Mat img_joint;
        img_joint.create(2*mImg_HEIGHT,mImg_WIDTH,LastImg_RGB.type());
        cv::Mat topImg=img_joint(Rect(0,0,mImg_WIDTH,mImg_HEIGHT));
        LastImg_RGB.copyTo(topImg);
        cv::Mat bottomImg=img_joint(Rect(0,mImg_HEIGHT,mImg_WIDTH,mImg_HEIGHT));
        CurrImg_RGB.copyTo(bottomImg);

        cv::Point pt1,pt2;

        for(size_t i=0,N=mCurrentFrame.mvKeysUn.size();i<N;i++)
        {
            if(vnMatches12[i]>0)
            {
                pt1=Point(Curr_mvKeysROI[i].pt.x,Curr_mvKeysROI[i].pt.y+mImg_HEIGHT);
                pt2=Last_mvKeysROI[vnMatches12[i]].pt;
                cv::line(img_joint,pt1,pt2,Scalar(255,0,0));
            }
        }
        cv::imshow("Match",img_joint);
        waitKey();
    }

    void ORBTest::SaveResult(const string& SaveFileName)
    {
        //cv::Mat src_im=image.clone();   //copy source image
        cv::Mat LabelROI_im=mCurrentImg.clone();
        cvtColor(LabelROI_im,LabelROI_im,CV_GRAY2RGB);
        cv::rectangle(LabelROI_im,mROI,Scalar(255,0,0),2);   //框出兴趣区域 颜色BGR
        //cv::imshow("ROIsrc",LabelROI_im);
        //TO-DO:show keypoint location and quantities
        //vector<cv::KeyPoint> KeyPoints(orbTest.mvKeys);
        int nKeys=Curr_mvKeysROI.size();
        ///draw KeyPoints
        //cout<<"draw KeyPoints"<<endl;
        const float r = 5;
        //cv::Point origin=roi.tl();
        for(int ni=0;ni<nKeys;ni++)
        {
            cv::Point2f pt1,pt2;
            pt1.x=Curr_mvKeysROI[ni].pt.x-r;//+origin.x
            pt1.y=Curr_mvKeysROI[ni].pt.y-r;//origin.y+
            pt2.x=Curr_mvKeysROI[ni].pt.x+r;//origin.x+
            pt2.y=Curr_mvKeysROI[ni].pt.y+r;//origin.y+
            if(mCurrentFrame.mnId==0||vnMatches12.size()<1)
            {
                cv::rectangle(LabelROI_im,pt1,pt2,cv::Scalar(0,255,0));
                cv::circle(LabelROI_im,Curr_mvKeysROI[ni].pt,
                        //Point(origin.x+KeyPoints[ni].pt.x,origin.y+KeyPoints[ni].pt.y),
                           2,cv::Scalar(0,255,0),-1);
            }
            //TODO debug
            else
            {
                if(vnMatches12[ni]>0){
                    cv::rectangle(LabelROI_im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(LabelROI_im,Curr_mvKeysROI[ni].pt,
                            //Point(origin.x+KeyPoints[ni].pt.x,origin.y+KeyPoints[ni].pt.y),
                               2,cv::Scalar(0,255,0),-1);
                }
                else{
                    cv::rectangle(LabelROI_im,pt1,pt2,cv::Scalar(0,0,255));
                    cv::circle(LabelROI_im,Curr_mvKeysROI[ni].pt,
                            //Point(origin.x+KeyPoints[ni].pt.x,origin.y+KeyPoints[ni].pt.y),
                               2,cv::Scalar(0,0,255),-1);
                }

            }



        }
        ///putText
        //cout<<"putText"<<endl;
        stringstream ss;
        ss<<nKeys;
        string text=ss.str()+" ORBKeyPoints";

        int font_face=cv::FONT_HERSHEY_COMPLEX;
        double font_scale=0.5;
        int thickness=1;
        int baseline;
        cv::Size text_size=cv::getTextSize(text,font_face,font_scale,thickness,&baseline);
        cv::putText(LabelROI_im,text,Point(mROI.x,mROI.y-20),font_face,font_scale,
                    cv::Scalar(255,0,0),thickness,8,0);

        //cout<<SaveFileName<<endl;
        //cv::imshow("Save",LabelROI_im);
        //waitKey();
        if(false==cv::imwrite(SaveFileName,LabelROI_im))
            cout<<"fail to save."<<endl;
    }
    void ORBTest::ORBMatch()
    {
        if(mCurrentFrame.mnId==0)
            return;

        ORBmatcher matcher(0.9,true);
        int matches=matcher.SearchForInitialization(mCurrentFrame,mLastFrame,vnMatches12,25);
        cout<<matches<<" matches."<<endl;
    }
    void ORBTest::CopyKeys()
    {
        Curr_mvKeysROI.assign(mCurrentFrame.mvKeys.begin(),mCurrentFrame.mvKeys.end());
        Shift_Keys_From_ROI_To_Origin();
    }

    void ORBTest::Shift_Keys_From_ROI_To_Origin()
    {
        size_t N=Curr_mvKeysROI.size();
        for(size_t ni=0;ni<N;ni++)
        {
            //mvKeys[ni].pt.x+=mROIOrigin.x;
            //mvKeys[ni].pt.y+=mROIOrigin.y;
            Curr_mvKeysROI[ni].pt.x+=mROI.tl().x;
            Curr_mvKeysROI[ni].pt.y+=mROI.tl().y;
        }
    }

    //cv::Mat DrawROI(cv::InputArray image,const int start_col,const int start_row, const int width,const int height)
    void ORBTest::DrawROI(cv::Mat& image)
    {
        rectangle(image,mROI,Scalar(255,0,0),2);
        //cv::imshow("ROIsrc",LabelROI_im);
        //ROIimage=image(Rect(x,y,width,height));//.clone(); //这样可能并没有发生复制，只是ROIimage指向了ROI区域？
        //cv::imshow("ROI",ROIimage);
        //imwrite("../ROI_image/");
        //waitKey();
    }

}

