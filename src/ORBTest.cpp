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

#include <pcl/point_types.h>

#include "ORBTest.h"
#include "../Thirdparty/DBoW2/DUtils/Random.h"
#include "Initializer.h"
#include "LidarProcess.h"

using namespace std;
using namespace cv;
namespace ORB_SIFT {

    /*
    //vector<cv::Mat> vmatGT;
    cv::Mat GT00_1(3,4,CV_32F);
    GT00_1<< 9.999978e-01  << 5.272628e-04 << -2.066935e-03 << -4.690294e-02
          << -5.296506e-04 << 9.999992e-01 << -1.154865e-03 << -2.839928e-02
          << 2.066324e-03  << 1.155958e-03 << 9.999971e-01  << 8.586941e-01;
          */

    ORBTest::ORBTest(){};

    ORBTest::ORBTest(std::string strSettingPath):mbFirstImg(true),mMatches(0){
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

        cv::Mat P0=cv::Mat::zeros(3,4,CV_32F);
        K.copyTo(P0.rowRange(0,3).colRange(0,3));
        P0.at<float>(0,3)=-fx*mbf;
        P0.copyTo(mP0);
        cout<<"Project matrix of camera_0 P0: "<<endl<<mP0<<endl;

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

        cout << endl  << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        ROI_middle_col=fSettings["ROI.middle_col"];
        ROI_lower_row=fSettings["ROI.lower_row"];

        //load lidar to camera transform

        cv::Mat Tr = cv::Mat::eye(4, 4, CV_32F);
        Tr.at<float>(0, 0) = fSettings["Tr.R11"];
        Tr.at<float>(0, 1) = fSettings["Tr.R12"];
        Tr.at<float>(0, 2) = fSettings["Tr.R13"];
        
        Tr.at<float>(0, 3) = fSettings["Tr.t1"];
        
        Tr.at<float>(1, 0) = fSettings["Tr.R21"];
        Tr.at<float>(1, 1) = fSettings["Tr.R22"];
        Tr.at<float>(1, 2) = fSettings["Tr.R23"];
        
        Tr.at<float>(1, 3) = fSettings["Tr.t2"];
        
        Tr.at<float>(2, 0) = fSettings["Tr.R31"];
        Tr.at<float>(2, 1) = fSettings["Tr.R32"];
        Tr.at<float>(2, 2) = fSettings["Tr.R33"];
        
        Tr.at<float>(2, 3) = fSettings["Tr.t3"];

        Tr.copyTo(mTr);
        cout<<"lidar to camera transform Tr:"<<endl<<mTr<<endl;

        mpORBextractor = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    }

    void ORBTest::GrabImage(const cv::Mat &img, const double &timestamp,pcl::PointCloud<PointType>::Ptr &orgin_cloud) {

        mCurrentImg=img;
        if(mCurrentImg.channels()==3)
        {
            if(mbRGB)
                cvtColor(mCurrentImg,mCurrentImg,CV_RGB2GRAY);
            else
                cvtColor(mCurrentImg,mCurrentImg,CV_BGR2GRAY);
        }
        else if(mCurrentImg.channels()==4)
        {
            if(mbRGB)
                cvtColor(mCurrentImg,mCurrentImg,CV_RGBA2GRAY);
            else
                cvtColor(mCurrentImg,mCurrentImg,CV_BGRA2GRAY);
        }
        
        //将点云投影到图像，投影点保存在lidarProjPts
        
        lego_loam::getParamFromYAML("/home/bene-robot/CLionProjects/ORB_SIFT/SettingFiles/loam_config.yaml");
        lego_loam::resetParameters();
        lego_loam::test();
        pcl::PointCloud<PointType>::Ptr full_cloud (new pcl::PointCloud<PointType>);	//FixME智能指针什么时候销毁？
        pcl::PointCloud<PointType>::Ptr ground_cloud (new pcl::PointCloud<PointType>);
		lego_loam::projectPointCloud(orgin_cloud,full_cloud);
		lego_loam::groundRemoval(full_cloud,ground_cloud);
		
		
		lidarProjPts.clear();
        ProjectLidarCloud(ground_cloud,lidarProjPts);
		//full_cloud.reset(new pcl::PointCloud<PointType>());
        //计算ROI区域
        if(mbFirstImg)
        {
            Compute_HW_ROI();
            mbFirstImg=false;

            cv::Mat t0(3,1,CV_32F);
            t0.at<double>(0)=-0.04690294;
            t0.at<double>(1)=-0.02839928;
            t0.at<double>(2)=0.8586941;

            //t0<<-0.04690294<<-0.02839928<<0.8586941;
            t0=t0/cv::norm(t0);
            cout<<"t0:"<<cv::format(t0,cv::Formatter::FMT_C)<<endl;
        }

        //提取当前图像的ROI
        WarpROI();

        //对图像进行特征提取、校正等操作
        mLastFrame = Frame(mCurrentFrame);
        mCurrentFrame = Frame(mROI_Img, timestamp, mpORBextractor, mK, mDistCoef, mbf);     //通过ROI提取特征
        //TODO 通过激光地面点投影到图像提取特征

        //mCurrentFrame = Frame(mCurrentImg, timestamp, mpORBextractor, mK, mDistCoef, mbf);
        Last_mvKeysROI=Curr_mvKeysROI;
        //复制用于展示的特征点
        CopyKeys();
        //从第二张图开始估计运动
        if(mCurrentFrame.mnId>0)
        {
            ORBMatch();
            DrawMatches();

            cv::Mat R21,t21;
            float score=0.0;

            HomoDecomp H_Decompor(mK,mLastFrame,mCurrentFrame,vnMatches12,20);
            H_Decompor.DecompHomography(score,R21,t21);

            vector<bool> vbTriangulated;
            vector<cv::Point3f> vIniP3D;
/*
            if(mMatches>10)
            {

                Initializer OrbHdecomposer(mLastFrame,1.0,200);
                OrbHdecomposer.Initialize(mCurrentFrame,vnMatches12,R21,t21,vIniP3D,vbTriangulated);

            }
*/
            cout<<"Score: "<<score<<endl;
            cout<<"R21:"<<endl<<cv::format(R21,cv::Formatter::FMT_C)<<endl;
            cout<<"t21:"<<endl<<cv::format(t21,cv::Formatter::FMT_C)<<endl;

        }
        mLastImg=mCurrentImg;
    }

    void ORBTest::Compute_HW_ROI()
    {
        const double middle_col=ROI_middle_col;
        const double lower_row=ROI_lower_row;

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

        for(size_t i=0,N=mLastFrame.mvKeysUn.size();i<N;i++)
        {
            if(vnMatches12[i]>0)
            {
                pt1=Point(Curr_mvKeysROI[vnMatches12[i]].pt.x,Curr_mvKeysROI[vnMatches12[i]].pt.y+mImg_HEIGHT);
                pt2=Last_mvKeysROI[i].pt;  //
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
        //cv::Mat P=mk*GT00_1;
        //cv::Point origin=roi.tl();
        for(int ni=0;ni<nKeys;ni++)
        {
            cv::Point2f pt1,pt2;
            pt1.x=Curr_mvKeysROI[ni].pt.x-r;//+origin.x
            pt1.y=Curr_mvKeysROI[ni].pt.y-r;//origin.y+
            pt2.x=Curr_mvKeysROI[ni].pt.x+r;//origin.x+
            pt2.y=Curr_mvKeysROI[ni].pt.y+r;//origin.y+
            /*
            cv::Point2f pt3,pt4;
            cv::Point2f gt_key=P*;
            pt3.x=Curr_mvKeysROI[ni].pt.x-r;
            pt3.y=Curr_mvKeysROI[ni].pt.y-r;
            pt4.x=Curr_mvKeysROI[ni].pt.x+r;
            pt4.y=Curr_mvKeysROI[ni].pt.y+r;
            */

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


                /*
                else{
                    cv::rectangle(LabelROI_im,pt1,pt2,cv::Scalar(0,0,255));
                    cv::circle(LabelROI_im,Curr_mvKeysROI[ni].pt,
                            //Point(origin.x+KeyPoints[ni].pt.x,origin.y+KeyPoints[ni].pt.y),
                               2,cv::Scalar(0,0,255),-1);
                }*/

            }



        }
        ///show project points
        for(size_t i=0,N=lidarProjPts.size();i<N;i++)
        {
        	cv::circle(LabelROI_im,lidarProjPts[i],2,cv::Scalar(0,0,255),-1);
        
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
        mMatches=matcher.SearchForInitialization(mLastFrame,mCurrentFrame,vnMatches12,30);
        cout<<"Id: "<<mCurrentFrame.mnId<<" find "<<mMatches<<" matches."<<endl;
    }

    void ORBTest::ProjectLidarCloud(pcl::PointCloud<PointType>::Ptr &ground_cloud
                                        ,vector<cv::Point2f> &validProjectionPoints)
    {
   		validProjectionPoints.clear();
   		//Tr transforms a point from velodyne coordinates into the
		//left rectified camera coordinate system. In order to map a point X from the
		//velodyne scanner to a point x in the i'th image plane, you thus have to
		//transform it like:
		//					x = Pi * Tr * X
		//
        for(size_t i=0,N=ground_cloud->points.size();i<N;i++)
        {
            //ProjectToImg(ground_cloud[i]);
            cv::Mat Point3d4=cv::Mat::zeros(4,1,CV_32F);
            Point3d4.at<float>(0)=ground_cloud->points[i].x;
            Point3d4.at<float>(1)=ground_cloud->points[i].y;
            Point3d4.at<float>(2)=ground_cloud->points[i].z;
            Point3d4.at<float>(3)=1;
             cv::Mat pt_hc=mP0*mTr*Point3d4; //将X投影到图像上 P0和Tr在calib中给出
            if(pt_hc.at<float>(2)>0)	//ground_point中有些投影后全为负的值，也能投影到图像上（上部），
            							//它们其实是激光雷达后方的点，在图像上是不可见的
            {
             cv::Point2f pt(pt_hc.at<float>(0)/pt_hc.at<float>(2),pt_hc.at<float>(1)/pt_hc.at<float>(2) );
            	if(pt.x>0&&pt.y>0&&pt.x<mImg_WIDTH&&pt.y<mImg_HEIGHT)
            	    validProjectionPoints.push_back(pt);
            }
        }
        
        cout<<"number of validProjPts: "<<validProjectionPoints.size()<<endl;
    }

    void ORBTest::CopyKeys()
    {
        Curr_mvKeysROI.assign(mCurrentFrame.mvKeysUn.begin(),mCurrentFrame.mvKeysUn.end());
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

    }



}

