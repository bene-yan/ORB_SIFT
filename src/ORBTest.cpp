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
#include "../Thirdparty/DBoW2/DUtils/Random.h"

using namespace std;
using namespace cv;
namespace ORB_SIFT {

    ORBTest::ORBTest(){};

    ORBTest::ORBTest(std::string strSettingPath):
    mbFirstImg(true),mMaxIterations(200),mMatches(0),mSigma(1.0) {
        mSigma2=(mSigma*mSigma);
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
            findHomography();

            if(mMatches>10)
            {
                vector<bool> vbMatches;
                cv::Mat H_orb;
                float score=0.0;
                findHomography(vbMatches,score,H_orb);
                cout<<"H_orb:"<<endl<<cv::format(H_orb,cv::Formatter::FMT_C)<<endl;
            }


        }
        mLastImg=mCurrentImg;


    }

    void ORBTest::Compute_HW_ROI()
    {
        const double middle_col=0.333333;
        const double lower_row=0.5;

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
        mMatches=matcher.SearchForInitialization(mLastFrame,mCurrentFrame,vnMatches12,50);
        cout<<mMatches<<" matches."<<endl;
    }
    void ORBTest::findHomography()
    {
        vector<cv::Point2f> KeyPoints1(vnMatches12.size()); //上一帧中有对象的特征点
        vector<cv::Point2f> KeyPoints2(vnMatches12.size()); //当前帧中有对象的特征点

        for(size_t i=0,iend=vnMatches12.size();i<iend;i++)
        {
            if(vnMatches12[i]>0)
            {

                KeyPoints1.push_back(Curr_mvKeysROI[vnMatches12[i]].pt);
                KeyPoints2.push_back(Last_mvKeysROI[i].pt);//是这里发生了访问越界
            }

        }
        cv::Mat H_orb=cv::findHomography(KeyPoints2,KeyPoints1,cv::RANSAC);
        cout<<"H_cv:"<<cv::format(H_orb,cv::Formatter::FMT_C)<<";"<<endl;
    }
    void ORBTest::GenerateSets()
    {

        mvMatches12.clear();
        mvMatches12.reserve(Curr_mvKeysROI.size());
        mvbMatched1.resize(Last_mvKeysROI.size());
        for(size_t i=0, iend=vnMatches12.size();i<iend; i++)
        {
            if(vnMatches12[i]>=0)
            {
                mvMatches12.push_back(make_pair(i,vnMatches12[i]));
                mvbMatched1[i]=true;
            }
            else
                mvbMatched1[i]=false;
        }

        const int N = mvMatches12.size();

        // Indices for minimum set selection
        vector<size_t> vAllIndices;
        vAllIndices.reserve(N);
        vector<size_t> vAvailableIndices;

        for(int i=0; i<N; i++)
        {
            vAllIndices.push_back(i);
        }

        // Generate sets of 8 points for each RANSAC iteration
        mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

        DUtils::Random::SeedRandOnce(0);

        for(int it=0; it<mMaxIterations; it++)
        {
            vAvailableIndices = vAllIndices;

            // Select a minimum set
            for(size_t j=0; j<8; j++)//TO-DO fix this for loop 匹配特征应该大于8
            {
                int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);

                int idx = vAvailableIndices[randi];

                mvSets[it][j] = idx;

                vAvailableIndices[randi] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }
        }
    }
    void ORBTest::findHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
    {
        GenerateSets();

        // Number of putative matches
        const int N = mvMatches12.size();

        // Normalize coordinates
        vector<cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        Normalize(Curr_mvKeysROI,vPn1, T1);
        Normalize(Last_mvKeysROI,vPn2, T2);
        cv::Mat T2inv = T2.inv();

        // Best Results variables
        score = 0.0;
        vbMatchesInliers = vector<bool>(N,false);

        // Iteration variables
        vector<cv::Point2f> vPn1i(8);
        vector<cv::Point2f> vPn2i(8);
        cv::Mat H21i, H12i;
        vector<bool> vbCurrentInliers(N,false);
        float currentScore;

        // Perform all RANSAC iterations and save the solution with highest score
        for(int it=0; it<mMaxIterations; it++)
        {
            // Select a minimum set
            for(size_t j=0; j<8; j++)
            {
                int idx = mvSets[it][j];

                vPn1i[j] = vPn1[mvMatches12[idx].first];
                vPn2i[j] = vPn2[mvMatches12[idx].second];
            }

            cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
            H21i = T2inv*Hn*T1;
            H12i = H21i.inv();

            currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

            if(currentScore>score)
            {
                H21 = H21i.clone();
                vbMatchesInliers = vbCurrentInliers;
                score = currentScore;
            }
        }
    }

    cv::Mat ORBTest::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
    {
        const int N = vP1.size();

        cv::Mat A(2*N,9,CV_32F);

        for(int i=0; i<N; i++)
        {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            A.at<float>(2*i,0) = 0.0;
            A.at<float>(2*i,1) = 0.0;
            A.at<float>(2*i,2) = 0.0;
            A.at<float>(2*i,3) = -u1;
            A.at<float>(2*i,4) = -v1;
            A.at<float>(2*i,5) = -1;
            A.at<float>(2*i,6) = v2*u1;
            A.at<float>(2*i,7) = v2*v1;
            A.at<float>(2*i,8) = v2;

            A.at<float>(2*i+1,0) = u1;
            A.at<float>(2*i+1,1) = v1;
            A.at<float>(2*i+1,2) = 1;
            A.at<float>(2*i+1,3) = 0.0;
            A.at<float>(2*i+1,4) = 0.0;
            A.at<float>(2*i+1,5) = 0.0;
            A.at<float>(2*i+1,6) = -u2*u1;
            A.at<float>(2*i+1,7) = -u2*v1;
            A.at<float>(2*i+1,8) = -u2;

        }

        cv::Mat u,w,vt;

        cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        return vt.row(8).reshape(0, 3);
    }

    float ORBTest::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
    {
        const int N = mvMatches12.size();

        const float h11 = H21.at<float>(0,0);
        const float h12 = H21.at<float>(0,1);
        const float h13 = H21.at<float>(0,2);
        const float h21 = H21.at<float>(1,0);
        const float h22 = H21.at<float>(1,1);
        const float h23 = H21.at<float>(1,2);
        const float h31 = H21.at<float>(2,0);
        const float h32 = H21.at<float>(2,1);
        const float h33 = H21.at<float>(2,2);

        const float h11inv = H12.at<float>(0,0);
        const float h12inv = H12.at<float>(0,1);
        const float h13inv = H12.at<float>(0,2);
        const float h21inv = H12.at<float>(1,0);
        const float h22inv = H12.at<float>(1,1);
        const float h23inv = H12.at<float>(1,2);
        const float h31inv = H12.at<float>(2,0);
        const float h32inv = H12.at<float>(2,1);
        const float h33inv = H12.at<float>(2,2);

        vbMatchesInliers.resize(N);

        float score = 0;

        const float th = 5.991;

        const float invSigmaSquare = 1.0/(sigma*sigma);

        for(int i=0; i<N; i++)
        {
            bool bIn = true;

            const cv::KeyPoint &kp1 = Curr_mvKeysROI[mvMatches12[i].first];
            const cv::KeyPoint &kp2 = Last_mvKeysROI[mvMatches12[i].second];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            // Reprojection error in first image
            // x2in1 = H12*x2

            const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
            const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
            const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

            const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

            const float chiSquare1 = squareDist1*invSigmaSquare;

            if(chiSquare1>th)
                bIn = false;
            else
                score += th - chiSquare1;

            // Reprojection error in second image
            // x1in2 = H21*x1

            const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
            const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
            const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

            const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

            const float chiSquare2 = squareDist2*invSigmaSquare;

            if(chiSquare2>th)
                bIn = false;
            else
                score += th - chiSquare2;

            if(bIn)
                vbMatchesInliers[i]=true;
            else
                vbMatchesInliers[i]=false;
        }

        return score;
    }
    void ORBTest::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
    {
        float meanX = 0;
        float meanY = 0;
        const int N = vKeys.size();

        vNormalizedPoints.resize(N);

        for(int i=0; i<N; i++)
        {
            meanX += vKeys[i].pt.x;
            meanY += vKeys[i].pt.y;
        }

        meanX = meanX/N;
        meanY = meanY/N;

        float meanDevX = 0;
        float meanDevY = 0;

        for(int i=0; i<N; i++)
        {
            vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
            vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

            meanDevX += fabs(vNormalizedPoints[i].x);
            meanDevY += fabs(vNormalizedPoints[i].y);
        }

        meanDevX = meanDevX/N;
        meanDevY = meanDevY/N;

        float sX = 1.0/meanDevX;
        float sY = 1.0/meanDevY;

        for(int i=0; i<N; i++)
        {
            vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
            vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
        }

        T = cv::Mat::eye(3,3,CV_32F);
        T.at<float>(0,0) = sX;
        T.at<float>(1,1) = sY;
        T.at<float>(0,2) = -meanX*sX;
        T.at<float>(1,2) = -meanY*sY;
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

