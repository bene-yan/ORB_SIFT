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

    /*
    //vector<cv::Mat> vmatGT;
    cv::Mat GT00_1(3,4,CV_32F);
    GT00_1<< 9.999978e-01  << 5.272628e-04 << -2.066935e-03 << -4.690294e-02
          << -5.296506e-04 << 9.999992e-01 << -1.154865e-03 << -2.839928e-02
          << 2.066324e-03  << 1.155958e-03 << 9.999971e-01  << 8.586941e-01;
          */

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

        cout << endl  << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        ROI_middle_col=fSettings["ROI.middle_col"];
        ROI_lower_row=fSettings["ROI.lower_row"];

        mpORBextractor = new ORBextractor(2*nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    }

    void ORBTest::GrabImage(const cv::Mat &img, const double &timestamp) {

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



        if(mbFirstImg)
        {
            Compute_HW_ROI();
            mbFirstImg=false;
        }

        WarpROI();

        mLastFrame = Frame(mCurrentFrame);
        mCurrentFrame = Frame(mROI_Img, timestamp, mpORBextractor, mK, mDistCoef, mbf);
        //mCurrentFrame = Frame(mCurrentImg, timestamp, mpORBextractor, mK, mDistCoef, mbf);
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
                cout<<"Score: "<<score<<endl;
                cout<<"H_orb:"<<endl<<cv::format(H_orb,cv::Formatter::FMT_C)<<endl;
                cv::Mat R21,t21;
                vector<cv::Point3f> vP3D;
                vector<bool> vbTriangulated;
                if(ReconstructH(vbMatches,H_orb,mK,R21,t21,vP3D,vbTriangulated,1.0,50))
                {
                    cout<<"R21:"<<endl<<cv::format(R21,cv::Formatter::FMT_C)<<endl;
                    cout<<"t21:"<<endl<<cv::format(t21,cv::Formatter::FMT_C)<<endl;
                }
                else
                    cout<<"fail to ReconstructH."<<endl;


            }


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

    {
        vector<cv::Point2f> KeyPoints1(vnMatches12.size()); //上一帧中有对象的特征点
        vector<cv::Point2f> KeyPoints2(vnMatches12.size()); //当前帧中有对象的特征点

        for(size_t i=0,iend=vnMatches12.size();i<iend;i++)
        {
            if(vnMatches12[i]>0)
            {

                KeyPoints1.push_back(mCurrentFrame.mvKeysUn[vnMatches12[i]].pt);
                KeyPoints2.push_back(mLastFrame.mvKeysUn[i].pt);//是这里发生了访问越界
            }

        }
        cv::Mat H_cv=cv::findHomography(KeyPoints2,KeyPoints1,cv::RANSAC);
        cout<<"H_cv:"<<cv::format(H_cv,cv::Formatter::FMT_C)<<";"<<endl;
    }
    void ORBTest::GenerateSets()
    {

        mvMatches12.clear();
        mvMatches12.reserve(mCurrentFrame.mvKeysUn.size());

        mvbMatched1.resize(mLastFrame.mvKeysUn.size());

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
    //TODO 如何在大量误匹配数据中找到正确的Homography
    void ORBTest::findHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
    {
        GenerateSets();

        // Number of putative matches
        const int N = mvMatches12.size();

        // Normalize coordinates
        vector<cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        Normalize(mLastFrame.mvKeysUn,vPn1, T1);
        Normalize(mCurrentFrame.mvKeysUn,vPn2, T2);
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

            //这里采用的方法是通过H矩阵计算匹配点的对称重投影误差
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

            const cv::KeyPoint &kp1 = mLastFrame.mvKeysUn[mvMatches12[i].first];
            const cv::KeyPoint &kp2 = mCurrentFrame.mvKeysUn[mvMatches12[i].second];

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

    bool ORBTest::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                                   cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
    {
        int N=0;
        for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
            if(vbMatchesInliers[i])
                N++;

        // We recover 8 motion hypotheses using the method of Faugeras et al.
        // Motion and structure from motion in a piecewise planar environment.
        // International Journal of Pattern Recognition and Artificial Intelligence, 1988

        cv::Mat invK = K.inv();
        cv::Mat A = invK*H21*K;

        cv::Mat U,w,Vt,V;
        cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
        V=Vt.t();

        float s = cv::determinant(U)*cv::determinant(Vt);

        float d1 = w.at<float>(0);
        float d2 = w.at<float>(1);
        float d3 = w.at<float>(2);

        if(d1/d2<1.00001 || d2/d3<1.00001)
        {
            cout<<"SVD abnormal!"<<endl;
            return false;
        }

        vector<cv::Mat> vR, vt, vn;
        vR.reserve(8);
        vt.reserve(8);
        vn.reserve(8);

        //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
        float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
        float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
        float x1[] = {aux1,aux1,-aux1,-aux1};
        float x3[] = {aux3,-aux3,aux3,-aux3};

        //case d'=d2
        float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

        float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
        float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

        for(int i=0; i<4; i++)
        {
            cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
            Rp.at<float>(0,0)=ctheta;
            Rp.at<float>(0,2)=-stheta[i];
            Rp.at<float>(2,0)=stheta[i];
            Rp.at<float>(2,2)=ctheta;

            cv::Mat R = s*U*Rp*Vt;
            vR.push_back(R);

            cv::Mat tp(3,1,CV_32F);
            tp.at<float>(0)=x1[i];
            tp.at<float>(1)=0;
            tp.at<float>(2)=-x3[i];
            tp*=d1-d3;

            cv::Mat t = U*tp;
            vt.push_back(t/cv::norm(t));

            cv::Mat np(3,1,CV_32F);
            np.at<float>(0)=x1[i];
            np.at<float>(1)=0;
            np.at<float>(2)=x3[i];

            cv::Mat n = V*np;
            if(n.at<float>(2)<0)
                n=-n;
            vn.push_back(n);
        }

        //case d'=-d2
        float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

        float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
        float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

        for(int i=0; i<4; i++)
        {
            cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
            Rp.at<float>(0,0)=cphi;
            Rp.at<float>(0,2)=sphi[i];
            Rp.at<float>(1,1)=-1;
            Rp.at<float>(2,0)=sphi[i];
            Rp.at<float>(2,2)=-cphi;

            cv::Mat R = s*U*Rp*Vt;
            vR.push_back(R);

            cv::Mat tp(3,1,CV_32F);
            tp.at<float>(0)=x1[i];
            tp.at<float>(1)=0;
            tp.at<float>(2)=x3[i];
            tp*=d1+d3;

            cv::Mat t = U*tp;
            vt.push_back(t/cv::norm(t));

            cv::Mat np(3,1,CV_32F);
            np.at<float>(0)=x1[i];
            np.at<float>(1)=0;
            np.at<float>(2)=x3[i];

            cv::Mat n = V*np;
            if(n.at<float>(2)<0)
                n=-n;
            vn.push_back(n);
        }


        int bestGood = 0;
        int secondBestGood = 0;
        int bestSolutionIdx = -1;
        float bestParallax = -1;
        vector<cv::Point3f> bestP3D;
        vector<bool> bestTriangulated;

        // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
        // We reconstruct all hypotheses and check in terms of triangulated points and parallax
        for(size_t i=0; i<8; i++)
        {
            float parallaxi;
            vector<cv::Point3f> vP3Di;
            vector<bool> vbTriangulatedi;
            //总是有其中两个解得到相同数量的nGood
            //增加约束Both frames,F* and F must be in the same side of the object plane.
            //1+n.t()*R.t()*t>0
            //cv::Mat nt=vn[i].t();
            cv::Mat Rt=vR[i].t();
            cv::Mat t=vt[i];
            double d=vn[i].dot(Rt*t);
            //double d=vn[i].t()*vR[i].t()*vt[i]+1;
            if(d<=0.000001)
                continue;

            int nGood = CheckRT(vR[i],vt[i],mLastFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);
            cout<<"R"<<i<<":"<<endl<<cv::format(vR[i],cv::Formatter::FMT_C)<<endl;
            cout<<"t"<<i<<":"<<endl<<cv::format(vt[i],cv::Formatter::FMT_C)<<endl;
            cout<<"nGood "<<nGood<<"v.s."<<" bestGood"<<bestGood<<endl;
            if(nGood>bestGood)
            {
                secondBestGood = bestGood;
                bestGood = nGood;
                bestSolutionIdx = i;
                bestParallax = parallaxi;
                bestP3D = vP3Di;
                bestTriangulated = vbTriangulatedi;
            }
            else if(nGood>secondBestGood)
            {
                secondBestGood = nGood;
            }
        }

        if(false==(secondBestGood<0.75*bestGood))
        {
            cout<<"false==(secondBestGood<0.75*bestGood)"<<endl;
            cout<<"secondBestGood is "<<secondBestGood<<" while "<<
            "0.75*bestGood is "<<0.75*bestGood<<endl;
        }
        if(false==(bestParallax>=minParallax))
        {
            cout<<"false==(bestParallax>=minParallax)"<<endl;
            cout<<"bestParallax: "<<bestParallax<<endl;
        }
        if(false==(bestGood>minTriangulated))
        {
            cout<<"false==(bestGood>minTriangulated)"<<endl;
        }
        if(false==(bestGood>0.9*N))
        {
            cout<<"false==(bestGood>0.9*N)"<<endl;
        }
        //if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
        if( bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
        {
            vR[bestSolutionIdx].copyTo(R21);
            vt[bestSolutionIdx].copyTo(t21);
            vP3D = bestP3D;
            vbTriangulated = bestTriangulated;

            return true;
        }
        cout<<"get to end, so false"<<endl;
        return false;
    }

    int ORBTest::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                             const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                             const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
    {
        // Calibration parameters
        const float fx = K.at<float>(0,0);
        const float fy = K.at<float>(1,1);
        const float cx = K.at<float>(0,2);
        const float cy = K.at<float>(1,2);

        vbGood = vector<bool>(vKeys1.size(),false);
        vP3D.resize(vKeys1.size());

        vector<float> vCosParallax;
        vCosParallax.reserve(vKeys1.size());

        // Camera 1 Projection Matrix K[I|0]
        cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
        K.copyTo(P1.rowRange(0,3).colRange(0,3));

        cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

        // Camera 2 Projection Matrix K[R|t]
        cv::Mat P2(3,4,CV_32F);
        R.copyTo(P2.rowRange(0,3).colRange(0,3));
        t.copyTo(P2.rowRange(0,3).col(3));
        P2 = K*P2;

        cv::Mat O2 = -R.t()*t;

        int nGood=0;

        for(size_t i=0, iend=vMatches12.size();i<iend;i++)
        {
            if(!vbMatchesInliers[i])
                continue;

            const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
            const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
            cv::Mat p3dC1;

            Triangulate(kp1,kp2,P1,P2,p3dC1);

            if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
            {
                vbGood[vMatches12[i].first]=false;
                continue;
            }

            // Check parallax
            cv::Mat normal1 = p3dC1 - O1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = p3dC1 - O2;
            float dist2 = cv::norm(normal2);

            float cosParallax = normal1.dot(normal2)/(dist1*dist2);

            // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
                continue;

            // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            cv::Mat p3dC2 = R*p3dC1+t;

            if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
                continue;

            // Check reprojection error in first image
            float im1x, im1y;
            float invZ1 = 1.0/p3dC1.at<float>(2);
            im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
            im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

            float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

            if(squareError1>th2)
                continue;

            // Check reprojection error in second image
            float im2x, im2y;
            float invZ2 = 1.0/p3dC2.at<float>(2);
            im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
            im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

            float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

            if(squareError2>th2)
                continue;

            //总是有其中两个解得到相同数量的nGood
            //增加约束Both frames,F* and F must be in the same side of the object plane.
            //1+n.t()*R.t()*t>0

            vCosParallax.push_back(cosParallax);
            vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
            nGood++;

            if(cosParallax<0.99998)
                vbGood[vMatches12[i].first]=true;
        }

        if(nGood>0)
        {
            sort(vCosParallax.begin(),vCosParallax.end());

            size_t idx = min(50,int(vCosParallax.size()-1));
            parallax = acos(vCosParallax[idx])*180/CV_PI;
        }
        else
            parallax=0;

        return nGood;
    }

    void ORBTest::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
    {
        cv::Mat A(4,4,CV_32F);

        A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
        A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
        A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
        A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

        cv::Mat u,w,vt;
        cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
        x3D = vt.row(3).t();
        x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
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


}

