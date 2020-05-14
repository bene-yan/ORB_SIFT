//
// Created by bene on 2020/5/11.
//
#include <iostream>
#include<algorithm>
#include<fstream>
#include <string>
#include <vector>
#include<iomanip>

#include <opencv2/opencv.hpp>   //rectanlge need this header
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>

#include "../include/ORBTest.h"
#include "../include/SIFTTest.h"

using namespace std;
using namespace ORB_SIFT;
using namespace cv;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}

void Compute_ROI(const cv::Mat& image,
                 const double lower_row,const double middle_col,
                 Rect& roi)
{
    //cv::Mat im_Mat=image.getMat();
    //cout<<lower_row<<" "<<middle_col<<endl;
    //cout<<"Colums:"<<image.cols<<" Rows:"<<image.rows<<endl;
    const int x=(0.5-0.5*middle_col)*(image.cols);  //起始列
    const int y=(1-lower_row)*(image.rows); //起始行
    const int width=middle_col*(image.cols);
    const int height=lower_row*(image.rows);
    //cout<<"x:"<<x<<" y:"<<y<<" w:"<<width<<" h:"<<height<<endl;
    roi=Rect(x,y,width,height);     //TODO:如何保证Rect不被修改
}

void WarpROI(const cv::Mat& image,
             Rect rect,
             cv::Mat& ROIimage)
{


    ROIimage=image(rect);//.clone(); //这样可能并没有发生复制，只是ROIimage指向了ROI区域？
    //cv::imshow("ROI",ROIimage);
    //imwrite("../ROI_image/");
    //waitKey();



}
void SaveResult(const cv::Mat &image,const string& SaveFileName,Rect roi,
        vector<cv::KeyPoint>& KeyPoints)
{
    //cv::Mat src_im=image.clone();   //copy source image
    cv::Mat LabelROI_im=image.clone();
    cvtColor(LabelROI_im,LabelROI_im,CV_GRAY2RGB);
    cv::rectangle(LabelROI_im,roi,Scalar(255,0,0),2);   //框出兴趣区域 颜色BGR
    //cv::imshow("ROIsrc",LabelROI_im);
    //TODO:show keypoint location and quantities
    int nKeys=KeyPoints.size();
    ///draw KeyPoints
    const float r = 5;
    //cv::Point origin=roi.tl();

    for(int ni=0;ni<nKeys;ni++)
    {
        cv::Point2f pt1,pt2;
        pt1.x=KeyPoints[ni].pt.x-r;//+origin.x
        pt1.y=KeyPoints[ni].pt.y-r;//origin.y+
        pt2.x=KeyPoints[ni].pt.x+r;//origin.x+
        pt2.y=KeyPoints[ni].pt.y+r;//origin.y+

        cv::rectangle(LabelROI_im,pt1,pt2,cv::Scalar(0,255,0));
        cv::circle(LabelROI_im,KeyPoints[ni].pt,
                //Point(origin.x+KeyPoints[ni].pt.x,origin.y+KeyPoints[ni].pt.y),
                2,cv::Scalar(0,255,0),-1);

    }
    ///putText
    stringstream ss;
    ss<<nKeys;
    string text=ss.str()+" ORBKeyPoints";

    int font_face=cv::FONT_HERSHEY_COMPLEX;
    double font_scale=0.5;
    int thickness=1;
    int baseline;
    cv::Size text_size=cv::getTextSize(text,font_face,font_scale,thickness,&baseline);
    cv::putText(LabelROI_im,text,Point(roi.x,roi.y-20),font_face,font_scale,
            cv::Scalar(255,0,0),thickness,8,0);

    //cout<<SaveFileName<<endl;
    //cv::imshow("Save",LabelROI_im);
    //waitKey();
    if(false==cv::imwrite(SaveFileName,LabelROI_im))
        cout<<"fail to save."<<endl;
}
//void DrawSIFT(cv::Mat& img, vector<cv::KeyPoint>& sift_keys){}
void SaveResult_SIFT(const cv::Mat &image,const string& SaveFileName,Rect roi,
                vector<cv::KeyPoint>& KeyPoints)
{
    //cv::Mat src_im=image.clone();   //copy source image
    cv::Mat LabelROI_im=image.clone();
    cvtColor(LabelROI_im,LabelROI_im,CV_GRAY2RGB);
    cv::rectangle(LabelROI_im,roi,Scalar(255,0,0),2);   //框出兴趣区域 颜色BGR
    //cv::imshow("ROIsrc",LabelROI_im);
    //TODO:show keypoint location and quantities
    int nKeys=KeyPoints.size();
    ///draw KeyPoints
    cv::drawKeypoints(LabelROI_im,KeyPoints,LabelROI_im,Scalar::all(-1),4);

    ///putText
    stringstream ss;
    ss<<nKeys;
    string text=ss.str()+" SIFTKeyPoints";

    int font_face=cv::FONT_HERSHEY_COMPLEX;
    double font_scale=0.5;
    int thickness=1;
    int baseline;
    cv::Size text_size=cv::getTextSize(text,font_face,font_scale,thickness,&baseline);
    cv::putText(LabelROI_im,text,Point(roi.x,roi.y-20),font_face,font_scale,
                cv::Scalar(255,0,0),thickness,8,0);

    //cout<<SaveFileName<<endl;
    //cv::imshow("Save",LabelROI_im);
    //waitKey();
    if(false==cv::imwrite(SaveFileName,LabelROI_im))
        cout<<"fail to save."<<endl;
}

int main(int argc, char** argv)
{
    if(argc!=3)
    {
        std::cerr<<"Usage: ./Test path_to_settings path_to_sequence"<<std::endl;
        return 1;
    }

// Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[2]), vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();
    if(nImages==0)
    {
        cerr<<"No image in"<<string(argv[2])<<endl;
    }
    cout<<endl<<nImages<<" pictures."<<endl;

    ORBTest ORB_Test(argv[1]);
    SIFTTest SIFT_Test(argv[1]);

    cv::Mat im;
    cv::Mat image_ROI;
    Rect ROI;
    string SavePath="../Result_imgs/";
    string Sequence=string(argv[2]);
    size_t len=Sequence.length();
    Sequence=Sequence.substr(len-2,2);

    for(int ni=0;ni<nImages;ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }
        if(ni==0)   //Compute_ROI need only first time
        {
            cout<<"image size:"<<im.cols<<" cols, "<<im.rows<<" rows."<<endl;
            Compute_ROI(im,0.333333,0.2,ROI);
            cout<<endl<<"ROI Position(x,y):"<<ROI.x<<","<<ROI.y<<" ."<<endl;//<<ROI.area()<<endl;
            ORB_Test.GetROIOrigin(ROI); //获取ROI原点坐标
            SIFT_Test.GetROIOrigin(ROI);
        }

        WarpROI(im,ROI,image_ROI);
        //ORBTest
        ORB_Test.Extract_ORB(image_ROI);
        //ORB_Test.Extract_ORB(im);

        //SIFTTest
        SIFT_Test.Extract_SIFT(image_ROI);
        //SIFT_Test.Extract_SIFT(im);

        cout<<"Extract "<<ORB_Test.mvKeys.size()<<" ORBPoints."<<endl;
        cout<<"Extract "<<SIFT_Test.mSift_keys.size()<<" SIFTPoints."<<endl;


        //Show result
        stringstream ss;
        ss << setfill('0') << setw(6) << ni;
        string FileName= SavePath +Sequence+"/"+ ss.str() + ".png";
        string Sift_FileName= SavePath +Sequence+"/SIFT/"+ ss.str() + ".png";

        SaveResult(im,FileName,ROI,ORB_Test.mvKeys);
        SaveResult_SIFT(im,Sift_FileName,ROI,SIFT_Test.mSift_keys);


    }

}

