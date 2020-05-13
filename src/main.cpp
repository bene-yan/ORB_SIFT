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

#include "ORBTest.h"
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
    stringstream ss;
    ss<<nKeys;

    string text=ss.str()+" ORBKeyPoints";
    int font_face=cv::FONT_HERSHEY_COMPLEX;
    double font_scale=0.5;
    int thickness=1;
    int baseline;
    cv::Size text_size=cv::getTextSize(text,font_face,font_scale,thickness,&baseline);
    cv::putText(LabelROI_im,text,Point(roi.x,roi.y-20),font_face,font_scale,
            cv::Scalar(255,255,255),thickness,8,0);

    cout<<SaveFileName<<endl;
    cv::imshow("Save",LabelROI_im);
    waitKey();
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
    cout<<endl<<nImages<<" pictures."<<endl;
    ORBTest ORBExtractorTest(argv[1]);
    cv::Mat im;
    cv::Mat image_ROI;
    Rect ROI;
    string SavePath="../Result_imgs/";
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
        }

        WarpROI(im,ROI,image_ROI);
        //ORBTest
        ORBExtractorTest.Extract_ORB(image_ROI);

        cout<<"Extract "<<ORBExtractorTest.mvKeys.size()<<" ORBPoints."<<endl;


        //Show result
        stringstream ss;
        ss << setfill('0') << setw(6) << ni;
        string FileName= SavePath + ss.str() + ".png";
        SaveResult(im,FileName,ROI,ORBExtractorTest.mvKeys);

    }

}

