//
// Created by bene on 2020/5/11.
//
#include <iostream>
#include<algorithm>
#include<fstream>
#include <string>
#include <vector>
#include<iomanip>

#include<chrono>

#include <opencv2/opencv.hpp>   //rectanlge need this header
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

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
//必须在LoadImages()后执行，否则没有vTimeStamps
void LoadLidarClouds(const string &strPathToSequence, vector<string> &vstrLidarFilenames, const vector<double> vTimestamps)
{

    //LoadLidarCloud
    string strLidarPathPrefix = strPathToSequence + "/velodyne/";

    const int nTimes = vTimestamps.size();
    vstrLidarFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        //ss << setfill('0') << setw(6) << i;
        //vstrLidarFilenames[i] = strLidarPathPrefix + ss.str() + ".bin";
        //读取原始激光点云
        ss << setfill('0') << setw(10) << i+3;	//前三帧丢弃
        vstrLidarFilenames[i] = strLidarPathPrefix + ss.str() + ".txt";
    }
}

void read_lidar_bindata(const std::string lidar_data_path,pcl::PointCloud<PointType>::Ptr &cloud)
{
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    //refer C++ primer page.676
    lidar_data_file.seekg(0, std::ios::end);    //将输入流中标记定位到流结束的位置
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);    //获取lidar_data_file的当前位置
                                                                            //由于上一条语句，将获取到流的结束处，也就是文件的大小
    lidar_data_file.seekg(0, std::ios::beg);    //将输入流中标记定位到流开始的位置

    std::vector<float> lidar_data(num_elements);
    //从文件流lidar_data_file中提取num_elements个字符保存到lidar_data_buffer
    //http://www.cplusplus.com/reference/istream/istream/read/
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data[0]), num_elements*sizeof(float));
    std::cout << "INFO_in_main: ";
    std::cout << "totally " << lidar_data.size() / 4.0 << " points in this lidar frame \n";
    //pcl::PointCloud<pcl::PointXYZI> laser_cloud;    //保存激光点云（三维坐标+强度）
       
    for (std::size_t i = 0; i < lidar_data.size(); i += 4)
    {
        pcl::PointXYZI point;
        point.x = lidar_data[i    ];
        point.y = lidar_data[i + 1];
        point.z = lidar_data[i + 2];
        point.intensity = lidar_data[i + 3];
        cloud->push_back(point);
    }
   
}
void read_lidar_textdata(const std::string lidar_data_path,pcl::PointCloud<PointType>::Ptr &cloud)
{
	std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in);
	string line;
	int point_size=0;
	while(getline(lidar_data_file,line))
	{
		istringstream LidarPoint(line);
		pcl::PointXYZI point;
		LidarPoint>>point.x>>point.y>>point.z>>point.intensity;
		cloud->push_back(point);
		point_size++;
	}
	std::cout << "INFO_in_main: ";
	std::cout << "totally " << point_size << " points in this lidar frame \n";
}

int pcd_read(const std::string &pcd_file,pcl::PointCloud<PointType>::Ptr &cloud)
{
    //创建PointCloud<pcl::PointXYZ> boost共享指针并初始化
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    //从磁盘加载PointCloud数据到二进制Blob
    if (pcl::io::loadPCDFile<PointType> (pcd_file, *cloud) == -1) //* load the file
    {
        string error_info="Couldn't read file " +pcd_file+ " \n";
        PCL_ERROR (error_info.c_str());
        return (-1);
    }
    //std::cout << "Loaded "
              //<< cloud->width * cloud->height
              //<< " data points from "<<pcd_file
              //<< std::endl;

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

void SaveResult_SIFT(const cv::Mat &image,const string& SaveFileName,Rect roi,
                vector<cv::KeyPoint>& KeyPoints)
{
    //cv::Mat src_im=image.clone();   //copy source image
    cv::Mat LabelROI_im=image.clone();
    cvtColor(LabelROI_im,LabelROI_im,CV_GRAY2RGB);
    cv::rectangle(LabelROI_im,roi,Scalar(255,0,0),2);   //框出兴趣区域 颜色BGR
    //cv::imshow("ROIsrc",LabelROI_im);
    //TO-DO:show keypoint location and quantities
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

    vector<string> vstrLidarFilenames;
    LoadLidarClouds(string(argv[2]),vstrLidarFilenames,vTimestamps);

    int nImages = vstrImageFilenames.size();
    if(nImages==0)
    {
        cerr<<"No image in"<<string(argv[2])<<endl;
    }
    cout<<endl<<nImages<<" pictures."<<endl;

    int nClouds = vstrImageFilenames.size();
    if(nImages==0)
    {
        cerr<<"No Cloud in"<<string(argv[2])<<endl;
    }

    ORBTest ORB_Test(argv[1]);
    //SIFTTest SIFT_Test(argv[1]);

    cv::Mat im;
    //cv::Mat Last_img;
    pcl::PointCloud<PointType>::Ptr origin_cloud (new pcl::PointCloud<PointType>);
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

        //Read pcd
        //pcd_read(vstrLidarFilenames[ni],origin_cloud);
		//read lidar.bin
		//read_lidar_bindata(vstrLidarFilenames[ni],origin_cloud);
		read_lidar_textdata(vstrLidarFilenames[ni],origin_cloud);

        //ORBTest
        #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point Start;
                std::chrono::steady_clock::time_point Done;
        #else
                std::chrono::monotonic_clock::time_point Start;
                std::chrono::monotonic_clock::time_point Done;
        #endif

        #ifdef COMPILEDWITHC11
                Start=std::chrono::steady_clock::now();
        #else
                Start=std::chrono::monotonic_clock::now();
        #endif
        //ORB_Test.Extract_ORB(image_ROI);
        ORB_Test.GrabImage(im,vTimestamps[ni],origin_cloud);
        //ORB_Test.Extract_ORB(im);
        #ifdef COMPILEDWITHC11
                Done=std::chrono::steady_clock::now();
        #else
                Done=std::chrono::monotonic_clock::now();
        #endif

        //std::chrono::duration<double,std::milli>
        std::chrono::duration<double> Time_cost
                =std::chrono::duration_cast<std::chrono::duration<double>>(Done-Start);
        std::cout << "INFO_in_main: ";
        std::cout<<"Spend "<<Time_cost.count()<<" Second."<<endl;
        
        //Show result
        stringstream ss;
        ss << setfill('0') << setw(6) << ni;
        string FileName= SavePath +Sequence+"/"+ ss.str() + ".png";
        string Sift_FileName= SavePath +Sequence+"/SIFT/"+ ss.str() + ".png";

        //SaveResult(im,FileName,ROI,ORB_Test.mvKeys);
        //SaveResult(im,FileName,ROI,ORB_Test);
        ORB_Test.SaveResult(FileName);

        //Last_img=im.clone();

    }

}

