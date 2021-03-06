#include <stdio.h>
#include <iostream>
 #include<ctime>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "gMM.h"
cv::Scalar getColor(const int clusterID)
{
	cv::Scalar clusterColor[20]={
	cv::Scalar(65,105,225),
	cv::Scalar(0,255,255),
	cv::Scalar(56,94,15),
	cv::Scalar(64,224,208),
	cv::Scalar(127,255,0),
	cv::Scalar(255,250,240),
	cv::Scalar(188,143,143),
	cv::Scalar(156,102,31),
	cv::Scalar(94,38,18),
	cv::Scalar(0,0,255),
	cv::Scalar(3,168,158),
	cv::Scalar(11,23,70),
	cv::Scalar(51,161,201),
	cv::Scalar(0,199,140),
	cv::Scalar(160,32,240),
	cv::Scalar(218,112,214),
	cv::Scalar(192,192,192),
	cv::Scalar(112,128,105),
	cv::Scalar(255,153,18),
	cv::Scalar(0,0,0)
	};
	if(clusterID>=sizeof(clusterColor)/sizeof(clusterColor[0]))
		return clusterColor[0];
	return clusterColor[clusterID];	
}

void readBenchmarkData(std::vector<PointWithC>& points)
{
    // load point cloud
    FILE *stream;
    stream = fopen ("../data/benchmark_hepta.dat","ra");

    unsigned int minpts, num_points, cluster, i = 0;
    double epsilon;
    fscanf(stream, "%u\n", &num_points);

    PointWithC *p = (PointWithC *)calloc(num_points, sizeof(PointWithC));

    while (i < num_points)
    {
          fscanf(stream, "%f,%f,%f,%d\n", &(p[i].x), &(p[i].y), &(p[i].z), &cluster);
          p[i].clusterID = UNCLASSIFIED;
          points.push_back(p[i]);
          ++i;
    }

    free(p);
    fclose(stream);
}

void printResults(const std::vector<PointWithC> points)
{
    int i = 0;
    printf("Number of points: %u\n"
        " x     y     z     cluster_id\n"
        "-----------------------------\n"
        , (unsigned int)(points.size()));
    while (i < points.size())
    {
          printf("%5.2lf %5.2lf %5.2lf: %d\n",
                 points[i].x,
                 points[i].y, points[i].z,
                 points[i].clusterID);
          ++i;
    }
}

void showPointWithCSet(cv::Mat& img, const std::vector<PointWithC> dataset, float eps=0.75)
{
	int s=50;
	int offX = 250;
	int offY = offX;
	cv::circle(img, cv::Point2d(img.cols*6.0/7.0,img.rows*1.0/7.0), eps*s, getColor(19), 2);
	for(int i=0;i<dataset.size();i++)
	{
		cv::circle(img, cv::Point2d(dataset[i].x*s+offX, dataset[i].y*s+offY), 4, getColor(dataset[i].clusterID), -1);
	}
}

void showPointWithCSet(cv::Mat& img, const std::vector<std::vector<cv::Point2d>> dataset, float eps=15.0)
{
	int s=1.0;
	int offX = 0;
	int offY = offX;
	cv::circle(img, cv::Point2d(img.cols*6.0/7.0,img.rows*1.0/7.0), eps*s, getColor(19), 2);
	for(int i=0;i<dataset.size();i++)
	{
		cv::Scalar color;
		if(dataset[i].size()>0)
		{
			color = getColor(i);
		}else{
			continue;
		}
		
		for(int j=0;j<dataset[i].size();j++)
		{
			cv::circle(img, cv::Point2d(dataset[i][j].x*s+offX, dataset[i][j].y*s+offY), 4, color , -1);
		}
		
	}
}

std::vector<PointWithC> getRowPoints(const cv::Mat ipm_mask)
{
	std::vector<PointWithC> points;
	for(int x = 0; x < ipm_mask.cols; x++)
	{
		for(int y = 0; y < ipm_mask.rows; y++)
		{
			if(ipm_mask.at<uchar>(y, x) > 0)
			{
				points.push_back(PointWithC{float(x), float(y), 0.0, -1});
			}
		}
	}
	return points;
}


std::vector<std::vector<cv::Point2d>> getCluster(const cv::Mat lane_mask)
{
	unsigned int clusterNum = 3;     // minimum number of cluster
	std::vector<PointWithC> points = getRowPoints(lane_mask);
	gMM gmm(clusterNum, points);
	std::vector<std::vector<cv::Point2d>> result = gmm.getCluster();
	return result;
}

void getParserInfo(std::string& fileName, int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{help||}{testImg|../data/39-1001150009181229150644600.png|Test image instance segmentation}");
	fileName = parser.get<std::string>("testImg");
}

const char * usage =
"\n"
"./testGMMImage -testImg=../data/39-1001150009181229150644600.png"
"\n";

static void help()
{
	std::cout << usage;
}

int main(int argc, char** argv)
{
	cv::Size resize(512, 270);
	if(argc<2)
	{
		help();
		return 0;
	}
	
	clock_t startTime,endTime;
	std::string fileName;
	getParserInfo(fileName, argc, argv);
	cv::Mat lane_mask = cv::imread(fileName, 0);
	cv::resize(lane_mask, lane_mask, resize);
	cv::Mat beforeCluster= lane_mask.clone();
	cv::imshow("beforeGMMCluster", beforeCluster);
	startTime = clock();//计时开始
	std::vector<std::vector<cv::Point2d>> result = getCluster(lane_mask);
	endTime = clock();//计时结束
	std::cout << "The run time is:" <<(double)(endTime - startTime)*1000 / CLOCKS_PER_SEC << "ms" << std::endl;
//	cv::Mat afterCluster = lane_mask.clone();
	cv::Mat afterCluster(lane_mask.rows,lane_mask.cols, CV_8UC3, cv::Scalar(255,255,255));
	showPointWithCSet(afterCluster, result);
	cv::imshow("afterGMMCluster", afterCluster);
	
	cv::waitKey(0);  

	return 0;
}
