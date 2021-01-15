#include <stdio.h>
#include <iostream>
#include "dbScan.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

cv::Scalar getColor(const int clusterID)
{
	cv::Scalar clusterColor[20]={
	cv::Scalar(0,0,0),
	cv::Scalar(192,192,192),
	cv::Scalar(112,128,105),
	cv::Scalar(255,153,18),
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
	};
	if(clusterID>=sizeof(clusterColor)/sizeof(clusterColor[0]))
		return clusterColor[0];
	return clusterColor[clusterID];	
}

void getParserInfo(std::string& fileName, int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{help||}{testData|../data/benchmark_hepta.dat|Test data}");

	fileName = parser.get<std::string>("testData");
}

std::vector<PointWithC> readBenchmarkData(const std::string fileName)
{
	std::vector<PointWithC> points;
	// load point cloud
	FILE *stream;
	stream = fopen(fileName.c_str(),"ra");

	unsigned int minpts, num_points, cluster, i = 0;
	double epsilon;
	fscanf(stream, "%u\n", &num_points);

	PointWithC *p = (PointWithC *)calloc(num_points, sizeof(PointWithC));

	while (i < num_points)
	{
		fscanf(stream, "%f,%f,%f,%d\n", &(p[i].x), &(p[i].y), &(p[i].z), &cluster);
		p[i].z = 0.0;
		p[i].clusterID = UNCLASSIFIED;
		points.push_back(p[i]);
		++i;
	}

	free(p);
	fclose(stream);
	return points;
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

const char * usage =
"\n"
"./testDbScan -testData=../data/benchmark_hepta.dat"
"\n";

static void help()
{
	std::cout << usage;
}

int main(int argc, char** argv)
{
	if(argc<2)
	{
		help();
		return 0;
	}
	std::string fileName;
	getParserInfo(fileName, argc, argv);
	
	unsigned int MINIMUM_POINTS = 4;     // minimum number of cluster
	float EPSILON = 0.75;  // distance for clustering, metre^2  
	cv::Mat src(500,1000,CV_8UC3, cv::Scalar(255,255,255));  
	
	
	// read point data
	std::vector<PointWithC> points = readBenchmarkData(fileName);
	printResults(points);
	cv::Mat beforeCluster= src.clone();
	showPointWithCSet(beforeCluster, points);
	cv::imshow("beforeCluster", beforeCluster);
	

	// constructor
	DBSCAN ds(MINIMUM_POINTS, EPSILON, points);
	std::vector<PointWithC> result = ds.run();
	

	// result of DBSCAN algorithm
	printResults(result);
	
	cv::Mat afterCluster= src.clone();
	showPointWithCSet(afterCluster, result);
	cv::imshow("afterCluster", afterCluster);
	
	cv::waitKey(0);  

	return 0;
}
