#ifndef MEANSHIFT_H
#define MEANSHIFT_H
#include <iostream>
#include <opencv2/opencv.hpp>
#define PI 3.1415926535898
#define EPSILON 0.01
#define CLUSTER_EPSILON 20

#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3

typedef struct Point_
{
    float x, y, z;  // X, Y, Z position
    int clusterID;  // clustered ID
}PointWithC;
struct MeanShift
{
	cv::Point2f pos;
	cv::Point2f res;
	int label = -1;
};

class MeanShiftClass
{
public:
	MeanShiftClass(const std::vector<PointWithC> points, int _bandwidth, cv::Mat rawImg);
	~MeanShiftClass();
	std::vector<std::vector<cv::Point2d>> getCluster();
private:
	bool stop;
	int point_num;
	int cluster_num;
	int kernel_bandwidth;
	std::vector<MeanShift> dataset;

	std::vector<std::vector<cv::Point2d>> LabelClusters();
	void ShowClusterResult();
	void ShiftOnce(MeanShift& p);
	int GetManhattanDistance(cv::Point2f p0, cv::Point2f p1);
	float GaussianKernel(int distance, int bandwidth);
	float GetEuclideanDistance(cv::Point2f p0, cv::Point2f p1);
	cv::Mat img;
};
#endif
