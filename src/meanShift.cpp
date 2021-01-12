//#include "pch.h"
#include "meanShift.h"


MeanShiftClass::MeanShiftClass(const std::vector<PointWithC> points, int _bandwidth, cv::Mat rawImg)
{
	for (int i = 0; i < points.size(); ++i)
	{
		MeanShift meanShiftP;
		meanShiftP.pos.x = points[i].x;
		meanShiftP.pos.y = points[i].y;
		meanShiftP.res.x = points[i].x;
		meanShiftP.res.y = points[i].y;
		dataset.push_back(meanShiftP);
	}
	point_num = dataset.size();
	kernel_bandwidth = _bandwidth;
	stop = false;
	img = rawImg;
}


MeanShiftClass::~MeanShiftClass()
{
}

std::vector<std::vector<cv::Point2d>> MeanShiftClass::getCluster()
{
	for (int i = 0; i < point_num; i++)
	{
		stop = false;
		while (!stop)
		{
			ShiftOnce(dataset[i]);
		}
//		cout <<"("<< dataset[i].res.x << "," << dataset[i].res.y<<")" << endl;
	}
	std::vector<std::vector<cv::Point2d>> reCluster = LabelClusters();
	return reCluster;
}

void MeanShiftClass::ShowClusterResult()
{
	cv::Mat src = cv::imread("2.jpg");
	for (int i = 0; i < dataset.size(); i++)
	{
		cv::circle(src, dataset[i].pos, 3, cv::Scalar(0, 0, 255), -1);
	}
	cv::imshow("src", src);
	cv::waitKey(0);
}

int MeanShiftClass::GetManhattanDistance(cv::Point2f p0, cv::Point2f p1)
{
	return abs(p0.x - p1.x) + abs(p0.y - p1.y);
}

float MeanShiftClass::GetEuclideanDistance(cv::Point2f p0, cv::Point2f p1)
{
	return sqrt((p0.x - p1.x)*(p0.x - p1.x) + (p0.y - p1.y)*(p0.y - p1.y));
}

float MeanShiftClass::GaussianKernel(int distance, int bandwidth)
{
	return exp(-0.5*(distance*distance) / (bandwidth*bandwidth));
}

void MeanShiftClass::ShiftOnce(MeanShift& p)
{
	float x_sum = 0;
	float y_sum = 0;
	float weight_sum = 0;
	for (int i = 0; i < point_num; i++)
	{
		float tmp_distance = GetEuclideanDistance(p.res, dataset[i].pos);
		float weight = GaussianKernel(tmp_distance, kernel_bandwidth);
		x_sum += dataset[i].pos.x * weight;
		y_sum += dataset[i].pos.y * weight;
		weight_sum += weight;
	}
	cv::Point2f shift_vector(x_sum/ weight_sum, y_sum/ weight_sum);
	float shift_distance = GetEuclideanDistance(p.res, shift_vector);
	

	if (shift_distance < EPSILON)
		stop = true;

	p.res = shift_vector;
}

std::vector<std::vector<cv::Point2d>> MeanShiftClass::LabelClusters()
{
	std::vector<std::vector<cv::Point2d>> reCluster;
	int current_label = -1;
	for (int i = 0; i < point_num; i++)
	{
		if (dataset[i].label != -1)
			continue;
		current_label++;
		std::vector<cv::Point2d> oneCluster;
		for (int j = 0; j < point_num; j++)
		{
			if (GetEuclideanDistance(dataset[i].res, dataset[j].res) < CLUSTER_EPSILON)
			{
				dataset[j].label = current_label;
				oneCluster.push_back(cv::Point2d(dataset[i].pos.x, dataset[i].pos.y));
			}
		}
		reCluster.push_back(oneCluster);	
	}
	return reCluster;
}
