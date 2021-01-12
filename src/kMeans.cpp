#include "kMeans.h"

kMeans::kMeans(const int classNum, const std::vector<PointWithC> points)
{
	K = classNum;
	const int samples_number{ int(points.size()) }, categories_number{ K };
	samples_data.create(samples_number, 3, CV_32FC1);
	for (int i = 0; i < samples_number; ++i)
	{
		cv::Mat tmp = samples_data.rowRange(i,i+1);
		cv::Mat image(1,3, CV_32FC1);
		image.at<float>(0,0) = points[i].x;
		image.at<float>(0,1) = points[i].y;
		image.at<float>(0,2) = points[i].z;
		image.copyTo(tmp);
	}
	
}
kMeans::~kMeans(){}

int kMeans::getClusterIndex(const std::vector<int> clusterIndexSet, int curPointClass)
{
	for(int i=0;i<clusterIndexSet.size();i++)
	{
		if(clusterIndexSet[i] == curPointClass)
		{
			return i;
		}
	}
	return -1;
}
std::vector<std::vector<cv::Point2d>> kMeans::getCluster()
{
	
	cv::Mat labels = run();
	std::vector<std::vector<cv::Point2d>> reCluster;
	std::vector<int> clusterIndexSet;
	for(int i=0;i<labels.rows;i++)
	{
		int index = getClusterIndex(clusterIndexSet, labels.at<int>(i));
		if(index>=0)
		{
			reCluster[index].push_back(cv::Point2d(samples_data.at<float>(i,0),samples_data.at<float>(i,1)));
		}else{
			std::vector<cv::Point2d> newPoint;
			newPoint.push_back(cv::Point2d(samples_data.at<float>(i,0),samples_data.at<float>(i,1)));
			reCluster.push_back(newPoint);
			clusterIndexSet.push_back(labels.at<int>(i));
		}	
	}
	return reCluster;
}

cv::Mat kMeans::run()
{
	const int attemps{ 5 };
	const cv::TermCriteria term_criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.01);
	cv::Mat labels_, centers_;
	double value = cv::kmeans(samples_data, K, labels_, term_criteria, attemps, cv::KMEANS_RANDOM_CENTERS, centers_);
	double sum=0;
	for(int i=0;i<labels_.rows;i++)
	{
		sum += labels_.at<int>(i);
	}
 
	return labels_;
}

