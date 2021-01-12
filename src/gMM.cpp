#include "gMM.h"

gMM::gMM(const int classNum, const std::vector<PointWithC> points)
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
gMM::~gMM(){}

int gMM::getClusterIndex(const std::vector<int> clusterIndexSet, int curPointClass)
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
std::vector<std::vector<cv::Point2d>> gMM::getCluster()
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

cv::Mat gMM::run()
{
	cv::Mat labels;
	cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
	em_model->setClustersNumber(K);
	em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
	em_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.1));
	em_model->trainEM(samples_data, cv::noArray(), labels, cv::noArray());

	return labels;
}

