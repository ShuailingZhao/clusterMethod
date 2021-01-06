#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3

//using namespace std;

typedef struct Point_
{
    float x, y, z;  // X, Y, Z position
    int clusterID;  // clustered ID
}PointWithC;

class DBSCAN {
public:    
	DBSCAN(unsigned int minPts, float eps, std::vector<PointWithC> points);
	~DBSCAN();

	std::vector<PointWithC> run();
	std::vector<std::vector<cv::Point2d>> getCluster();
	int getTotalPointSize();
	int getMinimumClusterSize();
	int getEpsilonSize();
    
private:
	int getClusterIndex(const std::vector<int> clusterIndexSet, int curPointClass);
	std::vector<int> calculateCluster(PointWithC point);
	int expandCluster(PointWithC point, int clusterID);
	inline double calculateDistance(PointWithC pointCore, PointWithC pointTarget);
	
	std::vector<PointWithC> m_points;
	unsigned int m_pointSize;
	unsigned int m_minPoints;
	float m_epsilon;
};
#endif // DBSCAN_H
