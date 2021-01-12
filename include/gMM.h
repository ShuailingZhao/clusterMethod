#ifndef KMEANS_H
#define KMEANS_H
#include <vector>
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

typedef struct Point_
{
    float x, y, z;  // X, Y, Z position
    int clusterID;  // clustered ID
}PointWithC;
class gMM {
public:
	gMM(const int classNum, const std::vector<PointWithC> points);
	cv::Mat run();
	std::vector<std::vector<cv::Point2d>> getCluster();
	~gMM();


    
private:
	int getClusterIndex(const std::vector<int> clusterIndexSet, int curPointClass);
	int K;
	cv::Mat samples_data;

};
#endif // DBSCAN_H
