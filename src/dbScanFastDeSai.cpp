#include "dbScanFast.h"

DBSCAN::DBSCAN(unsigned int minPts, float eps, const std::vector<PointWithC> points, const cv::Mat lane_mask, const cv::Mat Index)
{
        m_minPoints = minPts;
        m_epsilon = eps;
        m_points = points;
        mask = lane_mask;
        maskIndex = Index;
        m_pointSize = points.size();
}
DBSCAN::~DBSCAN(){}
std::vector<PointWithC> DBSCAN::run()
{
	int clusterID = 1;
	for(std::vector<PointWithC>::iterator iter = m_points.begin(); iter != m_points.end(); ++iter)
	{
		if ( iter->clusterID == UNCLASSIFIED )
		{
			if ( expandCluster(*iter, clusterID) != FAILURE )
			{
				clusterID += 1;
			}
		}

	}
	

	return m_points;
}

int DBSCAN::getClusterIndex(const std::vector<int> clusterIndexSet, int curPointClass)
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
std::vector<std::vector<cv::Point2d>> DBSCAN::getCluster()
{
	std::vector<PointWithC> pointsWithC = run();
	std::vector<std::vector<cv::Point2d>> reCluster;
	std::vector<int> clusterIndexSet;
	for(int i=0;i<pointsWithC.size();i++)
	{
		int index = getClusterIndex(clusterIndexSet, pointsWithC[i].clusterID);
		if(index>=0)
		{
			reCluster[index].push_back(cv::Point2d(pointsWithC[i].x, pointsWithC[i].y));
		}else{
			std::vector<cv::Point2d> newPoint;
			newPoint.push_back(cv::Point2d(pointsWithC[i].x, pointsWithC[i].y));
			reCluster.push_back(newPoint);
			clusterIndexSet.push_back(pointsWithC[i].clusterID);
		}	
	}
	return reCluster;
}


int DBSCAN::expandCluster(PointWithC point, int clusterID)
{    
    std::vector<int> clusterSeeds = calculateCluster(point);

    if ( clusterSeeds.size() < m_minPoints )
    {
        point.clusterID = NOISE;
        return FAILURE;
    }
    else
    {
        int index = 0, indexCorePoint = 0;
        std::vector<int>::iterator iterSeeds;
        for( iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            m_points.at(*iterSeeds).clusterID = clusterID;
            if (m_points.at(*iterSeeds).x == point.x && m_points.at(*iterSeeds).y == point.y && m_points.at(*iterSeeds).z == point.z )
            {
                indexCorePoint = index;
            }
            ++index;
        }
        clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);

        for(std::vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
        {
            std::vector<int> clusterNeighors = calculateCluster(m_points.at(clusterSeeds[i]));
            if ( clusterNeighors.size() >= m_minPoints )
            {
                std::vector<int>::iterator iterNeighors;
                for ( iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors )
                {
                    if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED || m_points.at(*iterNeighors).clusterID == NOISE )
                    {
                        if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED )
                        {
                            clusterSeeds.push_back(*iterNeighors);
                            n = clusterSeeds.size();
                        }
                        m_points.at(*iterNeighors).clusterID = clusterID;
                    }
                }
            }
        }

        return SUCCESS;
    }
}

std::vector<int> DBSCAN::calculateCluster(PointWithC point)
{
    int index = 0;
    std::vector<PointWithC>::iterator iter;
    std::vector<int> clusterIndex;
    for( iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( calculateDistance(point, *iter) <= m_epsilon )
        {
            clusterIndex.push_back(index);
        }
        index++;
    }
    return clusterIndex;
}

inline double DBSCAN::calculateDistance( PointWithC pointCore, PointWithC pointTarget )
{
    return sqrt(pow(pointCore.x - pointTarget.x,2)+pow(pointCore.y - pointTarget.y,2)+pow(pointCore.z - pointTarget.z,2));
}

int DBSCAN::getTotalPointSize()
{
	return m_pointSize;
}
int DBSCAN::getMinimumClusterSize()
{
	return m_minPoints;
}
int DBSCAN::getEpsilonSize()
{
	return m_epsilon;
}


