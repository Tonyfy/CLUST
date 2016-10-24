#ifndef ARECOG_H__
#define ARECOG_H__
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "fastCluster.h"
//#include "mysqlCPP.h"

typedef struct AlphaFeature
{

	cv::Mat feature;
}AFeature;

typedef struct AlphaRect
{
	cv::Rect rect;
	std::vector<cv::Point> ld;  //le,re,nose,lm,rm,cm;
	float face_score;
}ARect;

struct CFace
{
	std::string srcpath;
	ARect facerect;
	AFeature facefeature;
	int facelabel;
	bool isclustCenter;
	double x_width;
	double y_height;
};

#define REG_SUCCESS                        0
#define REG_DOUBLENAME_OTHERPERSON         1
#define REG_SMALL_SIZE                     2
#define REG_NOFACE                         3
#define REG_MANYFACE                       4
#define REG_ERROR_SQL                      5

class ARECOG
{
public:
	ARECOG()=default;
	~ARECOG()=default;
	virtual int A_Init(const char *modelpath)=0;
	virtual int A_UnInit() = 0;
	virtual void getnormface(const std::string& path)=0;

	virtual int AFaceProcess_ReadImage(const std::string &imgpath, cv::Mat& img) = 0;
	virtual int AFaceProcess_SaveImage(const cv::Mat& eimg, std::string& savepath) = 0;
	virtual int AFaceProcess_Facedetect(const cv::Mat& image, int& list_size,std::vector<ARect> &face_rect_list, int method = 0) = 0;
	virtual int AFaceProcess_Landmark(cv::Mat& gray, cv::Rect& r,ARect& ar) = 0;
	virtual int AFaceProcess_Getface(cv::Mat& image, ARect& facerect, cv::Mat& face) = 0;
	virtual int AFaceProcess_GetfaceFeature(cv::Mat& face, AFeature& feature) = 0;
	virtual int AFaceProcess_GetFaceFeature(cv::Mat& image, ARect& facerect,AFeature& feature) = 0;
	virtual int AFaceProcess_FeatureCompare(const AFeature& query_feature,const AFeature& ref_feature, double& similarity) = 0;
	virtual int compareFace(cv::Mat& queryface, cv::Mat& refface, double& similarity)=0;
	
	virtual int AFaceProcess_GetDist(const std::vector<CFace>& cfaces, cv::Mat &dist) = 0;

	virtual int AFaceProcess_Clust(const std::vector<CFace> &cfaces, Cluster cltr, std::vector<datapoint> &result) = 0;


	virtual int AFaceProcess_regImage(const std::string &imgpath, const std::string &id) = 0;
public:

};

#endif //ARECOG_H__