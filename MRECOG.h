#ifndef MRECOG_H__
#define MRECOG_H__

#include <opencv2/objdetect/objdetect.hpp>
#include "featureExByCaffe.h"
#include "ARECOG.h"
#include "faceAlign.h"
#include "caffeFaceVal.h"


class MRECOG :public ARECOG
{
public:

	int A_Init(const char *modulepath) override;
	int A_UnInit() override;
	int AFaceProcess_ReadImage(const std::string &imgpath, cv::Mat& img) override;
	int AFaceProcess_SaveImage(const cv::Mat& img, std::string &savepath) override;
	int AFaceProcess_Facedetect(const cv::Mat& image, int& list_size,std::vector<ARect> &face_rect_list, int method = 0) override;
    int compareFace(cv::Mat& queryface, cv::Mat& refface, double& similarity) override;
	int AFaceProcess_FeatureCompare(const AFeature& query_feature,const AFeature& ref_feature, double& similarity)override;
	int AFaceProcess_GetfaceFeature(cv::Mat& face, AFeature& feature) override;
	int AFaceProcess_Landmark(cv::Mat& gray, cv::Rect& r,ARect& ar) override;
	int AFaceProcess_Getface(cv::Mat& image, ARect& facerect, cv::Mat& face)override;
	int AFaceProcess_GetFaceFeature(cv::Mat& image, ARect& facerect,AFeature& feature) override;

	void rotateFaceOrin(cv::Mat& srcimg, ARect& src_r, cv::Mat& dstimg, ARect& dst_r);
	int AFaceProcess_RotateOneFace(cv::Mat& image, ARect &face_rect_list, cv::Mat& dstImage, ARect &dst_efr);
	
public:
	CascadeClassifier kcc;
	CascadeClassifier kccp;
	CaffeFaceValidator *cfv;
	Landmarker lder;
	string protonet;
	string caffemodel;
	featureExer *fe;

};

#endif //MRECOG_H__