#ifndef __FACEALIGN_H__
#define __FACEALIGN_H__

#include <cassert>
#include <vector>
#include <string>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

struct BBox {
	int x, y;
	int width, height;
	cv::Rect rect;

	BBox(int x, int y, int w, int h);
	BBox(const cv::Rect &rect);
	BBox(BBox const &other);
	void Project(const std::vector<cv::Point2f> &absLandmark, std::vector<cv::Point2f> &relLandmark) const;
	void ReProject(const std::vector<cv::Point2f> &realLandmark, std::vector<cv::Point2f> &absLandmark) const;
	BBox subBBox(float left, float right, float top, float bottom) const;
};

struct FaceDetector {
	cv::CascadeClassifier cc;

	void LoadXML(const std::string &path);
	int DetectFace(const cv::Mat &img, std::vector<cv::Rect> &rects);
};

struct CNN {
	caffe::Net<float> *cnn;

	CNN(const std::string &network, const std::string &model);
	std::vector<cv::Point2f> forward(const cv::Mat &data, const std::string &layer);
};

struct Landmarker {
	CNN *F_1;

	void LoadModel(const std::string &path);
	std::vector<cv::Point2f> DetectLandmark(const cv::Mat &img, const BBox &bbox);
};

#endif // __FACEALIGN_H__
