#ifndef UTILS_H__
#define UTILS_H__
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void showimg(cv::Mat &img);
void showface(cv::Mat& img, cv::Rect &r);


void showLandmarks(cv::Mat& image, cv::Rect& bbox, std::vector<cv::Point2f> &landmarks);
void ExpandRect(const cv::Mat& src, cv::Mat& dst, cv::Rect& rect, cv::Rect& realPosi, cv::Rect& posiInbf);
int cosSimilarity(const cv::Mat& q, const cv::Mat& r, double& similarity);
int ouSimilarity(const cv::Mat& q, const cv::Mat& r, double& similarity);


#endif //UTILS_H__