#ifndef UTILS_H__
#define UTILS_H__
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ARECOG.h"
#include "filesystem.h"

void showimg(cv::Mat &img);
void showface(cv::Mat& img, cv::Rect &r);
void showAface(cv::Mat& img, ARect &r);

void showLandmarks(cv::Mat& image, cv::Rect& bbox, std::vector<cv::Point2f> &landmarks);
void adjustfaceRect(cv::Mat& src,cv::Rect& facerect,cv::Mat& bigface,cv::Rect& dst_efr);
void getNormfaceInbigface(cv::Mat& bigface,cv::Rect& efr,cv::Rect& r);

void rotateFaceOrin(cv::Mat& srcimg, ARect& src_r, cv::Mat& dstimg, ARect& dst_r);

int AFaceProcess_RotateOneFace(cv::Mat& image, ARect &face_rect_list, cv::Mat& dstImage, ARect &dst_efr);

void ExpandRect(const cv::Mat& src, cv::Mat& dst, cv::Rect& rect, cv::Rect& realPosi, cv::Rect& posiInbf);

#endif //UTILS_H__