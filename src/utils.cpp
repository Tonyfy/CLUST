#include <iostream>
#include "../ARecog/utils.h"
#include "../ARecog/MRECOG.h"
#include "../ARecog/filesystem.h"

using namespace cv;
using namespace std;
void showimg(Mat &img)
{
	cv::imshow("img", img);
	cv::waitKey(0);
}

//void showAface(Mat& img, ARect& r)
//{
//	Mat img3ch(img.rows, img.cols, CV_8UC3);
//	cvtColor(img, img3ch, CV_GRAY2BGR);
//	rectangle(img3ch, r.rect, Scalar(255, 0, 0), 2);
//	showimg(img3ch);
//
//	circle(img3ch, Point(r.rect.x, r.rect.y), 2, Scalar(0, 255, 0), -1);
//	circle(img3ch, Point(r.rect.x+r.rect.width-1, r.rect.y+r.rect.height-1), 2, Scalar(0, 255, 0), -1);
//	for (int i = 0; i < r.ld.size(); i++)   //画出形状
//	{
//		circle(img3ch, r.ld[i], 2, Scalar(0, 255, 0), -1);
//	}
//	showimg(img3ch);
//}
void showface(Mat& img, Rect& r)
{

	Mat img3ch(img.rows, img.cols, CV_8UC3);
	cvtColor(img, img3ch, CV_GRAY2BGR);
	rectangle(img3ch, r, Scalar(255, 0, 0), 2);
	showimg(img3ch);
}

void showLandmarks(cv::Mat& image, Rect& bbox, vector<Point2f>& landmarks)
{
	Mat img;
	image.copyTo(img);
	rectangle(img, bbox, Scalar(0, 0, 255), 2);
	for (int i = 0; i < landmarks.size(); i++) {
		Point2f &point = landmarks[i];
		circle(img, point, 2, Scalar(0, 255, 0), -1);
	}
	showimg(img);
}


void ExpandRect(const Mat& src, Mat& dst, Rect& rect, Rect &realPosi, Rect &posiInbf)
{

	// 归一化人脸和矩形框
	const Mat& m = src;
	Mat expanded;
	Mat tmp;
	m.copyTo(tmp);
	//rectangle(tmp, rect, Scalar(0, 0, 255));
	//imshow("face", tmp);
	//waitKey(0);

	const double kPercentX = 0.4;
	const double kPercentY = 0.4;

	// 矩形框扩展后的点
	int left, top, right, bottom;
	left = rect.x - rect.width*kPercentX;
	top = rect.y - rect.height*kPercentY;
	right = rect.x + rect.width + rect.width*kPercentX;
	bottom = rect.y + rect.height + rect.height*kPercentY;
	// 实际图像中能够扩展到的点
	int real_left, real_top, real_right, real_bottom;
	real_left = max(0, left);
	real_top = max(0, top);
	real_right = min(right, m.cols - 1);
	real_bottom = min(bottom, m.rows - 1);
	// 新图像中的点
	int inner_left, inner_top, inner_right, inner_bottom;
	inner_left = real_left - left;
	inner_top = real_top - top;
	inner_right = real_right - left;
	inner_bottom = real_bottom - top;
	// 复制扩展后人脸区域到新图像
	int rows = bottom - top + 1;
	int cols = right - left + 1;
	expanded = Mat::zeros(rows, cols, m.type());
	Rect r1(inner_left, inner_top, inner_right - inner_left + 1, inner_bottom - inner_top + 1);
	Rect r2(real_left, real_top, real_right - real_left + 1, real_bottom - real_top + 1);

	//cout << "m\n" << r2 << endl;
	//cout << "expanded\n" << r1 << endl;
	//cout << expanded.size() << endl;

	m(r2).copyTo(expanded(r1));

	dst = expanded;
	realPosi = r2;
	posiInbf = r1;
}

int cosSimilarity(const Mat& q, const Mat& r, double& similarity)
{
    assert((q.rows==r.rows)&&(q.cols==r.cols));
    double fenzi = q.dot(r);
    double fenmu = sqrt(q.dot(q)) * sqrt(r.dot(r));
    similarity = fenzi/fenmu;
    return 0;
}
int ouSimilarity(const Mat& q, const Mat& r, double& similarity)
{
    assert((q.rows==r.rows)&&(q.cols==r.cols));
    similarity = (q - r).dot(q - r)/((q.dot(q))*(r.dot(r)));
    return 0;
}

