#include "utils.h"
#include "MRECOG.h"

void showimg(cv::Mat &img)
{
	cv::imshow("img", img);
	cv::waitKey(0);
}

void showAface(Mat& img, ARect& r)
{
	Mat img3ch(img.rows, img.cols, CV_8UC3);
	cvtColor(img, img3ch, CV_GRAY2BGR);
	rectangle(img3ch, r.rect, Scalar(255, 0, 0), 2);
	showimg(img3ch);

	circle(img3ch, Point(r.rect.x, r.rect.y), 2, Scalar(0, 255, 0), -1);
	circle(img3ch, Point(r.rect.x+r.rect.width-1, r.rect.y+r.rect.height-1), 2, Scalar(0, 255, 0), -1);
	for (int i = 0; i < r.ld.size(); i++)   //画出形状
	{
		circle(img3ch, r.ld[i], 2, Scalar(0, 255, 0), -1);
	}
	showimg(img3ch);
}
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

void adjustfaceRect(Mat& src, ARect &facerect, Mat& bigface, ARect &dst_efr)
{
	double eup = 0.60;
	double eleft = 0.60;
	double eright = 0.60;
	double edown = 0.60;

	int width = facerect.rect.width;
	int height = facerect.rect.height;

	// 矩形框扩展后的点
	int left, top, right, bottom;
	left = facerect.rect.x - width*eleft;
	top = facerect.rect.y - height*eup;
	right = facerect.rect.x + width + width*eright;
	bottom = facerect.rect.y + height + height*edown;

	// 实际图像中能够扩展到的点
	int real_left, real_top, real_right, real_bottom;
	real_left = max(0, left);
	real_top = max(0, top);
	real_right = min(right, src.cols - 1);
	real_bottom = min(bottom, src.rows - 1);
	// 新图像中的点
	int inner_left, inner_top, inner_right, inner_bottom;
	inner_left = real_left - left;
	inner_top = real_top - top;
	inner_right = real_right - left;
	inner_bottom = real_bottom - top;
	// 复制扩展后人脸区域到新图像
	int rows = bottom - top + 1;
	int cols = right - left + 1;
	int RC = min(rows, cols);  //防止rows cols差1
	Mat tmp = Mat::zeros(RC, RC, CV_8UC1);
	int WH = min(inner_right - inner_left + 1, inner_bottom - inner_top + 1);
	Rect r1(inner_left, inner_top, WH, WH);
	Rect r2(real_left, real_top, WH, WH);

	//cout << "m\n" << r2 << endl;
	//cout << "expanded\n" << r1 << endl;
	//cout << expanded.size() << endl;

	src(r2).copyTo(tmp(r1));
	tmp.copyTo(bigface);

	//将大脸放回原图,由于bf是大脸实际能取到的原图中的区域，
	//realposi是bf区域在大脸中的位置（当未出现越界时，bf和大脸一样大）
	Rect realposi = r2;
	Rect bf = r1;
	//showimg(dstimg);
	// Result  从大脸坐?晗当浠坏皆甲晗?
	int x_shift = bf.x - realposi.x;
	int y_shift = bf.y - realposi.y;


	dst_efr.rect.x = facerect.rect.x + x_shift;
	dst_efr.rect.y = facerect.rect.y + y_shift;
	dst_efr.rect.width = facerect.rect.width;
	dst_efr.rect.height = facerect.rect.height;
	//x_shift和y_shift可能不同。需要再次变为正方形
	int W_H2 = max(dst_efr.rect.width, dst_efr.rect.height);
	dst_efr.rect.x -= (W_H2 - dst_efr.rect.width) / 2;
	dst_efr.rect.y -= (W_H2 - dst_efr.rect.height) / 2;
	dst_efr.rect.width +=  (W_H2 - dst_efr.rect.width);
	dst_efr.rect.height = dst_efr.rect.width;  //强制变成正方形

	dst_efr.ld.clear();
	dst_efr.ld.push_back(Point(facerect.ld[0].x + x_shift,facerect.ld[0].y + y_shift));
	dst_efr.ld.push_back(Point(facerect.ld[1].x + x_shift,facerect.ld[1].y + y_shift));
	dst_efr.ld.push_back(Point(facerect.ld[2].x + x_shift,facerect.ld[2].y + y_shift));
	dst_efr.ld.push_back(Point(facerect.ld[3].x + x_shift,facerect.ld[3].y + y_shift));
	dst_efr.ld.push_back(Point(facerect.ld[4].x + x_shift,facerect.ld[4].y + y_shift));
	dst_efr.ld.push_back(Point(facerect.ld[5].x + x_shift,facerect.ld[5].y + y_shift));
	
	dst_efr.face_score = facerect.face_score;

}

void getNormfaceInbigface(Mat& bigface, ARect &efr, Rect &r)
{
	//showEface(bigface, efr);
	//归一化到需要的大小和位置
	int ec_y = efr.rect.y;
	int ec_mc_y = efr.ld[5].y - ec_y;

	int newWH = (int)(ec_mc_y*(1 + 80 / 48.0));
	r.width = newWH;
	r.height = newWH;
	r.y = efr.ld[0].y - ec_mc_y * 40 / 48.0;
	r.x = (bigface.cols - r.width) / 2;
}

void rotateFaceOrin(Mat &srcimg, ARect &efr, Mat &dstimg, ARect &dst_efr)
{
	assert((dstimg.rows == srcimg.rows) && (dstimg.cols == srcimg.cols) && (dstimg.type() == srcimg.type()));

	Mat dsttmp;
	Rect bf;   //记录大脸dsttmp实际取到的位置realgetR
	Rect realposi;
	ExpandRect(srcimg, dsttmp, efr.rect, bf, realposi);  //将原人脸区域frect扩展20%，得到dsttmp大脸，在dsttmp上进行旋转。
	//showimg(dsttmp);

	const double PIE = CV_PI;
	Point A = Point(efr.ld[0].x, efr.ld[0].y);
	Point B = Point(efr.ld[1].x, efr.ld[1].y);

	double angle = 180 * atan((B.y - A.y) / (double)(B.x - A.x + 1e-12)) / PIE;  //角度制
	double scale = 1.0;
	double cita = atan((B.y - A.y) / (double)(B.x - A.x + 1e-12));//弧度制

	vector<Point> attr;  //6个属性点在大脸中的坐标
	attr.push_back(Point(dsttmp.cols * 4 / 18.0, dsttmp.rows * 4 / 18.0));
	attr.push_back(Point(dsttmp.cols * 14 / 18.0, dsttmp.rows * 14 / 18.0));
	attr.push_back(Point(efr.ld[0].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[0].y - efr.rect.y + dsttmp.rows * 4 / 18.0));
	attr.push_back(Point(efr.ld[1].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[1].y - efr.rect.y + dsttmp.rows * 4 / 18.0));
	attr.push_back(Point(efr.ld[2].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[2].y - efr.rect.y + dsttmp.rows * 4 / 18.0));
	attr.push_back(Point(efr.ld[3].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[3].y - efr.rect.y + dsttmp.rows * 4 / 18.0));	
	attr.push_back(Point(efr.ld[4].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[4].y - efr.rect.y + dsttmp.rows * 4 / 18.0));
	attr.push_back(Point(efr.ld[5].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[5].y - efr.rect.y + dsttmp.rows * 4 / 18.0));  //centermouth

	for (int i = 0; i < attr.size(); i++)   //转存到旋转后的形状pA2
	{
		double dis = (double)sqrt(pow((double)attr[i].x - dsttmp.cols / 2.0, 2)
			+ pow((double)attr[i].y - dsttmp.rows / 2.0, 2));
		double cita0;
		if (attr[i].x > dsttmp.cols / 2.0)  //1,4象限
		{
			cita0 = -atan((double)(attr[i].y - dsttmp.rows / 2.0) / (attr[i].x - dsttmp.cols / 2.0 + 1e-12));
		}
		else if ((attr[i].x < dsttmp.cols / 2.0) && (attr[i].y <= dsttmp.rows / 2.0)) //2象限
		{
			cita0 = PIE - atan((attr[i].y - dsttmp.rows / 2.0) / (attr[i].x - dsttmp.cols / 2.0 + 1e-12));
		}
		else if ((attr[i].x < dsttmp.cols / 2.0) && (attr[i].y > dsttmp.rows / 2.0))  //3象限
		{
			cita0 = -1 * PIE - atan((attr[i].y - dsttmp.rows / 2.0) / (attr[i].x - dsttmp.cols / 2.0 + 1e-12));
		}
		else if (attr[i].x == dsttmp.cols / 2.0)    //垂直于y轴
		{
			if (attr[i].y > dsttmp.rows / 2.0)
			{
				cita0 = -1 * PIE / 2.0;
			}
			else if (attr[i].y <= dsttmp.rows / 2.0)
			{
				cita0 = PIE / 2.0;
			}
		}
		else{ cout << "error" << endl; }
		if ((((int)((dsttmp.cols / 2.0) + dis*cos(cita0 + cita))) >= 0) && ((int)(((dsttmp.cols / 2.0) + dis*cos(cita0 + cita))) <= dsttmp.cols - 1))
		{
			attr[i].x = (int)((dsttmp.cols / 2.0) + dis*cos(cita0 + cita));
		}
		else if (((int)(((dsttmp.cols / 2.0) + dis*cos(cita0 + cita)))) < 0)
		{
			attr[i].x = 0;
		}
		else if ((((int)((dsttmp.cols / 2.0) + dis*cos(cita0 + cita)))) >= dsttmp.cols)
		{
			attr[i].x = dsttmp.cols - 1;
		}
		else
		{
			cout << "error with x cordinate= " << attr[i].x << endl;
		}

		if ((((int)((dsttmp.rows / 2.0) - dis*sin(cita0 + cita))) >= 0) && (((int)((dsttmp.rows / 2.0) - dis*sin(cita0 + cita))) <= dsttmp.rows - 1))
		{
			attr[i].y = (int)((dsttmp.rows / 2.0) - dis*sin(cita0 + cita));
		}
		else if ((((int)((dsttmp.rows / 2.0) - dis*sin(cita0 + cita))) < 0))
		{
			attr[i].y = 0;
		}
		else if ((int)(((dsttmp.rows / 2.0) - dis*sin(cita0 + cita))) >= dsttmp.rows)
		{
			attr[i].y = dsttmp.rows - 1;
		}
		else
		{
			cout << "error with y cordinate= " << attr[i].y << endl;
		}
	}
	cv::Mat rot_mat(2, 3, CV_32FC1);
	Point center = Point(dsttmp.cols / 2, dsttmp.rows / 2);
	//std::cout<<"angle= "<<angle<<endl;
	rot_mat = getRotationMatrix2D(center, angle, scale);

	Mat rottmp(dsttmp.rows, dsttmp.cols, dsttmp.type());
	cv::warpAffine(dsttmp, rottmp, rot_mat, dsttmp.size());

	//在大脸上完成旋转,观测旋转后各信息是否正确
	//由于旋转后，人脸框的左上和右下两点发生变化，会变成一个长方形。要将其拓宽为正方形。
	int oriW = attr[1].x - attr[0].x;
	int oriH = attr[1].y - attr[0].y;
	int W_H = max(oriW, oriH);

	attr[0].x -= (W_H - oriW) / 2;
	attr[0].y -= (W_H - oriH) / 2;
	attr[1].x += (W_H - oriW) / 2;
	attr[1].y = attr[0].y + (attr[1].x - attr[0].x); //强制使左上和右下之间构成正方形
	assert(attr[1].x - attr[0].x == attr[1].y - attr[0].y);

	//将大脸放回原图,由于bf是大脸实际能取到的原图中的区域，
	//realposi是bf区域在大脸中的位置（当未出现越界时，bf和大脸一样大）
	rottmp(realposi).copyTo(dstimg(bf));

	//showimg(dstimg);
	// Result  从大脸坐标系变换到原图坐标系
	int x_shift = bf.x - realposi.x;
	int y_shift = bf.y - realposi.y;


	dst_efr.rect.x = attr[0].x + x_shift;
	dst_efr.rect.y = attr[0].y + y_shift;
	dst_efr.rect.width = attr[1].x - attr[0].x;
	dst_efr.rect.height = attr[1].y - attr[0].y;
	//x_shift和y_shift可能不同。需要再次变为正方形
	int W_H2 = max(dst_efr.rect.width, dst_efr.rect.height);
	
	dst_efr.rect.x -= (W_H2 - dst_efr.rect.width) / 2;
	dst_efr.rect.y -= (W_H2 - dst_efr.rect.height) / 2;
	dst_efr.rect.width += (W_H2 - dst_efr.rect.width);
	dst_efr.rect.height += (W_H2 - dst_efr.rect.height);

	dst_efr.ld.clear();
	dst_efr.ld.push_back(Point(attr[2].x + x_shift,attr[2].y + y_shift));
	dst_efr.ld.push_back(Point(attr[3].x + x_shift,attr[3].y + y_shift));
	dst_efr.ld.push_back(Point(attr[4].x + x_shift,attr[4].y + y_shift));
	dst_efr.ld.push_back(Point(attr[5].x + x_shift,attr[5].y + y_shift));
	dst_efr.ld.push_back(Point(attr[6].x + x_shift,attr[6].y + y_shift));	
	dst_efr.ld.push_back(Point(attr[7].x + x_shift,attr[7].y + y_shift));
	
	dst_efr.face_score = efr.face_score;

	//showface(dstimg,dst_efr);

}

int AFaceProcess_RotateOneFace(Mat& image, ARect &face_rect_list,
	Mat& dstImage, ARect& dst_efr)
{
	rotateFaceOrin(image, face_rect_list, dstImage, dst_efr);
	return 0;
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

