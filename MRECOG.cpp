#include "MRECOG.h"
#include "utils.h"

#include "filesystem.h"


using namespace cv;
using namespace std;

int MRECOG::A_Init(const char *modulepath)
{
	string kccpath = string(modulepath) + "/face.xml";
	string kccppath = string(modulepath) + "/pface.xml";
	string cfvnetpath = string(modulepath) + "/cppnet.prototxt";
	string cfvmodelpath = string(modulepath) + "/_iter_450000.caffemodel";
	string lderpath = string(modulepath) + "/deeplandmark";
	string prototxtpath = string(modulepath) + "/B.prototxt";
	string caffemodelpath = string(modulepath) + "/B.caffemodel";
	if (!FileSystem::isExists(string(modulepath)))
	{
		cerr << "module path is invalid!" << endl;
		return -1;
	}
	else if ((!FileSystem::isExists(kccpath)) || (!FileSystem::isExists(kccppath))
		|| (!FileSystem::isExists(cfvnetpath)) || (!FileSystem::isExists(cfvmodelpath))
		||(!FileSystem::isExists(lderpath)) || (!FileSystem::isExists(prototxtpath))
		|| (!FileSystem::isExists(caffemodelpath)))
	{
		cerr << "check your kccpath or kccppath or cfvnetpath or cfvmodelpath or lderpath or facerecog's proto or model." << endl;
		return -1;
	}
	//初始化M的模型
	kcc = CascadeClassifier(kccpath);
	kccp = CascadeClassifier(kccppath);
	cfv =new CaffeFaceValidator(cfvnetpath, cfvmodelpath);
	lder.LoadModel(lderpath);
	//protonet = string(modulepath) + "/DeepFacenet.prototxt";
	protonet = prototxtpath;
	//caffemodel = string(modulepath) + "/DeepFacenet_iter.caffemodel";
	caffemodel = caffemodelpath;
	fe = new featureExer(protonet, caffemodel);

	return 0;
}

int MRECOG::A_UnInit()
{
	delete cfv;
	delete fe;

	return 0;
}

int MRECOG::AFaceProcess_ReadImage(const std::string &imgpath, Mat& img)
{
	img = imread(imgpath, 0);
	return 0;
}

int MRECOG::AFaceProcess_SaveImage(const Mat& img, std::string &savepath)
{

	imwrite(savepath, img);
	return 0;
}


int MRECOG::AFaceProcess_Facedetect(const Mat& image, int& list_size,
	std::vector<ARect> &face_rect_list, int method)
{
	Mat gray, gray2;
	assert(image.data != NULL);
	if (image.type() == CV_8UC3) 
	{
		cvtColor(image, gray, CV_BGR2GRAY);
		cvtColor(image, gray2, CV_BGR2GRAY);
	}
	else if (image.type() == CV_8UC1) 
	{
		image.copyTo(gray);
		image.copyTo(gray2);
	}
	else
	{
		return -1;
	}

	int flag = 1;
	int norm_width, norm_height;
	if (((gray.rows > 600) && (gray.rows <= 1200)) || ((gray.cols > 600) && (gray.cols <= 1200)))
	{
		flag = 2;
	}
	else if (((gray.rows > 1200) && (gray.rows <= 2400)) || ((gray.cols > 1200) && (gray.cols <= 2400)))
	{
		flag = 3;
	}
	else if ((gray.rows > 2400) || (gray.cols > 2400))
	{
		flag = 5;
	}

	norm_width = gray.cols / flag;
	norm_height = gray.rows / flag;

	resize(gray,gray,Size(norm_width,norm_height));

	equalizeHist(gray, gray);

	vector<Rect> rects;
	if (method == 0)
	{
		kcc.detectMultiScale(gray,
			rects,
			1.1,
			2,
			0 | CV_HAAR_SCALE_IMAGE,
			Size(50, 50),
			Size(gray.cols , gray.rows ));
	}
	else if (method==1)
	{
		kccp.detectMultiScale(gray,
			rects,
			1.2,
			2,
			0 | CV_HAAR_SCALE_IMAGE,
			Size(30, 30),
			Size(2 * gray.cols / 3, 2 * gray.rows / 3));
	}
	else
	{
		cerr << "method must be 0 or 1" << endl;
	}

	if (flag > 1)
	{
		for (int i = 0; i < rects.size(); i++)
		{
			rects[i].x *= flag;
			rects[i].y *= flag;
			rects[i].width *= flag;
			rects[i].height *= flag;
		}
	}

	list_size = rects.size();
	//对每一张人脸，记录其面部属性
	vector<Point2f> landmarks;
	for (int i = 0; i < rects.size(); i++)
	{
		//使用caffe判断是否为人脸
		bool result = true;
		float sc = 0;
		Mat imgforcaffe;
		gray2(rects[i]).convertTo(imgforcaffe, CV_32F, 1.0 / 255.0);
		cfv->validate(imgforcaffe, result, sc);
		if (result)
		{
			//如果是人脸，则提取相应属性
			
			ARect tmpfrect;
			tmpfrect.ld.clear();

			tmpfrect.rect.x = rects[i].x;
			tmpfrect.rect.y = rects[i].y;
			tmpfrect.rect.width = rects[i].width;
			tmpfrect.rect.height = rects[i].height;
			
			BBox bbox_ = BBox(rects[i]).subBBox(0.1, 0.9, 0.2, 1);
			landmarks = lder.DetectLandmark(gray2, bbox_);
			//showLandmarks(gray, bbox_.rect, landmarks);
			tmpfrect.ld.push_back(Point((int)landmarks[0].x,(int)landmarks[0].y));			
			tmpfrect.ld.push_back(Point((int)landmarks[1].x,(int)landmarks[1].y));
			tmpfrect.ld.push_back(Point((int)landmarks[2].x,(int)landmarks[2].y));			
			tmpfrect.ld.push_back(Point((int)landmarks[3].x,(int)landmarks[3].y));
			tmpfrect.ld.push_back(Point((int)landmarks[4].x,(int)landmarks[4].y));				
			int cm_x=(tmpfrect.ld[3].x+tmpfrect.ld[4].x)/2;
			int cm_y=(tmpfrect.ld[3].y+tmpfrect.ld[4].y)/2;
			tmpfrect.ld.push_back(Point(cm_x,cm_y));			
			
			tmpfrect.face_score = sc;
			face_rect_list.push_back(tmpfrect);
		}
		else
		{
			list_size--;
		}
	}
	return 0;
}


int MRECOG::compareFace(Mat& queryface, Mat& refface, double& similarity)
{
	assert((queryface.type()==CV_8UC1)&&(refface.type()==CV_8UC1));
	Mat q(queryface.rows, queryface.cols, CV_8UC1);
	Mat r(refface.rows, refface.cols, CV_8UC1);
	queryface.copyTo(q);
	refface.copyTo(r);
	resize(q, q, cvSize(128, 128));
	q.convertTo(q, CV_32FC1, 1.0 / 255.0);
	
	resize(r, r, cvSize(128, 128));
	r.convertTo(r, CV_32FC1, 1.0 / 255.0);

	Mat qf(256, 1, CV_32FC1);
	Mat rf(256, 1, CV_32FC1);
	fe->extractfeature(q, qf);
	fe->extractfeature(r, rf);

	cosSimilarity(qf,rf,similarity);

	return 0;
	
}

int MRECOG::AFaceProcess_FeatureCompare(const AFeature& query_feature,
	const AFeature& ref_feature, double& similarity)
{
	cosSimilarity(query_feature.feature, ref_feature.feature, similarity);
	return 0;
}

int MRECOG::AFaceProcess_GetFaceFeature(Mat& image, ARect& facerect,
	AFeature& feature)
{
	Mat face;
	AFaceProcess_Getface(image, facerect, face);

	AFaceProcess_GetfaceFeature(face, feature);
	return 0;
}


int MRECOG::AFaceProcess_GetfaceFeature(cv::Mat& face, AFeature &feature)
{
	assert(face.type() == CV_8UC1);
	resize(face, face, cvSize(128, 128));
	face.convertTo(face, CV_32FC1, 1.0 / 255.0);

	double start = cv::getTickCount();
	for (int i = 0; i < 1000;)
	{
		i++;
		fe->extractfeature(face, feature.feature);
	}
	double extractf_cost = (cv::getTickCount() - start) / cv::getTickFrequency();
	cout << "ave extract cost " << extractf_cost / 1000.0 << endl;

	fe->extractfeature(face, feature.feature);

	return 0;
}


int  MRECOG::AFaceProcess_Landmark(cv::Mat& gray, cv::Rect& r,ARect& ar)
{
	vector<Point2f> landmarks;
	BBox bbox_ = BBox(r).subBBox(0.1, 0.9, 0.2, 1);
	landmarks = lder.DetectLandmark(gray, bbox_);

	ar.rect=r;
	ar.ld.clear();
	ar.ld.push_back(landmarks[0]);
	ar.ld.push_back(landmarks[1]);
	ar.ld.push_back(landmarks[2]);
	ar.ld.push_back(landmarks[3]);
	ar.ld.push_back(landmarks[4]);
	ar.ld.push_back(Point((landmarks[3].x+landmarks[4].x)/2,(landmarks[3].y+landmarks[4].y)/2));
	ar.face_score=10000;
	return 0;
}

int MRECOG::AFaceProcess_Getface(cv::Mat& image, ARect& facerect, Mat& face)
{
	ARect dst_r;
	Mat dstimg(image.rows,image.cols,image.type());
	image.copyTo(dstimg);
	//showimg(image);
	rotateFaceOrin(image, facerect, dstimg, dst_r);
	//showimg(dstimg);
	ARect fr_bigface;
	Mat bigface;
	adjustfaceRect(dstimg, dst_r, bigface, fr_bigface);
	
	assert(bigface.cols==bigface.rows);

	Rect r; //获得所需要的人脸区域在bigface中的位置
	getNormfaceInbigface(bigface,fr_bigface,r);

	//showimg(bigface);

	Mat tmpface(r.height, r.width, CV_8UC1);
	bigface(r).copyTo(tmpface); 
	//showimg(tmpface);
	tmpface.copyTo(face);
	return 0;
}



void MRECOG::rotateFaceOrin(Mat &srcimg, ARect &efr, Mat &dstimg, ARect &dst_efr)
{
    assert((dstimg.rows == srcimg.rows) && (dstimg.cols == srcimg.cols) && (dstimg.type() == srcimg.type()));
    Mat dsttmp;
    Rect bf;   //靠靠dsttmp靠靠靠縭ealgetR
    Rect realposi;
    ExpandRect(srcimg, dsttmp, efr.rect, bf, realposi);  //靠靠靠frect靠20%靠縟sttmp靠靠dsttmp靠靠靠
    //showimg(dsttmp);
    const double PIE = CV_PI;
    Point A = Point(efr.ld[0].x, efr.ld[0].y);
    Point B = Point(efr.ld[1].x, efr.ld[1].y);
    double angle = 180 * atan((B.y - A.y) / (double)(B.x - A.x + 1e-12)) / PIE;  //靠?
    double scale = 1.0;
    double cita = atan((B.y - A.y) / (double)(B.x - A.x + 1e-12));//靠?
    vector<Point> attr;  //6靠靠靠靠靠?
    attr.push_back(Point(dsttmp.cols * 4 / 18.0, dsttmp.rows * 4 / 18.0));
    attr.push_back(Point(dsttmp.cols * 14 / 18.0, dsttmp.rows * 14 / 18.0));
    attr.push_back(Point(efr.ld[0].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[0].y - efr.rect.y + dsttmp.rows * 4 / 18.0));
    attr.push_back(Point(efr.ld[1].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[1].y - efr.rect.y + dsttmp.rows * 4 / 18.0));
    attr.push_back(Point(efr.ld[2].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[2].y - efr.rect.y + dsttmp.rows * 4 / 18.0));
    attr.push_back(Point(efr.ld[3].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[3].y - efr.rect.y + dsttmp.rows * 4 / 18.0));	
    attr.push_back(Point(efr.ld[4].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[4].y - efr.rect.y + dsttmp.rows * 4 / 18.0));
    attr.push_back(Point(efr.ld[5].x - efr.rect.x + dsttmp.cols * 4 / 18.0, efr.ld[5].y - efr.rect.y + dsttmp.rows * 4 / 18.0));  //centermouth
    for (int i = 0; i < attr.size(); i++)   //靠靠靠靠縫A2
    {
        double dis = (double)sqrt(pow((double)attr[i].x - dsttmp.cols / 2.0, 2)
            + pow((double)attr[i].y - dsttmp.rows / 2.0, 2));
        double cita0;
        if (attr[i].x > dsttmp.cols / 2.0)  //1,4靠
        {
            cita0 = -atan((double)(attr[i].y - dsttmp.rows / 2.0) / (attr[i].x - dsttmp.cols / 2.0 + 1e-12));
        }
        else if ((attr[i].x < dsttmp.cols / 2.0) && (attr[i].y <= dsttmp.rows / 2.0)) //2靠
        {
            cita0 = PIE - atan((attr[i].y - dsttmp.rows / 2.0) / (attr[i].x - dsttmp.cols / 2.0 + 1e-12));
        }
        else if ((attr[i].x < dsttmp.cols / 2.0) && (attr[i].y > dsttmp.rows / 2.0))  //3靠
        {
            cita0 = -1 * PIE - atan((attr[i].y - dsttmp.rows / 2.0) / (attr[i].x - dsttmp.cols / 2.0 + 1e-12));
        }
        else if (attr[i].x == dsttmp.cols / 2.0)    //靠縴?
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
    //靠靠靠靠,靠靠靠靠靠靠
    //靠靠靠靠靠靠靠靠靠靠靠靠靠靠靠靠靠靠靠靠?
    int oriW = attr[1].x - attr[0].x;
    int oriH = attr[1].y - attr[0].y;
    int W_H = max(oriW, oriH);
    attr[0].x -= (W_H - oriW) / 2;
    attr[0].y -= (W_H - oriH) / 2;
    attr[1].x += (W_H - oriW) / 2;
    attr[1].y = attr[0].y + (attr[1].x - attr[0].x); //靠靠靠靠靠靠靠?
    assert(attr[1].x - attr[0].x == attr[1].y - attr[0].y);
    //靠靠靠?靠bf靠靠靠靠靠靠靠靠
    //realposi縝f靠靠靠靠靠靠靠靠靠bf靠靠靠?
    rottmp(realposi).copyTo(dstimg(bf));
    //showimg(dstimg);
    // Result  靠靠靠靠靠靠靠
    int x_shift = bf.x - realposi.x;
    int y_shift = bf.y - realposi.y;
    dst_efr.rect.x = attr[0].x + x_shift;
    dst_efr.rect.y = attr[0].y + y_shift;
    dst_efr.rect.width = attr[1].x - attr[0].x;
    dst_efr.rect.height = attr[1].y - attr[0].y;
    //x_shift縴_shift靠靠靠靠靠靠靠
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
int MRECOG::AFaceProcess_RotateOneFace(Mat& image, ARect &face_rect_list,
    Mat& dstImage, ARect& dst_efr)
{
    rotateFaceOrin(image, face_rect_list, dstImage, dst_efr);
    return 0;
}


void MRECOG::adjustfaceRect(Mat& src, ARect &facerect, Mat& bigface, ARect &dst_efr)
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
	dst_efr.rect.width += (W_H2 - dst_efr.rect.width);
	dst_efr.rect.height = dst_efr.rect.width;  //强制变成正方形

	dst_efr.ld.clear();
	dst_efr.ld.push_back(Point(facerect.ld[0].x + x_shift, facerect.ld[0].y + y_shift));
	dst_efr.ld.push_back(Point(facerect.ld[1].x + x_shift, facerect.ld[1].y + y_shift));
	dst_efr.ld.push_back(Point(facerect.ld[2].x + x_shift, facerect.ld[2].y + y_shift));
	dst_efr.ld.push_back(Point(facerect.ld[3].x + x_shift, facerect.ld[3].y + y_shift));
	dst_efr.ld.push_back(Point(facerect.ld[4].x + x_shift, facerect.ld[4].y + y_shift));
	dst_efr.ld.push_back(Point(facerect.ld[5].x + x_shift, facerect.ld[5].y + y_shift));

	dst_efr.face_score = facerect.face_score;

}

void MRECOG::getNormfaceInbigface(Mat& bigface, ARect &efr, Rect &r)
{
	//showEface(bigface, efr);
	//归一化到需要的大小和位置 
	int ec_y = efr.ld[0].y;
	int ec_mc_y = efr.ld[5].y - ec_y;

	int newWH = (int)(ec_mc_y*(1 + 80 / 48.0));
	r.width = newWH;
	r.height = newWH;
	r.y = efr.ld[0].y - ec_mc_y * 40 / 48.0;
	r.x = (bigface.cols - r.width) / 2;
}