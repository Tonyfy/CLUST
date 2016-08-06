#include "MRECOG.h"
#include "common.h"
#include "utils.h"

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

int MRECOG::AFaceProcess_Getface(cv::Mat& image, ARect& facerect, Mat& face)
{
	ARect dst_r;
	Mat dstimg;
	rotateFaceOrin(image, facerect, dstimg, dst_r);
	//showEimg(*dstimg);
	Rect fr_bigface;
	Mat bigface;
	adjustfaceRect(dstimg, dst_r.rect, bigface, fr_bigface);
	
	assert(bigface.cols==bigface.rows);

	Rect r; //获得所需要的人脸区域在bigface中的位置
	getNormfaceInbigface(bigface,fr_bigface,r);

	Mat tmpface(r.height, r.width, CV_8UC1);
	bigface(r).copyTo(tmpface); 

	tmpface.copyTo(face);
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

int MRECOG::AFaceProcess_ReadImage(const std::string &imgpath, Mat& img)
{
	img = imread(imgpath,0);
	return 0;
}

int MRECOG::AFaceProcess_SaveImage(const Mat& img, std::string &savepath)
{

	imwrite(savepath,img);
	return 0;
}


int MRECOG::AFaceProcess_FeatureCompare(const AFeature& query_feature,
	const AFeature& ref_feature, double& similarity)
{
	cosSimilarity(query_feature.feature,ref_feature.feature,similarity);
	return 0;
}

int MRECOG::AFaceProcess_GetfaceFeature(cv::Mat& face, AFeature &feature)
{
	assert(face.type() == CV_8UC1);
	resize(face, face, cvSize(128, 128));
	face.convertTo(face, CV_32FC1, 1.0 / 255.0);
	fe->extractfeature(face, feature.feature);

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
int MRECOG::cosSimilarity(const Mat& q, const Mat& r, double& similarity)
{
	assert((q.rows==r.rows)&&(q.cols==r.cols));
	double fenzi = q.dot(r);
	double fenmu = sqrt(q.dot(q)) * sqrt(r.dot(r));
	similarity = fenzi/fenmu;

	return 0;
}

int MRECOG::ouSimilarity(const Mat& q, const Mat& r, double& similarity)
{
	assert((q.rows==r.rows)&&(q.cols==r.cols));

	similarity = (q - r).dot(q - r)/((q.dot(q))*(r.dot(r)));
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

