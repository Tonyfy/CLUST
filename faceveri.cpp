#include <cassert>
#include <caffe/caffe.hpp>
#include "featureExByCaffe.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils.h"
#include "MRECOG.h"
#include "filesystem.h"

using namespace cv;
using namespace std;
using namespace caffe;
int main()
{
	google::InitGoogleLogging(" ");
	FLAGS_alsologtostderr = false;
	ARECOG *r = new MRECOG();
	string m = "../Models";
	r->A_Init(m.c_str());
	
	string Apicpath = "../imgs/ym1.jpg";
	string Bpicpath = "../imgs/ym2.jpg";

	string Cpicpath = "../imgs/ym2.jpg";
	string Dpicpath = "../imgs/ldh1.jpg";
	
	if ((!FileSystem::isExists(Apicpath)) || (!FileSystem::isExists(Bpicpath)) ||
		(!FileSystem::isExists(Cpicpath)) || (!FileSystem::isExists(Dpicpath)))
	{
		cout << "file not found" << endl;
		return -1;
	}

	Mat Apic,Bpic,Cpic,Dpic;
	
	r->AFaceProcess_ReadImage(Apicpath,Apic);	
	r->AFaceProcess_ReadImage(Bpicpath,Bpic);	
	r->AFaceProcess_ReadImage(Cpicpath,Cpic);
	r->AFaceProcess_ReadImage(Dpicpath,Dpic);

	Rect facerect(68,68,114,114);
	
	int listsizeA, listsizeB, listsizeC, listsizeD;
	vector<ARect> frlA, frlB, frlC, frlD;
	r->AFaceProcess_Facedetect(Apic, listsizeA, frlA, 0);
	r->AFaceProcess_Facedetect(Bpic, listsizeB, frlB, 0);
	r->AFaceProcess_Facedetect(Cpic, listsizeC, frlC, 0);
	r->AFaceProcess_Facedetect(Dpic, listsizeD, frlD, 0);

	//showface (Apic,frlA[0].rect);
	//showface (Bpic,frlB[0].rect);
	//showface (Cpic,frlC[0].rect);
	//showface (Dpic,frlD[0].rect);
	cout<<"facenums1,2,3,4 is "<<listsizeA<<" "<<listsizeB<<" "<<listsizeC<<" "<<listsizeD<<endl;

	ARect Aefr, Befr, Cefr, Defr;
	AFeature Af,Bf,Cf,Df;
	Af.feature=Mat(256,1,CV_32FC1);
	Bf.feature=Mat(256,1,CV_32FC1);
	Cf.feature=Mat(256,1,CV_32FC1);
	Df.feature=Mat(256,1,CV_32FC1);
	
	//double start = cv::getTickCount();
	//for (int i = 0; i < 1000; )
	//{
	//	i++;
	//	r->AFaceProcess_GetFaceFeature(Apic, frlA[0], Af);
	//}
	//double extractf_cost = (cv::getTickCount() - start) / cv::getTickFrequency();
	//cout << "ave extract cost "<<extractf_cost/1000.0 << endl;

	r->AFaceProcess_GetFaceFeature(Apic, frlA[0], Af);

	r->AFaceProcess_GetFaceFeature(Bpic, frlB[0], Bf);
	double simAB = 0.0;
	r->AFaceProcess_FeatureCompare(Af, Bf, simAB);
	cout<<"simAB is " << simAB << "\n";

	r->AFaceProcess_GetFaceFeature(Cpic, frlC[0], Cf);
	r->AFaceProcess_GetFaceFeature(Dpic, frlD[0], Df);
	double simCD = 0.0;
	r->AFaceProcess_FeatureCompare(Cf, Df, simCD);
	cout<<"simCD is "<<simCD<<endl;

	r->A_UnInit();
	return 0;

}
