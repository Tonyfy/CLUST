///*
//to clust people by facefeature,
//we need to extract all face feature of all image firstly, feature[facenum]
//then calculate the distance between each face pair.  dist[n*n].
//then start clust.
//
//by tony.
//time:2016Äê9ÔÂ13ÈÕ 16:43:46
//
//*/
//#include <cassert>
//#include <caffe/caffe.hpp>
//#include "ARecog/featureExByCaffe.h"
//#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include "ARecog/utils.h"
//#include "ARecog/MRECOG.h"
//#include "ARecog/filesystem.h"
//
//using namespace cv;
//using namespace std;
//using namespace caffe;
//
//int main()
//{
//	google::InitGoogleLogging(" ");
//	FLAGS_alsologtostderr = false;
//	ARECOG *r = new MRECOG();
//	string m = "../Models";
//	r->A_Init(m.c_str());
//
//	//input:images path
//	//output:all face feature.and calculated distance result.
//	string massdir = "";
//	vector<string> imgs;
//	FileSystem::readDir(massdir, "jpg", imgs);
//	string outdir = "";
//
//	ofstream clustlist(outdir + "/clustlist.txt");
//	ofstream facefeaturelist(outdir + "/facefeaturelist.txt");
//	int serielid = 0;
//	vector<AFeature> allfacefeature;
//	for (int i = 0; i < imgs.size(); i++)
//	{
//		allfacefeature.clear();
//		string imgname = imgs[i];
//		string imgpath = massdir + "/" + imgname;
//		Mat img;
//		r->AFaceProcess_ReadImage(imgpath, img);
//		
//		int facenum;
//		vector<ARect> facerectlist;
//
//		r->AFaceProcess_Facedetect(img, facenum, facerectlist, 0);
//		if (1 == facenum)
//		{
//			string savepath = outdir + "/" + imgname;
//			r->AFaceProcess_SaveImage(img, savepath);
//			clustlist << imgpath << " " << serielid++ << endl;
//			AFeature feature;
//			r->AFaceProcess_GetFaceFeature(img, facerectlist[0], feature);
//			facefeaturelist << feature.feature << endl;
//			allfacefeature.push_back(feature);
//
//			//calculate dist matrix.
//		}
//		else
//		{
//			//face is too much.
//			continue;
//		}
//	}
//	return 0;
//}