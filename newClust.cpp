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
#include "fastCluster.h"
#include  "common.h"
#include <string>
using namespace std;
using namespace cv;

int main(int argc,char* argv[])
{
	google::InitGoogleLogging(" ");
	FLAGS_alsologtostderr = false;
	//1，初始化：识别模型，加载配置文件
	char buff[256];
	sprintf(buff, "initialize begins");
	logger(buff);

	loadConfig("../config.txt");
	string model = Config["model"];
	ARECOG *ar = new MRECOG();
	ar->A_Init(model.c_str());

	sprintf(buff, "initialize ends");
	logger(buff);

	//2，读入待聚类的图片集，提取每张人脸的特征，得到feature池。（vector<EFeature>）
	string faceFolder = Config["faceFolder"];
	if (!FileSystem::isExists(faceFolder))
	{
		sprintf(buff, "faceFolder does not exist!");
		logger(buff);
		return -1;
	}
	string classFolder = Config["classFolder"];   //存储聚类结果的文件目录
	char buff_out[256];
	sprintf(buff_out, "%s", classFolder.c_str());
	FileSystem::makeDir(buff_out);

	vector<string> facePaths;
	FileSystem::readDir(faceFolder,"jpg",facePaths);

	vector<CFace> cfaces;
	for (int i = 0; i < facePaths.size(); i++)
	{
		string imgname = facePaths[i];
		string facepath = faceFolder + "/" + imgname;

		Mat bigface;
		ar->AFaceProcess_ReadImage(facepath, bigface);
		Rect centerface = Rect(bigface.cols*0.6 / 2.2, bigface.rows*0.6 / 2.2, bigface.cols / 2.2, bigface.rows / 2.2);
		ARect tar;
		ar->AFaceProcess_Landmark(bigface, centerface, tar);

		AFeature facefeature;
		logger("extracting face feature");
		ar->AFaceProcess_GetFaceFeature(bigface, tar, facefeature);
		logger("get feature successfully!");

		CFace tmpcface;
		tmpcface.isclustCenter = false;
		tmpcface.facefeature = facefeature;
		tmpcface.srcpath = facepath;
		tmpcface.x_width = 0.5;
		tmpcface.y_height = 0.5;
		tmpcface.facelabel = -1;
		tmpcface.facerect = tar;

		cfaces.push_back(tmpcface);
	}

	ofstream faceIdtxt("faceid.txt");
	for (int i = 0; i < cfaces.size(); i++)
	{
		faceIdtxt << i << " " << cfaces[i].srcpath << endl;
	}
	faceIdtxt.close();

	//4，调用fastCluster聚类算法进行聚类，输出每个feature所属的类别，即每张人脸的类别。
	/*人脸聚类*/
	if (cfaces.size() == 0)
	{
		cerr << "no face " << endl;
	}
	else if (cfaces.size() == 1)
	{
		//只有一个人脸，无需聚类
		Mat face = imread(cfaces[0].srcpath);  //取出人脸
		string dirinclass = classFolder + "/" + to_string(0);
		if (!FileSystem::isExists(dirinclass))
		{
			FileSystem::makeDir(dirinclass);
		}
		string savepath;
		savepath = dirinclass + "/" + cfaces[0].srcpath.substr(cfaces[0].srcpath.find_last_of("/") + 1);
		imwrite(savepath, face);
	}
	else
	{
		//人脸多于1张，聚类。
		vector<datapoint> result;
		ar->AFaceProcess_Clust(cfaces, result);

		for (int i = 0; i < result.size(); i++)
		{
			Mat face = imread(cfaces[i].srcpath);  //取出人脸
			string dirinclass = classFolder + "/" + to_string(result[i].label);
			if (!FileSystem::isExists(dirinclass))
			{
				FileSystem::makeDir(dirinclass);
			}
			string savepath;
			if (result[i].clustcenter)
			{
				savepath = dirinclass + "/center_" +
					cfaces[i].srcpath.substr(cfaces[i].srcpath.find_last_of("/") + 1);
			}
			else
			{
				savepath = dirinclass + "/" +
					cfaces[i].srcpath.substr(cfaces[i].srcpath.find_last_of("/") + 1);
			}
			imwrite(savepath, face);
		}
	}
	return 0;
}
