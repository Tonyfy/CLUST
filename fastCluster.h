/*
���پ����㷨
���룺N�������У���������֮��ľ���d(i,j)
�����m������Ĵ��ģ��������������Լ�ÿһ��������������𣨴��ģ�

Author��zhao
Date��2016��1��4�� 20:34:56
*/

#ifndef FASTCLUSTER_H__
#define FASTCLUSTER_H__

#include <iostream>
#include <vector>
#include <array>
#include <opencv/cxcore.hpp>
//#include "MRECOG.h"

struct node
{
	double rho;
	double delta;
	double gamma;
	int idx;
};

struct cluster
{
	int classid;
	int centerid;
	int nelement;
	int ncore;
	int nhalo;
	double centerrho;
	std::vector<int> elements;
};

struct androidpoint
{
	int label;
	std::string filename;
	double x_width;
	double y_height;
};

struct picture
{
	std::array<int, 6> date;
	std::string filepath;
	int orien;
	std::string filename;
};

struct picsInoneTime
{
	std::vector<picture> pic;
};

struct datapoint
{
	int label;
	bool clustcenter;
};

bool comp(node x, node y);
bool comppics(picture x, picture y);
bool comprho(cluster x, cluster y);
bool compgamma(node x, node y);

class Cluster{

public :

	Cluster(char *filename);
	//{
	//	data_analyse = std::ofstream("data.txt");
	//}
	//~Cluster()
	//{
	//	data_analyse.close();
	//}

	void fillval(std::vector<double> &a, double &val);
	void fillval(std::vector<int> &a, int &val);

	void splitpicsOntime(std::vector<picture>& pics, int rule, std::vector<picsInoneTime>& picsOT);

	void regressionsplit(picture& pic1, picture& pic2, int& i, int s, picsInoneTime& tmp, std::vector<picsInoneTime> & picsOT);
	//double compareMatFromMatlab(char* fp, mat& data);
	//double compareVecFromMatlab(char* fp, vector<double>& data);
	//void Mat2mat(cv::Mat &Matdata, mat &matdata);
	//double compareVecIntFromMatlab(char* fp, vector<int>& data);
	//void Mat2matInt(cv::Mat &Matdata, mat &matdata);

	double getaverNeighrate(const cv::Mat &dist);

	double getDc(cv::Mat &dist, double& percent);
	void calculateRho(cv::Mat &dist, double &dc, std::vector<double>& rho);
	void sortRho(std::vector<double>& rho, std::vector<double>& sorted_rho,
		std::vector<int>& ordrho);
	void sortgamma(std::vector<double>& gamma, std::vector<double>& sorted_gamma,
		std::vector<int>& ordgamma);
	void calculateDelta(cv::Mat& dist, std::vector<double>& rho,
		std::vector<double>& sorted_rho, std::vector<int>& ordrho,
		std::vector<double>& delta, std::vector<int>& nneigh,
		std::vector<double>& gamma, std::vector<int>& ordgamma);

	void fastClust(cv::Mat &dist, std::vector<datapoint>& clustResult);

	bool isSameOne(cv::Mat &dist, cluster& A, cluster& B);

	void mergeCluster(cluster& A, cluster& B, cluster& C);

public:
	//std::ofstream data_analyse;
};


#endif //FASTCLUSTER_H__