#include <iostream>
#include <fstream>
#include "../ARecog/fastCluster.h"

using namespace std;
using namespace cv;
//using namespace arma;

bool comp(node x, node y)
{
	return x.rho > y.rho;
}
bool compgamma(node x, node y)
{
	return x.gamma > y.gamma;
}
bool comppics(picture x, picture y)
{

	for (int i = 0; i < x.date.size();)
	{
		if (x.date.at(i) == y.date.at(i))
		{
			i++;
		}
		else
		{
			return x.date.at(i) < y.date.at(i);
		}
	}
	return false;
}
bool comprho(cluster x, cluster y)
{
	return x.centerrho > y.centerrho;
}

Cluster::Cluster(char *filename)
{
	//data_analyse = ofstream(filename);
}

void Cluster::regressionsplit(picture& pic1, picture& pic2, int& i, int s,
	picsInoneTime& tmp, std::vector<picsInoneTime>& picsOT)
{
	while (i < s + 1)
	{
		if (pic2.date.at(i) != pic1.date.at(i))
		{
			picsOT.push_back(tmp);
			tmp.pic.clear();
			tmp.pic.push_back(pic2);
			i += s + 1;
		}
		else
		{
			i++;
			regressionsplit(pic1, pic2, i, s, tmp, picsOT);
			if (i == s + 1)
			{
				tmp.pic.push_back(pic2);
				i += s + 1;
				break;
			}
		}
	}

}
void Cluster::splitpicsOntime(std::vector<picture>& pics, int rule,
	std::vector<picsInoneTime> & picsOT)
{
	assert((rule == 0) || (rule == 1) || (rule == 2) 
		|| (rule == 3) || (rule == 4) || (rule == 5));
	picsOT.clear();
	if (pics.size() < 2)
	{
		picsInoneTime tmp;
		tmp.pic.push_back(pics[0]);
		picsOT.push_back(tmp);
	}
	sort(pics.begin(), pics.end(), comppics);

	picsInoneTime tmp;
	tmp.pic.push_back(pics[0]);
	for (int i = 1; i < pics.size(); i++)
	{
		int ss = 0;
		regressionsplit(pics[i - 1], pics[i], ss, rule, tmp, picsOT);
	}
	picsOT.push_back(tmp);
	//if (rule == "year")
	//{
	//	picsInoneTime tmp;
	//	tmp.pic.push_back(pics[0]);
	//	for (int i = 1; i < pics.size(); i++)
	//	{
	//		if (pics[i].date.at(0) != pics[i - 1].date.at(0))
	//		{
	//			//�������µ�ʱ��ֶ�
	//			picsOT.push_back(tmp);
	//			tmp.pic.clear();
	//			tmp.pic.push_back(pics[i]);
	//		}
	//		else
	//		{
	//			tmp.pic.push_back(pics[i]);
	//		}
	//	}
	//	picsOT.push_back(tmp);  //���һ��tmp ������һ���ֶ�
	//}

}

void Cluster::fillval(vector<int> &a, int &val)
{
	for (int i = 0; i < a.size(); i++)
	{
		a[i] = val;
	}
}

void Cluster::fillval(vector<double> &a, double &val)
{
	for (int i = 0; i < a.size(); i++)
	{
		a[i] = val;
	}
}

double Cluster::getaverNeighrate(const Mat &dist)
{
	ofstream dataana("data.txt", ios::app);
	int N = dist.rows;
	int nneigh = 0;
	double averdist = 0.0;
	for (int i = 0; i < N - 1; i++)
	{
		for (int j = i + 1; j < N; j++)
		{
			averdist += dist.at<double>(i, j);
		}
	}
	double aver = 2 * averdist / (N*(N - 1));    //�������е�ƽ������

	cout <<"���������ľ�ֵ "<< aver << endl;
	Mat mean, stddev;
	meanStdDev(dist, mean, stddev);
	cout << mean.at<double>(0, 0);
	dataana << "���������ľ�ֵ ";
	dataana << mean.at<double>(0, 0);
	dataana << endl;
	dataana << "���������ķ��� " << stddev.at<double>(0, 0) << endl;
	double dc = max(aver - 0.35, 0.1);
	cout << "Ԥ���ľ���dcΪ " << dc << endl;
	dataana << "Ԥ���ľ���dc " << dc << endl;

	for (int i = 0; i < N - 1; i++)
	{
		for (int j = i + 1; j < N; j++)
		{
			if (dist.at<double>(i, j) < dc)
			{
				nneigh++;
			}
		}
	}
	double percents = 100.0*nneigh / (double)(N*(N - 1) / 2);
	cout << "�ھ���Ϊ " << percents << " %." << endl;
	dataana << "�ھ��� " << percents << endl;
	dataana.close();
	return percents;
}

double Cluster::getDc(cv::Mat &dist, double& percent)
{
	int Num = dist.rows;
	int N = Num*(Num - 1) / 2;  //���о��������
	if ((((int)percent) >= 100) || (((int)percent) <= 0))
	{
		cerr << "the percent must be [1-100],2-5 maybe ok." << endl;
	}
	cout << "average percentage of neighbours(hard code) :  " << percent << "% ." << endl;

	//��ýضϲ���dc,�Ƽ�ֵ��ʹ��ƽ��ÿ������ھ���Ϊ����������1%-2%
	int position = (int)(N*percent / 100);
	cv::Mat alldis(1, N, CV_64FC1);

	for (int i = 0; i < Num - 1; i++)
	{
		for (int j = i + 1; j < Num; j++)
		{
			alldis.at<double>(0, (1 + 2 * (Num - 1) - i)*i / 2 + (j - i) - 1) = dist.at<double>(i, j);
		}
	}
	vector<double> sda = (vector<double>)alldis;
	sort(sda.begin(), sda.end());  //��������
	double dc = sda[position];
	//double dc = 0.4;
	cout << "Computing Rho with gaussian kernal of radius(dc) : " << dc << endl;
	return dc;
}

void Cluster::calculateRho(cv::Mat &dist, double &dc, std::vector<double>& rho)
{
	int Num = dist.rows;
	assert(rho.size() == Num);
	double bdouble = 0.0;
	fillval(rho, bdouble);
	//Gaussian kernal
	for (int i = 0; i < Num - 1; i++)
	{
		for (int j = i + 1; j < Num; j++)
		{
			double distmp = dist.at<double>(i, j);
			rho[i] = rho[i] + exp(-pow(distmp / dc, 2));
			rho[j] = rho[j] + exp(-pow(distmp / dc, 2));
		}
	}
}

void Cluster::sortRho(vector<double>& rho, vector<double>& sorted_rho, vector<int>& ordrho)
{
	assert((sorted_rho.size() == rho.size()) && (ordrho.size() == rho.size()));
	//���ֲ��ܶ�rho[Num] ���ս������У��������±�ŵĴ���
	vector<node> _rho(rho.size());
	for (int i = 0; i < rho.size(); i++)
	{
		_rho[i].rho = rho[i];
		_rho[i].idx = i;
	}
	sort(_rho.begin(), _rho.end(), comp);//comp����ָ�������ǽ���

	for (int i = 0; i < _rho.size(); i++)
	{
		sorted_rho[i] = _rho[i].rho;
		ordrho[i] = _rho[i].idx;
	}
}
void Cluster::sortgamma(vector<double>& gamma, vector<double>& sorted_gamma,
	vector<int>& ordgamma)
{
	assert((sorted_gamma.size() == gamma.size()) && (ordgamma.size() == gamma.size()));
	//���ֲ��ܶ�rho[Num] ���ս������У��������±�ŵĴ���
	vector<node> _gamma(gamma.size());
	for (int i = 0; i < gamma.size(); i++)
	{
		_gamma[i].gamma = gamma[i];
		_gamma[i].idx = i;
	}
	sort(_gamma.begin(), _gamma.end(), compgamma);//comp����ָ�������ǽ���

	for (int i = 0; i < _gamma.size(); i++)
	{
		sorted_gamma[i] = _gamma[i].gamma;
		ordgamma[i] = _gamma[i].idx;
	}
}

void Cluster::calculateDelta(cv::Mat& dist, vector<double>& rho, vector<double>& sorted_rho,
	vector<int>& ordrho, vector<double>& delta, vector<int>& nneigh, std::vector<double>& sorted_gamma, std::vector<int>& ordgamma)
{
	assert((delta.size() == rho.size()) && (nneigh.size() == rho.size())
		&& (sorted_gamma.size() == rho.size()) && (ordgamma.size() == rho.size()));
	delta[ordrho[0]] = -1.0;
	nneigh[ordrho[0]] = 0;
	double maxd, mind;
	minMaxIdx(dist, &mind, &maxd);  //ȡ�����ľ���ֵ maxd.
	for (int ii = 1; ii < rho.size(); ii++)
	{
		delta[ordrho[ii]] = maxd;
		for (int jj = 0; jj < ii; jj++)
		{
			//cout << "dist.at<double>(ordrho["<<ii<<"], ordrho["<<jj<<"]) =dist["<<ordrho[ii]<<","<<ordrho[jj]<<"] = "<<dist.at<double>(ordrho[ii], ordrho[jj]) << endl;
			if (dist.at<double>(ordrho[ii], ordrho[jj]) < delta[ordrho[ii]])
			{
				delta[ordrho[ii]] = dist.at<double>(ordrho[ii], ordrho[jj]);
				nneigh[ordrho[ii]] = ordrho[jj];
			}
		}
	}

	cv::Mat delta_Mat = cv::Mat(delta);   //������delta��ֵ��Mat����ȡ���ֵ��
	double mindel, maxdel;
	minMaxIdx(delta_Mat, &mindel, &maxdel);
	delta[ordrho[0]] = maxdel;  //delta[Num]��ֵ���

	/*��rho[Num]  �� delta[Num] ����gamma[Num].*/
	vector<int> ind(rho.size());
	vector<double> gamma(rho.size());
	double minrho = sorted_rho[sorted_rho.size() - 1];
	double maxrho = sorted_rho[0];
	double difrho = maxrho - minrho;
	double difdelta = maxdel - mindel;
	for (int i = 0; i < rho.size(); i++)
	{
		ind[i] = i;
		gamma[i] = 1e-6 + (rho[i] - minrho) * (delta[i] - mindel) / (difrho*difdelta);
	}
	sorted_gamma = gamma;
	sortgamma(gamma, sorted_gamma, ordgamma);

}

void Cluster::fastClust(cv::Mat &dist, vector<datapoint>& clustResult)
{
	ofstream dataana("data.txt", ios::app);

	assert(dist.rows == dist.cols);
	int Num = dist.rows;
	double percent = getaverNeighrate(dist); //ָ��ƽ���ھ����İٷֱ�
	//percent = 5.0;
	double dc = getDc(dist, percent);
	cout << "�����ھ��ʼ����dc " << dc << endl;
	dataana << "�����ھ��ʼ����dc " << dc << endl;
	vector<double> rho(Num);
	calculateRho(dist, dc, rho);

	vector<double> rho_sorted(Num);
	vector<int> ordrho(Num);
	sortRho(rho, rho_sorted, ordrho);   //��������

	vector<double> delta(Num);
	vector<int> nneigh(Num);
	vector<double> sorted_gamma(Num);
	vector<int> ordgamma(Num);
	calculateDelta(dist, rho, rho_sorted, ordrho, delta, nneigh, sorted_gamma, ordgamma);

	vector<double> delta_sort(Num);
	delta_sort = delta;
	sort(delta_sort.begin(), delta_sort.end());

	vector<double> loggamma(Num);
	ofstream loggammatxt("log.txt");
	for (int i = 0; i < Num; i++)
	{
		loggamma[i] = log(sorted_gamma[i]);
		loggammatxt << loggamma[i] << " " << sorted_gamma[i] << endl;
	}
	loggammatxt.close();

	//��sorted_gamma�еõ��ϲ�λ�ã��ϲ�֮ǰ��Ϊ�������ġ�
	//double maxgammadif = 0;
	//int maxdifId = 0;
	//for (int i = 1; i < sorted_gamma.size(); i++)
	//{
	//	double tmp = sorted_gamma[i - 1] - sorted_gamma[i];
	//	if ((tmp > maxgammadif)&&(i>3))
	//	{
	//		maxgammadif = tmp;
	//		maxdifId = i;
	//	}	
	//}

	//��sorted_gamma��β���������������ҵ������maxgammadif�൱����Id��maxdifId��ġ�
	//for (int i = sorted_gamma.size() - 1; i > sorted_gamma.size() - maxdifId - 1; i--)
	//{
	//	double tmp = sorted_gamma[i - 1] - sorted_gamma[i];
	//	if (tmp > 0.5*maxgammadif)
	//	{
	//		maxdifId = i;
	//	}
	//}

	/*��ʼ����*/
	double maxdel = delta[ordrho[0]];
	//double rhomin = rho_sorted[rho_sorted.size()*max((100 - percent * 3), 10.0) / 100];  //ָ��rhomin����deltamin
	//double deltamin = delta_sort[ordrho.size()*(min(percent * 2, 70.0)) / 100];	double rhomin = rho_sorted[rho_sorted.size()*max((100 - percent * 3), 10.0) / 100];  //ָ��rhomin����deltamin
	double rhomin = rho_sorted[0] / 8;
	double deltamin = maxdel / 4;
	int NCLUST = 0;
	vector<int> cl(Num);  //��ʼ��cl[Num]��ȫΪ-1
	int bint = -1;
	fillval(cl, bint);
	vector<int> icl;

	ofstream rhodelta("rhodelta.txt");
	for (int i = 0; i < Num; i++)
	{
		rhodelta << i << " " << rho[i] << " " << delta[i] << endl;
	}
	rhodelta << rhomin << " " << deltamin << endl;
	rhodelta.close();

	dataana << "rhomin " << rhomin << endl;
	dataana << "deltamin " << deltamin << endl;
	dataana << "rhomin_max " << rho_sorted[0] << endl;
	dataana << "delta_max " << maxdel << endl;

	//for (int i = 0; i < maxdifId; i++)
	//{
	//	cl[ordgamma[i]] = NCLUST;
	//	icl.push_back(ordgamma[i]);
	//	NCLUST++;
	//}
	for (int i = 0; i < Num; i++)
	{
		if ((rho[i]>rhomin) && (delta[i] > deltamin))
		{
			cl[i] = NCLUST;  //��i�������������NCLUST
			icl.push_back(i);  //��NCLUST����������������i.
			NCLUST++;
		}
	}
	//���ҳ����еľ������ġ�
	cout << "NUMBER OF CLUSTERS : " << NCLUST << endl;
	dataana << "������������ " << NCLUST << endl;
	/*��ʼ������������ķ��䣬�������е�����*/
	cout << "Performing assignation" << endl;
	for (int i = 0; i < Num; i++)
	{
		if (cl[ordrho[i]] == -1)
		{
			//����Ѱ�Ҿ�������ʱ��ֻ�д����ĵ�cl�Ų���-1.��ÿһ���Ǵ����ĵ�����
			//�����Ϊ����nneigh����𣬼�ĳ�������ĵ����
			cl[ordrho[i]] = cl[nneigh[ordrho[i]]];
		}
	}
	/*halo��������������֮�����µĲ��ּ�Ϊ��nneigh����Щ���ݵ㡪������*/
	vector<int> halo(Num);
	for (int i = 0; i < Num; i++)
	{
		halo[i] = cl[i];
	}

	vector<double> bord_rho(NCLUST);
	if (NCLUST > 0)
	{
		for (int i = 0; i < NCLUST; i++)
		{
			bord_rho[i] = 0;
		}

		for (int i = 0; i < Num - 1; i++)
		{
			for (int j = i + 1; j < Num; j++)
			{
				if ((cl[i] != cl[j]) && (dist.at<double>(i, j) <= dc))
				{
					//i��j�����ͬ����������ľ���С�ڽضϾ��롣
					double rho_aver = (rho[i] + rho[j]) / 2.0;
					if (rho_aver>bord_rho[cl[i]])
					{
						//i��������bord_rho��ƽ���ֲ��ܶ�С,��bord_rho���ŵ�rho_aver.
						bord_rho[cl[i]] = rho_aver;
					}
					if (rho_aver>bord_rho[cl[j]])
					{
						//ͬ��������j��������bord_rho.
						bord_rho[cl[j]] = rho_aver;
					}
				}
			}
		}

		for (int i = 0; i < Num; i++)
		{
			if (rho[i] < bord_rho[cl[i]])
			{
				halo[i] = -1;  //��Ⱥ�� -1
			}
		}
	}

	if (NCLUST > 0)
	{
		//����ÿ������nc��nh
		vector<cluster> allclust(NCLUST);
		vector<int> nc(NCLUST);
		vector<int> nh(NCLUST);
		int ling = 0;
		fillval(nc, ling);
		fillval(nh, ling);
		dataana << "��ش�С ";
		for (int i = 0; i < NCLUST; i++)
		{
			int ncc = 0;
			int nhh = 0;
			for (int j = 0; j < Num; j++)
			{
				//ͳ��ÿ��������������������halo������
				if (cl[j] == i)
				{
					ncc++;
				}
				if (halo[j] == i)
				{
					nhh++;
				}
			}
			nc[i] = ncc;
			nh[i] = nhh;
			cout << "CLUSTER : " << i << "  CENTER: " << icl[i] << " ELEMENT: " << nc[i]
				<< "  CORE: " << nh[i] << "  HALO: " << nc[i] - nh[i] << endl;
			dataana << nc[i] << " ";
		}
		dataana << endl;
		for (int i = 0; i < NCLUST; i++)
		{
			allclust[i].classid = i;
			allclust[i].centerid = icl[i];
			allclust[i].nelement = nc[i];
			allclust[i].ncore = nh[i];
			allclust[i].nhalo = nc[i] - nh[i];
			allclust[i].centerrho = rho[icl[i]];
		}
		for (int i = 0; i < Num; i++)
		{
			int nclass = cl[i];
			allclust[nclass].elements.push_back(i);
		}
		//�Ծ��������ж��ι��࣬����rho���н�������
		//vector<cluster> newallclust = allclust;
		//sort(newallclust.begin(),newallclust.end(),comprho);
		//sort(allclust.begin(), allclust.end(), comprho);
		//�ҳ����е���ͼƬ �ࡣM*N����
		ofstream MNsimi("MNsimi.txt");
		vector<cluster> onepicclust;
		int single_class_element_size = 3; //�����Ѿ۳���صĳߴ���й�����
		dataana << "�����׼��С�ڵ��ڣ� " << single_class_element_size << endl;

		for (int i = 0; i < NCLUST; i++)
		{
			if (allclust[i].nelement <= single_class_element_size)
			{
				onepicclust.push_back(allclust[i]);
			}
		}
		int Monepiccluster = onepicclust.size();
		cout << " �������" << Monepiccluster << endl;
		dataana << "�������� " << Monepiccluster << endl;

		double merge_th = 0.7;
		dataana << "����ϲ���׼ " << merge_th << endl;

		int merge_time = 0;
		for (int i = 0; i < Monepiccluster; i++)
		{
			int maxid = -1;
			double maxsimi = 0.0;
			int index = onepicclust[i].classid;
			//int start = onepicclust[i].centerid;
			for (int j = 0; j < NCLUST; j++)  //�벻�ǵ���i������������Ƚ�
			{
				double dist_mn = 1 - dist.at<double>(allclust[index].centerid, allclust[j].centerid);
				MNsimi << dist_mn << " ";
				//����õ�������һ����������ƶ����1-���룩���õ���ӽ���һ�����
				if ((dist_mn > maxsimi) && (dist_mn<0.95))
				{
					maxsimi = dist_mn;
					maxid = j;
				}
			}

			if (maxsimi > merge_th)
			{
				merge_time++;
				//��ӽ��ĳ̶ȳ���merge_th����Ϊ��Ҫ�ϲ�
				for (int n = 0; n < allclust[index].elements.size(); n++)
				{
					allclust[maxid].nhalo += 1;
					allclust[maxid].elements.push_back(allclust[index].elements[n]);
					cl[allclust[index].elements[n]] = cl[allclust[maxid].centerid];
				}
				//allclust[maxid].nhalo += 1;
				//cl[onepicclust[i].centerid] = cl[allclust[maxid].centerid];
				/*cout << "���� " << allclust[index].classid << "�������� " << allclust[index].centerid
					<< "���� " << allclust[index].elements.size() << " ������"
					<< "����� " << maxid << "(���� " << allclust[index].elements.size() << "������)"
					<< "�ϲ� ������������ " << allclust[maxid].centerid << endl;*/
				icl[allclust[index].classid] = icl[allclust[maxid].classid];
				allclust[index].centerid = allclust[maxid].centerid;  //�ı���������������id

			}
			MNsimi << "\n";
		}
		MNsimi.close();
		dataana << "����ϲ����� " << merge_time << endl;

		ofstream dist_center_ij("dist_center_ij.txt");

		//dist_center_ij << "dc = " << dc << "\n";
		for (int i = 0; i < NCLUST; i++)
		{
			for (int j = 0; j < NCLUST; j++)
			{
				double dist_clust_ij = 1 - dist.at<double>(allclust[i].centerid, allclust[j].centerid);
				dist_center_ij << dist_clust_ij;
				dist_center_ij << "  ";
			}
			dist_center_ij << "\n";
		}
		dist_center_ij.close();

		clustResult.clear();
		//vector<datapoint> clustresult(Num);
		for (int i = 0; i < Num; i++)
		{
			datapoint cresult;
			cresult.label = cl[i];
			cresult.clustcenter = false;
			clustResult.push_back(cresult);
		}
		for (int i = 0; i < NCLUST; i++)
		{
			int centerID = icl[i];
			clustResult[centerID].clustcenter = true;
		}
	}
	else
	{
		//û�о������� ����Ϊһ�����
		for (int i = 0; i < Num; i++)
		{
			datapoint cresult;
			cresult.label = i;
			cresult.clustcenter = true;
			clustResult.push_back(cresult);
		}
	}
	
	dataana.close();

}

bool Cluster::isSameOne(cv::Mat &dist, cluster& A, cluster& B)
{
	int M = A.elements.size();
	int N = B.elements.size();
	double ABsim=0;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			double simij = 1 - dist.at<double>(A.elements[i], B.elements[j]);
			ABsim += simij;
		}
	}

	if (ABsim / (M*N) > 0.6)
	{
		//�������ƶȴ���0.6 ��Ϊ��Ҫ�ϲ�
		return true;
	}
	return false;

}

void Cluster::mergeCluster(cluster& A, cluster& B, cluster& C)
{
	assert((A.nelement>0) && (B.nelement > 0));
	A.nelement > B.nelement
		? (C.centerid = A.centerid, C.centerrho = A.centerrho, C.classid = A.classid)
		: (C.centerid = B.centerid, C.centerrho = B.centerrho, C.classid = B.classid);
	C.ncore = A.ncore + B.ncore;
	C.nhalo = A.nhalo + B.nhalo;
	C.nelement = A.nelement + B.nelement;
	for (int i = 0; i < A.nelement; i++)
	{
		C.elements.push_back(A.elements[i]);
	}

	for (int i = 0; i < B.nelement; i++)
	{
		C.elements.push_back(A.elements[i]);
	}
}