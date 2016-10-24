#ifndef featureExByCaffe_H
#define featureExByCaffe_H

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <caffe/caffe.hpp>
 
using namespace std;
using namespace cv;
using namespace caffe;
class featureExer {
public:
	featureExer() = default;
	featureExer(const std::string &net, const std::string &model) {
		network = new caffe::Net<float>(net, caffe::TEST);
		assert(network);
		network->CopyTrainedLayersFrom(model);
	}
	~featureExer() {
		delete network;
	}
	void featureExtract(const std::string &input, const std::string &output);
	void extractfeature(const cv::Mat &img, cv::Mat &feature) {
		cv::Mat tmp;
		//std::cout << img.depth();
		if (img.type() == CV_32FC3) {
			cv::cvtColor(img, tmp, CV_BGR2GRAY);
		}
		else {
			img.copyTo(tmp);
		}
		//cv::resize(tmp, tmp, cv::Size(31, 31));
		std::vector<cv::Mat> img_(1, tmp);
		cv::Mat tmp2 = cv::Mat(256, 1, CV_32FC1);
		std::vector<cv::Mat> feature_(1, tmp2);
#ifdef TIME_TEST_ON
		TIME_TEST_BEGIN;
		validate_(img_, feature_);
		double t;
		TIME_ESTIMATE(t);
		LOG(INFO) << "Validate " << result_.size() << " faces using " << t * 1000 / result_.size() << " ms/per";
		TIME_TEST_END;
#else
		extractfeature_(img_, feature_);
#endif
		feature = feature_[0];
	}

	void extractfeature(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &features) {
		std::vector<cv::Mat> imgs_;
		features.clear();
		for (int i = 0; i<imgs.size(); i++) {
			cv::Mat tmp;
			if (imgs[i].type() == CV_32FC3) {
				cv::cvtColor(imgs[i], tmp, CV_BGR2GRAY);
			}
			else {
				imgs[i].copyTo(tmp);
			}

			//cv::resize(tmp, tmp, cv::Size(31, 31));
			imgs_.push_back(tmp);
		}
#ifdef TIME_TEST_ON
		TIME_TEST_BEGIN;
		extractfeature_(imgs_, features);
		double t;
		TIME_ESTIMATE(t);
		LOG(INFO) << "Validate " << results.size() << " faces using " << t * 1000 / results.size() << " ms/per";
		TIME_TEST_END;
#else
		extractfeature_(imgs_, features);
#endif
	}

public:
	void extractfeature_(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &features)
	{
		assert(imgs.size()==features.size());
		for (int i = 0; i < imgs.size(); i++)
		{
			cv::Mat featuretmp(256,1,CV_32FC1);
			extractfeatureoneimg(imgs[i],featuretmp);
			features[i] = featuretmp;
		}
	}
	void extractfeatureoneimg(const cv::Mat &imgs, cv::Mat &features)
	{
		float loss = 0.0;
		const vector<Blob<float>*> &intput_blobs = network->input_blobs();
		float *blob_data = intput_blobs[0]->mutable_cpu_data();
		const float *ptr = NULL;
		for (int i = 0; i < imgs.rows; i++) {
			ptr = imgs.ptr<float>(i);
			for (int j = 0; j < imgs.cols; j++) {
				blob_data[i*imgs.cols + j] = ptr[j];
			}
		}

		network->ForwardFromTo(0, network->layers().size() - 1);
		//boost::shared_ptr<caffe::Blob<float> > prob = network->blob_by_name("eltwise6");
		boost::shared_ptr<caffe::Blob<float> > prob = network->blob_by_name("eltwise_fc1");

		for (int j = 0; j < 256; j++)
		{
				//second dim --feature serials
			features.at<float>(j, 0) = prob->data_at(0, j, 0, 0);
		}

	}


	caffe::Net<float> *network;
};

#endif // featureExByCaffe_H
