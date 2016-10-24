#ifndef CAFFE_COMPONENT_H
#define CAFFE_COMPONENT_H

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>


class CaffeFaceValidator {
public:
    CaffeFaceValidator(const std::string &net, const std::string &model) {
        network = new caffe::Net<float>(net, caffe::TEST);
        assert(network);
        network->CopyTrainedLayersFrom(model);
    }
    ~CaffeFaceValidator() {
        delete network;
    }

    void validate(const cv::Mat &img, bool &result, float &score) {
        cv::Mat tmp;
        if (img.type() == CV_8UC3) {
            cv::cvtColor(img, tmp, CV_BGR2GRAY);
        } else {
            img.copyTo(tmp);
        }
        cv::resize(tmp, tmp, cv::Size(36, 36));
        std::vector<cv::Mat> img_(1, tmp);
        std::vector<bool> result_(1, false);
        std::vector<float> score_(1, 0);
        validate_(img_, result_, score_);
        result = result_[0];
        score = score_[0];
    }


private:
 
	void validate_(const std::vector<cv::Mat> &imgs, std::vector<bool> &results, std::vector<float> &scores) {
		assert(imgs.size()==results.size());
		assert(imgs.size() == scores.size());
		for (int i = 0; i < imgs.size(); i++)
		{
			bool re;
			float sc;
			validoneimg(imgs[i],re,sc);
			results[i] = re;
			scores[i] = sc;

		}
    }

	void validoneimg(const cv::Mat &imgs, bool &result,float &score)
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
		boost::shared_ptr<caffe::Blob<float> > prob = network->blob_by_name("prob");
		float F = prob->data_at(0, 0, 0, 0);
		float T = prob->data_at(0, 1, 0, 0);
		result = (T > (F + 0.2)) ? (true) : (false);
		score = T;

	}
    caffe::Net<float> *network;
};


#endif // CAFFE_COMPONENT_H
