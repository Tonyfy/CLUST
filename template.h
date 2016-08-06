#ifndef TEMPLATE_H
#define TEMPLATE_H

#include <vector>
#include <opencv2/core/core.hpp>

namespace fc {

// 图像表征类
class Template : public std::vector<cv::Mat> {
public:
	Template(){}
	~Template(){}
	Template(cv::Mat& m){ push_back(m); }
	Template(const Template& t);

public:
	void operator=(const Template& t);
	Template& operator<<(const cv::Mat& m);
	Template& operator<<(const Template& t);
	const cv::Mat& m() const {
		static cv::Mat noMat;
		return (empty() ? noMat : (*this)[size()-1]);
	}
	bool isNull() { return empty() || !m().data; }
};

class TemplateList : public std::vector < Template > {
public:
	TemplateList(){}
	~TemplateList(){}
	TemplateList(const Template& t){ push_back(t); }
};

// 抽象变换类
class Transform {
public:
	Transform(){}
	~Transform(){}
	Transform(const Transform& t);

	// 将 src 变换成 dst
	// 任何继承自该类的Tranform必须实现这个方法 
	virtual void project(const Template& src, Template& dst) = 0;
	//virtual void project(const TemplateList& src, TemplateList& dst) = 0;

	virtual Transform* copy() { return this; }
};

class TransformList : public Transform {
public:
	TransformList();
	~TransformList();

	// 需要保证变换序列中的变换能够适应 src = dst 的情况
	void project(const Template& src, Template& dst);

	// usage transformList << new LBPTransform << new PCATransform(params, ...);
	TransformList& operator<<(const Transform* t);
	TransformList& operator<<(const Transform& t);
	void clear();
private:
	std::vector<Transform*> transform_list;
};

// 根据变换名字生成Transform类
// TODO
Transform* makeTransform(const std::string& transform_name);

} // namespace

#endif // TEMPLATE_H
