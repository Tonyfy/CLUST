#include <iostream>
#include "template.h"

using namespace std;
using namespace cv;

namespace fc {

Template::Template(const Template& t) {
	for (int i = 0; i < t.size(); i++) {
		auto& m = t[i];
		push_back(m);
	}
}

void Template::operator=(const Template& t) {
	// 清空
	clear();
	// 赋值
	for (int i = 0; i < t.size(); i++) {
		auto& one = t[i];
		push_back(one);
	}
}

Template& Template::operator<<(const Mat& m) {
	push_back(m);
	return *this;
}

Template& Template::operator<<(const Template& t) {
	for (int i = 0; i < t.size(); i++) {
		auto& m = t[i];
		push_back(m);
	}
	return *this;
}

TransformList::TransformList() {}
TransformList::~TransformList() {
	for (int i = 0; i < transform_list.size(); i++) {
		auto& t = transform_list[i];
		if (t) {
			delete t;
		}
	}
}

TransformList& TransformList::operator<<(const Transform* t) {
	// 去除const限定
	transform_list.push_back(const_cast<Transform*>(t));
	return *this;
}

TransformList& TransformList::operator<<(const Transform& t) {
	transform_list.push_back(const_cast<Transform*>(&t)->copy());
	return *this;
}

void TransformList::clear(){
	for (int i = 0; i < transform_list.size(); i++) {
		auto& t = transform_list[i];
		if (t) {
			delete t;
		}
	}
	transform_list.clear();
}

void TransformList::project(const Template& src, Template& dst) {
	Template _(src);
	for (int i = 0; i < transform_list.size(); i++) {
		auto& t = transform_list[i];
		t->project(_, _);
		if (_.isNull()) {
			// 变换出现错误
			break;
		}
	}
	dst.clear();
	dst = _;
}

Transform* makeTransform(const string transform_name) {
	// TODO
	//return nullptr;
	return NULL;
}

} // namespace
