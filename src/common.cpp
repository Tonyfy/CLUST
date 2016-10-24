#include <ctime>
#include <iostream>
#include <fstream>
#include <map>
#include "../ARecog/common.h"
#include "../ARecog/filesystem.h"

using namespace std;

map<string, string> Config;

int loadConfig(const std::string& configFilePath)
{
	if (!FileSystem::isExists(configFilePath)) {
		return -1;
	}
	ifstream infile(configFilePath);
	if (!infile) {
		// 读取文件失败
		return -1;
	}
	string key;
	string value;
	while (infile >> key && infile >> value){
		auto kv = Config.insert(make_pair(key, value));
		if (!kv.second) {
			// 插入失败
			return -1;
		}

		logger(key + " --> " + value);
	}
	return 0;
}

int logger(const string& msg) {
	time_t t = time(NULL);
	char buff[256];
	strftime(buff, sizeof(buff), "[%H:%M:%S]", localtime(&t));
	cout << string(buff) << msg << "\n" << endl;
	return 0;
}
