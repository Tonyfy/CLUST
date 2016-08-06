#ifndef COMMON_H
#define COMMON_H

#include <map>
#include <string>

// 全局配置信息
extern std::map<std::string, std::string> Config;

// 加载全局配置信息
int loadConfig(const std::string& configFilePath);

// 控制台日志
int logger(const std::string& msg);


#endif // COMMON_H