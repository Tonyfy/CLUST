#ifndef COMMON_H
#define COMMON_H

#include <map>
#include <string>

// ȫ��������Ϣ
extern std::map<std::string, std::string> Config;

// ����ȫ��������Ϣ
int loadConfig(const std::string& configFilePath);

// ����̨��־
int logger(const std::string& msg);


#endif // COMMON_H