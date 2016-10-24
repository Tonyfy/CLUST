#include "../ARecog/filesystem.h"

#ifdef WIN32
#include <windows.h>
#include <io.h>
#include <direct.h>

#else
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#endif // WIN32

using namespace std;

bool FileSystem::isExists(const string& path) {
	if (access(path.c_str(), 0) != -1) {
		return true;
	}
	else {
		return false;
	}
}

bool FileSystem::makeDir(const string& dp) {
	if (isExists(dp)) {
		return true;
	}
	int ret_code;

#ifdef WIN32
	ret_code = mkdir(dp.c_str());
#else
	// mode: -rwxrwxr-x
	ret_code = mkdir(dp.c_str(), 0755);
#endif // WIN32

	if (ret_code == 0) {
		return true;
	}
	else {
		return false;
	}
}

bool FileSystem::deleteFile(const string& fp) {
	int rc = remove(fp.c_str());
	if (rc == 0) {
		return true;
	}
	else {
		return false;
	}
}

bool FileSystem::moveFile(const string& src, const string& dst) {
	int rc = rename(src.c_str(), dst.c_str());
	if (!rc) {
		return false;
	}
	else {
		return true;
	}
}

string FileSystem::getFileName(const string& fp) {
	int pos = fp.find_last_of("/");
	if (pos == string::npos) { // xxx.yyy
		return fp.substr(0, fp.find_last_of("."));
	}
	else { // /path/to/xxx.yyy
		string _;
		_ = fp.substr(pos + 1);
		return _.substr(0, _.find_last_of("."));
	}
}

string FileSystem::getFileExt(const string& fp) {
	int pos = fp.find_last_of("/");
	if (pos == string::npos) {
		return fp.substr(fp.find_last_of(".") + 1);
	}
	else {
		string _;
		_ = _.substr(pos + 1);
		return _.substr(_.find_last_of(".") + 1);
	}
}

bool FileSystem::readDir(const string& dp, const string& ext, vector<string>& fn, int limit) {
	if (!isExists(dp)) {
		return false;
	}
	int counter = 0;

#ifdef WIN32
	long fd;
	_finddata_t fileInfo;
	string matchPath = dp + "/*." + ext;
	if ((fd = _findfirst(matchPath.c_str(), &fileInfo)) == -1) {
		// can't access dp
		return false;
	}
	do {
		fn.push_back(fileInfo.name);
		if ((++counter) == limit) break;
	} while (_findnext(fd, &fileInfo) == 0);
	_findclose(fd);
#else
	DIR *d;
	struct dirent *file;
	struct stat sb;
	if (!(d = opendir(dp.c_str()))) {
		return false;
	}
	while ((file = readdir(d)) != NULL) {
		// remove '.', '..', and all hidden files
		if (strncmp(file->d_name, ".", 1) == 0) {
			continue;
		}
		// remain specific extensions
		string _(file->d_name);
		if (getFileExt(_) == ext) {
			fn.push_back(_);
			if ((++counter) == limit) break;
		}
	}
#endif // WIN32

	return true;
}

string FileSystem::join(const string &part1, const string &part2) {
#ifdef WIN32
	return part1 + "\\" + part2;
#else
	return part1 + "/" + part2;
#endif // WIN32
}