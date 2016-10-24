#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <string>
#include <vector>

class FileSystem {
public:
	/**
	 * check a file or a directory exists
	 * @param[in] path		a file path or a directory path
	 * @return				true if path exists else false
	 */
	static bool isExists(const std::string& path);

	/**
	 * create a directory if not exists, **the parent directory must exist**
	 * @param[in] dp		directory path
	 * @reutrn				true if create success else false
	 */
	static bool makeDir(const std::string& dp);

	/**
	 * delete a file if exists
	 * if the file doesn't exists, nothing will be done and return true
	 * @param[in] fp		full file path in filesystem
	 * @return				true if deleted else false
	 */
	static bool deleteFile(const std::string& fp);

	/**
	 * move file from src to dst, both parent directory of src and dst must exists
	 * @param[in] src		file source
	 * @param[in] dst		file destination
	 * @return				true if moved else false
	 */
	static bool moveFile(const std::string& src, const std::string& dst);

	/**
	 * get file name from file path
	 * /path/to/xxx.yyy -> xxx or xxx.yyy -> xxx
	 * @param[in] fp		file path
	 * return				file name without extension name
	 */
	static std::string getFileName(const std::string& fp);

	/**
	 * get file extension name from file path
	 * /path/to/xxx.yyy -> yyy or xxx.yyy -> yyy
	 * @param[in] fp		file path
	 * @return				file extension name
	 */
	static std::string getFileExt(const std::string& fp);

	/**
	 * get file list in a directory with specific extension name
	 * @param[in] dp		directory path
	 * @param[in] ext		extension name
	 * @param[out] fn		files
	 * @paran[in] limit		numbers of files to load
	 * @return				true if success else false
	 */
	static bool readDir(const std::string& dp, const std::string& ext, \
		std::vector<std::string>& fn, int limit=0);

	/**
	 * join path for part1 and part2
	 * @param[in] part1, part2		filesystem path
	 * @return						filesystem path for part1/part2
	 */
	static std::string join(const std::string &part1, const std::string &part2);
};

#endif // FILESYSTEM_H