/// @file
/// @brief 平台适配器共享的路径拼接（内部头，不安装）。

#ifndef STACKLOG_PLATFORM_PATHS_HPP
#define STACKLOG_PLATFORM_PATHS_HPP

#include <string>

namespace stacklog
{
namespace platform
{

/// @brief 按指定分隔符拼接目录与文件名；目录已以分隔符结尾时不重复添加。
inline std::string join_path_with_separator(const std::string& dir, const std::string& name, const char* separator)
{
    if (dir.empty())
    {
        return name;
    }
    const char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\')
    {
        return dir + name;
    }
    return dir + separator + name;
}

} // namespace platform
} // namespace stacklog

#endif // STACKLOG_PLATFORM_PATHS_HPP
