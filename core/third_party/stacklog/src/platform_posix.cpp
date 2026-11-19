/// @file
/// @brief stacklog 平台端口的 POSIX 实现（内部文件，不安装）。

#ifndef _WIN32

#include "platform.hpp"
#include "platform_dirs.hpp"
#include "platform_paths.hpp"
#include "platform_stream.hpp"
#include "platform_time.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>

namespace stacklog
{
namespace platform
{
namespace
{

/// @brief POSIX 建目录规则：没有特殊根，单级用 mkdir(0755)。
struct posix_dir_rules
{
  static bool
  special_root(const std::string& dir, bool& handled, bool& result)
  {
    (void)dir;
    handled = false;
    result = false;
    return false;
  }

  static std::string::size_type
  start_index(const std::string& dir)
  {
    (void)dir;
    return 1;
  }

  static bool
  make_one_dir(const std::string& path)
  {
    if (path.empty() || path_exists(path))
      {
        return true;
      }
    return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
  }
};

/// @brief 以追加写打开文件的 POSIX 适配器。
class posix_file_sink : public stream_file_sink
{
public:
  explicit posix_file_sink(const std::string& path)
  {
    stream_.open(path.c_str(),
                 std::ios::out | std::ios::app | std::ios::binary);
  }
};

} // namespace

std::unique_ptr<file_sink>
open_append(const std::string& path)
{
  std::unique_ptr<file_sink> sink(new posix_file_sink(path));
  if (!sink->is_open())
    {
      return nullptr;
    }
  return sink;
}

bool
valid_utf8_path(const std::string& path)
{
  (void)path;
  return true;
}

bool
path_exists(const std::string& path)
{
  struct stat info;
  return ::stat(path.c_str(), &info) == 0;
}

bool
regular_file_size(const std::string& path, std::uint64_t& size)
{
  struct stat info;
  if (::stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode))
    {
      return false;
    }
  size = static_cast<std::uint64_t>(info.st_size);
  return true;
}

bool
make_dirs(const std::string& directory)
{
  return make_dirs_with_rules<posix_dir_rules>(directory);
}

bool
remove_file(const std::string& path)
{
  return std::remove(path.c_str()) == 0;
}

bool
rename_file(const std::string& from, const std::string& to)
{
  return std::rename(from.c_str(), to.c_str()) == 0;
}

std::string
join_path(const std::string& dir, const std::string& name)
{
  return join_path_with_separator(dir, name, "/");
}

std::string
timestamp()
{
  return make_timestamp([](std::time_t seconds, std::tm& out) {
    return localtime_r(&seconds, &out) != nullptr;
  });
}

} // namespace platform
} // namespace stacklog

#endif // !defined(_WIN32)
