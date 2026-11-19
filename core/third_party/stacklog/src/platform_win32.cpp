/// @file
/// @brief stacklog 平台端口的 Windows 实现（内部文件，不安装）。

#ifdef _WIN32

#include "platform.hpp"
#include "platform_dirs.hpp"
#include "platform_paths.hpp"
#include "platform_stream.hpp"
#include "platform_time.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace stacklog
{
namespace platform
{
namespace
{

/// @brief 把 UTF-8 路径转换为 Windows 宽字符路径；含无效 UTF-8 时返回 false。
bool
utf8_to_wide(const std::string& path, std::wstring& wide)
{
  if (path.empty())
    {
      wide.clear();
      return true;
    }

  const int length = ::MultiByteToWideChar(CP_UTF8,
                                           MB_ERR_INVALID_CHARS,
                                           path.data(),
                                           static_cast<int>(path.size()),
                                           nullptr,
                                           0);
  if (length <= 0)
    {
      return false;
    }

  wide.resize(static_cast<std::wstring::size_type>(length));
  // C++11 的 std::wstring::data() 只返回 const wchar_t*，写入需要可写缓冲区；
  // C++17 起才能直接使用 wide.data()。
  wchar_t* buffer = &wide[0]; // NOLINT(readability-container-data-pointer):
      // C++11 的 data() 只读
  return ::MultiByteToWideChar(CP_UTF8,
                               MB_ERR_INVALID_CHARS,
                               path.data(),
                               static_cast<int>(path.size()),
                               buffer,
                               length)
         == length;
}

/// @brief 判断路径是否以 \\?\ 扩展长度前缀开头。
bool
has_extended_prefix(const std::string& path)
{
  return path.size() >= 4 && path[0] == '\\' && path[1] == '\\'
         && path[2] == '?' && path[3] == '\\';
}

/// @brief 判断去掉末尾分隔符后的路径是否是盘符根（"C:" 或 "\\?\C:"）。
bool
is_drive_root(const std::string& path)
{
  if (path.size() == 2 && path[1] == ':')
    {
      return true;
    }
  return path.size() == 6 && has_extended_prefix(path) && path[5] == ':';
}

/// @brief 判断是否是 UNC 共享根（"\\\\server\\share"）。
bool
is_unc_root(const std::string& path)
{
  if (path.size() < 3 || (path[0] != '\\' && path[0] != '/')
      || (path[1] != '\\' && path[1] != '/'))
    {
      return false;
    }
  if (has_extended_prefix(path))
    {
      return false;
    }
  const std::string::size_type server_end = path.find_first_of("/\\", 2);
  if (server_end == std::string::npos)
    {
      return false;
    }
  return std::string::npos == path.find_first_of("/\\", server_end + 1);
}

/// @brief Windows 建目录规则：盘符/UNC 根只校验存在，单级用 CreateDirectoryW。
struct win32_dir_rules
{
  static bool
  special_root(const std::string& dir, bool& handled, bool& result)
  {
    if (is_drive_root(dir))
      {
        handled = true;
        result = path_exists(dir + "\\");
        return true;
      }
    if (is_unc_root(dir))
      {
        handled = true;
        result = path_exists(dir);
        return true;
      }
    return false;
  }

  static std::string::size_type
  start_index(const std::string& dir)
  {
    if (dir.size() >= 2 && dir[1] == ':')
      {
        return 2;
      }
    if (dir.size() >= 2 && (dir[0] == '\\' || dir[0] == '/')
        && (dir[1] == '\\' || dir[1] == '/'))
      {
        const std::string::size_type server_end = dir.find_first_of("/\\", 2);
        const std::string::size_type share_end
            = (server_end == std::string::npos)
                  ? std::string::npos
                  : dir.find_first_of("/\\", server_end + 1);
        return (share_end == std::string::npos) ? dir.size() : share_end + 1;
      }
    return 1;
  }

  static bool
  make_one_dir(const std::string& path)
  {
    if (path.empty() || path_exists(path))
      {
        return true;
      }
    std::wstring wide;
    if (!utf8_to_wide(path, wide))
      {
        return false;
      }
    if (0 != ::CreateDirectoryW(wide.c_str(), nullptr))
      {
        return true;
      }
    return ::GetLastError() == ERROR_ALREADY_EXISTS;
  }
};

/// @brief 以追加写打开文件的 Windows 适配器。
class win32_file_sink : public stream_file_sink
{
public:
  explicit win32_file_sink(const std::string& path)
  {
    std::wstring wide;
    if (!utf8_to_wide(path, wide))
      {
        return;
      }
    // C++11 标准没有 wchar_t 路径的流重载；MinGW 的 libstdc++ 与 MSVC STL 都
    // 提供该非标准扩展（MSVC 文档中的 basic_ofstream(const wchar_t*)）。
    stream_.open(wide.c_str(),
                 std::ios::out | std::ios::app | std::ios::binary);
  }
};

} // namespace

std::unique_ptr<file_sink>
open_append(const std::string& path)
{
  std::unique_ptr<file_sink> sink(new win32_file_sink(path));
  if (!sink->is_open())
    {
      return nullptr;
    }
  return sink;
}

bool
valid_utf8_path(const std::string& path)
{
  std::wstring wide;
  return utf8_to_wide(path, wide);
}

bool
path_exists(const std::string& path)
{
  std::wstring wide;
  if (!utf8_to_wide(path, wide))
    {
      return false;
    }
  // 用 Win32 原生 API 而不是 _wstat64：UCRT 的 _wstat64 支持 \\?\ 扩展路径，
  // 但 MSVCRT 运行时（例如 MinGW-w64 的 mingw64 环境）会返回 ENOENT，
  // 把真实存在的扩展路径误判为不存在。
  return ::GetFileAttributesW(wide.c_str()) != INVALID_FILE_ATTRIBUTES;
}

bool
regular_file_size(const std::string& path, std::uint64_t& size)
{
  // GetFileAttributesExW 提供 64 位大小与目录标记，并原生支持 \\?\ 扩展
  // 路径；_wstat64 在 MSVCRT 运行时上做不到（见 path_exists 注释）。
  std::wstring wide;
  WIN32_FILE_ATTRIBUTE_DATA info;
  if (!utf8_to_wide(path, wide)
      || 0 == ::GetFileAttributesExW(wide.c_str(), GetFileExInfoStandard, &info)
      || 0 != (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
    {
      return false;
    }
  ULARGE_INTEGER value;
  value.HighPart = info.nFileSizeHigh;
  value.LowPart = info.nFileSizeLow;
  size = static_cast<std::uint64_t>(value.QuadPart);
  return true;
}

bool
make_dirs(const std::string& directory)
{
  return make_dirs_with_rules<win32_dir_rules>(directory);
}

bool
remove_file(const std::string& path)
{
  std::wstring wide;
  return utf8_to_wide(path, wide) && 0 != ::DeleteFileW(wide.c_str());
}

bool
rename_file(const std::string& from, const std::string& to)
{
  std::wstring wide_from;
  std::wstring wide_to;
  // MoveFileW 与 _wrename 语义一致（目标存在时失败），但原生支持扩展路径。
  return utf8_to_wide(from, wide_from) && utf8_to_wide(to, wide_to)
         && 0 != ::MoveFileW(wide_from.c_str(), wide_to.c_str());
}

std::string
join_path(const std::string& dir, const std::string& name)
{
  // Windows 的 \\?\ 扩展长度路径不做 '/' 归一化，路径拼接必须用反斜杠。
  return join_path_with_separator(dir, name, "\\");
}

std::string
timestamp()
{
  return make_timestamp([](std::time_t seconds, std::tm& out) {
    return localtime_s(&out, &seconds) == 0;
  });
}

} // namespace platform
} // namespace stacklog

#endif // defined(_WIN32)
