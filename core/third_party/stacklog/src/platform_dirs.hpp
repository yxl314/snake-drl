///
/// \file
/// \brief          平台适配器共享的逐级建目录流程（内部头，不安装）。
///
/// POSIX 与 Windows 的差异收敛到 Rules 的三个静态钩子：根路径特判、起始位置、
/// 单目录创建；遍历与末尾分隔符处理只有一份。
///

#ifndef STACKLOG_PLATFORM_DIRS_HPP
#define STACKLOG_PLATFORM_DIRS_HPP

#include <string>

namespace stacklog
{
namespace platform
{

///
/// \brief          按 Rules 逐级创建目录；已存在视为成功。
///
template <typename Rules>
bool
make_dirs_with_rules(const std::string& directory)
{
  if (directory.empty() || directory == ".")
    {
      return true;
    }

  // 末尾分隔符会让 Windows 的 mkdir 失败；逐级创建前先去掉（保留根路径）。
  std::string dir = directory;
  while (dir.size() > 1
         && (dir[dir.size() - 1] == '/' || dir[dir.size() - 1] == '\\'))
    {
      dir.erase(dir.size() - 1);
    }

  // 盘符根/UNC 共享根等系统提供的已存在根交给规则判断，不逐级 mkdir。
  bool handled = false;
  bool result = false;
  if (Rules::special_root(dir, handled, result))
    {
      return result;
    }

  const std::string::size_type start = Rules::start_index(dir);
  for (std::string::size_type i = start; i < dir.size(); ++i)
    {
      const char ch = dir[i];
      if (ch != '/' && ch != '\\')
        {
          continue;
        }
      const std::string parent = dir.substr(0, i);
      if (parent.empty() || parent == "/" || parent == "\\"
          || (parent.size() == 2 && parent[1] == ':'))
        {
          continue;
        }
      if (!Rules::make_one_dir(parent))
        {
          return false;
        }
    }
  return Rules::make_one_dir(dir);
}

} // namespace platform
} // namespace stacklog

#endif // STACKLOG_PLATFORM_DIRS_HPP
