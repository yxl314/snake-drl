/// @file
/// @brief stacklog 内部平台端口；不属于公开头，不安装。
///
/// 核心逻辑只依赖这里的接口；POSIX 与 Windows 适配器分别实现，测试用
/// fake_platform 实现。端口不暴露平台头文件，也不向上层泄漏平台宏。

#ifndef STACKLOG_PLATFORM_HPP
#define STACKLOG_PLATFORM_HPP

#include <cstdint>
#include <memory>
#include <string>

namespace stacklog
{
namespace platform
{

/// @brief 追加写日志文件的句柄；把「写一行」与运行库差异隔离在适配器内。
class file_sink
{
public:
    /// @brief 允许派生类析构。
    virtual ~file_sink() = default;
    file_sink() = default;
    file_sink(const file_sink&) = delete;
    file_sink& operator=(const file_sink&) = delete;

    /// @brief 文件是否处于可写状态。
    virtual bool is_open() const = 0;

    /// @brief 追加一行（自动补换行并 flush）；失败返回 false。
    virtual bool write_line(const std::string& text) = 0;

    /// @brief 刷新缓冲区。
    virtual void flush() = 0;

    /// @brief 关闭文件；可重复调用。
    virtual void close() = 0;
};

/// @brief 以追加方式打开文件；失败返回空指针。
std::unique_ptr<file_sink> open_append(const std::string& path);

/// @brief 路径是否是合法 UTF-8；POSIX 恒 true，Windows 做 UTF-8→UTF-16 预校验。
bool valid_utf8_path(const std::string& path);

/// @brief 路径是否存在（文件或目录）。
bool path_exists(const std::string& path);

/// @brief 取普通文件大小；不存在或不是普通文件返回 false。
bool regular_file_size(const std::string& path, std::uint64_t& size);

/// @brief 递归创建目录；已存在视为成功。
bool make_dirs(const std::string& directory);

/// @brief 删除文件；不存在或失败返回 false。
bool remove_file(const std::string& path);

/// @brief 重命名文件；目标已存在时失败。
bool rename_file(const std::string& from, const std::string& to);

/// @brief 按平台分隔符拼接目录与文件名。
std::string join_path(const std::string& dir, const std::string& name);

/// @brief 本地时间戳字符串（`%Y-%m-%d %H:%M:%S.mmm`）。
std::string timestamp();

} // namespace platform
} // namespace stacklog

#endif // STACKLOG_PLATFORM_HPP
