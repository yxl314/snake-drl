/// @file
/// @brief 基于 std::ofstream 的追加写句柄；平台适配器只负责按平台路径打开。

#ifndef STACKLOG_PLATFORM_STREAM_HPP
#define STACKLOG_PLATFORM_STREAM_HPP

#include "platform.hpp"

#include <fstream>
#include <string>

namespace stacklog
{
namespace platform
{

/// @brief std::ofstream 的 file_sink 适配；子类用平台路径完成打开。
class stream_file_sink : public file_sink
{
public:
    bool is_open() const override
    {
        return stream_.is_open();
    }

    bool write_line(const std::string& text) override
    {
        stream_.write(text.c_str(), static_cast<std::streamsize>(text.size()));
        stream_.put('\n');
        stream_.flush();
        return stream_.good();
    }

    void flush() override
    {
        stream_.flush();
    }

    void close() override
    {
        stream_.close();
    }

protected:
    stream_file_sink() = default;
    /// @brief 由子类按平台路径打开。
    std::ofstream stream_;
};

} // namespace platform
} // namespace stacklog

#endif // STACKLOG_PLATFORM_STREAM_HPP
