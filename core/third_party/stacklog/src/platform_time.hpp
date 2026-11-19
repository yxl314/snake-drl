///
/// \file
/// \brief          平台适配器共享的本地时间戳获取与格式化（内部头，不安装）。
///

#ifndef STACKLOG_PLATFORM_TIME_HPP
#define STACKLOG_PLATFORM_TIME_HPP

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

namespace stacklog
{
namespace platform
{

///
/// \brief          用给定的本地时间与毫秒部分拼出 `%Y-%m-%d %H:%M:%S.mmm`。
///
inline std::string
format_timestamp(const std::tm& tm_value, long long milliseconds)
{
  char buffer[32];
  (void)std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_value);
  char result[40];
  (void)std::snprintf(
      result, sizeof(result), "%s.%03lld", buffer, milliseconds);
  return { result };
}

///
/// \brief          取当前本地时间戳；local_time
///                 由平台适配器提供（localtime_r/localtime_s）。
///
template <typename LocalTime>
std::string
make_timestamp(LocalTime local_time)
{
  const std::chrono::system_clock::time_point now
      = std::chrono::system_clock::now();
  const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
  const long long milliseconds
      = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch())
            .count()
        % 1000;

  std::tm tm_value;
  std::memset(&tm_value, 0, sizeof(tm_value));
  (void)local_time(seconds, tm_value);
  return format_timestamp(tm_value, milliseconds);
}

} // namespace platform
} // namespace stacklog

#endif // STACKLOG_PLATFORM_TIME_HPP
