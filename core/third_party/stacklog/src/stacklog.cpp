#include "platform.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stacklog/stacklog.hpp>
#include <string>
#include <vector>

namespace stacklog
{
namespace
{

/// 线程内追踪栈、节点序号和线程注册序号。
struct trace_context
{
  std::vector<trace_frame> stack;
  std::uint64_t node_sequence = 0;
  std::uint64_t thread_sequence = 0;
  bool thread_registered = false;
};

/// 进程级 trace 序号。
std::atomic<std::uint64_t> trace_sequence(0);

/// 进程级线程注册序号。
std::atomic<std::uint64_t> thread_sequence_counter(0);

/// 运行期最低级别。
std::atomic<int> current_level(static_cast<int>(level::info));

/// 未初始化降级提示只打印一次。
std::atomic<int> fallback_hinted(0);

/// 日志器共享状态。
struct logger_state
{
  std::mutex mutex;
  std::unique_ptr<platform::file_sink> file;
  config options;
  std::string path;
  std::uint64_t written_bytes = 0;
  std::uint64_t writes_since_check = 0;
  std::uint64_t failed_writes = 0;
  std::uint64_t rotation_retry_size = 0;
  bool active = false;
  bool ready = false;
};

/// 进程常驻状态；不析构，避免静态析构顺序问题。
logger_state&
state()
{
  static auto* instance = new logger_state;
  return *instance;
}

/// 以追加方式重新打开当前日志文件，并从实际文件大小恢复写入计数。
bool
open_append_locked(logger_state& logger)
{
  if (logger.file)
    {
      logger.file->close();
      logger.file.reset();
    }
  logger.file = platform::open_append(logger.path);
  if (!logger.file || !logger.file->is_open())
    {
      logger.file.reset();
      return false;
    }

  std::uint64_t size = 0;
  logger.written_bytes
      = platform::regular_file_size(logger.path, size) ? size : 0;
  return true;
}

std::string
escape_field(const std::string& value)
{
  std::string result;
  result.reserve(value.size());
  for (char ch : value)
    {
      switch (ch)
        {
        case '\\':
          result += "\\\\";
          break;
        case '|':
          result += "\\|";
          break;
        case '\r':
          result += "\\r";
          break;
        case '\n':
          result += "\\n";
          break;
        default:
          result += ch;
          break;
        }
    }
  return result;
}

std::string
dash_if_empty(const std::string& value)
{
  return value.empty() ? std::string("-") : value;
}

const char*
level_name(level value)
{
  switch (value)
    {
    case level::debug:
      return "DEBUG";
    case level::info:
      return "INFO";
    case level::warn:
      return "WARN";
    case level::error:
      return "ERROR";
    case level::off:
      return "OFF";
    }
  return "UNKNOWN";
}

std::string
base_name(const char* file)
{
  if (nullptr == file)
    {
      return "-";
    }
  const char* slash = std::strrchr(file, '/');
  const char* back = std::strrchr(file, '\\');
  if (nullptr != back && (nullptr == slash || back > slash))
    {
      slash = back;
    }
  return (nullptr == slash) ? std::string(file) : std::string(slash + 1);
}

trace_context&
thread_trace_context()
{
  static thread_local trace_context context;
  return context;
}

/// 给每个线程分配进程内唯一的注册序号。
std::uint64_t
thread_registration_sequence(trace_context& context)
{
  if (!context.thread_registered)
    {
      context.thread_sequence = thread_sequence_counter.fetch_add(1) + 1;
      context.thread_registered = true;
    }
  return context.thread_sequence;
}

trace_frame*
current_frame()
{
  trace_context& context = thread_trace_context();
  return context.stack.empty() ? nullptr : &context.stack.back();
}

std::string
make_trace_id()
{
  const std::uint64_t sequence = trace_sequence.fetch_add(1) + 1;
  const long long milliseconds
      = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
  return "t-" + std::to_string(milliseconds) + "-" + std::to_string(sequence);
}

void
begin_trace(const std::string& name)
{
  trace_context& context = thread_trace_context();
  trace_frame frame;
  frame.scope = name.empty() ? std::string("unnamed") : name;

  if (context.stack.empty())
    {
      frame.trace_id = make_trace_id();
    }
  else
    {
      frame.trace_id = context.stack.back().trace_id;
      frame.parent_id = context.stack.back().node_id;
    }

  // 节点号加入线程注册序号，保证跨线程接续后仍进程内唯一。
  const std::uint64_t thread_id = thread_registration_sequence(context);
  ++context.node_sequence;
  frame.node_id = frame.trace_id + "-t" + std::to_string(thread_id) + "-"
                  + std::to_string(context.node_sequence);
  context.stack.push_back(frame);
}

void
end_trace()
{
  trace_context& context = thread_trace_context();
  if (!context.stack.empty())
    {
      context.stack.pop_back();
    }
}

void
restore_trace(const trace_frame& frame)
{
  if (frame.valid())
    {
      thread_trace_context().stack.push_back(frame);
    }
}

std::string
format_line(level severity,
            const trace_frame* frame,
            const std::string& message,
            const char* file,
            int line)
{
  const std::string scope_name
      = (nullptr == frame) ? std::string("-")
                           : escape_field(dash_if_empty(frame->scope));
  const std::string trace_id
      = (nullptr == frame) ? std::string("-")
                           : escape_field(dash_if_empty(frame->trace_id));
  const std::string node_id = (nullptr == frame)
                                  ? std::string("-")
                                  : escape_field(dash_if_empty(frame->node_id));
  const std::string parent_id
      = (nullptr == frame) ? std::string("-")
                           : escape_field(dash_if_empty(frame->parent_id));
  const std::string location
      = escape_field(base_name(file) + ":" + std::to_string(line));

  std::string result;
  result += platform::timestamp();
  result += '|';
  result += level_name(severity);
  result += '|';
  result += trace_id;
  result += '|';
  result += node_id;
  result += '|';
  result += parent_id;
  result += '|';
  result += scope_name;
  result += '|';
  result += location;
  result += '|';
  result += escape_field(dash_if_empty(message));
  return result;
}

void
rotate_locked(logger_state& logger)
{
  if (logger.file)
    {
      logger.file->flush();
      logger.file->close();
      logger.file.reset();
    }

  const std::string base = logger.path;
  bool ok = true;

  if (logger.options.max_files > 0)
    {
      const std::string oldest
          = base + "." + std::to_string(logger.options.max_files);
      if (platform::path_exists(oldest) && !platform::remove_file(oldest))
        {
          ok = false;
        }

      if (ok)
        {
          for (unsigned index = logger.options.max_files; index > 1; --index)
            {
              const std::string from = base + "." + std::to_string(index - 1);
              const std::string to = base + "." + std::to_string(index);
              if (!platform::path_exists(from))
                {
                  continue;
                }
              if (platform::path_exists(to) && !platform::remove_file(to))
                {
                  ok = false;
                  break;
                }
              if (!platform::rename_file(from, to))
                {
                  ok = false;
                  break;
                }
            }
        }

      if (ok && platform::path_exists(base))
        {
          const std::string first = base + ".1";
          if ((platform::path_exists(first) && !platform::remove_file(first))
              || !platform::rename_file(base, first))
            {
              ok = false;
            }
        }
    }
  else
    {
      ok = false;
    }

  if (!ok)
    {
      // 归档失败时继续写当前文件，并允许文件再增长一个上限后重试。
      logger.ready = open_append_locked(logger);
      logger.rotation_retry_size
          = logger.ready ? logger.written_bytes + logger.options.max_file_size
                         : 0;
      return;
    }

  logger.written_bytes = 0;
  logger.rotation_retry_size = 0;
  logger.ready = open_append_locked(logger);
}

void
fallback_to_stderr(const std::string& line)
{
  if (0 == fallback_hinted.fetch_add(1))
    {
      (void)std::fprintf(stderr,
                         "stacklog: 文件输出未就绪，日志降级到 stderr\n");
    }
  (void)std::fprintf(stderr, "%s\n", line.c_str());
}

} // namespace

bool
init(const config& cfg)
{
  logger_state& logger = state();
  std::lock_guard<std::mutex> lock(logger.mutex);

  current_level.store(static_cast<int>(cfg.min_level));
  logger.active = false;
  logger.ready = false;
  if (logger.file)
    {
      logger.file->close();
      logger.file.reset();
    }
  logger.options = cfg;
  logger.path.clear();
  logger.written_bytes = 0;
  logger.writes_since_check = 0;
  logger.failed_writes = 0;
  logger.rotation_retry_size = 0;

  if (cfg.dir.empty() || cfg.base_name.empty() || cfg.max_file_size == 0
      || cfg.max_files == 0)
    {
      (void)std::fprintf(stderr, "stacklog: 配置非法，初始化失败\n");
      return false;
    }

  logger.path = platform::join_path(cfg.dir, cfg.base_name + ".log");

  // Windows 需要把整条路径转成 UTF-16；非法 UTF-8 必须在创建目录之前失败，
  // 否则 init 返回 false 却会留下一个空目录。POSIX 端口恒返回 true。
  if (!platform::valid_utf8_path(logger.path))
    {
      (void)std::fprintf(stderr, "stacklog: 路径不是合法 UTF-8，初始化失败\n");
      return false;
    }

  if (!platform::make_dirs(cfg.dir))
    {
      (void)std::fprintf(
          stderr, "stacklog: 无法创建日志目录: %s\n", cfg.dir.c_str());
      return false;
    }

  if (!open_append_locked(logger))
    {
      (void)std::fprintf(
          stderr, "stacklog: 无法打开日志文件: %s\n", logger.path.c_str());
      return false;
    }

  logger.active = true;
  logger.ready = true;
  return true;
}

void
shutdown()
{
  logger_state& logger = state();
  std::lock_guard<std::mutex> lock(logger.mutex);
  if (logger.file)
    {
      logger.file->flush();
      logger.file->close();
      logger.file.reset();
    }
  logger.active = false;
  logger.ready = false;
  logger.writes_since_check = 0;
  logger.failed_writes = 0;
  logger.rotation_retry_size = 0;
}

void
set_level(level value) noexcept
{
  current_level.store(static_cast<int>(value));
}

bool
detail::log_line::is_enabled(level severity) noexcept
{
  const int configured = current_level.load();
  if (configured == static_cast<int>(level::off) || severity == level::off)
    {
      return false;
    }
  return static_cast<int>(severity) >= configured;
}

void
detail::log_line::write_line(level severity,
                             const char* file,
                             int line,
                             const std::string& message)
{
  if (!is_enabled(severity))
    {
      return;
    }

  const trace_frame* frame = current_frame();
  const std::string text = format_line(severity, frame, message, file, line);

  logger_state& logger = state();
  bool written = false;
  {
    const std::uint64_t check_interval = 1024;
    const std::uint64_t retry_interval = 32;

    std::lock_guard<std::mutex> lock(logger.mutex);

    if (logger.active && !logger.ready)
      {
        ++logger.failed_writes;
        if (logger.failed_writes >= retry_interval)
          {
            logger.failed_writes = 0;
            logger.ready = open_append_locked(logger);
            if (logger.ready)
              {
                logger.writes_since_check = 0;
                logger.rotation_retry_size = 0;
              }
          }
      }

    if (logger.ready)
      {
        ++logger.writes_since_check;
        if (logger.writes_since_check >= check_interval)
          {
            logger.writes_since_check = 0;
            std::uint64_t actual_size = 0;
            if (!platform::regular_file_size(logger.path, actual_size)
                || actual_size != logger.written_bytes)
              {
                // 文件被外部删除、截断或轮转后，重新打开路径上的文件。
                if (logger.file)
                  {
                    logger.file->flush();
                    logger.file->close();
                    logger.file.reset();
                  }
                logger.written_bytes = 0;
                logger.ready = open_append_locked(logger);
                logger.rotation_retry_size = 0;
              }
          }
      }

    if (logger.ready)
      {
        const auto required = static_cast<std::uint64_t>(text.size() + 1);
        const std::uint64_t next_size = logger.written_bytes + required;
        const bool over_limit = next_size > logger.options.max_file_size;
        const bool retry_ready = logger.rotation_retry_size == 0
                                 || next_size > logger.rotation_retry_size;
        if (over_limit && retry_ready)
          {
            rotate_locked(logger);
          }

        if (logger.ready && logger.file && logger.file->is_open())
          {
            if (logger.file->write_line(text))
              {
                logger.written_bytes += required;
                written = true;
              }
            else
              {
                logger.ready = false;
                logger.failed_writes = 0;
              }
          }
      }
  }

  if (!written)
    {
      fallback_to_stderr(text);
    }
}

trace_scope::trace_scope(const std::string& name)
{
  begin_trace(name);
}

trace_scope::~trace_scope() noexcept
{
  end_trace();
}

bool
trace_frame::valid() const noexcept
{
  return !trace_id.empty() && !node_id.empty();
}

trace_frame
capture_trace()
{
  const trace_frame* frame = current_frame();
  return (nullptr == frame) ? trace_frame() : *frame;
}

restore_scope::restore_scope(const trace_frame& frame) : linked_(frame.valid())
{
  if (linked_)
    {
      restore_trace(frame);
    }
}

restore_scope::~restore_scope() noexcept
{
  if (linked_)
    {
      end_trace();
    }
}

detail::restore_trace_scope::restore_trace_scope(const trace_frame& frame,
                                                 const std::string& name)
  : restore_(frame), trace_(name)
{
}

detail::restore_trace_scope::~restore_trace_scope() noexcept = default;

detail::log_line::log_line(level severity, const char* file, int line)
  : severity_(severity), file_(file), line_(line),
    enabled_(is_enabled(severity))
{
}

detail::log_line::~log_line() noexcept
{
  if (!enabled_)
    {
      return;
    }

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
  try
    {
      write_line(severity_, file_, line_, stream_.str());
    }
  catch (...)
    {
      // 日志写入不能从析构函数抛出；失败时只向 stderr 提示一次固定信息。
      (void)std::fprintf(stderr, "stacklog: 日志写入失败\n");
    }
#else
  // -fno-exceptions 构建下无法捕获；日志失败仍由进程的终止策略处理。
  write_line(severity_, file_, line_, stream_.str());
#endif
}

} // namespace stacklog
