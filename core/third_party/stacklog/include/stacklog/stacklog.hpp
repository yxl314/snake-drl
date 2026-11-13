/// @file
/// @brief stacklog 的日志与追踪公共接口。
///
/// 汇总日志级别、运行配置、文件轮转、流式日志与跨线程追踪的声明。

#ifndef STACKLOG_STACKLOG_HPP
#define STACKLOG_STACKLOG_HPP

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>

#include <stacklog/version.hpp>

namespace stacklog
{

/// @brief 日志级别；数值越大越严重。
///
/// off 关闭全部输出；其余级别不低于运行期最低级别时才写出。
enum class level
{
    debug = 0,
    info = 1,
    warn = 2,
    error = 3,
    off = 4
};

/// @brief 日志运行参数。
///
/// init 会拒绝空目录、空基础名和零上限的配置。
struct config
{
    /// @brief 日志目录。
    std::string dir = "logs";
    /// @brief 日志基础名；现行文件为目录下的 @c base_name.log，归档文件带序号后缀。
    std::string base_name = "app";
    /// @brief 单个日志文件的最大字节数；超过后轮转。
    std::uint64_t max_file_size = 5ULL * 1024 * 1024;
    /// @brief 保留的归档文件数；轮转时最旧的一份被移除。
    unsigned max_files = 5;
    /// @brief 运行期最低日志级别。
    level min_level = level::info;
};

/// @brief 跨线程传递的追踪帧快照。
struct trace_frame
{
    /// @brief 顶层链编号；同一次操作的多个步骤共享。
    std::string trace_id;
    /// @brief 当前节点编号；进程内唯一。
    std::string node_id;
    /// @brief 父节点编号；根节点为空。
    std::string parent_id;
    /// @brief 可读作用域名称。
    std::string scope;

    /// @brief 构造空帧。
    trace_frame() = default;

    /// @brief 是否是可用于接续的有效帧。
    ///
    /// @return trace_id 与 node_id 均非空时为 true。
    bool valid() const noexcept;
};

/// @brief 初始化日志系统；失败时返回 false，日志降级到 stderr。
///
/// @param cfg 日志目录、基础名、单文件上限、归档数量和最低级别。
/// @return 配置合法且目录与文件均就绪时为 true。
/// @note 可重复调用，每次都会关闭已有文件并重建状态。
/// @note 目录为空、基础名为空或任一上限为零时初始化失败。
bool init(const config& cfg = config());

/// @brief 关闭日志文件；可重复调用。
///
/// @note 关闭后未重新 init 时，日志输出落到 stderr，并且只提示一次降级信息。
void shutdown();

/// @brief 修改运行期最低日志级别。
///
/// @param value 新的最低级别；level::off 关闭全部输出。
void set_level(level value) noexcept;

/// @brief 当前线程上的追踪作用域；构造入栈，析构出栈。
///
/// @note 构造与析构必须在同一线程完成；节点编号在进程内唯一。
class trace_scope
{
public:
    /// @brief 创建一个追踪节点。
    ///
    /// @param name 可读作用域名称；为空时记为 unnamed。
    explicit trace_scope(const std::string& name);

    /// @brief 离开当前追踪节点。
    ~trace_scope() noexcept;

    /// @brief 禁止拷贝：入栈与出栈必须一一对应。
    trace_scope(const trace_scope&) = delete;
    /// @brief 禁止赋值：入栈与出栈必须一一对应。
    trace_scope& operator=(const trace_scope&) = delete;
};

/// @brief 捕获当前线程栈顶追踪帧的副本，用于跨线程接续。
///
/// @return 栈顶帧的副本；当前线程没有追踪节点时返回空帧。
trace_frame capture_trace();

/// @brief 将外部 trace_frame 接续到当前线程；构造入栈，析构出栈。
///
/// 须位于作用域最外层，并保证其内所有 trace_scope 先析构。
///
/// @note 帧无效时不入栈，析构也不出栈，表现为空操作。
class restore_scope
{
public:
    /// @brief 接续一个捕获到的帧；无效帧退化为空操作。
    ///
    /// @param frame capture_trace 得到的追踪帧。
    explicit restore_scope(const trace_frame& frame);

    /// @brief 断开接续并出栈。
    ~restore_scope() noexcept;

    /// @brief 禁止拷贝：接续与断开必须一一对应。
    restore_scope(const restore_scope&) = delete;
    /// @brief 禁止赋值：接续与断开必须一一对应。
    restore_scope& operator=(const restore_scope&) = delete;

private:
    /// @brief 构造时帧是否有效；决定析构是否出栈。
    bool linked_;
};

namespace detail
{

/// @brief 接续外部帧并创建子作用域；成员析构顺序保证 restore 最后出栈。
class restore_trace_scope
{
public:
    /// @brief 先接续 frame，再创建 name 子作用域。
    ///
    /// @param frame 要接续的追踪帧；无效帧时只在当前线程新建节点。
    /// @param name 子作用域名称；为空时记为 unnamed。
    restore_trace_scope(const trace_frame& frame, const std::string& name);

    /// @brief 先结束 name 子作用域，再断开 frame。
    ~restore_trace_scope() noexcept;

    /// @brief 禁止拷贝：接续与断开必须一一对应。
    restore_trace_scope(const restore_trace_scope&) = delete;
    /// @brief 禁止赋值：接续与断开必须一一对应。
    restore_trace_scope& operator=(const restore_trace_scope&) = delete;

private:
    /// @brief 外部帧的接续；比 trace_ 晚析构。
    restore_scope restore_;
    /// @brief name 对应的子作用域；比 restore_ 早析构。
    trace_scope trace_;
};

/// @brief 流式日志行；析构时落盘。
///
/// 构造时判断本行是否启用；未启用时 operator<< 只返回自身，不缓冲内容。
class log_line
{
public:
    /// @brief 记录级别、文件名和行号。
    ///
    /// @param severity 本行级别。
    /// @param file 源文件名；写入时只取文件名部分。
    /// @param line 源码行号。
    log_line(level severity, const char* file, int line);

    /// @brief 写入缓冲内容。
    ///
    /// @note 析构不抛出；文件输出未就绪时整行落到 stderr，并只提示一次降级信息。
    ~log_line() noexcept;

    /// @brief 禁止拷贝：一行日志只写出一次。
    log_line(const log_line&) = delete;
    /// @brief 禁止赋值：一行日志只写出一次。
    log_line& operator=(const log_line&) = delete;

    /// @brief 追加任意可流式输出的值。
    ///
    /// @param value 待输出的值；本行未启用时不写入缓冲。
    /// @return 自身引用，便于连续输出。
    template <typename T> log_line& operator<<(const T& value)
    {
        if (enabled_)
        {
            stream_ << value;
        }
        return *this;
    }

    /// @brief 支持 std::endl 等流操纵符。
    ///
    /// @param manipulator 作用于内部缓冲的操纵符。
    /// @return 自身引用，便于连续输出。
    log_line& operator<<(std::ostream& (*manipulator)(std::ostream&))
    {
        if (enabled_)
        {
            stream_ << manipulator;
        }
        return *this;
    }

private:
    /// @brief 判断某个级别当前是否启用。
    ///
    /// @param severity 待判断的级别。
    /// @return 配置与待判断级别都不是 off，且 severity 不低于当前最低级别时为 true。
    static bool is_enabled(level severity) noexcept;

    /// @brief 写入一行日志；由析构函数调用。
    ///
    /// @param severity 本行级别。
    /// @param file 源文件名。
    /// @param line 源码行号。
    /// @param message 已拼装的消息正文。
    static void write_line(level severity, const char* file, int line, const std::string& message);

    /// @brief 本行级别。
    level severity_;
    /// @brief 源文件名。
    const char* file_;
    /// @brief 源码行号。
    int line_;
    /// @brief 构造时级别是否启用；未启用则跳过全部输出。
    bool enabled_;
    /// @brief 消息缓冲，析构时取出写入。
    std::ostringstream stream_;
};

} // namespace detail
} // namespace stacklog

/// @brief 通用日志宏：自动注入文件和行号。
#define STACKLOG_LOG(severity) ::stacklog::detail::log_line((severity), __FILE__, __LINE__)

/// @brief 调试日志。
#define STACKLOG_DEBUG() STACKLOG_LOG(::stacklog::level::debug)
/// @brief 普通信息。
#define STACKLOG_INFO() STACKLOG_LOG(::stacklog::level::info)
/// @brief 警告。
#define STACKLOG_WARN() STACKLOG_LOG(::stacklog::level::warn)
/// @brief 错误。
#define STACKLOG_ERROR() STACKLOG_LOG(::stacklog::level::error)

/// @brief 把两个记号直接拼接（预处理阶段展开）。
/// @param a 左侧记号。
/// @param b 右侧记号。
#define STACKLOG_DETAIL_CONCAT_IMPL(a, b) a##b

/// @brief 先展开参数再拼接，避免参数本身是宏时拼接结果出错。
/// @param a 左侧记号（会被预先展开）。
/// @param b 右侧记号（会被预先展开）。
#define STACKLOG_DETAIL_CONCAT(a, b) STACKLOG_DETAIL_CONCAT_IMPL(a, b)

/// @brief 用行号给名字加后缀，得到当前作用域内唯一的标识符。
/// @param name 名字前缀。
#define STACKLOG_DETAIL_UNIQUE(name) STACKLOG_DETAIL_CONCAT(name, __LINE__)

/// @brief 创建当前作用域的追踪节点；同一行不能重复使用，不同行可正常嵌套。
#define STACKLOG_TRACE(name) ::stacklog::trace_scope STACKLOG_DETAIL_UNIQUE(stacklog_trace_scope_)(name)

/// @brief 接续外部帧并立即创建子作用域；等价于 restore_scope + trace_scope。
///
/// 同一行不能重复使用。
#define STACKLOG_RESTORE_TRACE(frame, name)                                                                            \
    ::stacklog::detail::restore_trace_scope STACKLOG_DETAIL_UNIQUE(stacklog_restore_trace_scope_)((frame), (name))

#endif // STACKLOG_STACKLOG_HPP
