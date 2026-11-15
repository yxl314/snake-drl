#pragma once

// 仅供 clang-tidy 静态分析使用（由 core/audit.toml 的 [tidy].extra_args 强制
// 包含）：MinGW-w64/UCRT 的 assert.h 把 assert 失败转给 _assert/_wassert，但
// 这两个函数在头文件里没有 noreturn 标注；clang-analyzer 会认为 assert 失败后
// 继续执行，把 assert 保护的指针访问误报成空指针解引用/未定义返回。
// 这里补上声明属性，让分析器按“assert 失败即终止”建模。
// 普通编译（g++/MSVC）看不到这些内容，对构建与运行零影响。
#if defined(_WIN32) && defined(__clang__)
#ifdef __cplusplus
extern "C" {
#endif
void __cdecl _assert(const char* message, const char* file, unsigned line)
    __attribute__((noreturn));
void __cdecl _wassert(const wchar_t* message, const wchar_t* file, unsigned line)
    __attribute__((noreturn));
#ifdef __cplusplus
}
#endif
#endif
