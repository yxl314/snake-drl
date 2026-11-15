#!/bin/sh
# 仓库级门禁入口的 POSIX 薄壳：真正的分派逻辑在 check.py，Windows 直接用
# python check.py 即可（cmd/PowerShell 都可以）。用法与原来一致：
#
#     ./check.sh [all|core|python|lint]
#
# POSIX 上需要 sh（本脚本）与 python3（跑 check.py）；C++ 核心阶段的
# cxx_ci / cxx_audit 从仓库同级目录引用，clang-tidy / clang-format 走 PATH。
# 环境变量 JOBS 覆盖并行度，缺省用本机逻辑核数。
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
# 与旧脚本一致：先进入脚本所在目录，从其它目录调用时项目根仍是脚本所在工程。
cd "$DIR" || exit 2
exec python3 "$DIR/check.py" "$@"
