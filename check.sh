#!/bin/sh
# 仓库级门禁入口：C++ 核心、Python 测试、Python 静态检查各跑一遍。
# 用法：./check.sh [all|core|python|lint]
#
#   core    在 core/ 里跑 C++ 核心门禁（构建+ctest、排版、clang-tidy/CPD/Valgrind）
#   python  .venv 里的 pytest 套件
#   lint    .venv 里的 ruff 检查（ruff 未安装时跳过并提示，不算失败）
#   all     依次执行以上三项，任一步失败则整体失败，最后打印汇总
#
# 与 core/check.sh 的分工：core/check.sh 只负责 C++ 核心；本脚本是仓库根入口，
# 负责调用它并补上 Python 侧。两个脚本都可以单独使用。
#
# 环境前置（C++ 门禁用）：cxx_ci / cxx_audit 从仓库同级目录引用（../../），
# clang-tidy / clang-format 走 PATH。系统 clang 版本与项目的门禁配置不一致时
# （例如 clang-tidy 21 会多报 clang-analyzer 的 opt-in 检查），先切到项目自带
# 的 clang-17：
#     export PATH="$HOME/.local/bin:$HOME/.local/clang17/bin:$PATH"
#     export LD_LIBRARY_PATH="$HOME/.local/clang17/usr/lib/x86_64-linux-gnu"
#
# 并行度：JOBS 环境变量，默认本机核数，会传给 core/check.sh。
set -u

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
STAGE=${1:-all}
PY="$ROOT/.venv/bin/python"

STATUS=0
FAILED=""
SKIPPED=""
RESULTS=""

note() {
    # note <stage> <result>
    RESULTS="$RESULTS
  $1: $2"
}

fail() {
    STATUS=1
    FAILED="$FAILED $1"
}

skip() {
    SKIPPED="$SKIPPED $1"
}

run() {
    echo "[check] $*"
    "$@"
}

require_venv() {
    if [ -x "$PY" ]; then
        return 0
    fi
    echo "[check] no Python virtualenv at $ROOT/.venv" >&2
    echo "[check] create it from the repo root with:" >&2
    echo "[check]   python3 -m venv .venv && .venv/bin/pip install -e \".[plot,play,dev]\"" >&2
    return 1
}

gate_core() {
    ( cd "$ROOT/core" && run env JOBS="$JOBS" ./check.sh all )
}

gate_python() {
    require_venv || return 1
    run "$PY" -m pytest tests -q
}

gate_lint() {
    require_venv || return 1
    if ! "$PY" -m ruff --version >/dev/null 2>&1; then
        echo "[check] ruff is not installed -- lint skipped" >&2
        echo "[check] install it with: .venv/bin/pip install -e \".[dev]\"" >&2
        skip lint
        return 0
    fi
    run "$PY" -m ruff check python tests
}

run_stage() {
    # run_stage <name> <function>
    stage_name=$1
    stage_func=$2
    if "$stage_func"; then
        case " $SKIPPED " in
            *" $stage_name "*) note "$stage_name" "skipped" ;;
            *) note "$stage_name" "passed" ;;
        esac
    else
        note "$stage_name" "FAILED"
        fail "$stage_name"
    fi
}

echo "[check] repository: $ROOT"
echo "[check] parallelism: JOBS=$JOBS"

case "$STAGE" in
    all)
        run_stage core gate_core
        run_stage python gate_python
        run_stage lint gate_lint
        ;;
    core) run_stage core gate_core ;;
    python) run_stage python gate_python ;;
    lint) run_stage lint gate_lint ;;
    *)
        echo "usage: $0 [all|core|python|lint]" >&2
        exit 2
        ;;
esac

printf '%s\n' "[check] summary:${RESULTS}"
if [ "$STATUS" -ne 0 ]; then
    echo "[check] FAILED:$FAILED"
    exit 1
fi
echo "[check] passed"
exit 0
