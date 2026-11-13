#!/bin/sh
# C++ 核心的门禁入口：构建/测试、排版检查、静态审计。
# 用法：./check.sh [all|ci|style|audit]
#
# 三个工具从同级目录引用（与该项目其余部分一致），不安装到系统。
set -u

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
# 并行度自适应本机核数；显式给 JOBS=... 可覆盖
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
STAGE=${1:-all}
# 三个工具仓库与本仓库同级（../../ 是 code/ 目录）
TOOLS=../..
STATUS=0

run() {
    echo "[check] $*"
    "$@" || STATUS=1
}

case "$STAGE" in
    all)
        run python3 "$TOOLS/cxx_ci/ci/ci.py" --jobs "$JOBS"
        run python3 "$TOOLS/cxx_ci/standardize/standardize.py" --check --jobs "$JOBS"
        run env PYTHONPATH="$TOOLS/cxx_audit/src" python3 -m cpp_audit -c audit.toml \
            --jobs "$JOBS" all
        ;;
    ci)
        run python3 "$TOOLS/cxx_ci/ci/ci.py" --jobs "$JOBS"
        ;;
    style)
        run python3 "$TOOLS/cxx_ci/standardize/standardize.py" --check --jobs "$JOBS"
        ;;
    audit)
        run env PYTHONPATH="$TOOLS/cxx_audit/src" python3 -m cpp_audit -c audit.toml \
            --jobs "$JOBS" all
        ;;
    *)
        echo "usage: $0 [all|ci|style|audit]" >&2
        exit 2
        ;;
esac

if [ "$STATUS" -ne 0 ]; then
    echo "[check] FAILED"
else
    echo "[check] passed"
fi
exit "$STATUS"
