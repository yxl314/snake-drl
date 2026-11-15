#!/usr/bin/env python3
"""跨平台门禁入口：C++ 核心、Python 测试、Python 静态检查各跑一遍。

POSIX 的 ``check.sh`` 只是调用本脚本的薄壳；Windows（cmd/PowerShell）上直接::

    python check.py [all|core|python|lint]

分派与退出码语义与原来的 ``check.sh`` 一致：0 全部通过 / 1 有阶段失败 /
2 用法或配置错误 / 130 用户中断；输出保留 ``[check]`` 前缀与汇总格式。

Windows 与 POSIX 的差异只有两处，都收在本文件里：

* C++ 核心阶段：POSIX 直接跑 ``core/check.sh all``（保持旧输出与行为）；
  Windows 没有 ``sh``，等价地逐条运行它内部的三个命令（cxx_ci 的 ci 与
  standardize、cxx_audit 的 cpp_audit）。两个工具都只依赖 Python 标准库，
  所以用当前解释器即可。
* venv 位置：Windows 是 ``.venv/Scripts/python.exe``，POSIX 是
  ``.venv/bin/python``。CI 或沙箱环境里 venv 不在仓库内时，用环境变量
  ``SNAKE_DRL_PYTHON`` 指向解释器即可。

环境变量 ``JOBS`` 覆盖并行度，缺省用本机逻辑核数（对应 POSIX 的 ``nproc``）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

STAGES = ("all", "core", "python", "lint")
USAGE = "usage: check.py [all|core|python|lint]"
VENV_ENV = "SNAKE_DRL_PYTHON"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

# run_stage 的三种结果：True = 通过，False = 失败，SKIPPED = 跳过（不算失败）。
SKIPPED = "skipped"


def default_jobs() -> int:
    """返回并行度：``JOBS`` 环境变量优先，否则用本机逻辑核数。

    Returns:
        int: 正整数并行度；取不到核数时退回 4（与 ``nproc`` 缺失时的旧行为一致）。
    """
    raw = os.environ.get("JOBS", "").strip()
    if raw:
        try:
            jobs = int(raw)
        except ValueError:
            jobs = 0
        if jobs > 0:
            return jobs
    # Python 3.13+ 的 process_cpu_count() 与 POSIX 的 sched_getaffinity 都尊重
    # CPU 亲和性，比 os.cpu_count() 更接近 nproc 的语义。
    for probe in (
        lambda: os.process_cpu_count(),  # type: ignore[attr-defined]
        lambda: len(os.sched_getaffinity(0)),  # type: ignore[attr-defined]
        os.cpu_count,
    ):
        try:
            count = probe()
        except (AttributeError, OSError):
            continue
        if count:
            return int(count)
    return 4


def _print_command(command: list[str]) -> None:
    """按旧 ``check.sh`` 的 ``run()`` 格式打印即将执行的命令。"""
    print("[check] " + " ".join(command), flush=True)


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> bool:
    """打印并执行一条命令，返回是否成功；失败只记录，不抛异常。

    Args:
        command: argv 形式的命令。
        cwd: 工作目录。
        env: 完整的环境变量；None 表示继承当前进程。

    Returns:
        bool: 退出码为 0 时为 True。
    """
    _print_command(command)
    try:
        completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    except OSError as exc:
        print(f"[check] cannot run {command[0]}: {exc}", file=sys.stderr)
        return False
    return completed.returncode == 0


def _venv_hint(root: Path) -> str:
    """返回当前平台创建 venv 的一条可复制命令。"""
    if os.name == "nt":
        return 'py -m venv .venv && .venv\\Scripts\\pip install -e ".[plot,play,dev]"'
    return 'python3 -m venv .venv && .venv/bin/pip install -e ".[plot,play,dev]"'


def find_venv_python(root: Path) -> Path | None:
    """定位跑 pytest / ruff 的 Python 解释器。

    顺序：``SNAKE_DRL_PYTHON`` → 仓库内 ``.venv`` 的平台路径。找不到时打印
    与旧脚本同样的可操作提示。

    Args:
        root: 仓库根目录。

    Returns:
        Path | None: 可用的解释器路径；找不到时为 None。
    """
    override = os.environ.get(VENV_ENV, "").strip()
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
        print(f"[check] {VENV_ENV} does not exist: {override}", file=sys.stderr)
        print(f"[check] point {VENV_ENV} at a Python interpreter", file=sys.stderr)
        return None

    venv = root / ".venv"
    candidate = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if candidate.is_file():
        return candidate
    print(f"[check] no Python virtualenv at {venv}", file=sys.stderr)
    print("[check] create it from the repo root with:", file=sys.stderr)
    print(f"[check]   {_venv_hint(root)}", file=sys.stderr)
    return None


def gate_core(root: Path, jobs: int) -> bool:
    """C++ 核心门禁：构建 + ctest + 排版 + clang-tidy/CPD/Valgrind。

    Args:
        root: 仓库根目录。
        jobs: 并行度。

    Returns:
        bool: 三条命令全部成功时为 True。
    """
    core = root / "core"
    if os.name != "nt":
        # POSIX 上保持旧行为：所有分派、退出码合并都交给 core/check.sh。
        return run(["env", f"JOBS={jobs}", "./check.sh", "all"], cwd=core)

    # Windows 没有 sh，core/check.sh 不能直接执行；逐条运行它内部的三个命令，
    # 与 core/check.sh 的 run() 一样：某条失败后继续跑后面的，最后合并退出码。
    tools = root.parent
    env = dict(os.environ)
    audit_src = tools / "cxx_audit" / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(audit_src) + (os.pathsep + existing if existing else "")
    ok = run(
        [sys.executable, str(tools / "cxx_ci" / "ci" / "ci.py"), "--jobs", str(jobs)],
        cwd=core,
    )
    ok &= run(
        [
            sys.executable,
            str(tools / "cxx_ci" / "standardize" / "standardize.py"),
            "--check",
            "--jobs",
            str(jobs),
        ],
        cwd=core,
    )
    ok &= run(
        [sys.executable, "-m", "cpp_audit", "-c", "audit.toml", "--jobs", str(jobs), "all"],
        cwd=core,
        env=env,
    )
    return ok


def gate_python(root: Path, jobs: int) -> bool:
    """Python 测试门禁（jobs 未使用，保留统一签名）。

    Args:
        root: 仓库根目录。
        jobs: 并行度（pytest 自己决定并行度，这里不用）。

    Returns:
        bool: pytest 退出码为 0 时为 True。
    """
    python = find_venv_python(root)
    if python is None:
        return False
    return run([str(python), "-m", "pytest", "tests", "-q"], cwd=root)


def gate_lint(root: Path, jobs: int) -> bool | str:
    """Python 静态检查门禁；ruff 未安装时跳过（不算失败）。

    Args:
        root: 仓库根目录。
        jobs: 并行度（ruff 自己决定并行度，这里不用）。

    Returns:
        bool | str: 通过 True、失败 False、跳过 :data:`SKIPPED`。
    """
    python = find_venv_python(root)
    if python is None:
        return False
    probe = subprocess.run(
        [str(python), "-m", "ruff", "--version"],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        print("[check] ruff is not installed -- lint skipped", file=sys.stderr)
        if os.name == "nt":
            print(
                '[check] install it with: .venv\\Scripts\\pip install -e ".[dev]"',
                file=sys.stderr,
            )
        else:
            print(
                '[check] install it with: .venv/bin/pip install -e ".[dev]"',
                file=sys.stderr,
            )
        return SKIPPED
    return run([str(python), "-m", "ruff", "check", "python", "tests"], cwd=root)


# 阶段名 -> 执行函数；顺序与旧 check.sh 的 all 一致。
GATES = {"core": gate_core, "python": gate_python, "lint": gate_lint}


def main(argv: list[str] | None = None) -> int:
    """命令行入口；返回进程退出码。

    Args:
        argv: 参数列表（不含程序名）；None 时用 ``sys.argv[1:]``。

    Returns:
        int: 0 通过 / 1 失败 / 2 用法错误 / 130 用户中断。
    """
    args = list(sys.argv[1:] if argv is None else argv)
    stage = args[0] if args else "all"
    if stage not in STAGES:
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE

    root = Path(__file__).resolve().parent
    os.chdir(root)
    jobs = default_jobs()
    # 旧 check.sh 在 all 时固定 core -> python -> lint；单阶段时只跑一个。
    planned = list(GATES) if stage == "all" else [stage]

    print(f"[check] repository: {root}")
    print(f"[check] parallelism: JOBS={jobs}")

    results: list[tuple[str, str]] = []
    failed: list[str] = []
    try:
        for name in planned:
            outcome = GATES[name](root, jobs)
            if outcome is False:
                results.append((name, "FAILED"))
                failed.append(name)
            elif outcome == SKIPPED:
                results.append((name, "skipped"))
            else:
                results.append((name, "passed"))
    except KeyboardInterrupt:
        print("\n[check] interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED

    print("[check] summary:")
    for name, outcome in results:
        print(f"  {name}: {outcome}")
    if failed:
        print("[check] FAILED:" + "".join(f" {name}" for name in failed))
        return EXIT_FAILED
    print("[check] passed")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
