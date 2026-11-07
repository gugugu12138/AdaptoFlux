"""Run the supplementary studies in a new, isolated copy of this release.

所有正式证据均从发布包只读取得；旧脚本只会在新建输出目录的副本中执行。
运行器不使用当前仓库的 ATF，不修改冻结协议，也不覆盖已有输出目录。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


SOURCE = Path(__file__).resolve().parent
PACKAGE_RELATIVE = Path("experiments/rebuttal_20261106")
TASKS = ("type-audit", "encoding", "scaling-audit", "action", "type-search", "scaling")
DEPENDENCIES = ("numpy", "networkx", "pandas", "matplotlib", "scipy", "psutil")
IGNORED_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", "cache"}


def digest(path: Path) -> str:
    """分块读取文件，避免大型轨迹文件在检查时额外占用大量内存。"""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def is_link(path: Path) -> bool:
    """同时拒绝符号链接和 Windows junction，防止复制或改名越出边界。"""
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def inventory(root: Path) -> dict[str, str]:
    """对发布文件做字节级清单；运行产生的缓存不属于发布证据。"""
    result = {}
    for directory, folders, files in os.walk(root, followlinks=False):
        current = Path(directory)
        folders[:] = sorted(name for name in folders if name not in IGNORED_DIRECTORIES)
        for name in folders + files:
            path = current / name
            if is_link(path):
                raise RuntimeError(f"Linked files/directories are not supported: {path}")
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = current / name
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def verify_manifest(source: Path) -> dict:
    """验证公开版本的摘要；原始摘要仅用于溯源，绝不偷偷改写为新摘要。

    匿名化文件允许同时保留 source_sha256 与 transformation。这里比较
    sha256（发布字节），原冻结协议与发布映射之间的关系由发布验证器解释。
    """
    path = source / "release_manifest.json"
    if not path.is_file():
        raise RuntimeError("Missing release_manifest.json; use the complete release package.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
        raise RuntimeError("Unsupported release_manifest.json schema.")
    if not manifest["files"]:
        raise RuntimeError("The release manifest contains no files.")
    for relative, entry in manifest["files"].items():
        candidate = (source / relative.replace("\\", "/")).resolve()
        if not candidate.is_relative_to(source) or not candidate.is_file():
            raise RuntimeError(f"Invalid release manifest path: {relative}")
        if not isinstance(entry, dict) or digest(candidate) != entry.get("sha256"):
            raise RuntimeError(f"Published file checksum mismatch: {relative}")
    return manifest


def checked_output(value: str) -> Path:
    """输出必须不存在，且不能位于发布包或其所在仓库内。

    resolve 会先解析已有父目录的链接。除了 .git 根，还识别 ATF 源码根，
    因而从没有 .git 的源码压缩包启动时也不会把结果混入源码目录。
    """
    requested = Path(value).expanduser()
    if requested.exists() or is_link(requested):
        raise RuntimeError("--output must name a new directory that does not exist.")
    output = requested.resolve()
    protected = [SOURCE]
    for parent in SOURCE.parents:
        if (parent / ".git").exists() or (parent / "ATF/core/adaptoflux.py").is_file():
            protected.append(parent)
    for root in protected:
        if output.is_relative_to(root):
            raise RuntimeError(f"--output must be outside the source package/repository: {root}")
    return output


def require_files(task: str) -> None:
    """在创建输出目录前检查核心资产，缺件时不给出看似成功的空复现。"""
    files = [
        "runtime_v0/ATF/__init__.py",
        "runtime_v0/ATF/GraphProcessor/graph_processor.py",
        "runtime_v0/experiments/minimal_stateful_comparison/env.py",
        "runtime_v0/experiments/minimal_stateful_comparison/methods_constrained.py",
    ]
    study_files = {
        "type-audit": ["type_search/audit_results.py", "type_search/formal/candidate_catalog.jsonl"],
        "type-search": ["type_search/run_type_search.py", "type_search/protocol_frozen.json"],
        "encoding": ["state_encoding/run_encoding.py", "state_encoding/summarize_encoding.py",
                     "type_search/formal/candidate_catalog.jsonl"],
        "scaling-audit": ["scaling/audit_and_plot.py", "scaling/completion_manifest.json"],
        "scaling": ["scaling/run_scaling.py", "scaling/audit_and_plot.py"],
        "action": ["action_example/verify_native.py", "action_example/reference/reproduce.py",
                   "action_example/reference/kernel.py", "action_example/reference/world_test.npz",
                   "action_example/reference/final_raw.npz",
                   "action_example/reference/models/system_s104.json",
                   "action_example/reference/models/graph_s104_bird.json"],
    }
    missing = [name for name in files + study_files[task] if not (SOURCE / name).is_file()]
    if missing:
        raise RuntimeError("Incomplete release: " + ", ".join(missing))
    if task == "scaling" and (os.name != "nt" or shutil.which("powershell") is None):
        raise RuntimeError("The frozen scaling runner requires Windows PowerShell; use scaling-audit elsewhere.")


def copy_workspace(output: Path) -> tuple[Path, Path]:
    """保留实验相对路径，并把固定 V0 运行器放在旧脚本预期的工作根目录。

    copy2 保留源码字节（包括 CRLF），不会让文件换行转换破坏冻结摘要。
    不导入源仓库，因此所有缓存和旧脚本的固定输出位置都落在 scratch 中。
    """
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    package = workspace / PACKAGE_RELATIVE
    shutil.copytree(SOURCE, package, copy_function=shutil.copy2,
                    ignore=shutil.ignore_patterns(*IGNORED_DIRECTORIES, "*.pyc", "*.pyo"))
    shutil.copytree(package / "runtime_v0/ATF", workspace / "ATF", copy_function=shutil.copy2)
    shutil.copytree(package / "runtime_v0/experiments/minimal_stateful_comparison",
                    workspace / "experiments/minimal_stateful_comparison", copy_function=shutil.copy2)
    # 明确建立本地 experiments 包，避免同名第三方包覆盖这份固定环境。
    (workspace / "experiments/__init__.py").write_text(
        "# 复现工作区：固定实验环境的包入口。\n", encoding="utf-8")
    return workspace, package


def archive_copies(output: Path, package: Path, relative_paths: list[str]) -> None:
    """只移动本次复制产生的精确路径，既清空旧脚本输出位，又保留旧证据。

    本函数不使用递归删除，也不接受源发布包作为路径。每个目标在操作前
    都检查解析后的绝对范围；归档位置必须是本次新输出目录内的新路径。
    """
    expected_package = output / "workspace" / PACKAGE_RELATIVE
    if package.resolve() != expected_package.resolve():
        raise RuntimeError("Refusing to archive outside this run's scratch package.")
    for relative in relative_paths:
        source = (package / relative).resolve()
        destination = (output / "archived_published" / relative).resolve()
        if not source.is_relative_to(package.resolve()) or source == package.resolve():
            raise RuntimeError("Unsafe scratch archive source.")
        if not destination.is_relative_to(output.resolve()):
            raise RuntimeError("Unsafe scratch archive destination.")
        if source.exists():
            if is_link(source) or destination.exists():
                raise RuntimeError(f"Refusing to replace an archive path: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)


def child_environment(workspace: Path) -> dict[str, str]:
    """固定哈希随机种子，禁用字节码写入，并把所有解释器缓存限制在副本。"""
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                       PYTHONPATH=str(workspace), MPLBACKEND="Agg",
                       MPLCONFIGDIR=str(workspace / ".cache/matplotlib"),
                       XDG_CACHE_HOME=str(workspace / ".cache"),
                       NUMBA_CACHE_DIR=str(workspace / ".cache/numba"))
    return environment


def execute(arguments: list[str], name: str, output: Path, workspace: Path) -> None:
    """子进程以当前虚拟环境启动；输出同步显示并单独保存，失败不掩盖。"""
    command = [sys.executable, "-B", *arguments]
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    print(f"Running {name} ...", flush=True)
    with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=workspace, env=child_environment(workspace),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            status = process.wait()
        except BaseException:
            # 用户中断后终止本次子进程，避免它继续在后台改写工作副本。
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if status:
        raise RuntimeError(f"{name} failed (exit {status}); see {logs / (name + '.log')}")


def run_study(task: str, output: Path, workspace: Path, package: Path) -> None:
    """所有分支调用既有 CLI，不重写算法、不改变种子或候选选择规则。"""
    def run(relative: str, *arguments: str) -> None:
        # pilot 与 formal 共用脚本但分别留日志，避免后一阶段覆盖前一阶段记录。
        name = Path(relative).stem
        if arguments and arguments[0] in {"pilot", "formal"}:
            name += "_" + arguments[0]
        execute([str(package / relative), *arguments], name, output, workspace)

    if (package / "verify_release.py").is_file():
        run("verify_release.py")
    execute(["-c", "from pathlib import Path; import ATF; "
             "assert Path(ATF.__file__).resolve() == Path('ATF/__init__.py').resolve(); "
             "print('Verified isolated V0 runtime:', ATF.__file__)"],
            "runtime_import", output, workspace)
    if task == "type-search":
        archive_copies(output, package, ["type_search/formal", "type_search/environment.json",
                                         "type_search/RESULTS.txt"])
        run("type_search/run_type_search.py", "formal")
        run("type_search/audit_results.py")
    elif task == "type-audit":
        run("type_search/audit_results.py")
    elif task == "encoding":
        catalog = package / "type_search/formal/candidate_catalog.jsonl"
        count = sum(bool(line.strip()) for line in catalog.read_text(encoding="utf-8").splitlines())
        if count != 1007:
            raise RuntimeError(f"The published encoding audit requires all 1,007 records; found {count}.")
        archive_copies(output, package, ["state_encoding/formal", "state_encoding/RESULTS.txt"])
        run("state_encoding/run_encoding.py", "--catalog", str(catalog), "--phase", "formal",
            "--timing", "--repeats", "5")
        run("state_encoding/summarize_encoding.py")
    elif task == "action":
        # 使用新标签保存重放输出；公开 formal 结果即使在工作副本内也保留。
        run("action_example/verify_native.py", "--tasks", "bird", "--episodes", "20",
            "--label", "reproduction")
    elif task == "scaling-audit":
        run("scaling/audit_and_plot.py")
    elif task == "scaling":
        names = ["pilot.json", "frozen_protocol.json", "formal_results.json", "completion_manifest.json",
                 "per_graph.csv", "timing_samples.csv", "summary.csv", "audit.json", "REPORT.txt",
                 "FIGURE_CAPTION.txt", "scaling_summary.png", "scaling_summary.svg"]
        archive_copies(output, package, ["scaling/" + name for name in names])
        run("scaling/run_scaling.py", "pilot")
        run("scaling/run_scaling.py", "formal")
        run("scaling/audit_and_plot.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--output", required=True, help="New output directory outside the repository/package.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Verify and copy the release without executing any experiment.")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required; the recorded environment used Python 3.12.9.")
    output = checked_output(args.output)
    require_files(args.task)
    verify_manifest(SOURCE)
    # 保存整个公开目录的前后摘要，既覆盖冻结证据，也覆盖辅助说明与运行器。
    before = inventory(SOURCE)
    versions = {}
    if not args.prepare_only:
        for name in DEPENDENCIES:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as error:
                raise RuntimeError(f"Missing dependency {name}; install requirements.txt first.") from error
    workspace, package = copy_workspace(output)
    record = {"task": args.task, "prepare_only": args.prepare_only,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "packages": versions,
              "source_files_sha256": before, "status": "prepared"}
    record_path = output / "reproduction_record.json"
    try:
        if not args.prepare_only:
            record["status"] = "running"
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            run_study(args.task, output, workspace, package)
            record["status"] = "completed"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = str(error)
        raise
    finally:
        record["source_files_unchanged"] = inventory(SOURCE) == before
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if not record["source_files_unchanged"]:
            raise RuntimeError("The source release changed during this run; see reproduction_record.json.")
    print(f"{record['status'].capitalize()}. Working results: {package}")
    print(f"Source release unchanged. Run record: {record_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Reproduction error: {error}", file=sys.stderr)
        raise SystemExit(1)
