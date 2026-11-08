"""隔离工作流对照的导入和缓存，不改动系统环境或冻结的运行器。"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V0 = REPO / "experiments/rebuttal_20261106/runtime_v0"
sys.dont_write_bytecode = True

# 安装时和受限进程的临时目录可能不同；允许复现者显式指定依赖位置，
# 也允许在自己的虚拟环境中正常安装 PyTensor，不要求使用本机的隔离目录。
locations = [os.environ.get("ATF_WORKFLOW_DEPS"),
             str(Path(tempfile.gettempdir()) / "atf_pytensor_workflow_20261108/deps"),
             str(Path.home() / "AppData/Local/Temp/atf_pytensor_workflow_20261108/deps")]
for candidate in locations:
    if candidate and (Path(candidate) / "pytensor").is_dir():
        sys.path.insert(0, candidate)
        break
sys.path.insert(0, str(V0))

# 所有即时编译/字体缓存均放在临时区，且禁用字节码，避免污染实验源或衍生仓库。
CACHE = Path(tempfile.gettempdir()) / "atf_workflow_runtime_20261108"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE / "numba"))
os.environ.setdefault("PYTENSOR_FLAGS", f"floatX=float64,cxx=,base_compiledir={CACHE / 'pytensor'}")
