"""小型动作效果锚定的执行模式检查，不运行完整学习控制器。

每种优化模式使用独立 Python 进程，重用冻结动作探针和 ActionSink。
只在进程内替换构建 Mode 的函数，不编辑动作适配器或第三方源码。
额外检查 shared updates 能作为效果根，使用户可调用函数无需返回令牌。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_message(error):
    """保留可诊断错误文本，但删除本机绝对路径和不稳定对象地址。"""
    message = str(error)
    message = re.sub(r"[A-Za-z]:[\\/][^\n\r]*", "<local-path>", message)
    message = re.sub(r"0x[0-9a-fA-F]+", "<address>", message)
    return " ".join(message.split())[:500]


def capture(function):
    try:
        return {"success": True, "result": function()}
    except Exception as error:
        # 模式切换可能暴露真实实现边界；记录失败，不替换执行路径来制造成功。
        return {"success": False, "error_class": type(error).__name__,
                "message": safe_message(error)}


def source_hashes(action):
    """只读核验两个核心源树及原动作适配器，独立于其正式控制器评估。"""
    roots = {"ATF_V0": action.bootstrap.V0 / "ATF",
             "PyTensor": Path(action.pytensor.__file__).resolve().parent}
    hashes = {"action_comparison.py": sha(HERE / "action_comparison.py")}
    for name, root in roots.items():
        for path in sorted(root.rglob("*.py")):
            hashes[f"{name}/{path.relative_to(root).as_posix()}"] = sha(path)
    return hashes


def shared_update_probe(action):
    """通过公开的 function(updates=...) 让动作执行可达，用户输出为空。"""
    np, pt, pytensor = action.np, action.pt, action.pytensor
    state, events, actions = {"position": 0}, [], []
    permits = [pt.scalar(f"update_permit_{i}", dtype="bool") for i in range(2)]
    shared_token = pytensor.shared(np.int64(0), name="action_completion_token")

    def apply_action(action_id):
        state["position"] = state["position"] * (action_id + 1) + (1 if action_id == 1 else 4)

    first = action.ActionSink(1, events, actions, apply_action)(permits[0], shared_token, np.zeros(4))
    second = action.ActionSink(2, events, actions, apply_action)(permits[1], first, np.zeros(4))
    # 完成令牌不属于调用者所请求的输出；shared update 本身保留该依赖链。
    function = pytensor.function(permits, [], updates={shared_token: second}, mode=action.mode())
    trials = [(True, False), (False, True), (False, False)]
    positions, tokens, outputs = [], [], []
    for trial in trials:
        result = function(*trial)
        assert result == []
        positions.append(state["position"])
        tokens.append(int(shared_token.get_value()))
        outputs.append(result)
    assert positions == [1, 7, 7]
    assert tokens == [2, 4, 6]
    assert actions == [1, 2] and events == [1, 2, 1, 2, 1, 2]
    return {"positions": positions, "action_order": actions,
            "sink_invocations": len(events), "shared_token_after_calls": tokens,
            "user_visible_outputs": outputs, "effect_root": "shared_variable_update",
            "common_permits": trials,
            "interpretation": "A shared-variable update anchors both callbacks even when the requested user-visible output list is empty."}


def decorated_callback_probe(action):
    """用官方 wrap_py 装饰原函数，不手写 Op 子类，测两种效果根及负对照。

    这项检查防止把自定义 Op 类的代码量误当成 PyTensor 的最低接入成本。
    permit 与 token 使用内置 bool/int64 类型；回调返回零维 int64 数组，
    满足官方包装器的存储约定。类型声明之外没有新增核心代码或运行器。
    """
    np, pt, pytensor = action.np, action.pt, action.pytensor
    bool_type = pt.scalar(dtype="bool").type
    token_type = pt.lscalar().type
    trials = [(True, False), (False, True), (False, False)]
    reports = {}
    for root in ("requested_output", "shared_update", "unanchored"):
        state, events, emitted_actions = {"position": 0}, [], []
        permits = [pt.scalar(f"decorated_permit_{i}", dtype="bool") for i in range(2)]

        def make_callback(action_id):
            @pytensor.wrap_py(itypes=[bool_type, token_type], otypes=[token_type])
            def callback(permit, token):
                events.append(action_id)
                if bool(permit):
                    emitted_actions.append(action_id)
                    state["position"] = state["position"] * (action_id + 1) + (1 if action_id == 1 else 4)
                return np.asarray(int(token) + 1, dtype="int64")
            return callback

        first_callback, second_callback = make_callback(1), make_callback(2)
        shared_token = pytensor.shared(np.int64(0)) if root == "shared_update" else None
        # wrap_py 使用基础 Op 的输入接口，不会替我们把 Python/NumPy 常量
        # 转成符号变量；因此常量令牌显式进入张量图。shared 已经是 Variable。
        token = shared_token if shared_token is not None else pt.as_tensor_variable(np.int64(0))
        first = first_callback(permits[0], token)
        second = second_callback(permits[1], first)
        if root == "requested_output":
            function = pytensor.function(permits, second, mode=action.mode())
        elif root == "shared_update":
            function = pytensor.function(permits, [], updates={shared_token: second}, mode=action.mode())
        else:
            # 回调图存在，但既未成为输出的依赖，也未接入 shared updates。
            function = pytensor.function(permits, pt.cast(permits[0], "int64") + pt.cast(permits[1], "int64"), mode=action.mode())
        positions, outputs, tokens = [], [], []
        for trial in trials:
            output = function(*trial)
            outputs.append(output if isinstance(output, list) else int(output))
            positions.append(state["position"])
            if shared_token is not None:
                tokens.append(int(shared_token.get_value()))
        if root == "unanchored":
            assert positions == [0, 0, 0] and events == [] and emitted_actions == []
        else:
            assert positions == [1, 7, 7]
            assert emitted_actions == [1, 2] and events == [1, 2, 1, 2, 1, 2]
            if root == "requested_output":
                assert outputs == [2, 2, 2]
            else:
                assert outputs == [[], [], []] and tokens == [2, 4, 6]
        reports[root] = {"positions": positions, "action_order": emitted_actions,
                         "sink_invocations": len(events), "user_visible_outputs": outputs,
                         "shared_token_after_calls": tokens}
    return {"decorator": "pytensor.wrap_py", "native_generated_op": type(first_callback).__name__,
            "handwritten_op_subclasses": 0, "input_types": ["bool scalar", "int64 scalar"],
            "output_types": ["int64 scalar"], "roots": reports,
            "interpretation": "The public decorator can wrap original Python action callbacks. Both explicit output anchoring and shared updates retain effects; an unanchored callback branch is not executed. A handwritten Op subclass is not necessary for this example."}


def worker(label):
    # 导入发生在工作进程；模块的 if __name__ 守卫确保不会执行完整动作实验。
    import action_comparison as action
    from pytensor.compile.mode import Mode

    before = source_hashes(action)
    optimizer = None if label == "unoptimized" else "fast_run"
    action.mode = lambda: Mode(linker="py", optimizer=optimizer)
    standard = capture(action.effect_probes)
    updates = capture(lambda: shared_update_probe(action))
    decorated = capture(lambda: decorated_callback_probe(action))
    after = source_hashes(action)
    unchanged = before == after
    return {
        "mode": label,
        "linker": "py",
        "optimizer": optimizer,
        "pytensor_version": action.pytensor.__version__,
        "effect_output_and_unanchored_probes": standard,
        "effect_shared_updates_with_empty_outputs": updates,
        "decorated_python_action_callbacks": decorated,
        "action_source_sha256": before["action_comparison.py"],
        "core_source_files_checked": len(before) - 1,
        "core_and_action_sources_unchanged": unchanged,
        "all_checks_passed": standard["success"] and updates["success"] and decorated["success"] and unchanged,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["unoptimized", "fast_run"])
    args = parser.parse_args()
    if args.worker:
        # 固定前缀让主进程能与第三方库偶发的 stdout 日志区分，不写临时结果文件。
        print("EFFECT_MODE_RESULT " + json.dumps(worker(args.worker)), flush=True)
        return
    results = []
    action_before = sha(HERE / "action_comparison.py")
    for label in ("unoptimized", "fast_run"):
        process = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", label],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
        lines = [line for line in process.stdout.splitlines() if line.startswith("EFFECT_MODE_RESULT ")]
        if process.returncode == 0 and len(lines) == 1:
            result = json.loads(lines[0].removeprefix("EFFECT_MODE_RESULT "))
        else:
            result = {"mode": label, "all_checks_passed": False,
                      "worker_returncode": process.returncode,
                      "worker_error": safe_message(process.stderr[-1500:])}
        results.append(result)
        print(f"{label}: passed={result['all_checks_passed']}", flush=True)
    action_unchanged = action_before == sha(HERE / "action_comparison.py")
    report = {
        "study": "Action-effect anchoring mode sensitivity",
        "effect_check_source_sha256": sha(__file__),
        "action_source_sha256": action_before,
        "independent_process_per_mode": True,
        "modes": results,
        "action_source_unchanged": action_unchanged,
        "all_checks_passed": action_unchanged and all(result["all_checks_passed"] for result in results),
        "scope": "Only the small mutually exclusive action-effect example is tested under both optimization modes. This does not validate the complete floating-point controller under FAST_RUN, measure speed, or establish automatic concurrency safety. Effects can be anchored by requested outputs or by shared-variable updates; user-visible completion-token outputs are not required. The native wrap_py decorator also supports these callbacks without a handwritten Op subclass.",
    }
    path = HERE / "results/effect_modes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "modes": len(results)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
