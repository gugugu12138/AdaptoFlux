"""将冻结动作控制器逐节点迁移到 PyTensor，并核验实际动作副作用接入。

本文件只读取仓库已公开的参考副本，绝不导入或运行 ATF-nodes。
数值控制器用原生张量运算构图；只有动作回调使用自定义 Op。
主实验是执行迁移与接口比较，不重新训练，也不把回合数当作独立训练种子。
"""
from __future__ import annotations

import bootstrap

import argparse
import ast
import copy
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import networkx as nx
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from ATF.GraphProcessor.graph_processor import GraphProcessor
from ATF.CollapseManager.collapse_functions import CollapseMethod

HERE = Path(__file__).resolve().parent
REFERENCE = bootstrap.REPO / "experiments/rebuttal_20261106/action_example"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_references():
    """仅加载受保护的 NumPy 参考和原类定义，跳过旧脚本的缓存路径设置。"""
    source = REFERENCE / "reference"
    spec = importlib.util.spec_from_file_location("action_portable_reference", source / "reproduce.py")
    portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portable)
    namespace = {"np": np, "nx": nx, "math": math, "dataclass": dataclass, "field": field,
                 "GraphProcessor": GraphProcessor, "CollapseMethod": CollapseMethod}
    tree = ast.parse((REFERENCE / "verify_native.py").read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in ("Context", "finite_feature", "NativeController")]
    assert len(selected) == 3
    # 只重用现成适配器，不执行其模块级副作用或主程序，不修改函数体。
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, "frozen_native_adapter", "exec"), namespace)
    tree = ast.parse((source / "kernel.py").read_text(encoding="utf-8"))
    tick = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tick")
    tick.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[tick], type_ignores=[]))
    exec(compile(module, "frozen_kernel_tick", "exec"), namespace)
    return portable, namespace["tick"], namespace["NativeController"]


def mode():
    """关闭优化以保留逐项运算顺序；本实验不比较优化后吞吐率。"""
    return Mode(linker="py", optimizer=None)


class ActionSink(Op):
    """动作回调返回完成令牌，使效果在输出驱动的图中保持可达。

    permit 决定是否发出动作，token 提供显式先后依赖，state 是同步状态
    完成的屏障。回调仍可改变 Python 环境对象；不是仅返回一个动作编号。
    每次控制器调用都独立重置接收列表，因此这里仅声称串行接口正确性。
    """

    __props__ = ("action_id",)

    def __init__(self, action_id, events, actions, effect=None):
        self.action_id = int(action_id)
        self.events, self.actions, self.effect = events, actions, effect

    def make_node(self, permit, token, state):
        permit, token, state = map(pt.as_tensor_variable, (permit, token, state))
        if permit.dtype != "bool" or permit.ndim or token.dtype != "int64" or token.ndim:
            raise TypeError("Action permits require scalar bool and scalar int64 token")
        if state.dtype != "float64" or state.ndim != 1:
            raise TypeError("State barrier requires a float64 vector")
        return Apply(self, [permit, token, state], [pt.lscalar().type()])

    def perform(self, node, inputs, output_storage):
        permit, token, _ = inputs
        self.events.append(self.action_id)
        if bool(permit):
            self.actions.append(self.action_id)
            if self.effect is not None:
                self.effect(self.action_id)
        output_storage[0][0] = np.asarray(int(token) + 1, dtype="int64")


class PyTensorController:
    """逐个构造冻结特征、乘法、加法、路由和显式状态更新节点。"""

    def __init__(self, record):
        self.record = record
        observation, old = pt.dvector("observation"), pt.dvector("old_state")
        uniform = pt.dscalar("routing_uniform")
        values = []
        for index, (operation, left, right, _) in enumerate(record["features"]):
            if operation == 0:
                value = observation[left]
            elif operation == 7:
                value = old[left] if record["mode"] else pt.constant(np.float64(0))
            elif operation == 1:
                value = values[left] * values[right]
            elif operation == 2:
                value = values[left] - values[right]
            elif operation == 3:
                value = values[left] + values[right]
            elif operation == 4:
                value = pt.cast(values[left] > values[right], "float64")
            elif operation == 5:
                value = pt.abs(values[left])
            else:
                value = pt.clip(values[left], np.float64(-1), np.float64(1))
            value = pt.switch(pt.or_(pt.isnan(value), pt.isinf(value)), np.float64(0),
                              pt.clip(value, np.float64(-4), np.float64(4)))
            value.name = f"feature_{index}"
            values.append(value)

        scores = []
        for action, weights in enumerate(record["weights"]):
            total = None
            # 保持原控制器逐项乘加的次序；不换成 BLAS 点积，避免舍入次序差异。
            for index, weight in enumerate(weights):
                product = np.float64(weight) * values[index]
                total = product if total is None else total + product
            total.name = f"score_{action}"
            scores.append(total)

        emitted = [pt.clip(values[index], np.float64(-1), np.float64(1))
                   for index in record["emissions"]]
        state_mode = record["mode"]
        rho = np.float64(.9 if state_mode in (2, 4) else 0)
        new = []
        for index in range(4):
            if state_mode in (3, 4):
                value = rho * np.float64(.5) * (old[index % 2] + old[2 + index % 2])
                value = value + emitted[index % 2] + emitted[2 + index % 2]
            elif state_mode:
                value = rho * old[index] + emitted[index]
            else:
                value = pt.constant(np.float64(0))
            new.append(pt.clip(value, np.float64(-1), np.float64(1)))

        count = len(scores)
        priorities = pt.stack(scores)
        ties = pt.eq(priorities, pt.max(priorities))
        ties_int = pt.cast(ties, "int64")
        n_ties = pt.sum(ties_int)
        epsilon = record.get("epsilon", 0)
        routed = (uniform - np.float64(epsilon)) / np.float64(1 - epsilon) if epsilon else uniform
        rank = pt.minimum(n_ties - 1, pt.cast(routed * n_ties, "int64"))
        tie_choice = pt.sum(pt.arange(count, dtype="int64")
                            * pt.cast(pt.and_(ties, pt.eq(pt.cumsum(ties_int) - 1, rank)), "int64"))
        if epsilon:
            explore = pt.minimum(np.int64(count - 1), pt.cast(uniform / np.float64(epsilon) * count, "int64"))
            choice = pt.switch(uniform < np.float64(epsilon), explore, tie_choice)
        else:
            choice = tie_choice
        if not record["enabled"]:
            choice = pt.constant(np.int64(-1))
        permits = [pt.eq(choice, np.int64(index)) for index in range(count)]
        state = pt.stack(new)
        self.events, self.actions = [], []
        token = pt.constant(np.int64(0))
        for index, permit in enumerate(permits):
            # 显式令牌是适配器提供的效果顺序，不是 PyTensor 自动推断的效果类型系统。
            token = ActionSink(index + 1, self.events, self.actions)(permit, token, state)
        self.function = pytensor.function(
            [observation, old, uniform], [pt.stack(values), priorities, pt.stack(permits), state, choice + 1, token],
            mode=mode(), on_unused_input="ignore", allow_input_downcast=False,
        )
        self.apply_nodes = len(self.function.maker.fgraph.apply_nodes)
        self.action_ops = sum(isinstance(node.op, ActionSink) for node in self.function.maker.fgraph.apply_nodes)
        assert self.action_ops == count

    def tick(self, observation, old, uniform):
        self.events.clear()
        self.actions.clear()
        values, scores, permits, new, action, token = self.function(
            np.asarray(observation, dtype="float64"), np.asarray(old, dtype="float64"), np.float64(uniform))
        assert int(token) == len(self.record["action_nodes"])
        assert self.events == list(range(1, int(token) + 1))
        assert self.actions == ([int(action)] if int(action) else [])
        assert sum(permits) == int(int(action) != 0)
        return int(action), new, values, scores, tuple(bool(x) for x in permits)


def effect_probes():
    """直接检查图是否执行具有真实环境写入的回调，以及输出锚定的必要性。"""
    state = {"position": 0}
    permits = [pt.scalar(f"permit_{index}", dtype="bool") for index in range(2)]
    events, actions = [], []

    def advance(action):
        # 非交换更新用于核验令牌的实际先后作用：0 -> 1 -> 7。
        state["position"] = state["position"] * (action + 1) + (1 if action == 1 else 4)

    first = ActionSink(1, events, actions, advance)(permits[0], np.int64(0), np.zeros(4))
    second = ActionSink(2, events, actions, advance)(permits[1], first, np.zeros(4))
    anchored = pytensor.function(permits, second, mode=mode())
    # 两个运行器使用完全相同的三个互斥许可输入；单次只执行一种真实环境更新。
    trials = [(True, False), (False, True), (False, False)]
    tensor_positions = []
    for allowed in trials:
        assert int(anchored(*allowed)) == 2
        tensor_positions.append(state["position"])
    assert tensor_positions == [1, 7, 7] and actions == [1, 2]
    anchored_result = {"positions": tensor_positions, "action_order": list(actions), "sink_invocations": len(events)}
    state["position"] = 0
    events.clear(); actions.clear()
    # 未把完成令牌接到任何所请求输出时，输出驱动编译不会保留这个动作分支。
    unanchored = pytensor.function(permits, pt.cast(permits[0], "int64") + pt.cast(permits[1], "int64"), mode=mode())
    for allowed in trials:
        unanchored(*allowed)
    assert state["position"] == 0 and not events

    native_state, native_actions, native_calls = {"position": 0}, [], []
    graph = nx.MultiDiGraph()
    graph.add_node("root", output_types=["ActionPermit", "ActionPermit"], layer=0)
    graph.add_node("collapse", layer=2)
    graph.add_edge("root", "collapse", output_index=0, data_coord=0, data_type="ActionPermit")
    methods = {}
    for index in range(2):
        action_id = index + 1
        def sink(allowed, action_id=action_id):
            native_calls.append(action_id)
            if allowed:
                native_actions.append(action_id)
                native_state["position"] = native_state["position"] * (action_id + 1) + (1 if action_id == 1 else 4)
            return []
        name = f"action_{action_id}"
        graph.add_node(name, method_name=name, layer=1, is_passthrough=False)
        graph.add_edge("root", name, output_index=index, input_slot=0, data_type="ActionPermit")
        methods[name] = {"function": sink, "input_count": 1, "output_count": 0,
                         "input_types": ["ActionPermit"], "output_types": [], "vectorized": False}
    native = GraphProcessor(graph, methods, collapse_method=CollapseMethod.IDENTITY)
    native_positions = []
    for allowed in trials:
        native.infer_with_graph([list(allowed)])
        native_positions.append(native_state["position"])
    assert native_positions == tensor_positions and native_actions == [1, 2] and len(native_calls) == 6
    return {"pytensor_anchored_effects": anchored_result,
            "pytensor_unanchored_branch": {"callbacks": 0, "position": 0},
            "atf_zero_output_sinks": {"positions": native_positions, "action_order": native_actions, "sink_invocations": len(native_calls)},
            "common_permits": trials, "environment_actions": ["position = 2*position + 1", "position = 3*position + 4"],
            "all_expected_checks_passed": True,
            "interpretation": "Both executors perform the same actual environment changes under mutually exclusive permits. ATF retains connected zero-output sinks; this PyTensor adapter adds an output-anchored completion token. This case needs no ordering between simultaneously permitted conflicting actions and proves no automatic concurrency safety."}


def boundary_checks(record, kernel, native_class):
    reports = []
    configurations = [("frozen", True, record.get("epsilon", 0), False, 4),
                      ("uniform_ties", True, 0, True, 4),
                      ("exploration_and_ties", True, .1, True, 4),
                      ("disabled", False, 0, False, 4),
                      ("no_state", True, 0, False, 0),
                      ("instant_shared", True, 0, False, 3),
                      ("private_memory", True, 0, False, 2)]
    observation, old = np.linspace(-7, 7, 157), np.asarray([.7, -.2, -.4, .8])
    for label, enabled, epsilon, zeros, state_mode in configurations:
        changed = copy.deepcopy(record)
        changed.update(enabled=enabled, epsilon=epsilon, mode=state_mode)
        if zeros:
            changed["weights"] = np.zeros_like(changed["weights"]).tolist()
        tensor, native = PyTensorController(changed), native_class(changed)
        for uniform in (0, .049, .05, .099999, .1, .5, .999999999):
            actual = tensor.tick(observation, old, uniform)
            other = native.tick(observation, old, uniform)
            expected = kernel(observation, np.asarray(changed["features"]), np.asarray(changed["weights"]),
                              np.asarray(changed["emissions"]), old, state_mode, uniform, enabled, -1, epsilon)
            assert actual[0] == other[0] == expected[0]
            errors = []
            for index, ref_index in ((1, 1), (2, 3), (3, 4)):
                np.testing.assert_allclose(actual[index], expected[ref_index], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(actual[index], other[index], rtol=1e-12, atol=1e-12)
                errors.append(float(np.max(np.abs(actual[index] - expected[ref_index]))))
            reports.append({"configuration": label, "u": uniform, "action": actual[0],
                            "max_errors_state_features_scores": errors})
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird-episodes", type=int, default=20)
    parser.add_argument("--other-episodes", type=int, default=2)
    parser.add_argument("--label", default="formal")
    parser.add_argument("--skip-boundaries", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.bird_episodes <= 20 and 1 <= args.other_episodes <= 20
    assert args.label.replace("_", "").isalnum()
    destination = HERE / "results"
    destination.mkdir(exist_ok=True)
    protected = [REFERENCE / "reference/reproduce.py", REFERENCE / "reference/kernel.py",
                 REFERENCE / "verify_native.py", REFERENCE / "reference/models/system_s104.json",
                 REFERENCE / "reference/world_test.npz", REFERENCE / "native_trace_bird_formal.npz",
                 bootstrap.V0 / "ATF/GraphProcessor/graph_processor.py",
                 bootstrap.V0 / "ATF/CollapseManager/collapse_functions.py"]
    protected.append(Path(__file__))
    before = {p.relative_to(bootstrap.REPO).as_posix(): sha(p) for p in protected}
    tensor_root = Path(pytensor.__file__).resolve().parent
    # 核心源树只读记录；新增适配器与第三方核心修改分开统计。
    core_files = [("pytensor/" + p.relative_to(tensor_root).as_posix(), p)
                  for p in tensor_root.rglob("*.py")]
    core_files += [("atf_v0/" + p.relative_to(bootstrap.V0).as_posix(), p)
                   for p in (bootstrap.V0 / "ATF").rglob("*.py")]
    core_before = {name: sha(path) for name, path in core_files}
    protocol = {"purpose": "Frozen learned-controller execution migration and action-interface comparison; no retraining",
                "pytensor_version": pytensor.__version__, "bird_episodes": args.bird_episodes,
                "other_task_episodes": args.other_episodes, "episode_start": 83000000,
                "environment_seed": 200000104, "selection": "Consecutive prefix of the existing frozen evaluation stream",
                "compilation": "Python linker, optimizer=None; explicit state and externally supplied routing randomness",
                "numeric_implementation": "Native PyTensor arithmetic, comparison, clipping, indexing, and selection nodes; no whole-controller callback Op",
                "action_adapter": "Real custom action callbacks with explicit chained completion tokens requested as an output",
                "environment_advance": "Outside both controller graphs, matching the frozen study",
                "tolerance": {"rtol": 1e-12, "atol": 1e-12}, "protected_source_sha256": before,
                "backend_core_source_sha256": core_before}
    (destination / f"action_protocol_{args.label}.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    portable, kernel, native_class = load_references()
    system = json.loads((REFERENCE / "reference/models/system_s104.json").read_text())
    with np.load(REFERENCE / "reference/world_test.npz", allow_pickle=False) as data:
        world = {k: data[k].copy() for k in data.files}
    with np.load(REFERENCE / "native_trace_bird_formal.npz", allow_pickle=False) as data:
        archived = {k: data[k].copy() for k in data.files}
    started, tasks, archive_index = time.perf_counter(), [], 0
    for name in ("catch", "maze", "snake", "bird"):
        record = system["modules"][name]["model"]
        tensor, native = PyTensorController(record), native_class(record)
        count = args.bird_episodes if name == "bird" else args.other_episodes
        outcomes, errors, checked = [], np.zeros(3), 0
        for episode in range(83000000, 83000000 + count):
            state = portable.initialize(record["task"], 200000104, episode, world)
            reference_state, memory = state.copy(), np.zeros(4)
            while not state[1]:
                observation = portable.observe(record["task"], 200000104, episode, state, world)
                uniform = portable.draw(200000104, episode, int(state[0]), 1900 + record["task"])
                old = memory.copy()
                actual = tensor.tick(observation, old, uniform)
                other = native.tick(observation, old, uniform)
                expected = kernel(observation, np.asarray(record["features"]), np.asarray(record["weights"]),
                                  np.asarray(record["emissions"]), old, record["mode"], uniform,
                                  record["enabled"], -1, record.get("epsilon", 0))
                assert actual[0] == other[0] == expected[0]
                for slot, (index, ref_index) in enumerate(((1, 1), (2, 3), (3, 4))):
                    np.testing.assert_allclose(actual[index], expected[ref_index], rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(actual[index], other[index], rtol=1e-12, atol=1e-12)
                    errors[slot] = max(errors[slot], float(np.max(np.abs(actual[index] - expected[ref_index]))))
                assert actual[4] == other[4]
                if name == "bird":
                    assert archived["episode"][archive_index] == episode
                    assert archived["step"][archive_index] == int(state[0])
                    assert archived["action"][archive_index] == actual[0]
                    for index, field_name in ((1, "new_state"), (2, "features"), (3, "scores"), (4, "permits")):
                        np.testing.assert_allclose(actual[index], archived[field_name][archive_index], rtol=1e-12, atol=1e-12)
                    archive_index += 1
                memory = actual[1]
                portable.advance(record["task"], 200000104, episode, state, actual[0], world)
                portable.advance(record["task"], 200000104, episode, reference_state, expected[0], world)
                np.testing.assert_array_equal(state, reference_state)
                checked += 1
            outcomes.append({"episode": episode, "success": int(state[2]), "steps": int(state[0]), "progress": float(state[3])})
            print(json.dumps({"task": name, "episode": episode, "steps": int(state[0]), "matched": True}), flush=True)
        tasks.append({"task": name, "episodes": count, "steps": checked, "all_matched": True,
                      "max_absolute_errors_state_features_scores": errors.tolist(), "outcomes": outcomes,
                      "features": len(record["features"]), "score_coefficients": int(np.size(record["weights"])),
                      "pytensor_apply_nodes": tensor.apply_nodes, "pytensor_action_ops": tensor.action_ops,
                      "atf_nodes": native.graph.number_of_nodes(), "atf_zero_output_sinks": len(record["action_nodes"])})
    boundaries = [] if args.skip_boundaries else boundary_checks(system["modules"]["bird"]["model"], kernel, native_class)
    effects = effect_probes()
    after = {p.relative_to(bootstrap.REPO).as_posix(): sha(p) for p in protected}
    assert before == after
    assert core_before == {name: sha(path) for name, path in core_files}
    result = {"all_checks_passed": True, "tasks": tasks, "total_steps": sum(r["steps"] for r in tasks),
              "bird_archived_steps_matched": archive_index, "boundary_probes": boundaries, "effect_probes": effects,
              "protected_sources_unchanged": True, "backend_core_sources_unchanged": True,
              "backend_core_files_checked": len(core_before), "seconds": time.perf_counter() - started,
              "interpretation": "Both backends execute the same learned action controllers. PyTensor needs explicit action anchoring and adapters; this tests execution and integration, not synthesis superiority or novel test generalization."}
    (destination / f"action_results_{args.label}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "steps": result["total_steps"], "archived_steps": archive_index}), flush=True)


if __name__ == "__main__":
    main()
