"""冻结 ATF V0 的工作流适配层与原生接口边界探针。

共同搜索器只负责提出图和决定何时入库；本文件将每张候选图真正交给
GraphProcessor，将复用模块真正封装成 EvolvedMethod。这里没有第二个
Python 图解释器，也不修改 ATF 核心。所有探针使用独立实例，避免影响
正式搜索中的方法库或把探针原语泄漏给搜索器。
"""
from __future__ import annotations

import bootstrap  # noqa: F401  # 先选择冻结 V0，并设置缓存、字节码等实验环境。

from dataclasses import dataclass
import re
from typing import Any, Callable

import networkx as nx
import numpy as np

from ATF.CollapseManager.collapse_functions import CollapseMethod
from ATF.GraphProcessor.graph_processor import GraphProcessor
from ATF.core.evolved_method import EvolvedMethod


def _active_nodes(program: dict) -> list[int]:
    """从唯一输出回溯依赖；共享节点只纳入一次，但输入边保留各自端口。"""
    input_count = len(program["input_types"])
    visited: set[int] = set()

    def visit(reference: int) -> None:
        if reference < input_count:
            return
        index = reference - input_count
        if index in visited:
            return
        visited.add(index)
        for source in program["nodes"][index]["inputs"]:
            visit(source)

    visit(program["output"])
    return sorted(visited)


def _graph(program: dict, methods: dict) -> nx.MultiDiGraph:
    """把公开的端口引用转换为原生多重边，不把整张图藏进一个回调节点。

    根输入占用引用 0..D-1，节点 i 的单输出引用为 D+i。数据边携带
    output_index 和 input_slot，所以 add(y, y) 是两条参数位置不同的边，
    并不会因为源节点相同而丢失第二个参数。非活跃节点不进入执行图。
    """
    graph = nx.MultiDiGraph()
    input_types = list(program["input_types"])
    input_count = len(input_types)
    graph.add_node("root", layer=0, output_types=input_types)
    graph.add_node("collapse", layer=len(program["nodes"]) + 1)

    def source_info(reference: int) -> tuple[str, int, str]:
        if reference < input_count:
            return "root", reference, input_types[reference]
        index = reference - input_count
        method = methods[program["nodes"][index]["method"]]
        return f"n{index}", 0, method["output_types"][0]

    for index in _active_nodes(program):
        node = program["nodes"][index]
        target = f"n{index}"
        graph.add_node(target, method_name=node["method"], layer=index + 1,
                       is_passthrough=False)
        for slot, reference in enumerate(node["inputs"]):
            source, output_index, data_type = source_info(reference)
            graph.add_edge(source, target, output_index=output_index,
                           input_slot=slot, data_type=data_type)
    source, output_index, data_type = source_info(program["output"])
    graph.add_edge(source, "collapse", output_index=output_index,
                   data_coord=0, data_type=data_type)
    return graph


def _json_value(value: Any) -> Any:
    """只为探针报告转换容器；不参与被比较方法的执行或类型判断。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _capture(function: Callable[[], Any]) -> dict:
    """把原生接受/拒绝行为记录下来，而不是把预期限制伪装为成功。

    错误类别保留，但不输出调用栈、机器绝对路径或对象内存地址。
    这里的探针输入均由本文件生成，错误信息不包含用户数据。
    """
    try:
        return {"status": "accepted", "result": _json_value(function())}
    except Exception as error:
        message = str(error)
        message = re.sub(r"[A-Za-z]:[\\/][^\n\r]*", "<local-path>", message)
        message = re.sub(r"0x[0-9a-fA-F]+", "<address>", message)
        message = " ".join(message.split())[:260]
        return {"status": "rejected", "error_type": type(error).__name__,
                "message": message}


@dataclass(frozen=True)
class Observation:
    """自定义 Python 对象探针：仅携带一个数值，使用确定性的 repr。"""
    value: float


class ATFBackend:
    name = "atf_v0"

    def __init__(self):
        # 注册表在正式流程中只追加，不覆盖已有名称；旧模块引用的原语语义
        # 因而不会被后续入库悄悄改变。学习与准入政策位于共同驱动器中。
        self.methods: dict[str, dict] = {}
        for name, count, function in [
            ("increment", 1, lambda x: x + 1.),
            ("add_3", 1, lambda x: x + 3.),
            ("multiply_by_2", 1, lambda x: x * 2.),
            ("multiply_by_05", 1, lambda x: x * .5),
            ("identity", 1, lambda x: x),
            ("add", 2, lambda x, y: x + y),
        ]:
            self._register_primitive(name, function, ["scalar"] * count)

    def _register_primitive(self, name: str, function: Callable,
                            input_types: list[str], output_type: str = "scalar") -> None:
        """通过既有方法字典约定接入函数；这属于适配层，不是新增核心 API。"""
        if name in self.methods:
            raise ValueError(f"Method name already registered: {name}")
        self.methods[name] = {
            "function": function,
            "input_count": len(input_types),
            "output_count": 1,
            "input_types": list(input_types),
            "output_types": [output_type],
            "group": "workflow_comparison",
            "weight": 1.0,
            "vectorized": False,
        }

    def register(self, name: str, program: dict) -> None:
        """将已获准的数值子图变成真实 EvolvedMethod，供后续候选调用。

        此接口不负责发现、筛选、去重或验证任务成功；这些决策必须由双方
        共用的学习流程作出。V0 会把模块输出转为 float，这个边界在下面
        单独测试，不通过包装器补齐任意对象/向量输出能力。
        """
        if name in self.methods:
            raise ValueError(f"Method name already registered: {name}")
        metadata = {"name": name, "input_types": list(program["input_types"]),
                    "output_types": ["scalar"],
                    "input_count": len(program["input_types"]), "output_count": 1,
                    "group": "workflow_comparison", "weight": 1.0,
                    "vectorized": False, "is_evolved": True}
        method = EvolvedMethod(name, _graph(program, self.methods), self.methods, metadata)
        self.methods[name] = dict(metadata, function=method)

    def compile(self, program: dict) -> Callable[[np.ndarray], np.ndarray]:
        """准备原生运行器；ATF 此处并不进行代码生成或新增形状验证。

        正式工作负载约定输入 NxD、输出每行一个数值。原生运行器接收
        输入后自行检查 batch 维度；不能在适配层预先过滤错误输入，从而
        掩盖两种系统实际的输入约束。向量/对象探针直接调用原生接口。
        """
        processor = GraphProcessor(_graph(program, self.methods), self.methods,
                                   collapse_method=CollapseMethod.IDENTITY)

        def evaluate(samples: np.ndarray) -> np.ndarray:
            return np.asarray(processor.infer_with_graph(samples), dtype=float).reshape(-1)

        return evaluate

    def probes(self) -> dict:
        """记录能力检查与边界观察；所有图均为确定性的独立小例子。"""
        samples = np.arange(5, dtype=float).reshape(-1, 1)

        def unary(name: str, input_type: str = "scalar") -> dict:
            return {"input_types": [input_type], "nodes": [{"method": name, "inputs": [0]}],
                    "output": 1}

        def scalar_module():
            backend = ATFBackend()
            body = {"input_types": ["scalar"], "nodes": [
                {"method": "increment", "inputs": [0]},
                {"method": "multiply_by_2", "inputs": [1]}], "output": 2}
            backend.register("learned_affine", body)
            output = backend.compile(unary("learned_affine"))(samples)
            np.testing.assert_array_equal(output, 2 * (samples[:, 0] + 1))
            return {"outputs": output, "expected_outputs": [2., 4., 6., 8., 10.]}

        def vector_module():
            backend = ATFBackend()
            backend._register_primitive("make_vector", lambda x: np.array([x, x + 1.]),
                                        ["scalar"], "vector")
            # 输出声明也设为 vector，让失败反映 EvolvedMethod 的实际转换，
            # 而不是人为写错元数据。直接调用不经过 compile 的输出展平。
            body = unary("make_vector")
            method = EvolvedMethod("vector_module", _graph(body, backend.methods),
                                   backend.methods, {"input_types": ["scalar"],
                                   "output_types": ["vector"], "output_count": 1})
            return method(2.)

        def vector_primitive():
            backend = ATFBackend()
            backend._register_primitive("make_vector", lambda x: np.array([x, x + 1.]),
                                        ["scalar"], "vector")
            processor = GraphProcessor(_graph(unary("make_vector"), backend.methods),
                                       backend.methods, collapse_method=CollapseMethod.IDENTITY)
            return processor.infer_with_graph_single([2.])

        def shared(nested: bool):
            backend = ATFBackend()
            calls = [0]

            def counted(x):
                # 计数只用于观测执行次数，不改变该原语返回值或候选结构。
                calls[0] += 1
                return x + 1.

            backend._register_primitive("counted", counted, ["scalar"])
            body = {"input_types": ["scalar"], "nodes": [
                {"method": "counted", "inputs": [0]},
                {"method": "add", "inputs": [1, 1]}], "output": 2}
            if nested:
                backend.register("inner_shared", body)
                outer = {"input_types": ["scalar"], "nodes": [
                    {"method": "inner_shared", "inputs": [0]},
                    {"method": "add", "inputs": [1, 1]}], "output": 2}
                backend.register("outer_shared", outer)
                program, factor = unary("outer_shared"), 4.
            else:
                program, factor = body, 2.
            output = backend.compile(program)(samples)
            np.testing.assert_array_equal(output, factor * (samples[:, 0] + 1.))
            assert calls[0] == len(samples)
            return {"outputs": output, "counted_primitive_calls": calls[0],
                    "expected_calls": len(samples), "sample_count": len(samples)}

        def local_replacement():
            backend = ATFBackend()
            original = GraphProcessor(_graph(unary("increment"), backend.methods),
                                      backend.methods, collapse_method=CollapseMethod.IDENTITY)
            # 在独立图副本上调用原生替换接口，再执行两个版本。允许两种后端
            # 重建/重新编译，不要求它们具有相同的缓存或动态绑定语义。
            revised = GraphProcessor(original.graph.copy(), backend.methods,
                                     collapse_method=CollapseMethod.IDENTITY)
            new_id = revised.replace_node_method("n0", "add_3")
            before = np.asarray(original.infer_with_graph(samples)).reshape(-1)
            after = np.asarray(revised.infer_with_graph(samples)).reshape(-1)
            np.testing.assert_array_equal(before, samples[:, 0] + 1.)
            np.testing.assert_array_equal(after, samples[:, 0] + 3.)
            assert original.graph.nodes["n0"]["method_name"] == "increment"
            return {"original_outputs": before, "revised_outputs": after,
                    "original_method_unchanged": True, "revised_node": new_id}

        def observation(module: bool, wrong: bool = False):
            backend = ATFBackend()
            backend._register_primitive("read_observation", lambda value: float(value.value),
                                        ["observation"])
            body = unary("read_observation", "observation")
            value = {"value": 2.5} if wrong else Observation(2.5)
            if module:
                backend.register("observation_module", body)
                output = backend.methods["observation_module"]["function"](value)
            else:
                processor = GraphProcessor(_graph(body, backend.methods), backend.methods,
                                           collapse_method=CollapseMethod.IDENTITY)
                output = processor.infer_with_graph_single([value])
            assert float(output) == 2.5
            return {"output": float(output), "input_kind": "Observation"}

        def append_primitive():
            backend = ATFBackend()
            count_before = len(backend.methods)
            old_evaluator = backend.compile(unary("increment"))
            backend._register_primitive("square", lambda value: value * value, ["scalar"])
            output = backend.compile(unary("square"))(samples)
            np.testing.assert_array_equal(output, samples[:, 0] ** 2)
            np.testing.assert_array_equal(old_evaluator(samples), samples[:, 0] + 1)
            return {"outputs": output, "methods_before": count_before,
                    "methods_after": len(backend.methods), "old_program_unchanged": True}

        return {
            "scalar_module": _capture(scalar_module),
            "vector_primitive": _capture(vector_primitive),
            "vector_single_output_module": _capture(vector_module),
            "wrong_input_rank": _capture(lambda: ATFBackend().compile(unary("increment"))(np.arange(5.))),
            "shared_fanout": _capture(lambda: shared(False)),
            "nested_shared_module": _capture(lambda: shared(True)),
            "local_replacement": _capture(local_replacement),
            "observation_python_method": _capture(lambda: observation(False)),
            "observation_scalar_module": _capture(lambda: observation(True)),
            "observation_wrong_object": _capture(lambda: observation(False, True)),
            "observation_module_wrong_object": _capture(lambda: observation(True, True)),
            "append_primitive": _capture(append_primitive),
        }
