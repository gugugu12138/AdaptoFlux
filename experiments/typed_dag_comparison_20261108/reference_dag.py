"""独立的类型化 DAG 参考后端，不依赖 ATF 的表示或执行代码。

程序使用全局整数引用：前 ``len(input_types)`` 项是外部输入，其后是
按列表顺序排列的节点输出。每个节点只有一个输出，但同一输出可以供
多个节点使用，也可以重复接入同一节点的不同参数位置。

复合方法与原语使用相同注册表。编译时展开每次复合方法调用，保留其
内部共享关系；执行时不会递归解释图或重复计算已共享的中间节点。
这里只缓存编译结果，不跨节点调用缓存数值，也不假定原语可交换。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _Plan:
    """编译后的紧凑指令表；第 i 条指令的输出槽为输入数加 i。"""

    input_types: tuple[str, ...]
    output_type: str
    instructions: tuple[tuple[Callable[..., Any], tuple[int, ...]], ...]
    output_slot: int


@dataclass(frozen=True)
class _Method:
    """已校验的方法接口，以及原语函数或已编译的复合方法体。"""

    input_types: tuple[str, ...]
    output_type: str
    function: Callable[..., Any] | None = None
    body: _Plan | None = None


@dataclass(frozen=True)
class _CachedModule:
    """记录缓存所属的注册表和条目，防止误用于另一个实验条件。"""

    registry: Mapping[str, Any]
    entry: Mapping[str, Any]
    method: _Method


def _types(value: Any, context: str) -> tuple[str, ...]:
    """类型是名义字符串，不能把标量存储相同当成声明类型相同。"""
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{context} 必须是由非空类型字符串组成的列表")
    return tuple(value)


def _reference(value: Any, available: int, context: str) -> int:
    # 同时接受 Python 和 NumPy 整数，但排除容易被当成 0/1 的布尔值。
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{context} 必须是整数引用")
    result = int(value)
    if result < 0 or result >= available:
        raise ValueError(f"{context} 引用 {result} 超出可用范围 [0, {available})")
    return result


class _Compiler:
    """一次编译内共享方法缓存，避免让可变注册表污染后续编译。"""

    def __init__(self, registry: Mapping[str, Any], module_cache: dict | None = None):
        if not isinstance(registry, Mapping):
            raise ValueError("registry 必须是方法名到方法声明的映射")
        self.registry = registry
        self.methods: dict[str, _Method] = {}
        # 外部可以在同一阶段的不同候选之间共享模块编译结果。注册表只能
        # 追加条目，既有方法及其依赖保持不变；修改它们后必须清空此缓存。
        # 原语自身仍在每次候选编译中读取，避免把整个注册表永久冻结。
        self.module_cache = module_cache if module_cache is not None else {}
        # 保存正在展开的方法名，可给出可读的递归依赖链。
        self.expanding: list[str] = []

    def method(self, name: Any) -> _Method:
        if not isinstance(name, str) or not name:
            raise ValueError("节点的 method 必须是非空方法名字符串")
        if name in self.methods:
            return self.methods[name]
        if name in self.expanding:
            start = self.expanding.index(name)
            cycle = self.expanding[start:] + [name]
            raise ValueError("复合方法存在递归调用循环：" + " -> ".join(cycle))
        if name not in self.registry:
            raise ValueError(f"未注册的方法：{name}")
        entry = self.registry[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"方法 {name} 的声明必须是映射")
        cached = self.module_cache.get(name)
        if (
            isinstance(cached, _CachedModule)
            and cached.registry is self.registry
            and cached.entry is entry
        ):
            self.methods[name] = cached.method
            return cached.method

        inputs = _types(entry.get("input_types"), f"方法 {name} 的 input_types")
        outputs = _types(entry.get("output_types"), f"方法 {name} 的 output_types")
        if len(outputs) != 1:
            raise ValueError(f"方法 {name} 必须恰有一个输出类型")
        has_function, has_body = "function" in entry, "body" in entry
        if has_function == has_body:
            raise ValueError(f"方法 {name} 必须且只能提供 function 或 body")

        if has_function:
            function = entry["function"]
            if not callable(function):
                raise ValueError(f"方法 {name} 的 function 不可调用")
            result = _Method(inputs, outputs[0], function=function)
        else:
            self.expanding.append(name)
            try:
                body = self.program(entry["body"], f"方法 {name} 的 body")
            finally:
                self.expanding.pop()
            if body.input_types != inputs:
                raise ValueError(f"方法 {name} 的方法体输入类型与注册接口不同")
            if body.output_type != outputs[0]:
                raise ValueError(f"方法 {name} 的方法体输出类型与注册接口不同")
            result = _Method(inputs, outputs[0], body=body)
            self.module_cache[name] = _CachedModule(self.registry, entry, result)

        self.methods[name] = result
        return result

    def program(self, program: Any, context: str = "program") -> _Plan:
        if not isinstance(program, Mapping):
            raise ValueError(f"{context} 必须是程序映射")
        input_types = _types(program.get("input_types"), f"{context}.input_types")
        nodes = program.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError(f"{context}.nodes 必须是节点列表")
        n_inputs = len(input_types)
        reference_types = list(input_types)
        checked: list[tuple[_Method, tuple[int, ...]]] = []

        # 先校验全部节点，包含不通向最终输出的节点。否则非法类型或循环
        # 可能因暂时没有被使用而藏在候选图或复合方法中。
        for index, node in enumerate(nodes):
            location = f"{context}.nodes[{index}]"
            if not isinstance(node, Mapping):
                raise ValueError(f"{location} 必须是节点映射")
            method = self.method(node.get("method"))
            raw_inputs = node.get("inputs")
            if not isinstance(raw_inputs, list):
                raise ValueError(f"{location}.inputs 必须是引用列表")
            if len(raw_inputs) != len(method.input_types):
                raise ValueError(f"{location} 的输入数与方法声明不符")
            refs = tuple(
                _reference(ref, n_inputs + index, f"{location}.inputs[{slot}]")
                for slot, ref in enumerate(raw_inputs)
            )
            for slot, (ref, expected) in enumerate(zip(refs, method.input_types)):
                if reference_types[ref] != expected:
                    raise ValueError(
                        f"{location}.inputs[{slot}] 类型不匹配："
                        f"需要 {expected!r}，实际为 {reference_types[ref]!r}"
                    )
            # 保留重复引用及其参数位置，不能用集合替代多重输入边。
            checked.append((method, refs))
            reference_types.append(method.output_type)

        output = _reference(program.get("output"), len(reference_types), f"{context}.output")

        # 只有最终输出的传递依赖参与执行。节点只能向前引用，因此之后按
        # 原列表顺序生成指令就已经满足拓扑顺序，无须每次求值重新排序。
        active: set[int] = set()
        pending = [output]
        while pending:
            ref = pending.pop()
            if ref < n_inputs:
                continue
            index = ref - n_inputs
            if index not in active:
                active.add(index)
                pending.extend(checked[index][1])

        instructions: list[tuple[Callable[..., Any], tuple[int, ...]]] = []
        slots = {index: index for index in range(n_inputs)}
        for index, (method, refs) in enumerate(checked):
            if index not in active:
                continue
            arguments = tuple(slots[ref] for ref in refs)
            if method.function is not None:
                instructions.append((method.function, arguments))
                slots[n_inputs + index] = n_inputs + len(instructions) - 1
                continue

            # 对本次模块调用建立独立的槽位映射。内部节点只展开一次，多个
            # 后继共享同一槽；不同调用各自计算，避免错误地合并两次调用。
            body = method.body
            assert body is not None
            inner_slots = dict(enumerate(arguments))
            for inner_index, (function, inner_refs) in enumerate(body.instructions):
                instructions.append((function, tuple(inner_slots[ref] for ref in inner_refs)))
                inner_slots[len(body.input_types) + inner_index] = n_inputs + len(instructions) - 1
            # 方法体可以直接返回输入，这种别名输出不需要额外执行节点。
            slots[n_inputs + index] = inner_slots[body.output_slot]

        return _Plan(input_types, reference_types[output], tuple(instructions), slots[output])


def validate_program(program: dict, registry: dict) -> None:
    """校验端口类型、前向引用、复合接口及递归循环；不调用任何原语。

    所有出现于图中的方法都会校验，包括未激活节点及其方法体。注册表中
    完全未被程序引用的方法不参与本程序的校验。非法声明抛出 ValueError。
    """
    _Compiler(registry).program(program)


def make_evaluator(
    program: dict, registry: dict, *, module_cache: dict | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """编译一次，返回接收二维数值批次并产生一维输出的求值函数。

    编译过程不修改程序或注册表。求值函数绑定编译时的原语函数对象；若
    调用方替换方法实现或结构，应重新调用本函数，避免隐式改变已编译程序。
    每个样本独立执行，同一个样本的每个激活原语只接收 Python float，
    不把整个数组交给原语，因此没有通过向量化改变对照的计算粒度。

    可选 module_cache 用于同阶段跨候选复用模块编译结果。只有注册表的
    既有条目及其依赖保持不变时才能复用；新增方法不影响原缓存。更改
    既有原语、模块体或类型声明后，调用方必须清空缓存或提供新字典。
    不传缓存时，每次编译均重新校验全部被引用的方法声明。
    """
    if module_cache is not None and not isinstance(module_cache, dict):
        raise ValueError("module_cache 必须是字典或 None")
    plan = _Compiler(registry, module_cache=module_cache).program(program)
    n_inputs = len(plan.input_types)
    instructions, output_slot = plan.instructions, plan.output_slot

    def evaluate(values: np.ndarray) -> np.ndarray:
        batch = np.asarray(values, dtype=float)
        if batch.ndim != 2 or batch.shape[1] != n_inputs:
            raise ValueError(f"输入必须是形状为 (样本数, {n_inputs}) 的二维数组")
        result = np.empty(batch.shape[0], dtype=float)
        for row_index, row in enumerate(batch):
            storage = [float(value) for value in row]
            for function, refs in instructions:
                # 常用的一元、二元方法直接传参，减少参考后端的解释器开销；
                # 其余元数仍遵循相同的按位置调用语义，重复参数也原样保留。
                if len(refs) == 1:
                    value = function(storage[refs[0]])
                elif len(refs) == 2:
                    value = function(storage[refs[0]], storage[refs[1]])
                else:
                    value = function(*(storage[ref] for ref in refs))
                storage.append(float(value))
            result[row_index] = storage[output_slot]
        return result

    return evaluate
