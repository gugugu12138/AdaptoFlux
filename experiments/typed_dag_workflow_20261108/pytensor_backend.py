"""用 PyTensor 原生计算图和 OpFromGraph 接入共同的子图复用流程。

这里仅负责把实验声明式程序翻译成 PyTensor 变量与算子。候选生成、
任务成功判据和模块准入由共同实验驱动器提供；它们不是 PyTensor 的
原生学习算法，也不是本适配器独立实现的一套图解释器。

重要版本细节：PyTensor 3.3.8 忽略传给 OpFromGraph 的编译 mode 参数。
因此只在外层 function 上明确设置 Python linker 和无优化模式，模块
使用原生的非内联、惰性链接路径，而不声称强行替换其内部链接器。
"""
from __future__ import annotations

# bootstrap 在第三方包首次导入之前配置临时依赖和编译缓存位置。
import bootstrap  # noqa: F401

from dataclasses import dataclass

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import Mode
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.graph.replace import clone_replace
from pytensor.graph.type import Type


def _mode():
    """每次创建明确模式，不依赖机器上的默认优化和 C 编译器设置。"""
    return Mode(linker="py", optimizer=None)


def _function(inputs, output):
    """建立真实 PyTensor 可调用对象；保留合法但未使用的程序输入。"""
    return pytensor.function(
        inputs,
        output,
        mode=_mode(),
        on_unused_input="ignore",
        allow_input_downcast=False,
    )


class PyTensorBackend:
    """按只追加的名称表组织原语和已封装模块，保持旧模块定义稳定。"""

    name = "pytensor"
    execution_mode = {
        "outer_linker": "py",
        "outer_optimizer": None,
        "module": "OpFromGraph",
        "inline": False,
        "inner_linking": "Native PyTensor 3.3.8 lazy inner-function linking; no ignored compile_kwargs supplied.",
        "runtime_version": pytensor.__version__,
    }

    def __init__(self):
        # 原语直接创建原生张量表达式，不把整张候选图藏进 Python 回调。
        # 浮点常数均显式指定 float64，避免用户 floatX 配置影响类型。
        self.methods = {
            "increment": lambda x: x + np.float64(1),
            "add_3": lambda x: x + np.float64(3),
            "multiply_by_2": lambda x: x * np.float64(2),
            "multiply_by_05": lambda x: x * np.float64(0.5),
            "identity": lambda x: x,
            "add": lambda x, y: x + y,
        }
        self.arities = {name: 1 for name in self.methods}
        self.arities["add"] = 2

    def _graph(self, program):
        """只翻译一次每个声明节点；多条边引用同一 Variable 保留共享。"""
        if any(name != "scalar" for name in program["input_types"]):
            raise ValueError("The common numeric workflow declares scalar input ports only")
        inputs = [pt.dscalar(f"input_{i}") for i in range(len(program["input_types"]))]
        values = list(inputs)
        for index, node in enumerate(program["nodes"]):
            refs = node["inputs"]
            # 这是对共同文件格式的边界检查，不代替 PyTensor 的类型检查。
            # 负数索引在 Python 中合法，但在本实验的前向引用格式中不合法。
            if any(type(ref) is not int or not 0 <= ref < len(values) for ref in refs):
                raise ValueError("Candidate graph contains an invalid forward reference")
            name = node["method"]
            if len(refs) != self.arities[name]:
                raise ValueError("Candidate method arity does not match its registered interface")
            value = self.methods[name](*(values[ref] for ref in refs))
            value.name = f"node_{index}"
            values.append(value)
        output = program["output"]
        if type(output) is not int or not 0 <= output < len(values):
            raise ValueError("Candidate graph contains an invalid output reference")
        return inputs, values[output]

    def register(self, name, program):
        """把当前程序封装成真实 OpFromGraph；已注册模块可以继续被嵌套。"""
        if name in self.methods:
            raise ValueError("Append-only method registry cannot overwrite a name")
        inputs, output = self._graph(program)
        module = OpFromGraph(inputs, [output], inline=False, name=name)
        self.methods[name] = module
        self.arities[name] = len(inputs)

    def compile(self, program):
        """将共同程序编译成样本矩阵接口，底层逐样本调用标量计算图。"""
        inputs, output = self._graph(program)
        function = _function(inputs, output)
        width = len(inputs)

        def evaluate(samples):
            values = np.asarray(samples, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != width:
                raise ValueError("Samples must be a matrix matching the input arity")
            return np.asarray([float(function(*row)) for row in values], dtype=np.float64)

        return evaluate

    def probes(self):
        """便于驱动器使用实例接口；探针本身不使用共同数值图适配器。"""
        return probes()


def _capture(action):
    """失败只记录异常类别，避免把本机路径或 traceback 写入匿名材料。"""
    try:
        result = action()
        return {"success": True, **result}
    except Exception as error:
        return {"success": False, "error_class": type(error).__name__}


def _scalar_module_probe():
    x = pt.dscalar("scalar_input")
    module = OpFromGraph([x], [2 * (x + 1)], inline=False, name="scalar_module_probe")
    function = _function([x], module(x))
    observed = float(function(np.float64(3)))
    assert observed == 8
    return {"observed": observed, "input_dtype": x.dtype, "input_rank": x.ndim,
            "native_module_class": type(module).__name__}


def _vector_module_probe():
    # 不经过共同的 scalar 声明；直接测原生模块能否保留向量端口和输出。
    x = pt.dvector("vector_input")
    module = OpFromGraph([x], [x * np.float64(2)], inline=False, name="vector_module_probe")
    function = _function([x], module(x))
    expected = np.asarray([2, 4, 6], dtype=np.float64)
    observed = np.asarray(function(np.asarray([1, 2, 3], dtype=np.float64)))
    np.testing.assert_array_equal(observed, expected)
    return {"observed": observed.tolist(), "output_rank": observed.ndim,
            "output_dtype": str(observed.dtype), "native_module_class": type(module).__name__}


def _rank_rejection_probe():
    scalar = pt.dscalar("scalar_input")
    vector = pt.dvector("wrong_rank_input")
    module = OpFromGraph([scalar], [scalar + 1], inline=False, name="scalar_rank_contract")
    try:
        module(vector)
    except TypeError as error:
        return {"observed": "rejected_before_compilation", "error_class": type(error).__name__,
                "expected_rank": 0, "supplied_rank": 1}
    raise AssertionError("Native scalar module unexpectedly accepted a vector input")


class _CountedIdentity(Op):
    """仅用于观察执行次数的探针 Op；不作为学习原语或现实效果安全声明。"""

    def __init__(self):
        self.calls = 0

    def make_node(self, x):
        x = pt.dscalar().type.filter_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        self.calls += 1
        outputs[0][0] = np.asarray(inputs[0], dtype=np.float64).copy()


def _shared_node_probe():
    x = pt.dscalar("shared_input")
    counted = _CountedIdentity()
    intermediate = counted(x)
    output = intermediate + intermediate
    function = _function([x], output)
    # 若编译阶段改变计数，单独清零；本探针测执行次数，不混入编译过程。
    counted.calls = 0
    observed = [float(function(np.float64(i))) for i in range(5)]
    assert observed == [0, 2, 4, 6, 8]
    assert counted.calls == 5
    assert output.owner.inputs[0] is output.owner.inputs[1]
    return {"observed": observed, "samples": 5, "primitive_calls": counted.calls,
            "two_edges_reference_one_variable": True}


def _local_replace_probe():
    x = pt.dscalar("replace_input")
    intermediate = x + np.float64(1)
    original = intermediate * np.float64(2)
    replacement = x + np.float64(3)
    changed = clone_replace(original, replace={intermediate: replacement})
    original_function = _function([x], original)
    changed_function = _function([x], changed)
    values = [np.float64(-1), np.float64(0), np.float64(2)]
    old_before = [float(original_function(v)) for v in values]
    new_values = [float(changed_function(v)) for v in values]
    old_after = [float(original_function(v)) for v in values]
    assert old_before == old_after == [0, 2, 6]
    assert new_values == [4, 6, 10]
    assert original.owner.inputs[0] is intermediate
    return {"original_before": old_before, "replaced": new_values,
            "original_after": old_after, "original_graph_unchanged": True}


@dataclass(frozen=True)
class _Observation:
    """本地不可变输入对象，避免把可变别名或隐式副作用混入该类型探针。"""

    position: float
    target: float


class _ObservationType(Type):
    """用 PyTensor 正式扩展接口声明 Python 对象的名义类型及值约束。"""

    __props__ = ()

    def filter(self, data, strict=False, allow_downcast=None):
        if not isinstance(data, _Observation):
            raise TypeError("Observation input requires the declared Observation class")
        return data

    def __str__(self):
        return "WorkflowObservation"


class _ObservationDistance(Op):
    """自定义 Op 将异质对象接到数值计算图，逻辑在 perform 中显式定义。"""

    __props__ = ()

    def make_node(self, observation):
        observation = _ObservationType().filter_variable(observation)
        return Apply(self, [observation], [pt.dscalar().type()])

    def perform(self, node, inputs, outputs):
        observation = inputs[0]
        outputs[0][0] = np.asarray(observation.target - observation.position, dtype=np.float64)


def _python_object_probe():
    observation = _ObservationType()("observation_input")
    distance = _ObservationDistance()(observation)
    function = _function([observation], distance)
    observed = float(function(_Observation(position=2, target=7)))
    assert observed == 5
    try:
        function({"position": 2, "target": 7})
    except TypeError as error:
        rejection = type(error).__name__
    else:
        raise AssertionError("Custom Type unexpectedly accepted an undeclared Python class")
    return {"observed": observed, "wrong_python_class_rejected": True,
            "error_class": rejection, "extension_classes": ["Type", "Op"],
            "core_source_modified": False}


def probes():
    """记录原生接口的成功或失败；不把探针失败隐藏成‘未测试’。"""
    return {
        "scalar_module": _capture(_scalar_module_probe),
        "vector_input_output_module": _capture(_vector_module_probe),
        "rank_incompatible_input_rejected": _capture(_rank_rejection_probe),
        "shared_intermediate_executed_once": _capture(_shared_node_probe),
        "local_replace_preserves_original": _capture(_local_replace_probe),
        "custom_python_object_type": _capture(_python_object_probe),
    }
