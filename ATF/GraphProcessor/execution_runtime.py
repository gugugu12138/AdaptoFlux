"""按独立的副作用执行计划运行数据图。

数据图仍负责传递方法参数；控制依赖只表示“前驱完成后才允许执行”。二者
不混在一起，因此已有隐式状态函数不需要额外接收令牌参数，零输出动作也能
通过调度器发出完成信号。每次调用都会重新创建就绪计数和完成状态，模型只
保留程序的语义顺序，不会误用上一个样本或上一次调用的完成信号。

这里的保证以方法正确声明副作用为前提；不会分析任意 Python 函数体来证明
它真的纯净。共享外部对象必须用同一个资源名声明，未知方法走保守排序。
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
from functools import wraps
from threading import RLock

import numpy as np

from .execution_plan import UID_KEY, compile_execution_plan, set_execution_order


def is_passthrough(data):
    """兼容旧模型的透传节点；透传只搬运引用，不调用用户方法。"""
    if "is_passthrough" in data:
        return bool(data["is_passthrough"])
    name = data.get("method_name")
    return name is None or str(name).lower() in {"null", "unmatched"}


def prepare_graph(graph, methods):
    """执行前检查数据端口，并把旧格式隐含的参数位置显式固定下来。

    旧图可能没有 input_slot，原执行器依赖入边的插入顺序绑定参数。仅固定
    节点调度顺序仍不够：保存重载后，减法等非交换算子的两个输入可能反转。
    因此首次准备时把原有绑定写成 input_slot；混合格式、越界输出、重复的
    聚合坐标都会在任何用户函数执行之前被拒绝。
    """
    if graph is None:
        raise ValueError("No graph has been configured")
    bindings = []
    for node, data in graph.nodes(data=True):
        if node in {"root", "collapse"}:
            continue
        incoming = list(graph.in_edges(node, keys=True, data=True))
        if is_passthrough(data):
            expected = 1
        else:
            name = data.get("method_name")
            if name not in methods:
                raise ValueError(f"Node {node!r} has unknown method {name!r}")
            expected = methods[name]["input_count"]
        if len(incoming) != expected:
            raise ValueError(f"Node {node!r} expects {expected} data inputs, got {len(incoming)}")
        slots = [edge[3].get("input_slot") for edge in incoming]
        if slots and all(slot is None for slot in slots):
            bindings.extend((edge[3], slot, "input_slot" in edge[3]) for slot, edge in enumerate(incoming))
        elif any(slot is None for slot in slots):
            raise ValueError(f"Node {node!r}: mixed use of input_slot")
        elif any(type(slot) is not int for slot in slots) or sorted(slots) != list(range(expected)):
            raise ValueError(f"Node {node!r} has inconsistent input_slot indices: {slots}")
    for source, target, data in graph.edges(data=True):
        index = data.get("output_index")
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError(f"Invalid output_index on {source!r} -> {target!r}")
        if source not in {"root", "collapse"} and index is not None:
            info = graph.nodes[source]
            count = 1 if is_passthrough(info) else methods[info["method_name"]]["output_count"]
            if index >= count:
                raise ValueError(f"Output index {index} exceeds {source!r}'s {count} outputs")
    coords = []
    for source, _, data in graph.in_edges("collapse", data=True):
        coord = data.get("data_coord")
        if data.get("output_index") is None or type(coord) is not int or coord < 0:
            raise ValueError(f"Edge from {source!r} to collapse needs output_index and integer data_coord")
        coords.append(coord)
    if len(coords) != len(set(coords)):
        raise ValueError("Collapse data_coord values must be unique to fix aggregation order")
    for data, slot, _ in bindings:
        data["input_slot"] = slot
    try:
        # 导出的计划必须包含已经冻结的端口，避免侧车计划与 graph.json 不一致。
        return compile_execution_plan(graph, methods)
    except Exception:
        # 顺序与新数据依赖冲突时撤销本次端口补全，不留下半完成的准备工作。
        for data, _, had_slot in bindings:
            if had_slot:
                data["input_slot"] = None
            else:
                data.pop("input_slot", None)
        raise


def transactional_graph_edit(operation):
    """让结构修改成为事务：验证失败就恢复原图、层号与丢弃节点配置。

    必须保留原 NetworkX 对象，而不仅给 self.graph 换一个引用，因为
    PathGenerator 等组件还持有该数据图。这个回滚只撤销图修改，不会撤销
    已执行用户函数的外部状态；本装饰器包裹的建图操作不应执行用户动作。
    """
    @wraps(operation)
    def edit(self, *args, **kwargs):
        with self._execution_lock:
            self.prepare_execution_plan()
            original = self.graph
            snapshot = copy.deepcopy(original)
            layer, discard = self.layer, self.discard_node_method_name
            try:
                result = operation(self, *args, **kwargs)
                self.prepare_execution_plan()
                return result
            except Exception:
                original.clear()
                original.graph.update(snapshot.graph)
                original.add_nodes_from(snapshot.nodes(data=True))
                original.add_edges_from(snapshot.edges(keys=True, data=True))
                self.graph = original
                self.layer, self.discard_node_method_name = layer, discard
                raise
    return edit


class ExecutionRuntime:
    """串行与线程执行共用一套绑定、节点调用、输出聚合语义。

    共用实现避免两条路径分别维护端口规则而产生差异。GraphProcessor 提供
    图、方法池与聚合器，本类只负责计划准备、调用隔离和依赖调度。
    """

    def __getstate__(self):
        # 锁是进程内同步对象，不应被 deepcopy/pickle 复制；候选模型应有自己的锁。
        state = self.__dict__.copy()
        state.pop("_execution_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._execution_lock = RLock()

    def prepare_execution_plan(self):
        """在内置评分或自定义评分前冻结/校验候选，供训练器显式调用。

        推理入口也会调用它，防止用户直接修改 graph 后使用过期依赖。当前
        以每次重检换取正确性；以后若加缓存，必须同时追踪图和效果声明版本。
        """
        with self._execution_lock:
            return prepare_graph(self.graph, self.methods)

    def set_execution_order(self, order):
        """显式修改程序的内部节点顺序，调用方必须重新评价任务表现。

        这不是运行时调度器随意选择顺序的接口。对相互冲突的状态访问重新
        排序会改变程序语义；只有作为经过评价的新候选才能在训练中被接受。
        """
        with self._execution_lock:
            previous = copy.deepcopy(self.graph.graph)
            previous_uids = {node: data.get(UID_KEY) for node, data in self.graph.nodes(data=True)}
            try:
                set_execution_order(self.graph, order)
                return self.prepare_execution_plan()
            except Exception:
                self.graph.graph.clear()
                self.graph.graph.update(previous)
                for node, uid in previous_uids.items():
                    if uid is None:
                        self.graph.nodes[node].pop(UID_KEY, None)
                    else:
                        self.graph.nodes[node][UID_KEY] = uid
                raise

    @staticmethod
    def _input_samples(values):
        """维持原二维输入接口；对象和数组作为单个特征时保留其引用。"""
        if isinstance(values, np.ndarray):
            if values.ndim != 2:
                raise ValueError("Input values must be a 2D array")
            return [list(row) for row in values]
        if isinstance(values, list) and all(isinstance(row, (list, tuple)) for row in values):
            return [list(row) for row in values]
        raise TypeError("values must be a 2D numpy array or list of lists")

    def _node_inputs(self, node, outputs):
        # 参数只读取数据边。计划的 predecessors 还包含控制前驱，不能拿它来传参。
        incoming = sorted(self.graph.in_edges(node, data=True), key=lambda e: e[2]["input_slot"])
        columns = []
        for source, _, data in incoming:
            index = data.get("output_index")
            columns.append([row if index is None else row[index] for row in outputs[source]])
        return columns

    def _call_node(self, node, columns, sample_count):
        """调用一个已经就绪的节点，统一检查逐样本/向量化输出数量。"""
        data = self.graph.nodes[node]
        if is_passthrough(data):
            return [[value] for value in columns[0]]
        name = data["method_name"]
        info = self.methods[name]
        function, output_count = info["function"], info["output_count"]
        if info.get("vectorized", False) and sample_count > 1:
            # 仅纯函数整批执行会走这里。有副作用时外层按完整样本逐个运行，
            # sample_count 恒为 1，因此不会因向量化失败而重复执行一次副作用。
            try:
                batched = []
                for column in columns:
                    if all(isinstance(v, (int, float, np.number)) for v in column):
                        batched.append(np.asarray(column))
                    elif all(isinstance(v, np.ndarray) for v in column):
                        batched.append(np.stack(column, axis=0))
                    else:
                        raise ValueError("Inputs cannot be stacked")
                result = function(*batched)
                if isinstance(result, tuple):
                    if len(result) != output_count or any(len(v) != sample_count for v in result):
                        raise ValueError("Invalid vectorized output shape")
                    return [[result[j][i] for j in range(output_count)] for i in range(sample_count)]
                result = np.asarray(result)
                if result.ndim == 1 and output_count == 1 and len(result) == sample_count:
                    return [[v] for v in result]
                if result.shape == (sample_count, output_count):
                    return result.tolist()
            except Exception:
                # 纯函数的向量化失败可以安全回退；纯净声明错误不属于可保证范围。
                pass
        results = []
        for index in range(sample_count):
            try:
                result = function(*(column[index] for column in columns))
            except Exception as error:
                raise RuntimeError(f"Error in method {name!r} at sample {index}: {error}") from error
            if output_count == 0 and result is None:
                result = []
            elif not isinstance(result, (list, tuple)):
                result = [result]
            if len(result) != output_count:
                raise ValueError(f"Method {name!r} returned {len(result)} outputs; expected {output_count}")
            results.append(list(result))
        return results

    def _collapse_results(self, outputs, sample_count):
        """全部运算与动作完成后，按固定 data_coord 顺序执行聚合。"""
        edges = sorted(self.graph.in_edges("collapse", data=True), key=lambda e: e[2]["data_coord"])
        if not edges:
            # 没有数值输出的动作图也必须把动作执行完，返回每个样本的空特征行。
            return [[] for _ in range(sample_count)]
        columns = [[row[data["output_index"]] for row in outputs[source]] for source, _, data in edges]
        return [self.collapse_manager.collapse(list(row)) for row in zip(*columns)]

    def _execute_samples(self, samples, plan, workers):
        """执行一次完整图调用；线程完成顺序可以变，语义依赖不能被越过。"""
        outputs = {"root": samples}
        if workers is None or workers == 1:
            for node in plan.order:
                outputs[node] = self._call_node(node, self._node_inputs(node, outputs), len(samples))
        else:
            # 数据前驱和控制前驱在计划中已去重；同一前驱完成只扣减一次。
            pending = {node: len(plan.predecessors[node]) for node in plan.order}
            ready = [node for node in plan.order if pending[node] == 0]
            rank = {node: i for i, node in enumerate(plan.order)}
            futures = {}
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                try:
                    while ready or futures:
                        ready.sort(key=rank.__getitem__, reverse=True)
                        while ready and len(futures) < workers:
                            node = ready.pop()
                            # 主线程先完成参数绑定，再提交工作线程。只有主线程写
                            # outputs/就绪计数，用户函数只接触自己的参数与声明资源。
                            columns = self._node_inputs(node, outputs)
                            futures[pool.submit(self._call_node, node, columns, len(samples))] = node
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        # 先检查本批所有已完成任务是否失败，再放行后继。原来的
                        # 队列轮询会在某前驱异常、后继永远不就绪时一直等待。
                        results = [(future, future.result()) for future in done]
                        for future, result in results:
                            node = futures.pop(future)
                            outputs[node] = result
                            completed += 1
                            for successor in plan.successors[node]:
                                pending[successor] -= 1
                                if pending[successor] == 0:
                                    ready.append(successor)
                    if completed != len(plan.order):
                        raise RuntimeError("Execution plan has unsatisfied dependencies")
                except BaseException:
                    # 不再提交后继，取消尚未启动任务；已运行的外部副作用不能回滚。
                    for future in futures:
                        future.cancel()
                    raise
        return self._collapse_results(outputs, len(samples))

    def _infer_with_execution_plan(self, values, workers=None):
        """推理公共入口：隔离同一实例的调用，并统一串行/并行的批量顺序。"""
        if workers is not None and (type(workers) is not int or workers < 1):
            raise ValueError("num_workers must be a positive integer")
        # 同一处理器实例一次只执行一个调用，防止两个 tick 混用外部状态。
        # 不同处理器实例若引用同一个外部环境，仍需调用方隔离环境或加共享锁。
        with self._execution_lock:
            samples = self._input_samples(values)
            plan = self.prepare_execution_plan()
            if not samples:
                return np.array([])
            # root 的端口数由本次输入决定，而不是方法注册决定。必须先检查整批
            # 样本，否则可能前一个动作已经写状态，后一个节点才发现输入越界。
            indices = [data.get("output_index") for _, _, data in self.graph.out_edges("root", data=True)]
            required = max((index for index in indices if index is not None), default=-1) + 1
            for index, sample in enumerate(samples):
                if len(sample) < required:
                    raise ValueError(f"Input sample {index} has {len(sample)} features; graph needs at least {required}")
            if plan.has_effects:
                # 全局/外部状态通常被多个样本共享。按“完整样本1→完整样本2”
                # 执行，避免原先按节点批处理形成“A1,A2,B1,B2”的交错语义。
                results = []
                for sample in samples:
                    results.extend(self._execute_samples([sample], plan, workers))
            else:
                results = self._execute_samples(samples, plan, workers)
            return np.array(results)
