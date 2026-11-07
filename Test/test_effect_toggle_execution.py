"""副作用排序开关的运行行为测试。

开关关闭后只解除额外的状态访问顺序约束，数据依赖、参数校验和异常传播
仍然有效。并发用 Barrier 证明两个工作线程确实同时进入方法，不依靠偶然的
线程完成次序判断是否关闭；有状态批量则通过完整事件序列检查语义差异。
"""
import copy
import threading
import unittest

import networkx as nx

from ATF.GraphProcessor.graph_processor import GraphProcessor
from ATF.GraphProcessor.execution_plan import normalize_effects


def method(function, **effects):
    """未提供效果字段时保持旧注册表形态，验证 unknown 也能显式关闭排序。"""
    info = dict(function=function, input_count=1, output_count=1, vectorized=False)
    if effects:
        info["effects"] = normalize_effects(**effects)
    return info


def independent_graph(names, output=None):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(["root", "collapse"])
    for name in names:
        graph.add_node(name, method_name=name, layer=1, is_passthrough=False)
        graph.add_edge("root", name, output_index=0, input_slot=0)
    if output is not None:
        graph.add_edge(output, "collapse", output_index=0, data_coord=0)
    return graph


class EffectToggleExecutionTests(unittest.TestCase):
    def state_program(self, enabled=True):
        state, events = {"value": 1}, []

        def add(dummy):
            state["value"] += 1
            events.append("add")
            return state["value"]

        def double(dummy):
            state["value"] *= 2
            events.append("double")
            return state["value"]

        def read(dummy):
            events.append("read")
            return state["value"]

        methods = {"add": method(add, writes=["env"]),
                   "double": method(double, writes=["env"]),
                   "read": method(read, reads=["env"])}
        graph = independent_graph(methods, output="read")
        return GraphProcessor(graph, methods, enable_effect_ordering=enabled), state, events

    def test_default_enabled_and_toggle_back_preserve_effect_declarations(self):
        gp, state, events = self.state_program()
        declarations = {name: copy.deepcopy(info["effects"]) for name, info in gp.methods.items()}
        self.assertTrue(gp.enable_effect_ordering)
        self.assertTrue(gp.prepare_execution_plan().control_edges)
        for enabled in (False, True, False, True):
            gp.set_effect_ordering(enabled)
            self.assertIs(gp.enable_effect_ordering, enabled)
            self.assertEqual(bool(gp.prepare_execution_plan().control_edges), enabled)
            self.assertEqual(declarations, {name: info["effects"] for name, info in gp.methods.items()})
            if enabled:
                state["value"] = 1
                events.clear()
                self.assertEqual(gp.infer_with_task_parallel([[0]], 3).tolist(), [4])
                self.assertEqual(events, ["add", "double", "read"])

    def test_disabled_unknown_and_shared_resource_methods_actually_overlap(self):
        for declared in (False, True):
            with self.subTest(declared=declared):
                barrier = threading.Barrier(2, timeout=3)

                def work(value):
                    barrier.wait()
                    return value + 1

                effects = {"writes": ["shared"]} if declared else {}
                methods = {name: method(work, **effects) for name in ("left", "right")}
                graph = independent_graph(methods)
                for coord, name in enumerate(methods):
                    graph.add_edge(name, "collapse", output_index=0, data_coord=coord)
                gp = GraphProcessor(graph, methods, enable_effect_ordering=False)
                self.assertEqual(gp.prepare_execution_plan().control_edges, ())
                self.assertEqual(gp.infer_with_task_parallel([[2]], 2).tolist(), [6])
                # 开关只改变调度，不得把 unknown 或声明写入的函数篡改成 pure。
                self.assertEqual("effects" in gp.methods["left"], declared)

    def test_disabled_batch_uses_node_major_order_and_enabled_uses_sample_major(self):
        for enabled, expected, sequence in (
            (True, [4, 10], ["add", "double", "read"] * 2),
            (False, [12, 12], ["add", "add", "double", "double", "read", "read"]),
        ):
            for workers in (None, 1):
                with self.subTest(enabled=enabled, workers=workers):
                    gp, state, events = self.state_program(enabled)
                    # 一个线程用于固定此测试的节点次序；关闭后的多线程共享状态
                    # 本就没有确定性承诺，不能把某种竞争结果写成必须满足的断言。
                    result = (gp.infer_with_graph([[0], [0]]) if workers is None else
                              gp.infer_with_task_parallel([[0], [0]], workers))
                    self.assertEqual(result.tolist(), expected)
                    self.assertEqual(events, sequence)

    def test_disabled_mode_keeps_data_dependencies_and_single_sample_wrappers(self):
        events = []

        def first(value):
            events.append("first")
            return value * 2

        def after(value):
            self.assertEqual(events[-1], "first")
            events.append("after")
            return value + 1

        # 插入顺序故意与依赖顺序相反，保证结果来自数据依赖而不是节点列表。
        graph = independent_graph(["after"])
        graph.remove_edge("root", "after")
        graph.add_node("first", method_name="first", layer=1, is_passthrough=False)
        graph.add_edge("root", "first", output_index=0, input_slot=0)
        graph.add_edge("first", "after", output_index=0, input_slot=0)
        graph.add_edge("after", "collapse", output_index=0, data_coord=0)
        gp = GraphProcessor(graph, {"first": method(first), "after": method(after)},
                            enable_effect_ordering=False)
        for parallel in (False, True):
            events.clear()
            self.assertEqual(gp.infer_with_graph_single([3], use_pipeline=parallel, num_workers=2), 7)
            self.assertEqual(events, ["first", "after"])

    def test_disabled_edit_can_reverse_old_order_and_reenable_freezes_new_program(self):
        methods = {"add": method(lambda x: x + 1, writes=["env"]),
                   "double": method(lambda x: x * 2, writes=["env"])}
        gp = GraphProcessor(independent_graph(methods, output="add"), methods)
        gp.prepare_execution_plan()
        changed = copy.deepcopy(gp.graph)
        changed.remove_edge("root", "add")
        changed.add_edge("double", "add", output_index=0, input_slot=0)
        before = copy.deepcopy(gp.graph)
        with self.assertRaises(ValueError):
            gp.set_graph(changed)
        self.assertTrue(nx.utils.graphs_equal(before, gp.graph))
        gp.set_effect_ordering(False)
        gp.set_graph(changed)
        self.assertEqual(gp.infer_with_task_parallel([[2]], 2).tolist(), [5])
        gp.set_effect_ordering(True)
        self.assertEqual(gp.prepare_execution_plan().order, ("double", "add"))
        self.assertEqual(gp.infer_with_graph([[2]]).tolist(), [5])

    def test_invalid_switch_values_do_not_change_graph_or_mode(self):
        gp, state, events = self.state_program()
        gp.prepare_execution_plan()
        before = copy.deepcopy(gp.graph)
        for value in (None, 0, 1, "false", "true", [], {}):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    gp.set_effect_ordering(value)
                self.assertTrue(gp.enable_effect_ordering)
                self.assertTrue(nx.utils.graphs_equal(before, gp.graph))
                with self.assertRaises((TypeError, ValueError)):
                    GraphProcessor(gp.graph, gp.methods, enable_effect_ordering=value)

    def test_disabled_validation_still_precedes_any_effect(self):
        gp, state, events = self.state_program(False)
        for parallel in (False, True):
            with self.assertRaisesRegex(ValueError, "Input sample 1"):
                if parallel:
                    gp.infer_with_task_parallel([[0], []], 3)
                else:
                    gp.infer_with_graph([[0], []])
            self.assertEqual(events, [])
            self.assertEqual(state["value"], 1)

    def test_deepcopy_retains_disabled_mode_and_can_toggle_independently(self):
        gp, state, events = self.state_program(False)
        cloned = copy.deepcopy(gp)
        self.assertFalse(cloned.enable_effect_ordering)
        self.assertIsNot(gp._execution_lock, cloned._execution_lock)
        cloned.set_effect_ordering(True)
        self.assertFalse(gp.enable_effect_ordering)
        self.assertTrue(cloned.enable_effect_ordering)

    def test_failed_enable_preserves_disabled_configuration(self):
        gp, state, events = self.state_program(False)
        gp.prepare_execution_plan()
        # 模拟调用方直接编辑 NetworkX 图后留下无效端口；开启操作必须先验证，
        # 失败时不能只改了开关却留下另一模式的旧计划或部分更新后的元数据。
        gp.graph.add_edge("root", "add", output_index=0, input_slot=1)
        before = copy.deepcopy(gp.graph)
        with self.assertRaises(ValueError):
            gp.set_effect_ordering(True)
        self.assertFalse(gp.enable_effect_ordering)
        self.assertTrue(nx.utils.graphs_equal(before, gp.graph))
        self.assertEqual(events, [])

    def test_disabled_worker_failure_does_not_release_data_dependents(self):
        events = []

        def fail(value):
            raise ValueError("intentional failure")

        graph = independent_graph(["first"])
        graph.add_node("after", method_name="after", layer=2, is_passthrough=False)
        graph.add_edge("first", "after", output_index=0, input_slot=0)
        graph.add_edge("after", "collapse", output_index=0, data_coord=0)
        methods = {"first": method(fail), "after": method(lambda x: events.append(x) or x)}
        gp = GraphProcessor(graph, methods, enable_effect_ordering=False)
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            gp.infer_with_task_parallel([[1]], 2)
        self.assertEqual(events, [])

    def test_explicit_order_on_enabled_processor_honors_previously_disabled_graph(self):
        gp, state, events = self.state_program(False)
        gp.prepare_execution_plan()
        # 新实例显式选开启，随后立即指定另一种合法程序顺序。此前图文件的
        # False 元数据不能让首次准备丢掉刚刚明确指定的 double→add→read。
        enabled = GraphProcessor(gp.graph, gp.methods, enable_effect_ordering=True)
        enabled.set_execution_order(["double", "add", "read"])
        self.assertEqual(enabled.infer_with_task_parallel([[0]], 3).tolist(), [3])
        self.assertEqual(events, ["double", "add", "read"])


if __name__ == "__main__":
    unittest.main()
