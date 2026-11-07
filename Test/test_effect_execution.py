"""串行/线程执行的可观察行为回归测试。

重点检查非交换状态写入、读写先后、存储重排、批量样本顺序和异常退出。
并发测试使用 Barrier 验证互不冲突分支确实可以重叠执行，避免实现靠把
所有程序无条件串行化来“通过”一致性测试。
"""
import copy
import random
import threading
import time
import unittest

import networkx as nx
import numpy as np

from ATF.GraphProcessor.graph_processor import GraphProcessor
from ATF.GraphProcessor.execution_plan import normalize_effects


def method(fn, inputs=1, outputs=1, **effects):
    info = dict(function=fn, input_count=inputs, output_count=outputs, vectorized=False)
    if effects:
        info["effects"] = normalize_effects(**effects)
    return info


def parallel_nodes(names):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(["root", "collapse"])
    for name in names:
        graph.add_node(name, method_name=name, layer=1, is_passthrough=False)
        graph.add_edge("root", name, output_index=0, input_slot=0)
    return graph


class EffectExecutionTests(unittest.TestCase):
    def state_program(self, declared=True):
        state, events = {"value": 1}, []

        def add(dummy):
            time.sleep(.001)
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

        writes = dict(writes=["environment"]) if declared else {}
        reads = dict(reads=["environment"]) if declared else {}
        methods = {"add": method(add, **writes), "double": method(double, **writes),
                   "read": method(read, **reads)}
        graph = parallel_nodes(methods)
        graph.add_edge("read", "collapse", output_index=0, data_coord=0)
        return GraphProcessor(graph, methods), state, events

    def test_conflicting_reads_and_writes_match_under_parallel_execution(self):
        for declared in [False, True]:
            gp, state, events = self.state_program(declared)
            original_edges = gp.graph.number_of_edges()
            for workers in [None, 1, 2, 4]:
                with self.subTest(declared=declared, workers=workers):
                    state["value"] = 1
                    events.clear()
                    output = gp.infer_with_graph([[0]]) if workers is None else gp.infer_with_task_parallel([[0]], workers)
                    self.assertEqual(output.tolist(), [4])
                    self.assertEqual(events, ["add", "double", "read"])
            self.assertEqual(original_edges, gp.graph.number_of_edges())
            self.assertTrue(gp.prepare_execution_plan().control_edges)

    def test_frozen_order_survives_node_and_edge_reinsertion(self):
        gp, state, events = self.state_program()
        gp.prepare_execution_plan()
        for seed in range(12):
            graph = nx.MultiDiGraph(**copy.deepcopy(gp.graph.graph))
            nodes, edges = list(gp.graph.nodes(data=True)), list(gp.graph.edges(keys=True, data=True))
            rng = random.Random(seed)
            rng.shuffle(nodes)
            rng.shuffle(edges)
            graph.add_nodes_from(copy.deepcopy(nodes))
            graph.add_edges_from(copy.deepcopy(edges))
            shuffled = GraphProcessor(graph, gp.methods)
            state["value"] = 1
            events.clear()
            self.assertEqual(shuffled.infer_with_task_parallel([[0]], 3).tolist(), [4])
            self.assertEqual(events, ["add", "double", "read"])

    def test_effectful_batch_runs_complete_program_per_sample(self):
        gp, state, events = self.state_program()
        for parallel in [False, True]:
            state["value"] = 1
            events.clear()
            output = gp.infer_with_task_parallel([[0], [0]], 3) if parallel else gp.infer_with_graph([[0], [0]])
            self.assertEqual(output.tolist(), [4, 10])
            self.assertEqual(events, ["add", "double", "read"] * 2)

    def test_independent_resources_can_overlap(self):
        barrier = threading.Barrier(2, timeout=2)

        def work(x):
            barrier.wait()
            return x + 1

        methods = {"left": method(work, writes=["left"]), "right": method(work, writes=["right"])}
        graph = parallel_nodes(methods)
        for coord, name in enumerate(methods):
            graph.add_edge(name, "collapse", output_index=0, data_coord=coord)
        gp = GraphProcessor(graph, methods)
        self.assertEqual(gp.prepare_execution_plan().control_edges, ())
        self.assertEqual(gp.infer_with_task_parallel([[2]], 2).tolist(), [6])

    def test_pure_branches_remain_parallel(self):
        barrier = threading.Barrier(2, timeout=2)

        def work(x):
            barrier.wait()
            return x

        methods = {name: method(work, pure=True) for name in ["left", "right"]}
        graph = parallel_nodes(methods)
        for coord, name in enumerate(methods):
            graph.add_edge(name, "collapse", output_index=0, data_coord=coord)
        self.assertEqual(GraphProcessor(graph, methods).infer_with_task_parallel([[3]], 2).tolist(), [6])

    def test_zero_output_actions_and_zero_input_methods(self):
        events = []
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(["root", "collapse"])
        graph.add_node("sink", method_name="sink", layer=1, is_passthrough=False)
        methods = {"sink": method(lambda: events.append("action"), inputs=0, outputs=0, writes=["env"])}
        gp = GraphProcessor(graph, methods)
        self.assertEqual(gp.infer_with_task_parallel([[]], 2).shape, (1, 0))
        self.assertEqual(events, ["action"])

    def test_worker_failure_does_not_run_dependents_or_hang(self):
        events = []

        def fail(x):
            raise ValueError("deliberate failure")

        graph = parallel_nodes(["fail"])
        graph.add_node("after", method_name="after", layer=2, is_passthrough=False)
        graph.add_edge("fail", "after", output_index=0, input_slot=0)
        graph.add_edge("after", "collapse", output_index=0, data_coord=0)
        gp = GraphProcessor(graph, {"fail": method(fail), "after": method(lambda x: events.append(x) or x)})
        with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
            gp.infer_with_task_parallel([[2]], 2)
        self.assertEqual(events, [])

    def test_legacy_argument_order_is_frozen_before_reinsertion(self):
        graph = parallel_nodes([])
        graph.add_node("subtract", method_name="subtract", layer=1, is_passthrough=False)
        graph.add_edge("root", "subtract", output_index=1)
        graph.add_edge("root", "subtract", output_index=0)
        graph.add_edge("subtract", "collapse", output_index=0, data_coord=0)
        gp = GraphProcessor(graph, {"subtract": method(lambda a, b: a - b, inputs=2, pure=True)})
        self.assertEqual(gp.infer_with_graph([[2, 7]]).tolist(), [5])
        edges = list(gp.graph.edges(keys=True, data=True))
        gp.graph.remove_edges_from(list(gp.graph.edges(keys=True)))
        gp.graph.add_edges_from(reversed(edges))
        self.assertEqual(gp.infer_with_task_parallel([[2, 7]], 2).tolist(), [5])

    def test_graph_copy_keeps_order_and_has_independent_call_lock(self):
        gp, state, events = self.state_program()
        gp.prepare_execution_plan()
        cloned = copy.deepcopy(gp)
        self.assertIsNot(gp._execution_lock, cloned._execution_lock)
        self.assertEqual(gp.prepare_execution_plan().to_dict(), cloned.prepare_execution_plan().to_dict())

    def test_replacement_keeps_identity_and_rejects_invalid_arity_atomically(self):
        gp, state, events = self.state_program()
        gp.prepare_execution_plan()
        uid = gp.graph.nodes["add"]["_atf_uid"]
        gp.methods["also_add"] = gp.methods["add"].copy()
        replacement = gp.replace_node_method("add", "also_add")
        self.assertEqual(gp.graph.nodes[replacement]["_atf_uid"], uid)
        self.assertEqual(gp.infer_with_task_parallel([[0]], 3).tolist(), [4])
        gp.methods["invalid"] = method(lambda a, b: a + b, inputs=2)
        before = copy.deepcopy(gp.graph)
        with self.assertRaises(ValueError):
            gp.replace_node_method(replacement, "invalid")
        self.assertTrue(nx.utils.graphs_equal(before, gp.graph))

    def test_vectorized_pure_output_and_safe_scalar_fallback(self):
        graph = parallel_nodes(["square"])
        graph.add_edge("square", "collapse", output_index=0, data_coord=0)

        def scalar_only(x):
            if not np.isscalar(x):
                raise TypeError("scalar only")
            return x * x

        for fn in [lambda x: x * x, scalar_only]:
            info = method(fn, pure=True)
            info["vectorized"] = True
            gp = GraphProcessor(graph, {"square": info})
            self.assertEqual(gp.infer_with_task_parallel([[2], [3]], 2).tolist(), [4, 9])

    def test_bad_later_sample_is_rejected_before_any_state_change(self):
        # 整批输入预检必须在第一条动作之前完成，不能执行一半才发现下一行缺列。
        gp, state, events = self.state_program()
        for parallel in [False, True]:
            with self.assertRaisesRegex(ValueError, "Input sample 1"):
                if parallel:
                    gp.infer_with_task_parallel([[0], []], 3)
                else:
                    gp.infer_with_graph([[0], []])
            self.assertEqual(state["value"], 1)
            self.assertEqual(events, [])

    def test_failed_growth_preserves_the_accepted_graph(self):
        gp, state, events = self.state_program()
        gp.prepare_execution_plan()
        before = copy.deepcopy(gp.graph)
        # 无效坐标会在append内部已经递增layer之后失败，事务必须回滚层号与图。
        layer = gp.layer
        with self.assertRaises((ValueError, KeyError)):
            gp.append_nx_layer({"valid_groups": {"add": [[999]]}, "unmatched_features": []})
        self.assertTrue(nx.utils.graphs_equal(before, gp.graph))
        self.assertEqual(gp.layer, layer)


if __name__ == "__main__":
    unittest.main()
