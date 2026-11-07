"""执行计划的行为验证，不读写外部文件。

除了资源声明和保存校验，还枚举一个小图的全部合法调度，检查读取结果与
最终状态；再用固定随机种子的声明组合核对所有冲突对均存在顺序约束。
这些检查验证执行语义，不把临时节点排列或某个实现细节当作成功标准。
"""
import copy
import json
import random
import unittest

import networkx as nx

from ATF.GraphProcessor.execution_plan import (
    ORDER_KEY, UID_KEY, VERSION_KEY, aggregate_effects, compile_execution_plan,
    effects_for_method, export_execution_plan, normalize_effects,
    normalize_effect_spec, restore_execution_plan, set_execution_order,
)


def method(*, pure=None, reads=None, writes=None):
    return {"function": lambda value: value, "input_count": 1, "output_count": 1,
            "input_types": ["scalar"], "output_types": ["scalar"], "vectorized": False,
            "effects": normalize_effects(pure=pure, reads=reads, writes=writes)}


def independent_graph(names):
    """构造数据上互不依赖的节点，便于单独观察副作用控制依赖。"""
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(["root", "collapse"])
    for name in names:
        graph.add_node(name, method_name=name, is_passthrough=False)
        graph.add_edge("root", name, output_index=0, input_slot=0, data_type="scalar")
        graph.add_edge(name, "collapse", output_index=0, data_coord=len(graph), data_type="scalar")
    return graph


def reordered_copy(graph):
    """只改变节点/边存储顺序，保留 UID、端口和冻结语义，模拟文件重排。"""
    new = nx.MultiDiGraph(**copy.deepcopy(graph.graph))
    new.add_nodes_from(reversed([(n, copy.deepcopy(d)) for n, d in graph.nodes(data=True)]))
    for source, target, key, attrs in reversed(list(graph.edges(keys=True, data=True))):
        new.add_edge(source, target, key=key, **copy.deepcopy(attrs))
    return new


class EffectDeclarationTests(unittest.TestCase):
    """确保旧方法保守迁移，注册声明与复合方法摘要不会丢失副作用。"""
    def test_missing_declaration_is_unknown_and_cannot_assert_purity_by_default(self):
        self.assertEqual(effects_for_method({})["kind"], "unknown")
        self.assertEqual(normalize_effects(pure=False)["kind"], "unknown")
        self.assertEqual(normalize_effects(reads=[], writes=[])["kind"], "pure")
        with self.assertRaises(ValueError):
            normalize_effects(pure=True, reads=["environment"])
        with self.assertRaises(ValueError):
            normalize_effects(reads="environment")
        with self.assertRaises(ValueError):
            normalize_effects(writes=[""])

    def test_explicit_normalization_is_stable_and_validates_serialized_specs(self):
        spec = normalize_effects(reads=["b", "a", "a"], writes={"c"})
        self.assertEqual(spec, {"kind": "declared", "reads": ["a", "b"], "writes": ["c"]})
        self.assertEqual(normalize_effect_spec(json.loads(json.dumps(spec))), spec)
        self.assertEqual(normalize_effect_spec({"pure": True})["kind"], "pure")
        for invalid in ({"kind": "pure", "writes": ["x"]},
                        {"kind": "declared"},
                        {"kind": "surprise"}, {"kind": "unknown", "reads": ["x"]},
                        {"kind": "pure", "typo": 1}):
            with self.assertRaises(ValueError):
                normalize_effect_spec(invalid)

    def test_function_declarations_survive_direct_registry_entries(self):
        def f(value):
            return value
        f.effects = normalize_effects(reads=["env"])
        self.assertEqual(effects_for_method({"function": f}), f.effects)
        self.assertEqual(effects_for_method({"function": f, "effects": None})["kind"], "unknown")

    def test_composite_summary_includes_all_resources_and_opaque_members(self):
        graph = independent_graph(["read", "write", "pure"])
        methods = {"read": method(reads=["a"]), "write": method(writes=["b"]),
                   "pure": method(pure=True)}
        self.assertEqual(aggregate_effects(graph, methods), normalize_effects(reads=["a"], writes=["b"]))
        methods["write"] = method()
        self.assertEqual(aggregate_effects(graph, methods)["kind"], "unknown")


class EffectPlanTests(unittest.TestCase):
    """验证顺序保持、资源冲突、显式变异与持久化的主要边界。"""
    def test_only_conflicting_resources_are_serialized(self):
        names = ["read_a", "read_a2", "write_a", "read_b", "write_b", "pure"]
        graph = independent_graph(names)
        methods = {"read_a": method(reads=["a"]), "read_a2": method(reads=["a"]),
                   "write_a": method(writes=["a"]), "read_b": method(reads=["b"]),
                   "write_b": method(writes=["b"]), "pure": method(pure=True)}
        plan = compile_execution_plan(graph, methods)
        self.assertEqual(set(plan.control_edges), {("read_a", "write_a"), ("read_a2", "write_a"), ("read_b", "write_b")})
        self.assertEqual(plan.predecessors["pure"], ())
        self.assertEqual(plan.predecessors["read_a2"], ())
        self.assertNotIn("root", plan.order)
        self.assertNotIn("collapse", plan.predecessors)
        self.assertEqual(graph.number_of_edges(), 2 * len(names))

    def test_unknown_effects_form_linear_barriers(self):
        names = [f"opaque_{i}" for i in range(250)]
        graph = independent_graph(names)
        methods = {name: method() for name in names}
        plan = compile_execution_plan(graph, methods)
        self.assertEqual(len(plan.control_edges), len(names) - 1)
        self.assertEqual(plan.control_edges, tuple(zip(names, names[1:])))

    def test_wildcard_read_waits_for_writes_but_not_other_reads(self):
        names = ["write_a", "read_b", "read_all", "write_b", "read_a"]
        graph = independent_graph(names)
        methods = {"write_a": method(writes=["a"]), "read_b": method(reads=["b"]),
                   "read_all": method(reads=["*"]), "write_b": method(writes=["b"]),
                   "read_a": method(reads=["a"])}
        plan = compile_execution_plan(graph, methods)
        self.assertIn(("write_a", "read_all"), plan.control_edges)
        self.assertNotIn(("read_b", "read_all"), plan.control_edges)
        self.assertIn(("read_all", "write_b"), plan.control_edges)
        self.assertIn(("read_b", "write_b"), plan.control_edges)
        self.assertNotIn(("write_b", "read_a"), plan.control_edges)

    def test_unknown_and_wildcard_write_order_all_state_accesses(self):
        for middle in (method(), method(writes=["*"])):
            graph = independent_graph(["read_a", "write_b", "barrier", "read_c", "write_d"])
            methods = {"read_a": method(reads=["a"]), "write_b": method(writes=["b"]),
                       "barrier": middle, "read_c": method(reads=["c"]), "write_d": method(writes=["d"])}
            plan = compile_execution_plan(graph, methods)
            self.assertEqual(set(plan.predecessors["barrier"]), {"read_a", "write_b"})
            self.assertIn("barrier", plan.predecessors["read_c"])
            self.assertIn("barrier", plan.predecessors["write_d"])

    def test_resource_frontiers_cover_every_conflicting_pair(self):
        # 此处以读写集合的对称冲突定义独立检查算法输出，而不是重复前沿算法。
        rng = random.Random(20261107)
        choices = [method(), method(pure=True), method(reads=["*"]), method(writes=["*"]),
                   method(reads=["a"]), method(writes=["a"]), method(reads=["b"]),
                   method(writes=["b"]), method(reads=["a"], writes=["b"]),
                   method(reads=["*"], writes=["b"])]
        def overlap(left, right):
            return bool(left and right and ("*" in left or "*" in right or set(left) & set(right)))
        def conflict(left, right):
            if "pure" in (left["kind"], right["kind"]):
                return False
            if "unknown" in (left["kind"], right["kind"]):
                return True
            return (overlap(left["writes"], right["reads"] + right["writes"])
                    or overlap(right["writes"], left["reads"]))
        for trial in range(150):
            names = [f"n{i}" for i in range(8)]
            methods = {name: rng.choice(choices) for name in names}
            plan = compile_execution_plan(independent_graph(names), methods)
            dependencies = nx.DiGraph()
            dependencies.add_nodes_from(names)
            dependencies.add_edges_from(plan.control_edges)
            for i, first in enumerate(names):
                for second in names[i + 1:]:
                    if conflict(plan.effects[first], plan.effects[second]):
                        self.assertTrue(nx.has_path(dependencies, first, second), (trial, first, second))

    def test_all_legal_schedules_preserve_observable_state_and_reads(self):
        # a 上的写→读→写必须有序，b 的独立链和纯节点可以交错。
        # 检查所有合法交错，避免只用一次顺利的拓扑排序证明一致性。
        names = ["increment_a", "read_a", "multiply_a", "increment_b", "read_b", "pure"]
        graph = independent_graph(names)
        methods = {"increment_a": method(writes=["a"]), "read_a": method(reads=["a"]),
                   "multiply_a": method(reads=["a"], writes=["a"]),
                   "increment_b": method(writes=["b"]), "read_b": method(reads=["b"]),
                   "pure": method(pure=True)}
        plan = compile_execution_plan(graph, methods)
        schedule = nx.DiGraph()
        schedule.add_nodes_from(plan.order)
        schedule.add_edges_from((pred, node) for node, preds in plan.predecessors.items() for pred in preds)
        count = 0
        for order in nx.all_topological_sorts(schedule):
            state, outputs = {"a": 1, "b": 10}, {}
            for node in order:
                if node == "increment_a": state["a"] += 1
                elif node == "multiply_a": state["a"] *= 2
                elif node == "increment_b": state["b"] += 1
                elif node == "read_a": outputs["a"] = state["a"]
                elif node == "read_b": outputs["b"] = state["b"]
                else: outputs["pure"] = 49
            self.assertEqual(state, {"a": 4, "b": 11})
            self.assertEqual(outputs, {"a": 2, "b": 11, "pure": 49})
            count += 1
        self.assertGreater(count, 1)

    def test_storage_reorder_and_rename_do_not_change_effect_order(self):
        graph = independent_graph(["plus", "times", "read"])
        methods = {name: method() for name in ("plus", "times", "read")}
        original = compile_execution_plan(graph, methods)
        shuffled = reordered_copy(graph)
        renamed = nx.relabel_nodes(shuffled, {"plus": "new_name"}, copy=True)
        plan = compile_execution_plan(renamed, methods)
        self.assertEqual(plan.order, ("new_name", "times", "read"))
        self.assertEqual([plan.uids[n] for n in plan.order], [original.uids[n] for n in original.order])

    def test_new_nodes_insert_without_reordering_old_program(self):
        graph = independent_graph(["first", "last"])
        methods = {name: method() for name in ("first", "middle", "last")}
        compile_execution_plan(graph, methods)
        graph.add_node("middle", method_name="middle", is_passthrough=False)
        graph.add_edge("first", "middle", output_index=0, input_slot=0)
        graph.add_edge("middle", "last", output_index=0, input_slot=1)
        self.assertEqual(compile_execution_plan(graph, methods).order, ("first", "middle", "last"))

    def test_conflicting_mutation_rejected_until_explicitly_reordered(self):
        graph = independent_graph(["first", "last"])
        methods = {name: method() for name in ("first", "last")}
        original = compile_execution_plan(graph, methods).to_dict()
        graph.add_edge("last", "first", output_index=0, input_slot=1)
        with self.assertRaisesRegex(ValueError, "frozen execution order"):
            compile_execution_plan(graph, methods)
        self.assertEqual(graph.graph[ORDER_KEY], original["order"])
        set_execution_order(graph, ["last", "first"])
        self.assertEqual(compile_execution_plan(graph, methods).order, ("last", "first"))

    def test_saved_plan_roundtrip_and_mismatched_declarations_fail_closed(self):
        graph = independent_graph(["write", "read"])
        methods = {"write": method(writes=["a"]), "read": method(reads=["a"])}
        saved = json.loads(json.dumps(export_execution_plan(graph, methods)))
        copied = reordered_copy(graph)
        copied.graph.clear()
        restored = restore_execution_plan(copied, methods, saved)
        self.assertEqual(restored.order, ("write", "read"))
        changed_methods = copy.deepcopy(methods)
        changed_methods["write"]["effects"] = normalize_effects(pure=True)
        with self.assertRaisesRegex(ValueError, "differs"):
            restore_execution_plan(copied, changed_methods, saved)
        copied["root"]["read"][0]["output_index"] = 1
        with self.assertRaisesRegex(ValueError, "differs"):
            restore_execution_plan(copied, methods, saved)

    def test_saved_plan_accepts_uid_preserving_display_rename(self):
        graph = independent_graph(["first", "last"])
        methods = {name: method() for name in ("first", "last")}
        saved = export_execution_plan(graph, methods)
        renamed = nx.relabel_nodes(graph, {"first": "renamed"})
        self.assertEqual(restore_execution_plan(renamed, methods, saved).order, ("renamed", "last"))

    def test_saved_plan_rejects_conflicting_embedded_metadata(self):
        graph = independent_graph(["first", "last"])
        methods = {name: method() for name in ("first", "last")}
        saved = export_execution_plan(graph, methods)
        graph.graph[ORDER_KEY].reverse()
        with self.assertRaisesRegex(ValueError, "disagree"):
            restore_execution_plan(graph, methods, saved)

    def test_duplicate_uid_invalid_saved_plan_and_invalid_boundaries_rejected(self):
        methods = {name: method() for name in ("a", "b")}
        graph = independent_graph(["a", "b"])
        saved = export_execution_plan(graph, methods)
        graph.nodes["b"][UID_KEY] = graph.nodes["a"][UID_KEY]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compile_execution_plan(graph, methods)
        for bad_edge in (("collapse", "a"), ("a", "root"), ("a", "a")):
            malformed = independent_graph(["a", "b"])
            malformed.add_edge(*bad_edge)
            with self.assertRaises(ValueError):
                compile_execution_plan(malformed, methods)
        for patch in ({"version": 123}, {"order": [saved["order"][0]]}, {"control_edges": []}):
            bad = {**saved, **patch}
            clean = independent_graph(["a", "b"])
            for old, new in zip(graph, clean):
                if old not in ("root", "collapse"):
                    clean.nodes[new][UID_KEY] = saved["order"][["a", "b"].index(old)]
            with self.assertRaises(ValueError):
                restore_execution_plan(clean, methods, bad)


if __name__ == "__main__":
    unittest.main()
