"""核验效果声明和保存重载：顺序改变不能悄悄改变外部状态的可观察结果。

测试采用两个不交换的状态操作（先加一、再乘二）及后续读取；结果不只检查
输出形状，还检查真实状态访问顺序。临时文件只创建在当前 ATF 的 tmp 目录。
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

from ATF.core.adaptoflux import AdaptoFlux
from ATF.GraphProcessor.execution_plan import export_execution_plan
from ATF.methods.decorators import method_profile


ROOT = Path(__file__).resolve().parents[1]


class EffectPersistenceTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "tmp"
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="effect-persistence-", dir=scratch)
        self.path = Path(self.directory.name).resolve()
        assert self.path.parent == scratch.resolve()
        self.addCleanup(self.directory.cleanup)

    @staticmethod
    def model():
        model = AdaptoFlux(values=np.array([[0.]]), input_types_list=["scalar"])
        state = {"value": 1., "events": []}

        @method_profile(input_types=["scalar"], output_types=["scalar"], reads={"world"}, writes={"world"})
        def increment(_):
            state["value"] += 1
            state["events"].append("increment")
            return state["value"]

        @method_profile(input_types=["scalar"], output_types=["scalar"], reads={"world"}, writes={"world"})
        def double(_):
            state["value"] *= 2
            state["events"].append("double")
            return state["value"]

        @method_profile(input_types=["scalar"], output_types=["scalar"], reads={"world"})
        def observe(_):
            state["events"].append("observe")
            return state["value"]

        for fn in [increment, double, observe]:
            model.add_method(fn.__name__, fn, 1, 1, fn.input_types, fn.output_types,
                             "state", 1., False)
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(["root", "collapse"])
        for index, fn in enumerate([increment, double, observe]):
            node = f"1_{index}_{fn.__name__}"
            graph.add_node(node, method_name=fn.__name__, layer=1, is_passthrough=False)
            graph.add_edge("root", node, input_slot=0, output_index=0, data_type="scalar")
        graph.add_edge("1_2_observe", "collapse", output_index=0, data_coord=0, data_type="scalar")
        model.graph_processor.set_graph(graph)
        model.path_generator.graph = model.graph
        return model, state

    def assert_behavior(self, model, state, parallel=False):
        state.update(value=1., events=[])
        result = (model.infer_with_task_parallel([[0.]], num_workers=3)
                  if parallel else model.infer_with_graph([[0.]]))
        np.testing.assert_array_equal(result, [4.])
        self.assertEqual(state["events"], ["increment", "double", "observe"])

    def test_declared_pure_and_default_unknown_registration(self):
        @method_profile(input_types=["scalar"], output_types=["scalar"])
        def unknown(x):
            return x

        @method_profile(input_types=["scalar"], output_types=["scalar"], pure=True)
        def pure(x):
            return x

        model = AdaptoFlux(values=np.array([[0.]]), input_types_list=["scalar"])
        for fn in (unknown, pure):
            model.add_method(fn.__name__, fn, 1, 1, ["scalar"], ["scalar"], "default", 1., False)
            self.assertEqual(model.methods[fn.__name__]["effects"], fn.effects)
        self.assertEqual(unknown.effects["kind"], "unknown")
        self.assertEqual(pure.effects, {"kind": "pure", "reads": [], "writes": []})
        with self.assertRaises(ValueError):
            method_profile(pure=True, writes={"world"})

    def test_invalid_registry_does_not_replace_accepted_methods(self):
        model, state = self.model()
        accepted_methods = model.methods
        accepted_graph = model.graph
        accepted_plan = export_execution_plan(model.graph, model.methods)
        invalid = {name: dict(info) for name, info in model.methods.items()}
        # 新方法声明要求两个参数，但已接受的数据图只提供一个。整个方法池更新
        # 必须失败，不能先替换外层注册表再让运行器发现错误，造成组件各用一版。
        invalid["double"]["input_count"] = 2
        with self.assertRaises(ValueError):
            model.set_methods(invalid)
        self.assertIs(model.methods, accepted_methods)
        self.assertIs(model.graph_processor.methods, accepted_methods)
        self.assertIs(model.path_generator.methods, accepted_methods)
        self.assertIs(model.graph, accepted_graph)
        self.assertEqual(export_execution_plan(model.graph, model.methods), accepted_plan)
        self.assert_behavior(model, state, parallel=True)

    def test_import_declared_effects_from_explicit_method_path(self):
        source = self.path / "operators.py"
        source.write_text(
            'from ATF.methods.decorators import method_profile\n'
            '@method_profile(input_types=["scalar"], output_types=["scalar"], reads={"position"})\n'
            'def observe(x): return x\n', encoding="utf-8")
        model = AdaptoFlux(values=np.array([[0.]]), input_types_list=["scalar"])
        with contextlib.redirect_stdout(io.StringIO()):
            model.import_methods_from_file(str(source))
        self.assertEqual(model.methods["observe"]["effects"],
                         {"kind": "declared", "reads": ["position"], "writes": []})

    def test_json_save_reload_survives_node_and_edge_reordering(self):
        model, state = self.model()
        self.assert_behavior(model, state)
        plan = export_execution_plan(model.graph, model.methods)
        model.save_model(self.path)
        graph_file = self.path / "graph.json"
        data = json.loads(graph_file.read_text(encoding="utf-8"))
        data["nodes"].reverse()
        data["edges"].reverse()
        graph_file.write_text(json.dumps(data), encoding="utf-8")
        model.load_model(self.path)
        self.assertEqual(export_execution_plan(model.graph, model.methods), plan)
        self.assertIs(model.path_generator.graph, model.graph)
        self.assert_behavior(model, state, parallel=True)

    def test_gexf_round_trip_uses_sidecar(self):
        model, state = self.model()
        model.save_model(self.path)
        model.load_model(self.path / "graph.gexf")
        self.assert_behavior(model, state, parallel=True)
        self.assertIs(model.path_generator.graph, model.graph)

    def test_json_self_contained_without_sidecar(self):
        model, state = self.model()
        model.save_model(self.path)
        (self.path / "execution_plan.json").unlink()
        model.load_model(self.path / "graph.json")
        self.assert_behavior(model, state)

    def test_gexf_without_sidecar_cannot_silently_choose_new_order(self):
        model, _ = self.model()
        model.save_model(self.path)
        original_graph = model.graph
        (self.path / "execution_plan.json").unlink()
        with self.assertRaisesRegex(ValueError, "requires a saved execution plan"):
            model.load_model(self.path / "graph.gexf")
        self.assertIs(model.graph, original_graph)

    def test_planned_json_without_both_plan_copies_is_rejected(self):
        model, state = self.model()
        model.save_model(self.path)
        original_graph = model.graph
        file = self.path / "graph.json"
        payload = json.loads(file.read_text(encoding="utf-8"))
        # 模拟新模型的两份计划都丢失。节点仍带持久 UID，不能走旧模型迁移路径，
        # 即使图自身还留有部分顺序字段，也不能据此跳过完整的声明和依赖校验。
        payload.pop("execution_plan")
        file.write_text(json.dumps(payload), encoding="utf-8")
        (self.path / "execution_plan.json").unlink()
        with self.assertRaisesRegex(ValueError, "requires a saved execution plan"):
            model.load_model(self.path)
        self.assertIs(model.graph, original_graph)
        self.assertIs(model.path_generator.graph, original_graph)
        self.assert_behavior(model, state)

    def test_invalid_sidecar_does_not_replace_live_graph(self):
        model, state = self.model()
        model.save_model(self.path)
        original_graph = model.graph
        (self.path / "execution_plan.json").write_text("null", encoding="utf-8")
        with self.assertRaises(ValueError):
            model.load_model(self.path)
        self.assertIs(model.graph, original_graph)
        self.assertIs(model.path_generator.graph, original_graph)
        self.assert_behavior(model, state)

    def test_changed_effect_declaration_rejected_before_replacement(self):
        model, _ = self.model()
        model.save_model(self.path)
        original_graph = model.graph
        model.methods["increment"]["effects"] = {"kind": "pure", "reads": [], "writes": []}
        with self.assertRaises(ValueError):
            model.load_model(self.path)
        self.assertIs(model.graph, original_graph)

    def test_changed_data_port_rejected_before_replacement(self):
        model, _ = self.model()
        model.save_model(self.path)
        original_graph = model.graph
        file = self.path / "graph.json"
        payload = json.loads(file.read_text(encoding="utf-8"))
        payload["edges"][0]["output_index"] = 1
        file.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            model.load_model(self.path)
        self.assertIs(model.graph, original_graph)

    def test_legacy_graph_receives_one_frozen_order(self):
        model, state = self.model()
        graph = model.graph.copy()
        graph.graph.clear()
        for _, data in graph.nodes(data=True):
            data.pop("_atf_uid", None)
        file = self.path / "legacy.json"
        file.write_text(json.dumps(json_graph.node_link_data(graph, edges="edges")), encoding="utf-8")
        model.load_model(file)
        self.assert_behavior(model, state)
        first = export_execution_plan(model.graph, model.methods)
        self.assert_behavior(model, state, parallel=True)
        self.assertEqual(export_execution_plan(model.graph, model.methods), first)

    def test_clone_preserves_plan_and_independent_metadata(self):
        model, state = self.model()
        self.assert_behavior(model, state)
        clone = model.clone()
        self.assertIsNot(clone.graph, model.graph)
        self.assertIs(clone.path_generator.graph, clone.graph)
        self.assertEqual(export_execution_plan(clone.graph, clone.methods),
                         export_execution_plan(model.graph, model.methods))
        self.assert_behavior(clone, state, parallel=True)
        clone.methods["increment"]["effects"]["writes"].append("other")
        self.assertEqual(model.methods["increment"]["effects"]["writes"], ["world"])


if __name__ == "__main__":
    unittest.main()
