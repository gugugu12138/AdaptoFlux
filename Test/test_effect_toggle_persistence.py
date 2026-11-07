"""核验公开开关及模型持久化，防止关闭状态在克隆/加载中被默认值覆盖。

这里使用两个读写同一状态、没有数据依赖的节点，并观察多样本结果。
开启时按完整样本执行，关闭时按节点处理整批；两种模式的真实输出不同，
因此测试不仅检查布尔字段，也能发现加载后调度语义没有随字段恢复的问题。
临时模型一律放入系统临时目录，不向项目仓库写入测试产物。
"""
import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

from ATF.CollapseManager.collapse_functions import CollapseMethod
from ATF.core.adaptoflux import AdaptoFlux


class EffectTogglePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="atf-effect-toggle-")
        self.path = Path(self.directory.name).resolve()
        self.addCleanup(self.directory.cleanup)

    @staticmethod
    def model(enabled=True):
        model = AdaptoFlux(values=np.array([[0.]]), input_types_list=["scalar"],
                           collapse_method=CollapseMethod.IDENTITY,
                           enable_effect_ordering=enabled)
        state = {"value": 1.}

        def increment(_):
            state["value"] += 1
            return state["value"]

        def double(_):
            state["value"] *= 2
            return state["value"]

        graph = nx.MultiDiGraph()
        graph.add_nodes_from(["root", "collapse"])
        for index, function in enumerate((increment, double)):
            name = function.__name__
            model.add_method(name, function, 1, 1, ["scalar"], ["scalar"],
                             "state", 1., False,
                             effects={"reads": ["world"], "writes": ["world"]})
            graph.add_node(name, method_name=name, layer=1, is_passthrough=False)
            graph.add_edge("root", name, input_slot=0, output_index=0, data_type="scalar")
            graph.add_edge(name, "collapse", output_index=0, data_coord=index,
                           data_type="scalar")
        model.graph_processor.set_graph(graph)
        model.path_generator.graph = model.graph
        return model, state

    def assert_mode_behavior(self, model, state, enabled):
        self.assertIs(model.enable_effect_ordering, enabled)
        plan = model.graph_processor.prepare_execution_plan()
        self.assertEqual(bool(plan.control_edges), enabled)
        state["value"] = 1.
        expected = [[2., 4.], [5., 10.]] if enabled else [[2., 6.], [3., 12.]]
        np.testing.assert_array_equal(model.infer_with_graph([[0.], [0.]]), expected)

    def test_constructor_and_public_setter_reject_non_boolean_without_changes(self):
        model, state = self.model(False)
        for invalid in (None, 0, 1, "False"):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    AdaptoFlux(input_types_list=["scalar"], enable_effect_ordering=invalid)
                accepted_graph = model.graph
                with self.assertRaises(ValueError):
                    model.set_effect_ordering(invalid)
                self.assertIs(model.graph, accepted_graph)
                self.assert_mode_behavior(model, state, False)
        model.set_effect_ordering(True)
        self.assert_mode_behavior(model, state, True)
        model.set_effect_ordering(False)
        self.assert_mode_behavior(model, state, False)

    def test_clone_and_factory_inherit_mode_and_custom_collapse(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                model, state = self.model(enabled)
                # 自定义聚合器分支也必须继承模式，不能只修复普通构造分支。
                model.set_custom_collapse(lambda values: values)
                cloned = model.clone()
                self.assertIsNot(cloned.graph, model.graph)
                self.assertIs(cloned.path_generator.graph, cloned.graph)
                self.assert_mode_behavior(cloned, state, enabled)
                replacement = model._create_graph_processor(model.graph.copy())
                self.assertIs(replacement.enable_effect_ordering, enabled)
                self.assertEqual(bool(replacement.prepare_execution_plan().control_edges), enabled)
                # 克隆后的切换不应修改原对象的配置。
                cloned.set_effect_ordering(not enabled)
                self.assertIs(model.enable_effect_ordering, enabled)

    def test_saved_setting_wins_for_json_and_gexf_and_keeps_batch_behavior(self):
        for enabled in (True, False):
            source, _ = self.model(enabled)
            directory = self.path / str(enabled)
            source.save_model(directory)
            payload = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
            self.assertIs(payload["execution_plan"]["enable_effect_ordering"], enabled)
            self.assertIs(payload["graph"]["_atf_enable_effect_ordering"], enabled)
            self.assertIs(source.enable_effect_ordering, enabled)
            for filename in ("graph.json", "graph.gexf"):
                with self.subTest(enabled=enabled, filename=filename):
                    target, state = self.model(not enabled)
                    target.load_model(directory / filename)
                    self.assertIs(target.path_generator.graph, target.graph)
                    self.assert_mode_behavior(target, state, enabled)

    def test_embedded_plan_alone_keeps_disabled_setting(self):
        source, _ = self.model(False)
        source.save_model(self.path)
        (self.path / "execution_plan.json").unlink()
        target, state = self.model(True)
        target.load_model(self.path / "graph.json")
        self.assert_mode_behavior(target, state, False)

    def test_old_plan_without_switch_field_defaults_to_enabled(self):
        source, _ = self.model(True)
        source.save_model(self.path)
        file = self.path / "graph.json"
        payload = json.loads(file.read_text(encoding="utf-8"))
        payload["execution_plan"].pop("enable_effect_ordering")
        payload["graph"].pop("_atf_enable_effect_ordering")
        file.write_text(json.dumps(payload), encoding="utf-8")
        (self.path / "execution_plan.json").write_text(
            json.dumps(payload["execution_plan"]), encoding="utf-8")
        target, state = self.model(False)
        target.load_model(file)
        self.assert_mode_behavior(target, state, True)

    def test_legacy_graph_without_plan_defaults_to_enabled(self):
        source, _ = self.model(False)
        graph = source.graph.copy()
        graph.graph.clear()
        for _, attrs in graph.nodes(data=True):
            attrs.pop("_atf_uid", None)
        file = self.path / "legacy.json"
        file.write_text(json.dumps(json_graph.node_link_data(graph, edges="edges")), encoding="utf-8")
        target, state = self.model(False)
        target.load_model(file)
        self.assert_mode_behavior(target, state, True)

    def test_malformed_switch_and_graph_plan_disagreement_do_not_replace_model(self):
        for invalid in (None, 0, 1, "False", True):
            with self.subTest(value=invalid):
                source, _ = self.model(False)
                source.save_model(self.path)
                file = self.path / "graph.json"
                payload = json.loads(file.read_text(encoding="utf-8"))
                # True 是合法布尔值，却与计划中的 False 不一致；其余值还违反
                # 类型约束。不能用 bool(value) 吞掉此类损坏的模型元数据。
                payload["graph"]["_atf_enable_effect_ordering"] = invalid
                if invalid is not True:
                    payload["execution_plan"]["enable_effect_ordering"] = invalid
                file.write_text(json.dumps(payload), encoding="utf-8")
                (self.path / "execution_plan.json").write_text(
                    json.dumps(payload["execution_plan"]), encoding="utf-8")
                target, state = self.model(True)
                accepted_graph = target.graph
                with self.assertRaises(ValueError):
                    target.load_model(file)
                self.assertIs(target.graph, accepted_graph)
                self.assertIs(target.path_generator.graph, accepted_graph)
                self.assert_mode_behavior(target, state, True)

    def test_sidecar_mode_disagreement_is_rejected_atomically(self):
        source, _ = self.model(False)
        source.save_model(self.path)
        file = self.path / "execution_plan.json"
        plan = json.loads(file.read_text(encoding="utf-8"))
        plan["enable_effect_ordering"] = True
        file.write_text(json.dumps(plan), encoding="utf-8")
        target, state = self.model(False)
        accepted_graph = target.graph
        with self.assertRaises(ValueError):
            target.load_model(self.path)
        self.assertIs(target.graph, accepted_graph)
        self.assert_mode_behavior(target, state, False)

    def test_disabled_plan_accepts_storage_reorder_but_still_checks_contract(self):
        source, _ = self.model(False)
        source.save_model(self.path)
        file = self.path / "graph.json"
        payload = json.loads(file.read_text(encoding="utf-8"))
        payload["nodes"].reverse()
        payload["edges"].reverse()
        file.write_text(json.dumps(payload), encoding="utf-8")
        target, _ = self.model(True)
        target.load_model(file)
        # 关闭模式允许文件重排产生另一合法数据拓扑序，因此只验证关闭契约，
        # 不能要求两个隐式状态写入在重载后仍取得原来的非交换运算结果。
        self.assertFalse(target.enable_effect_ordering)
        self.assertFalse(target.graph_processor.prepare_execution_plan().control_edges)
        accepted_graph = target.graph
        # 对节点存储顺序的宽容不应放宽方法声明检查；同时改两个计划副本，
        # 避开“侧车不同”这个早期错误，确保实际走到关闭模式的契约核验。
        payload["execution_plan"]["nodes"][0]["effects"] = {
            "kind": "pure", "reads": [], "writes": []}
        file.write_text(json.dumps(payload), encoding="utf-8")
        (self.path / "execution_plan.json").write_text(
            json.dumps(payload["execution_plan"]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            target.load_model(file)
        self.assertIs(target.graph, accepted_graph)
        self.assertFalse(target.enable_effect_ordering)


if __name__ == "__main__":
    unittest.main()
