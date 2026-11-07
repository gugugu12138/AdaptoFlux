"""验证关闭开关后，训练候选、子程序抽取和复合方法重载不会悄悄重新开启。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx
import numpy as np

from ATF.GraphProcessor.execution_plan import EFFECT_ORDERING_KEY
from ATF.ModelTrainer.model_trainer import ModelTrainer
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer
from ATF.ModelTrainer.GraphEvoTrainer.method_pool_evolver import MethodPoolEvolver
from ATF.ModelTrainer.GraphEvoTrainer.refinement_strategies import (
    refine_random_single_step, refine_full_sweep_step,
)
from ATF.ModelTrainer.LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer
from ATF.core.adaptoflux import AdaptoFlux
from ATF.core.evolved_method import EvolvedMethod
from Test.test_effect_training import make_case, Evaluator


class EffectToggleTrainingTests(unittest.TestCase):
    @staticmethod
    def scalar_model():
        values = np.array([[0.0], [1.0], [2.0]])
        af = AdaptoFlux(values=values, input_types_list=["scalar"],
                       enable_effect_ordering=False)
        af.add_method("plus_one", lambda value: float(value + 1), 1, 1,
                      ["scalar"], ["scalar"], "test", 1.0, False,
                      effects=dict(kind="pure", reads=[], writes=[]))
        return af, values

    def test_layer_growth_keeps_disabled_mode_after_accept_and_rollback(self):
        # 实际走一次接受和一次回退，避免只检查构造参数却漏掉训练重建处理器。
        af, values = self.scalar_model()
        trainer = LayerGrowTrainer(af, max_attempts=1, verbose=False, use_pipeline=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = trainer.train(values, values[:, 0] + 1, max_layers=2,
                                   max_total_attempts=2, save_model=False, enable_early_stop=False)
        self.assertEqual(result["layers_added"], 1)
        self.assertFalse(af.enable_effect_ordering)
        self.assertFalse(af.graph_processor.prepare_execution_plan().to_dict()["enable_effect_ordering"])
        np.testing.assert_array_equal(af.infer_with_graph(values), values[:, 0] + 1)

    def test_diverse_initialization_keeps_disabled_mode(self):
        af, values = self.scalar_model()
        af.append_nx_layer(af.process_random_method())
        trainer = GraphEvoTrainer(af, num_initial_models=1, max_init_layers=2,
                                  max_refinement_steps=1, enable_evolution=False,
                                  verbose=False, use_pipeline=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = trainer._phase_diverse_initialization(values, values[:, 0] + 2)
        final = result["best_model"]
        self.assertEqual(result["best_loss"], 0.0)
        self.assertFalse(final.enable_effect_ordering)
        self.assertFalse(final.graph_processor.prepare_execution_plan().control_edges)

    def test_refinement_candidates_and_accepted_model_keep_disabled_mode(self):
        for strategy in [refine_random_single_step, refine_full_sweep_step]:
            with self.subTest(strategy=strategy.__name__):
                af, state = make_case()
                af.graph_processor.set_effect_ordering(False)
                seen = []

                class ModeEvaluator(Evaluator):
                    def _evaluate_loss(self, inputs, target, adaptoflux_instance):
                        seen.append(adaptoflux_instance.graph_processor.enable_effect_ordering)
                        return super()._evaluate_loss(inputs, target, adaptoflux_instance)

                trainer = ModeEvaluator(state)
                result = strategy(trainer, af, np.array([[0.0]]), np.array([4.0]),
                                  ["1_0_noop"], 2.0, af.graph_processor)
                self.assertTrue(result[0])
                self.assertEqual(seen, [False])
                self.assertFalse(af.graph_processor.enable_effect_ordering)
                self.assertFalse(af.graph_processor.prepare_execution_plan().control_edges)

    def test_custom_evaluation_preparation_does_not_enable_ordering(self):
        af, _ = make_case()
        af.graph_processor.set_effect_ordering(False)

        class EvaluationOnlyTrainer(ModelTrainer):
            def train(self, *args, **kwargs):
                raise NotImplementedError

        trainer = EvaluationOnlyTrainer.__new__(EvaluationOnlyTrainer)
        trainer.adaptoflux = af
        seen = []

        def custom(model, values, target):
            plan = model.graph_processor.prepare_execution_plan()
            seen.append((model.graph_processor.enable_effect_ordering, plan.control_edges))
            return 1.0

        trainer.custom_loss_evaluator = custom
        trainer.custom_accuracy_evaluator = custom
        trainer._evaluate_loss([[0]], [0])
        trainer._evaluate_accuracy([[0]], [0])
        self.assertEqual(seen, [(False, ()), (False, ())])

    def test_disabled_extraction_skips_effect_filter_but_keeps_sink_limit(self):
        af, _ = make_case()
        af.graph_processor.set_effect_ordering(False)
        evolver = MethodPoolEvolver(type("Trainer", (), {"verbose": False})())
        mapping = {"1_0_noop": "A", "1_1_double": "B", "2_0_read": "C"}
        fragment = nx.MultiDiGraph()
        fragment.add_nodes_from(["A", "C"])
        graphs, counts = evolver._filter_effect_safe_candidates([fragment], [10], mapping, af)
        self.assertEqual((graphs, counts), ([fragment], [10]))
        # 零输出限制来自旧提取器的数值边界，关闭排序不应把该限制一起绕过。
        af.graph.remove_edge("2_0_read", "collapse")
        af.methods["read"]["output_count"] = 0
        af.methods["read"]["output_types"] = []
        self.assertEqual(evolver._filter_effect_safe_candidates([fragment], [10], mapping, af), ([], []))

    def test_reconstruction_and_registration_inherit_source_mode(self):
        af, _ = make_case()
        af.graph_processor.set_effect_ordering(False)
        mapping = {"1_0_noop": "A", "1_1_double": "B", "2_0_read": "C"}
        signatures = nx.MultiDiGraph()
        for original, signature in mapping.items():
            signatures.add_node(signature, **af.graph.nodes[original])
        signatures.add_edge("A", "C", output_index=0, input_slot=0,
                            data_coord=0, data_type="scalar")
        for _, _, attrs in af.graph.out_edges("root", data=True):
            attrs["data_coord"] = 0
        evolver = MethodPoolEvolver(type("Trainer", (), {"verbose": False})())
        graph, = evolver._reconstruct_executable_graphs([signatures], signatures, mapping, af)
        self.assertFalse(graph.graph[EFFECT_ORDERING_KEY])
        # 注册到开启模式的另一模型：被抽取子程序仍应沿用来源的关闭模式。
        for target_mode in [False, True]:
            with self.subTest(target_mode=target_mode):
                target = AdaptoFlux(values=np.array([[0.0]]), input_types_list=["scalar"],
                                   enable_effect_ordering=target_mode)
                target.set_methods(dict(af.methods))
                result = evolver._register_evolved_methods(target, [graph], [1])
                composite = target.methods[result["new_method_names"][0]]["function"]
                self.assertFalse(composite.enable_effect_ordering)
                self.assertEqual(target.enable_effect_ordering, target_mode)
                target.set_effect_ordering(False)
                target.set_effect_ordering(True)
                self.assertFalse(composite.enable_effect_ordering)

    def test_composite_disabled_mode_survives_lazy_rebuild_and_reload(self):
        af, state = make_case()
        composite = EvolvedMethod("disabled", af.graph, af.methods,
                                  dict(name="disabled", output_count=1),
                                  enable_effect_ordering=False)
        composite._processor = None
        state["value"] = 1.0
        self.assertEqual(composite(0.0), 2.0)
        self.assertFalse(composite._processor.enable_effect_ordering)
        # 导出测试写入系统临时目录，退出上下文自动清理，不向仓库放临时模型。
        with tempfile.TemporaryDirectory(prefix="atf-toggle-composite-") as folder:
            composite.save(folder)
            restored = EvolvedMethod.from_files(str(Path(folder) / "disabled"), methods_ref=af.methods)
            self.assertFalse(restored.enable_effect_ordering)
            self.assertFalse(restored._processor.prepare_execution_plan().control_edges)
            state["value"] = 1.0
            self.assertEqual(restored(0.0), 2.0)
            meta_path = Path(folder) / "disabled.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["enable_effect_ordering"] = True
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mode differs"):
                EvolvedMethod.from_files(str(Path(folder) / "disabled"), methods_ref=af.methods)

    def test_composite_rejects_non_boolean_mode_and_inconsistent_metadata(self):
        af, _ = make_case()
        for invalid in [0, 1, None, "false"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                EvolvedMethod("bad", af.graph, af.methods, enable_effect_ordering=invalid)
        with self.assertRaisesRegex(ValueError, "mode differs"):
            EvolvedMethod("bad", af.graph, af.methods,
                          {"enable_effect_ordering": False}, enable_effect_ordering=True)

    def test_standalone_constructor_default_is_explicitly_enabled(self):
        # 直接构造独立子程序时默认开启；从已有文件或来源快照迁移则由加载器、
        # 提取器显式传递开关。仅传入带关闭标记的图并不覆盖构造参数的选择。
        af, _ = make_case()
        af.graph_processor.set_effect_ordering(False)
        composite = EvolvedMethod("enabled", af.graph, af.methods)
        self.assertTrue(composite.enable_effect_ordering)
        self.assertTrue(composite._processor.prepare_execution_plan().control_edges)


if __name__ == "__main__":
    unittest.main()
