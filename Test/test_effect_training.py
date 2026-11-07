"""训练与复合方法的行为回归：评分/接受一致、抽取顺序、保存重载与效果契约。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io

import networkx as nx
import numpy as np

from ATF.GraphProcessor.graph_processor import GraphProcessor
from ATF.GraphProcessor.execution_plan import aggregate_effects
from ATF.ModelTrainer.model_trainer import ModelTrainer
from ATF.ModelTrainer.GraphEvoTrainer.refinement_strategies import (
    refine_random_single_step, refine_full_sweep_step,
)
from ATF.ModelTrainer.GraphEvoTrainer.method_pool_evolver import MethodPoolEvolver
from ATF.core.evolved_method import EvolvedMethod
from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer


def method(fn, effects):
    return dict(function=fn, input_count=1, output_count=1,
                input_types=["scalar"], output_types=["scalar"],
                group="test", weight=1.0, vectorized=False, effects=effects)


class Model:
    def __init__(self, graph, methods):
        self.methods = methods
        self.graph_processor = GraphProcessor(graph, methods)
        self.path_generator = type("PathHolder", (), {})()
        self.path_generator.graph = self.graph_processor.graph

    @property
    def graph(self):
        return self.graph_processor.graph


def make_case():
    state = {"value": 1.0}

    def noop(_):
        return 0.0

    def increment(_):
        state["value"] += 1.0
        return 0.0

    def double(_):
        state["value"] *= 2.0
        return 0.0

    def read(_):
        return state["value"]

    rw = dict(kind="declared", reads=["env"], writes=["env"])
    methods = {"noop": method(noop, rw), "increment": method(increment, rw),
               "double": method(double, rw),
               "read": method(read, dict(kind="declared", reads=["env"], writes=[]))}
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(["root", "collapse"])
    for node, name in [("1_0_noop", "noop"), ("1_1_double", "double"), ("2_0_read", "read")]:
        graph.add_node(node, method_name=name, layer=int(node[0]), is_passthrough=False)
    for destination in ["1_0_noop", "1_1_double"]:
        graph.add_edge("root", destination, output_index=0, input_slot=0, data_type="scalar")
    graph.add_edge("1_0_noop", "2_0_read", output_index=0, input_slot=0, data_type="scalar")
    graph.add_edge("2_0_read", "collapse", output_index=0, data_coord=0, data_type="scalar")
    return Model(graph, methods), state


class Evaluator:
    max_total_refinement_attempts = None
    _total_refinement_attempts = 0
    verbose = False

    def __init__(self, state):
        self.state = state
        self.evaluated_order = None

    def _get_compatible_methods_for_node(self, af, node):
        return ["noop", "increment"]

    def _evaluate_loss(self, inputs, target, adaptoflux_instance):
        gp = adaptoflux_instance.graph_processor
        self.state["value"] = 1.0
        output = gp.infer_with_graph(inputs)
        self.evaluated_order = [gp.graph.nodes[n]["method_name"] for n in gp.prepare_execution_plan().order]
        return float(abs(output[0] - target[0]))

    def _evaluate_accuracy(self, inputs, target, adaptoflux_instance):
        self.state["value"] = 1.0
        output = adaptoflux_instance.graph_processor.infer_with_graph(inputs)
        return float(output[0] == target[0])

    def _should_accept(self, old, new):
        return new < old


class EffectTrainingTests(unittest.TestCase):
    @staticmethod
    def scalar_model():
        values = np.array([[0.0], [1.0], [2.0]])
        af = AdaptoFlux(values=values, input_types_list=["scalar"])
        af.add_method("plus_one", lambda value: float(value + 1), 1, 1,
                      ["scalar"], ["scalar"], "test", 1.0, False,
                      effects=dict(kind="pure", reads=[], writes=[]))
        return af, values

    def test_layer_growth_accept_and_rollback_keep_valid_plans(self):
        # 第一层让目标恰好匹配，第二层反而变差，真实训练流程必须能够回退到
        # 第一层的合法计划，而不是留下删除节点对应的顺序或过期生成器引用。
        af, values = self.scalar_model()
        trainer = LayerGrowTrainer(af, max_attempts=1, verbose=False, use_pipeline=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = trainer.train(values, values[:, 0] + 1, max_layers=2,
                                   max_total_attempts=2, save_model=False, enable_early_stop=False)
        self.assertEqual(result["layers_added"], 1)
        self.assertEqual(len(af.graph_processor.prepare_execution_plan().order), 1)
        np.testing.assert_array_equal(af.infer_with_graph(values), values[:, 0] + 1)

    def test_graphevo_reinitialization_removes_old_plan_members(self):
        # 多样初始化先复制旧模型再退回边界图，随后重新生长；中间阶段也必须
        # 能通过严格校验，最终计划仅包含当前候选仍然存在的节点。
        af, values = self.scalar_model()
        af.append_nx_layer(af.process_random_method())
        old_ids = set(af.graph.graph["_atf_execution_order"])
        trainer = GraphEvoTrainer(af, num_initial_models=1, max_init_layers=2,
                                  max_refinement_steps=1, enable_evolution=False,
                                  verbose=False, use_pipeline=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = trainer._phase_diverse_initialization(values, values[:, 0] + 2)
        final = result["best_model"]
        self.assertEqual(result["best_loss"], 0.0)
        self.assertEqual(len(final.graph_processor.prepare_execution_plan().order), 2)
        self.assertTrue(old_ids.isdisjoint(final.graph.graph["_atf_execution_order"]))
        np.testing.assert_array_equal(final.infer_with_graph(values), values[:, 0] + 2)

    def test_evaluated_candidate_order_is_the_accepted_order(self):
        # 先加一再乘二得到 4；若接受时节点重建改变顺序，则会错误地得到 3。
        # 同时检查最终并行执行，避免只在候选串行评分路径上修好问题。
        for strategy in [refine_random_single_step, refine_full_sweep_step]:
            with self.subTest(strategy=strategy.__name__):
                af, state = make_case()
                trainer = Evaluator(state)
                improved, loss, accuracy, replacements, _ = strategy(
                    trainer, af, np.array([[0.0]]), np.array([4.0]),
                    ["1_0_noop"], 2.0, af.graph_processor)
                self.assertTrue(improved)
                self.assertEqual((loss, accuracy, replacements), (0.0, 1.0, 1))
                order = [af.graph.nodes[n]["method_name"] for n in af.graph_processor.prepare_execution_plan().order]
                self.assertEqual(order, ["increment", "double", "read"])
                self.assertEqual(order, trainer.evaluated_order)
                self.assertIs(af.path_generator.graph, af.graph)
                state["value"] = 1.0
                self.assertEqual(af.graph_processor.infer_with_task_parallel([[0.0]], 3)[0], 4.0)

    def test_no_improvement_preserves_the_original_graph(self):
        af, state = make_case()
        af.graph_processor.prepare_execution_plan()
        before = copy.deepcopy(af.graph)
        strategy = Evaluator(state)
        result = refine_random_single_step(strategy, af, [[0.0]], [4.0],
                                          ["1_0_noop"], -1.0, af.graph_processor)
        self.assertFalse(result[0])
        self.assertEqual(list(af.graph.nodes(data=True)), list(before.nodes(data=True)))
        self.assertEqual(af.graph.graph, before.graph)

    def test_custom_evaluators_see_an_already_prepared_plan(self):
        af, _ = make_case()
        class EvaluationOnlyTrainer(ModelTrainer):
            def train(self, *args, **kwargs):
                raise NotImplementedError

        trainer = EvaluationOnlyTrainer.__new__(EvaluationOnlyTrainer)
        trainer.adaptoflux = af
        seen = []

        def custom(model, values, target):
            seen.append(tuple(model.graph.graph["_atf_execution_order"]))
            return 1.0

        trainer.custom_loss_evaluator = custom
        trainer.custom_accuracy_evaluator = custom
        self.assertEqual(trainer._evaluate_loss([[0]], [0]), 1.0)
        self.assertEqual(trainer._evaluate_accuracy([[0]], [0]), 1.0)
        self.assertEqual(len(seen[0]), 3)
        self.assertEqual(seen[0], seen[1])

    def test_interleaved_stateful_extraction_is_rejected(self):
        # A 与 C 中间还有外部 B 的状态访问，不能把 A、C 封成不可中断的一次调用。
        af, _ = make_case()
        evolver = MethodPoolEvolver(type("Trainer", (), {"verbose": False})())
        mapping = {"1_0_noop": "A", "1_1_double": "B", "2_0_read": "C"}
        interleaved = nx.MultiDiGraph()
        interleaved.add_nodes_from(["A", "C"])
        contiguous = nx.MultiDiGraph()
        contiguous.add_nodes_from(["A", "B"])
        graphs, counts = evolver._filter_effect_safe_candidates(
            [interleaved, contiguous], [10, 7], mapping, af)
        self.assertEqual(graphs, [contiguous])
        self.assertEqual(counts, [7])

    def test_composite_effects_and_order_survive_reload(self):
        # 主动倒置保存文件的节点/边排列，检验恢复的是已保存的语义顺序。
        af, state = make_case()
        composite = EvolvedMethod("stateful", af.graph, af.methods,
                                  dict(name="stateful", output_count=1))
        self.assertEqual(composite.effects, dict(kind="declared", reads=["env"], writes=["env"]))
        root = Path(__file__).resolve().parents[1] / "tmp" / "effect_training_tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            composite.save(folder)
            path = Path(folder) / "stateful.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["nodes"].reverse()
            payload["edges"].reverse()
            path.write_text(json.dumps(payload), encoding="utf-8")
            restored = EvolvedMethod.from_files(str(Path(folder) / "stateful"), methods_ref=af.methods)
            state["value"] = 1.0
            self.assertEqual(restored(0.0), 2.0)
            self.assertEqual(restored.effects, composite.effects)
            self.assertEqual(restored.graph.graph["_atf_execution_order"], composite.graph.graph["_atf_execution_order"])

    def test_reconstructed_composite_uses_source_effect_order(self):
        af, state = make_case()
        af.graph.nodes["1_0_noop"]["method_name"] = "increment"
        af.graph_processor.prepare_execution_plan()
        mapping = {"1_0_noop": "A", "1_1_double": "B", "2_0_read": "C"}
        signatures = nx.MultiDiGraph()
        for source in reversed(list(mapping)):
            signatures.add_node(mapping[source], **af.graph.nodes[source])
        signatures.add_edge("A", "C", output_index=0, input_slot=0,
                            data_coord=0, data_type="scalar")
        # 旧重建器通过 data_coord 识别入口列；测试补齐这一真实输入规范。
        for _, _, attrs in af.graph.out_edges("root", data=True):
            attrs["data_coord"] = 0
        evolver = MethodPoolEvolver(type("Trainer", (), {"verbose": False})())
        graph, = evolver._reconstruct_executable_graphs(
            [signatures], signatures, mapping, af)
        processor = GraphProcessor(graph, af.methods)
        state["value"] = 1.0
        self.assertEqual(processor.infer_with_graph([[0.0, 0.0]])[0], 4.0)
        names = [graph.nodes[node]["method_name"] for node in processor.prepare_execution_plan().order]
        self.assertEqual(names, ["increment", "double", "read"])

    def test_composite_cannot_hide_an_unknown_constituent(self):
        af, _ = make_case()
        af.methods["double"].pop("effects")
        self.assertEqual(aggregate_effects(af.graph, af.methods)["kind"], "unknown")
        with self.assertRaisesRegex(ValueError, "effect declaration"):
            EvolvedMethod("unsafe", af.graph, af.methods,
                          dict(name="unsafe", effects=dict(kind="pure", reads=[], writes=[])))

    def test_composite_rejects_changed_effect_contract_before_execution(self):
        af, state = make_case()
        composite = EvolvedMethod("stateful", af.graph, af.methods,
                                  dict(name="stateful", output_count=1))
        af.methods["double"]["effects"] = dict(kind="unknown", reads=[], writes=[])
        with self.assertRaisesRegex(ValueError, "effects changed"):
            composite(0.0)
        self.assertEqual(state["value"], 1.0)

    def test_composite_load_checks_each_effect_not_just_the_union(self):
        af, _ = make_case()
        composite = EvolvedMethod("stateful", af.graph, af.methods,
                                  dict(name="stateful", output_count=1))
        root = Path(__file__).resolve().parents[1] / "tmp" / "effect_training_tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            composite.save(folder)
            changed = {name: dict(info) for name, info in af.methods.items()}
            changed["noop"]["effects"] = dict(kind="pure", reads=[], writes=[])
            # double/read 仍覆盖 env，总效果并集未变，但 noop 的已保存声明变了。
            self.assertEqual(aggregate_effects(composite.graph, changed), composite.effects)
            with self.assertRaisesRegex(ValueError, "Saved execution plan differs"):
                EvolvedMethod.from_files(str(Path(folder) / "stateful"), methods_ref=changed)

    def test_composite_load_rejects_missing_plan_in_new_format(self):
        af, _ = make_case()
        composite = EvolvedMethod("stateful", af.graph, af.methods,
                                  dict(name="stateful", output_count=1))
        root = Path(__file__).resolve().parents[1] / "tmp" / "effect_training_tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            composite.save(folder)
            path = Path(folder) / "stateful.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["execution_plan"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing its execution_plan"):
                EvolvedMethod.from_files(str(Path(folder) / "stateful"), methods_ref=af.methods)

    def test_composite_load_migrates_legacy_without_plan_or_uids(self):
        af, state = make_case()
        root = Path(__file__).resolve().parents[1] / "tmp" / "effect_training_tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            base = Path(folder) / "legacy"
            payload = nx.node_link_data(af.graph, edges="edges")
            self.assertTrue(all("_atf_uid" not in node for node in payload["nodes"]))
            Path(str(base) + ".json").write_text(json.dumps(payload), encoding="utf-8")
            Path(str(base) + ".meta.json").write_text(
                json.dumps(dict(name="legacy", output_count=1)), encoding="utf-8")
            restored = EvolvedMethod.from_files(str(base), methods_ref=af.methods)
            self.assertEqual(len(restored.graph.graph["_atf_execution_order"]), 3)
            state["value"] = 1.0
            self.assertEqual(restored(0.0), 2.0)


if __name__ == "__main__":
    unittest.main()
