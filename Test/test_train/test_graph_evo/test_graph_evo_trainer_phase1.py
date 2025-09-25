# test_graph_evo_trainer_phase1.py

import numpy as np
import logging
from typing import Any, Dict, List
import networkx as nx
from itertools import cycle

from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============== 模拟 AdaptoFlux 类 ===============
class MockAdaptoFlux:
    def __init__(self):
        self.graph_processor = MockGraphProcessor()
        self.graph_processor._parent = self  # 👈 添加这一行！
        self.methods = {
            "add": {"group": "math", "input_types": ["scalar"], "output_types": ["scalar"]},
            "multiply": {"group": "math", "input_types": ["scalar"], "output_types": ["scalar"]},
            "subtract": {"group": "math", "input_types": ["scalar"], "output_types": ["scalar"]},
            "relu": {"group": "activation", "input_types": ["scalar"], "output_types": ["scalar"]},
            "sigmoid": {"group": "activation", "input_types": ["scalar"], "output_types": ["scalar"]},
        }

    def clone(self):
        # 返回一个深拷贝
        import copy
        return copy.deepcopy(self)

    def process_random_method(self) -> Dict[str, Any]:
        # 随机选一个方法组
        group = np.random.choice(["math", "activation"])
        return {
            "valid_groups": [group],
            "selected_group": group,
            "method_candidates": [m for m, info in self.methods.items() if info.get('group') == group]
        }

    def append_nx_layer(self, plan: Dict[str, Any], discard_unmatched: str, discard_node_method_name: str):
        # 随机选一个方法添加到图中
        candidates = plan.get("method_candidates", [])
        if not candidates:
            return
        selected_method = np.random.choice(candidates)
        self.graph_processor.add_node_with_method(selected_method)

    def infer_with_graph(self, values: np.ndarray) -> np.ndarray:
        # 模拟：对输入加一个随机扰动，用于制造不同损失
        np.random.seed(hash(str(self.graph_processor.get_structure_signature())) % (2**32))
        noise = np.random.normal(0, 0.1, size=values.shape)
        return values + noise  # 模拟输出

    def save_model(self, folder: str):
        # 占位，不实际保存
        pass


class MockGraphProcessor:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0

    def add_node_with_method(self, method_name: str):
        node_id = f"node_{self.node_counter}"
        self.graph.add_node(node_id, method_name=method_name, group=method_name)
        self.node_counter += 1
        # 简单连接：如果存在前一个节点，就连起来
        if self.node_counter > 1:
            prev_node = f"node_{self.node_counter - 2}"
            self.graph.add_edge(prev_node, node_id)

    def _is_processing_node(self, node: str) -> bool:
        return True  # 所有节点都视为处理节点

    def get_structure_signature(self) -> str:
        """返回图结构的签名，用于判断是否不同"""
        methods = [
            self.graph.nodes[node]['method_name']
            for node in sorted(self.graph.nodes)
        ]
        return "_".join(methods)
    
    def infer_with_graph(self, values: np.ndarray) -> np.ndarray:
        # 委托给外层 adaptoflux 实例（我们假设它存在）
        # 在 MockAdaptoFlux 中，我们已定义 infer_with_graph
        # 所以这里直接调用外层实例的方法
        if hasattr(self, '_parent') and self._parent:
            return self._parent.infer_with_graph(values)
        else:
            # 备用：返回输入值加噪声
            np.random.seed(hash(str(self.get_structure_signature())) % (2**32))
            noise = np.random.normal(0, 0.1, size=values.shape)
            return values + noise


# =============== 测试函数 ===============
def test_phase1_fixed_mode():
    print("\n" + "="*60)
    print("🧪 TEST 1: Fixed Mode Initialization (All models with 3 layers)")
    print("="*60)

    # 创建 mock 实例
    mock_af = MockAdaptoFlux()

    # 创建 trainer
    trainer = GraphEvoTrainer(
        adaptoflux_instance=mock_af,
        num_initial_models=5,
        max_init_layers=3,
        init_mode="fixed",
        verbose=True
    )

    # 准备测试数据
    X = np.random.randn(10, 1).astype(np.float32)
    y = (X > 0).astype(np.float32)

    # 执行第一阶段
    result = trainer._phase_diverse_initialization(X, y)

    # 验证结果
    assert result is not None
    assert 'best_model' in result
    assert 'best_loss' in result
    assert 'best_accuracy' in result

    print(f"\n✅ Selected best model with Loss: {result['best_loss']:.6f}, Accuracy: {result['best_accuracy']:.4f}")

    # 验证是否真的生成了不同结构
    print("\n🔍 Checking if generated models have different structures...")
    structures = set()
    for i in range(trainer.num_initial_models):
        # 重新生成（为了测试，我们手动模拟生成过程）
        temp_af = mock_af.clone()
        trainer._randomly_initialize_graph(temp_af, num_layers_to_add=3)
        sig = temp_af.graph_processor.get_structure_signature()
        structures.add(sig)
        print(f"  Candidate {i+1} structure: {sig}")

    if len(structures) > 1:
        print(f"✅ Detected {len(structures)} unique structures out of {trainer.num_initial_models} candidates.")
    else:
        print("⚠️  Warning: All candidates have identical structure. Randomness may be broken.")


def test_phase1_list_mode():
    print("\n" + "="*60)
    print("🧪 TEST 2: List Mode Initialization (Custom layer counts: [1, 2, 3, 2, 1])")
    print("="*60)

    mock_af = MockAdaptoFlux()

    trainer = GraphEvoTrainer(
        adaptoflux_instance=mock_af,
        num_initial_models=5,
        init_mode="list",
        init_layers_list=[1, 2, 3, 2, 1],  # 每个候选模型的层数
        verbose=True
    )

    X = np.random.randn(10, 1).astype(np.float32)
    y = (X > 0).astype(np.float32)

    result = trainer._phase_diverse_initialization(X, y)

    assert result is not None
    print(f"\n✅ Selected best model with Loss: {result['best_loss']:.6f}, Accuracy: {result['best_accuracy']:.4f}")

    # 验证每个模型的层数是否符合预期
    print("\n🔍 Verifying layer counts per candidate...")
    expected_layers = [1, 2, 3, 2, 1]
    for i in range(len(expected_layers)):
        temp_af = mock_af.clone()
        trainer._randomly_initialize_graph(temp_af, num_layers_to_add=expected_layers[i])
        node_count = len(temp_af.graph_processor.graph.nodes)
        print(f"  Candidate {i+1}: expected {expected_layers[i]} layers, got {node_count} nodes")
        assert node_count == expected_layers[i], f"Candidate {i+1} layer count mismatch!"


def test_phase1_best_selection():
    print("\n" + "="*60)
    print("🧪 TEST 3: Best Model Selection (Does it pick the lowest loss?)")
    print("="*60)

    mock_af = MockAdaptoFlux()

    # 固定随机种子，让输出可预测（仅用于测试）
    np.random.seed(42)

    trainer = GraphEvoTrainer(
        adaptoflux_instance=mock_af,
        num_initial_models=3,
        max_init_layers=2,
        init_mode="fixed",
        verbose=True
    )

    X = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.0], [1.0], [1.0]], dtype=np.float32)

    # 我们 monkey-patch _evaluate_loss_with_instance 以便控制损失值
    original_eval_loss = trainer._evaluate_loss_with_instance

    # 模拟不同模型返回不同损失
    test_losses = [0.9, 0.3, 0.7]  # 第二个模型损失最低
    loss_iter = cycle(test_losses)

    def mock_eval_loss(instance, x, t):
        return next(loss_iter)

    trainer._evaluate_loss_with_instance = mock_eval_loss

    # 临时修改 accuracy 评估也返回固定值
    def mock_eval_acc(instance, x, t):
        return 1.0 - trainer._evaluate_loss_with_instance(instance, x, t)  # 简化

    trainer._evaluate_accuracy_with_instance = mock_eval_acc

    result = trainer._phase_diverse_initialization(X, y)

    print(f"\n🎯 Expected best candidate: ID=1 (loss=0.3), Got ID={result['best_model'].id if hasattr(result['best_model'], 'id') else 'unknown'}")
    print(f"📊 Returned best loss: {result['best_loss']:.4f}")

    # 重置方法
    trainer._evaluate_loss_with_instance = original_eval_loss

    # 手动验证：最小损失应为 0.3
    assert abs(result['best_loss'] - 0.3) < 1e-5, "Best model selection failed!"
    print("✅ Best model selection passed!")


# =============== 主程序 ===============
if __name__ == "__main__":
    print("🚀 Starting GraphEvoTrainer Phase 1 Tests...")

    test_phase1_fixed_mode()
    test_phase1_list_mode()
    test_phase1_best_selection()

    print("\n🎉 All Phase 1 tests completed successfully!")