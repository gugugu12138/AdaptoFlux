import unittest
import numpy as np
from collections import Counter
import math

from ATF import AdaptoFlux, CollapseMethod

class TestGraphEntropy(unittest.TestCase):

    def test_get_graph_entropy(self):
        # 初始化 AdaptoFlux 实例
        values = np.random.rand(10, 5)  # 10 个样本，5 个特征
        labels = np.random.randint(0, 2, size=10)
        dummy_methods_path = "methods.py"

        flux = AdaptoFlux(values, labels, dummy_methods_path)

        # 添加一些方法到 methods 字典（否则 method_name 不会被识别）
        flux.add_method("add", lambda x: x, input_count=2, output_count=1)
        flux.add_method("multiply", lambda x: x, input_count=2, output_count=1)
        flux.add_method("subtract", lambda x: x, input_count=2, output_count=1)

        # 清除默认的 root -> collapse 边以便重新构造
        flux.graph.remove_edges_from(list(flux.graph.in_edges("collapse")))

        # 添加带有 method_name 的节点和边
        layer_nodes = [
            ("layer1_add", "add"),
            ("layer1_mult", "multiply"),
            ("layer1_sub", "subtract"),
            ("layer1_add2", "add"),  # 再加一个 add，增加频率
        ]

        for node_name, method in layer_nodes:
            flux.graph.add_node(node_name, method_name=method)
            flux.graph.add_edge("root", node_name)
            flux.graph.add_edge(node_name, "collapse")

        # 测试 get_graph_entropy()
        entropy = flux.get_graph_entropy()

        # 输出信息用于调试
        print(f"当前图中各方法出现次数: {flux.get_method_counter()}")
        print(f"计算得到的图结构熵值: {entropy}")

        # 验证熵值范围是否合理（例如在 0 到 log2(N_methods) 之间）
        method_counter = flux.get_method_counter()
        total = sum(method_counter.values())
        probabilities = [count / total for count in method_counter.values()]
        expected_entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        
        self.assertAlmostEqual(entropy, expected_entropy, delta=1e-6)

if __name__ == '__main__':
    unittest.main()