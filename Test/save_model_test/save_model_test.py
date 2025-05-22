import numpy as np
from ATF import AdaptoFlux, CollapseMethod

# 创建模拟数据
values = np.random.rand(100, 5)     # 100 个样本，每个样本有 5 个特征
labels = np.random.randint(0, 2, 100)  # 二分类标签

# 假设你有一个 methods.py 文件，里面定义了方法函数
methods_path = "methods.py"

# 实例化 AdaptoFlux 模型
model = AdaptoFlux(values, labels, methods_path=methods_path, collapse_method=CollapseMethod.SUM)

# 导入方法（假设 methods.py 中定义了 add、multiply 等方法）
model.import_methods_from_file()

# 随机生成路径并添加一层网络结构
result = model.process_random_method()
model.append_nx_layer(result)

# 再次添加一层
result2 = model.replace_random_elements(result, n=2)
model.append_nx_layer(result2)

# 打印当前图结构信息
print("当前图节点数:", len(model.graph.nodes))
print("当前图边数:", len(model.graph.edges))

# 保存模型（包含 .gexf 和 .gpickle 图结构）
model.save_model(folder="Test/save_model_test/test_model_output")

print("✅ 模型和图结构已保存到 Test/save_model_test/test_model_output")

# ----------------------------
# 加载保存的图结构进行验证
# ----------------------------

import networkx as nx

# 从 .gexf 加载图（可读性强，适合调试）
graph_gexf = nx.read_gexf("Test/save_model_test/test_model_output/graph.gexf")
print("📊 从 .gexf 加载的图节点数:", len(graph_gexf.nodes))
print("📊 从 .gexf 加载的图边数:", len(graph_gexf.edges))

import os
import networkx as nx
from networkx.readwrite import json_graph
import json

json_file_path = "Test/save_model_test/test_model_output/graph.json"

if os.path.exists(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 使用 node_link_graph 将 JSON 数据还原为图
    graph_json = json_graph.node_link_graph(data)

    print("📊 从 .json 加载的图节点数:", len(graph_json.nodes))
    print("📊 从 .json 加载的图边数:", len(graph_json.edges))
else:
    print(f"❌ 文件 {json_file_path} 不存在，请确认路径或先运行保存模型步骤")