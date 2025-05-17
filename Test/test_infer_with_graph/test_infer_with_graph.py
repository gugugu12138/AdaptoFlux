import numpy as np
from ATF import AdaptoFlux, CollapseMethod

# 加载自定义方法
methods_path = "Test/test_infer_with_graph/methods_test.py"

# 模拟输入数据 (2个样本，每个有2个特征)
values = np.array([
    [1, 2],
    [3, 4]
])

# 初始化模型
model = AdaptoFlux(values=values, labels=np.array([0, 0]), methods_path=methods_path)

# 导入方法
model.import_methods_from_file()

# 清空 root -> collapse 的默认边
model.graph.remove_edges_from(list(model.graph.in_edges("collapse")))

# 自定义图结构
# 添加两个中间节点
model.graph.add_node("add_node1")
model.graph.nodes["add_node1"]["method_name"] = "add"

model.graph.add_node("add_node2")
model.graph.nodes["add_node2"]["method_name"] = "add"


model.graph.add_node("mul_node")
model.graph.nodes["mul_node"]["method_name"] = "multiply"

# 添加边
model.graph.add_edge("root", "add_node1", output_index=0, data_coord=0)   # root 特征0 → add_node
model.graph.add_edge("root", "add_node2", output_index=1, data_coord=1)   # root 特征1 → add_node
model.graph.add_edge("add_node1", "mul_node", output_index=0, data_coord=0)             # add_node 输出 → mul_node
model.graph.add_edge("add_node2", "mul_node", output_index=0, data_coord=1)             # add_node 输出 → mul_node
model.graph.add_edge("mul_node", "collapse", output_index=0, data_coord=0)            # mul_node 输出 → collapse

# 设置坍缩方式为 SUM
model.collapse_method = CollapseMethod.SUM

# 执行推理
result = model.infer_with_graph(values)

print("推理结果：")
print(result)