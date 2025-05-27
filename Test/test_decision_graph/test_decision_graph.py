import numpy as np
from ATF import AdaptoFlux

# 加载自定义方法
methods_path = "Test/test_decision_graph/logic_functions.py"

# 输入数据 (4 个样本)
values = np.array([
    [-2],   # 负数 & 偶数
    [0],    # 零
    [3],    # 正数 & 奇数
    [4]     # 正数 & 偶数
])

# 初始化模型
model = AdaptoFlux(values=values, labels=np.zeros((len(values),)), methods_path=methods_path)

# 导入方法
model.import_methods_from_file()

# 清空 root -> collapse 的默认边
model.graph.remove_edges_from(list(model.graph.in_edges("collapse")))

# 添加节点并绑定方法
model.graph.add_node("classify", method_name="classify_number")

# 添加边：root → classify
model.graph.add_edge("root", "classify", output_index=0, data_coord=0)  # 分类结果输出到 collapse

# 添加边：classify 输出的三个分支分别连接到 collapse 的不同位置
model.graph.add_edge("classify", "collapse", output_index=0, data_coord=0)  # 负数分支
model.graph.add_edge("classify", "collapse", output_index=1, data_coord=1)  # 零分支
model.graph.add_edge("classify", "collapse", output_index=2, data_coord=2)  # 正数分支

def custom_collapse(arr):
    return arr

model.add_collapse_method(custom_collapse)  # 注册自定义坍缩函数

# 执行推理
result = model.infer_with_graph(values)

print("推理结果：")
print(result)