# Test/test_multimodal.py
import numpy as np
from ATF import AdaptoFlux

# 定义多模态方法（直接内联，不依赖外部文件）
def process_text(txt):
    """处理字符串：返回长度和大写"""
    return [len(txt), txt.upper()]

def process_image(img):
    """处理图像（模拟）：返回均值和形状"""
    return [float(img.mean()), img.shape]

def fuse_features(num, txt_len, txt_upper, img_mean, img_shape):
    """融合所有特征，返回结构化结果"""
    return {
        "number": num,
        "text_length": txt_len,
        "text_upper": txt_upper,
        "image_mean": img_mean,
        "image_shape": img_shape
    }

# 手动注册方法（绕过 methods_path）
methods = {
    "process_text": {
        "function": process_text,
        "input_count": 1,
        "output_count": 2,
        "input_types": ["str"],
        "output_types": ["int", "str"],
        "group": "default",
        "weight": 1.0,
        "vectorized": False
    },
    "process_image": {
        "function": process_image,
        "input_count": 1,
        "output_count": 2,
        "input_types": ["image"],
        "output_types": ["float", "tuple"],
        "group": "default",
        "weight": 1.0,
        "vectorized": False
    },
    "fuse_features": {
        "function": fuse_features,
        "input_count": 5,
        "output_count": 1,
        "input_types": ["scalar", "int", "str", "float", "tuple"],
        "output_types": ["dict"],
        "group": "default",
        "weight": 1.0,
        "vectorized": False
    }
}

# 构造多模态输入：每个样本 = [数值, 字符串, 图像]
values = [
    [42, "hello", np.random.rand(8, 8, 3)],
    [-10, "world", np.random.rand(16, 16, 1)]
]

# 显式声明每列的语义类型
input_types_list = ["scalar", "str", "image"]

# 初始化 AdaptoFlux（不传 values/labels 以避免自动推断）
af = AdaptoFlux(input_types_list=input_types_list)
af.set_methods(methods)

# === 手动构建图 ===
G = af.graph

# 清空默认边（root → collapse）
G.remove_edges_from(list(G.in_edges("collapse")))

# 添加处理节点
G.add_node("text_proc", method_name="process_text", layer=1)
G.add_node("img_proc", method_name="process_image", layer=1)
G.add_node("fuser", method_name="fuse_features", layer=2)

# root → 处理节点
G.add_edge("root", "text_proc", output_index=1, data_coord=1, data_type="str")      # 字符串列
G.add_edge("root", "img_proc", output_index=2, data_coord=2, data_type="image")    # 图像列

# 处理节点 → fuser
# fuser 的输入顺序必须匹配 fuse_features(num, txt_len, txt_upper, img_mean, img_shape)

# 1. 原始数值 (num) → 参数 0
G.add_edge("root", "fuser", output_index=0, data_coord=0, data_type="scalar")

# 2. text_len → 参数 1
G.add_edge("text_proc", "fuser", output_index=0, data_coord=1, data_type="int")
# 3. text_upper → 参数 2
G.add_edge("text_proc", "fuser", output_index=1, data_coord=2, data_type="str")

# 4. img_mean → 参数 3
G.add_edge("img_proc", "fuser", output_index=0, data_coord=3, data_type="float")
# 5. img_shape → 参数 4
G.add_edge("img_proc", "fuser", output_index=1, data_coord=4, data_type="tuple")

# fuser → collapse
G.add_edge("fuser", "collapse", output_index=0, data_coord=0, data_type="dict")

# 自定义 collapse：直接返回结果（不聚合）
af.set_custom_collapse(lambda x: x[0])  # x 是 [dict]，取第一个

# 执行推理
results = af.infer_with_graph(values)

print("✅ 多模态推理成功！结果示例：")
for i, res in enumerate(results):
    print(f"\n样本 {i}:")
    print(f"  number: {res['number']}")
    print(f"  text_length: {res['text_length']}")
    print(f"  text_upper: {res['text_upper']}")
    print(f"  image_mean: {res['image_mean']:.4f}")
    print(f"  image_shape: {res['image_shape']}")

# Test/test_true_vectorized_speedup.py
import numpy as np
import time
import math
from ATF import AdaptoFlux

# ✅ 逐样本方法（纯 Python，无 NumPy 开销）
def slow_math(a, b):
    return math.sin(math.cos(math.exp(a * b)))

# ✅ 向量化方法（纯 NumPy，无 list 拷贝）
def fast_math(a, b):
    return np.sin(np.cos(np.exp(a * b)))

methods = {
    "slow_math": {
        "function": slow_math,
        "input_count": 2,
        "output_count": 1,
        "input_types": ["scalar", "scalar"],
        "output_types": ["scalar"],
        "vectorized": False
    },
    "fast_math": {
        "function": fast_math,
        "input_count": 2,
        "output_count": 1,
        "input_types": ["scalar", "scalar"],
        "output_types": ["scalar"],
        "vectorized": True
    }
}

def test_vectorized_speedup(method_name, description, N=200000):
    # ⚠️ 关键：直接使用 list 输入（避免 array → list 拷贝）
    values = [[float(x), float(y)] for x, y in zip(
        np.random.rand(N) * 0.5,  # 避免 exp 溢出
        np.random.rand(N) * 0.5
    )]
    
    af = AdaptoFlux(input_types_list=["scalar", "scalar"])
    af.set_methods(methods)
    G = af.graph
    
    G.remove_edges_from(list(G.in_edges("collapse")))
    G.add_node("math_node", method_name=method_name, layer=1)
    G.add_edge("root", "math_node", output_index=0, data_coord=0, data_type="scalar")
    G.add_edge("root", "math_node", output_index=1, data_coord=1, data_type="scalar")
    G.add_edge("math_node", "collapse", output_index=0, data_coord=0, data_type="scalar")
    
    af.set_custom_collapse(lambda x: x[0])
    
    # 计时
    start = time.perf_counter()
    results = af.infer_with_graph(values)
    elapsed = time.perf_counter() - start
    
    # 验证（前 10 个样本）
    test_vals = values[:10]
    if method_name == "slow_math":
        expected = [slow_math(a, b) for a, b in test_vals]
    else:
        a_arr = np.array([a for a, b in test_vals])
        b_arr = np.array([b for a, b in test_vals])
        expected = fast_math(a_arr, b_arr).tolist()
    
    assert np.allclose(results[:10], expected, atol=1e-6), f"验证失败: {method_name}"
    
    print(f"{description}: {elapsed:.4f} 秒 (N={N})")
    return elapsed

if __name__ == "__main__":
    print("🚀 测试真实向量化加速（避免拷贝开销）...\n")
    
    N = 200000  # 足够大以凸显加速，但内存友好
    
    time_slow = test_vectorized_speedup("slow_math", "逐样本 (vectorized=False)", N)
    time_fast = test_vectorized_speedup("fast_math", "向量化 (vectorized=True)", N)
    
    speedup = time_slow / time_fast
    print(f"\n🔥 加速比: {speedup:.2f}x")
    
    if speedup > 30:
        print("✅ 向量化成功！性能显著提升。")
    else:
        print("⚠️ 加速不足（检查计算复杂度或 NumPy 安装）")