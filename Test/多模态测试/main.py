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

import time

def slow_exp_sum(a, b):
    """逐样本：计算 exp(a) + exp(b)"""
    return np.exp(a) + np.exp(b)  # 注意：这里用 np.exp 但仍是逐样本！

def fast_exp_sum(a, b):
    """向量化：批量计算 exp(a) + exp(b)"""
    return np.exp(a) + np.exp(b)  # a, b 是 (N,) 数组

# ======================
# 2. 注册方法
# ======================
methods = {
    "slow_exp": {
        "function": slow_exp_sum,
        "input_count": 2,
        "output_count": 1,
        "input_types": ["scalar", "scalar"],
        "output_types": ["scalar"],
        "vectorized": False  # 逐样本
    },
    "fast_exp": {
        "function": fast_exp_sum,
        "input_count": 2,
        "output_count": 1,
        "input_types": ["scalar", "scalar"],
        "output_types": ["scalar"],
        "vectorized": True   # 向量化
    }
}

# ======================
# 3. 构建纯数值输入（无字符串/图像）
# ======================
N = 100000  # 大样本量
values = np.random.rand(N, 2)  # 纯数值矩阵
input_types_list = ["scalar", "scalar"]

# ======================
# 4. 测试函数（纯数值图）
# ======================
def test_pure_vectorized(method_name, description):
    af = AdaptoFlux(input_types_list=input_types_list)
    af.set_methods(methods)
    G = af.graph
    
    # 清空默认边
    G.remove_edges_from(list(G.in_edges("collapse")))
    
    # 添加纯数值节点
    G.add_node("exp_node", method_name=method_name, layer=1)
    G.add_edge("root", "exp_node", output_index=0, data_coord=0, data_type="scalar")
    G.add_edge("root", "exp_node", output_index=1, data_coord=1, data_type="scalar")
    G.add_edge("exp_node", "collapse", output_index=0, data_coord=0, data_type="scalar")
    
    # 简单 collapse
    af.set_custom_collapse(lambda x: x[0])
    
    # 计时
    start = time.time()
    results = af.infer_with_graph(values)
    elapsed = time.time() - start
    
    # 验证
    expected = np.exp(values[:, 0]) + np.exp(values[:, 1])
    assert np.allclose(results, expected, atol=1e-6)
    
    print(f"{description}: {elapsed:.4f} 秒 (N={N})")
    return elapsed

# ======================
# 5. 执行测试
# ======================
if __name__ == "__main__":
    print("🚀 测试纯数值向量化加速...\n")
    
    time_slow = test_pure_vectorized("slow_exp", "逐样本 (vectorized=False)")
    time_fast = test_pure_vectorized("fast_exp", "向量化 (vectorized=True)")
    
    speedup = time_slow / time_fast
    print(f"\n🔥 加速比: {speedup:.2f}x")
    
    if speedup > 10:
        print("✅ 向量化成功！性能显著提升。")
    else:
        print("⚠️ 仍未加速（检查 NumPy 安装或操作复杂度）")

import numpy as np
import time

N = 10000

# 逐样本
a = np.random.rand(N)
b = np.random.rand(N)

start = time.time()
result1 = [np.exp(ai) + np.exp(bi) for ai, bi in zip(a, b)]
print("逐样本:", time.time() - start)

# 向量化
start = time.time()
result2 = np.exp(a) + np.exp(b)
print("向量化:", time.time() - start)

print("加速比:", (time.time() - start) / (time.time() - start))  # 伪代码