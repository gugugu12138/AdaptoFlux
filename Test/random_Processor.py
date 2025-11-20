import random
from copy import deepcopy
import numpy as np
import networkx as nx
from ATF import PathGenerator

def test_type_matching_system():
    """测试类型检测系统是否正常工作"""
    
    # 创建测试图结构
    graph = nx.MultiDiGraph()
    
    # 添加 collapse 节点和带类型信息的边
    for i in range(10):
        if i < 3:
            # 添加3个类型为 'int' 的数据
            graph.add_edge(f"node_{i}", "collapse", data_type="int", value=i)
        elif i < 6:
            # 添加3个类型为 'float' 的数据
            graph.add_edge(f"node_{i}", "collapse", data_type="float", value=float(i))
        else:
            # 添加4个类型为 'str' 的数据
            graph.add_edge(f"node_{i}", "collapse", data_type="str", value=f"str_{i}")

    # 定义带输入类型的方法池
    methods = {
        "add_integers": {
            "input_count": 2, 
            "input_types": ["int", "int"],
            "output_count": 1
        },
        "multiply_floats": {
            "input_count": 2, 
            "input_types": ["float", "float"], 
            "output_count": 1
        },
        "concat_strings": {
            "input_count": 2, 
            "input_types": ["str", "str"], 
            "output_count": 1
        },
        "mixed_operation": {
            "input_count": 3, 
            "input_types": ["int", "float", "str"], 
            "output_count": 1
        },
        "single_int": {
            "input_count": 1, 
            "input_types": ["int"], 
            "output_count": 1
        }
    }

    print("=== 测试类型匹配系统 ===")
    print(f"图中 collapse 边数量: {len(graph.in_edges('collapse', data=True))}")
    
    # 测试基本类型匹配
    print("\n--- 测试1: 基本类型匹配 (remove_from_pool=False) ---")
    path_gen = PathGenerator(graph, methods, remove_from_pool=False)
    result = path_gen.process_random_method(shuffle_indices=False)
    
    print("结果分析:")
    print(f"  - index_map 长度: {len(result['index_map'])}")
    print(f"  - valid_groups: {result['valid_groups']}")
    print(f"  - unmatched: {result['unmatched']}")
    
    # 检查类型匹配是否正确
    collapse_edges = list(graph.in_edges("collapse", data=True))
    type_analysis = {}
    for idx, (u, v, data) in enumerate(collapse_edges):
        data_type = data.get("data_type")
        if idx in result['index_map']:
            assigned_method = result['index_map'][idx]['method']
            type_analysis[idx] = {
                'data_type': data_type,
                'assigned_method': assigned_method,
                'group': result['index_map'][idx]['group']
            }
    
    print("\n类型分配详情:")
    for idx, info in type_analysis.items():
        print(f"  索引 {idx}: 数据类型={info['data_type']}, 分配方法={info['assigned_method']}, 组={info['group']}")
    
    # 测试启用 remove_from_pool
    print("\n--- 测试2: 启用 remove_from_pool ---")
    path_gen_remove = PathGenerator(graph, methods, remove_from_pool=True)
    result_remove = path_gen_remove.process_random_method(shuffle_indices=False)
    
    print("启用 remove_from_pool 的结果:")
    print(f"  - index_map 长度: {len(result_remove['index_map'])}")
    print(f"  - valid_groups: {result_remove['valid_groups']}")
    print(f"  - unmatched: {result_remove['unmatched']}")
    
    # 测试相同类型输入优化
    print("\n--- 测试3: 相同类型输入优化 (optimize_same_type_inputs=True) ---")
    path_gen_optimize = PathGenerator(graph, methods, 
                                    remove_from_pool=False, 
                                    optimize_same_type_inputs=True)
    result_optimize = path_gen_optimize.process_random_method(shuffle_indices=False)
    
    print("启用优化的结果:")
    print(f"  - index_map 长度: {len(result_optimize['index_map'])}")
    print(f"  - valid_groups: {result_optimize['valid_groups']}")
    print(f"  - unmatched: {result_optimize['unmatched']}")
    
    # 测试类型不匹配情况
    print("\n--- 测试4: 类型不匹配测试 ---")
    mismatch_methods = {
        "only_ints": {
            "input_count": 2, 
            "input_types": ["int", "int"], 
            "output_count": 1
        }
    }
    path_gen_mismatch = PathGenerator(graph, mismatch_methods)
    result_mismatch = path_gen_mismatch.process_random_method(shuffle_indices=False)
    
    print("只有整数方法的结果:")
    print(f"  - index_map 长度: {len(result_mismatch['index_map'])}")
    print(f"  - valid_groups: {result_mismatch['valid_groups']}")
    print(f"  - unmatched: {result_mismatch['unmatched']}")
    
    # 统计各类型数据数量
    type_counts = {}
    for u, v, data in graph.in_edges("collapse", data=True):
        data_type = data.get("data_type")
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
    
    print(f"\n数据类型统计: {type_counts}")
    
    # 验证类型匹配逻辑
    print("\n--- 验证类型匹配逻辑 ---")
    expected_matches = {
        "int": 3,  # 3个int数据
        "float": 3,  # 3个float数据  
        "str": 4   # 4个str数据
    }
    print(f"预期数据分布: {expected_matches}")
    
    actual_matches = {}
    for idx in result['index_map']:
        edge_data = collapse_edges[idx][2]
        data_type = edge_data.get("data_type")
        assigned_method = result['index_map'][idx]['method']
        if assigned_method != "unmatched":
            if data_type not in actual_matches:
                actual_matches[data_type] = 0
            actual_matches[data_type] += 1
    
    print(f"实际被匹配的数据类型分布: {actual_matches}")
    
    print("\n=== 类型匹配系统测试完成 ===")

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 创建空图
    empty_graph = nx.MultiDiGraph()
    empty_graph.add_node("collapse")
    
    methods = {
        "test_method": {
            "input_count": 2, 
            "input_types": ["int", "int"], 
            "output_count": 1
        }
    }
    
    path_gen = PathGenerator(empty_graph, methods)
    try:
        result = path_gen.process_random_method()
        print("空图测试结果:", result)
    except Exception as e:
        print(f"空图测试异常: {e}")
    
    # 创建只有部分类型匹配的数据
    partial_graph = nx.MultiDiGraph()
    for i in range(3):
        partial_graph.add_edge(f"node_{i}", "collapse", data_type="int", value=i)
    
    path_gen_partial = PathGenerator(partial_graph, methods)
    result_partial = path_gen_partial.process_random_method()
    print("部分类型匹配测试结果:", result_partial)

if __name__ == "__main__":
    test_type_matching_system()
    test_edge_cases()