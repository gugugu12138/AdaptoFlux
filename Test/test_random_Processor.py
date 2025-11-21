import random
from copy import deepcopy
import numpy as np
import networkx as nx

# 假设您的类在当前文件中，所以直接从当前文件导入
# 如果您的类在其他文件中，请相应调整导入语句
from ATF import PathGenerator  # 修改为实际的导入路径

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
    print("\n--- 测试3: 相同类型输入优化 (optimize_same_type_inputs=True)目前该功能未实现，留出接口 ---")
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

def test_type_mapping_system():
    """测试类型映射系统是否正常工作"""
    
    # 创建测试图结构 - 使用映射表中的类型
    graph = nx.MultiDiGraph()
    
    # 添加 collapse 节点和带类型信息的边
    for i in range(8):
        if i < 2:
            # 添加2个类型为 'int32' 的数据 (应该映射到 scalar)
            graph.add_edge(f"node_{i}", "collapse", data_type="int32", value=i)
        elif i < 4:
            # 添加2个类型为 'float64' 的数据 (应该映射到 scalar)
            graph.add_edge(f"node_{i}", "collapse", data_type="float64", value=float(i))
        elif i < 6:
            # 添加2个类型为 'bool' 的数据 (应该映射到 scalar)
            graph.add_edge(f"node_{i}", "collapse", data_type="bool", value=True if i % 2 == 0 else False)
        else:
            # 添加2个类型为 'scalar' 的数据 (基础类型)
            graph.add_edge(f"node_{i}", "collapse", data_type="scalar", value=i)

    # 定义只接受 'scalar' 类型输入的方法池
    methods = {
        "scalar_operation": {
            "input_count": 2, 
            "input_types": ["scalar", "scalar"],  # 只接受 scalar 类型
            "output_count": 1
        },
        "single_scalar": {
            "input_count": 1, 
            "input_types": ["scalar"],  # 只接受 scalar 类型
            "output_count": 1
        }
    }

    print("\n=== 测试类型映射系统 ===")
    print(f"图中 collapse 边数量: {len(graph.in_edges('collapse', data=True))}")
    
    # 显示所有数据的类型
    collapse_edges = list(graph.in_edges("collapse", data=True))
    print("\n数据类型分布:")
    for idx, (u, v, data) in enumerate(collapse_edges):
        print(f"  索引 {idx}: {data['data_type']} -> {data['value']}")
    
    print("\n--- 测试1: 使用默认映射表 (int32, float64, bool 映射到 scalar) ---")
    path_gen = PathGenerator(graph, methods, remove_from_pool=False)
    result = path_gen.process_random_method(shuffle_indices=False)
    
    print("结果分析:")
    print(f"  - index_map 长度: {len(result['index_map'])}")
    print(f"  - valid_groups: {result['valid_groups']}")
    print(f"  - unmatched: {result['unmatched']}")
    
    # 详细分析每个索引的类型匹配情况
    type_analysis = {}
    for idx, (u, v, data) in enumerate(collapse_edges):
        data_type = data.get("data_type")
        if idx in result['index_map']:
            assigned_method = result['index_map'][idx]['method']
            type_analysis[idx] = {
                'original_type': data_type,
                'assigned_method': assigned_method,
                'group': result['index_map'][idx]['group']
            }
    
    print("\n类型匹配详情:")
    for idx, info in type_analysis.items():
        print(f"  索引 {idx}: 原始类型={info['original_type']}, 分配方法={info['assigned_method']}, 组={info['group']}")
    
    # 验证所有非 'scalar' 类型的数据是否都被成功映射
    mapped_count = 0
    for idx, info in type_analysis.items():
        if info['original_type'] in ['int32', 'float64', 'bool'] and info['assigned_method'] != 'unmatched':
            mapped_count += 1
            print(f"    ✓ {info['original_type']} -> {info['assigned_method']} (成功映射)")
    
    print(f"\n映射统计: {mapped_count}/6 非 scalar 类型数据被成功映射")
    
    # 测试自定义映射表
    print("\n--- 测试2: 使用自定义映射表 ---")
    custom_mapping = {
        "numeric": ["int32", "int64", "float32", "float64"],
        "logical": ["bool", "int8"]  # bool 映射到 logical
    }
    
    path_gen_custom = PathGenerator(graph, methods, 
                                  remove_from_pool=False,
                                  type_equivalence_map=custom_mapping)
    
    # 修改方法池以测试自定义映射
    custom_methods = {
        "numeric_operation": {
            "input_count": 2,
            "input_types": ["numeric", "numeric"],
            "output_count": 1
        },
        "logical_operation": {
            "input_count": 1,
            "input_types": ["logical"],
            "output_count": 1
        }
    }
    
    # 创建一个包含 numeric 和 logical 类型数据的图
    custom_graph = nx.MultiDiGraph()
    for i in range(6):
        if i < 3:
            custom_graph.add_edge(f"node_{i}", "collapse", data_type="int32", value=i)
        else:
            custom_graph.add_edge(f"node_{i}", "collapse", data_type="bool", value=True)
    
    path_gen_custom = PathGenerator(custom_graph, custom_methods, 
                                  remove_from_pool=False,
                                  type_equivalence_map=custom_mapping)
    result_custom = path_gen_custom.process_random_method(shuffle_indices=False)
    
    print("自定义映射表结果:")
    print(f"  - valid_groups: {result_custom['valid_groups']}")
    print(f"  - unmatched: {result_custom['unmatched']}")
    
    # 详细分析自定义映射
    custom_collapse_edges = list(custom_graph.in_edges("collapse", data=True))
    custom_analysis = {}
    for idx, (u, v, data) in enumerate(custom_collapse_edges):
        data_type = data.get("data_type")
        if idx in result_custom['index_map']:
            assigned_method = result_custom['index_map'][idx]['method']
            custom_analysis[idx] = {
                'original_type': data_type,
                'assigned_method': assigned_method,
                'group': result_custom['index_map'][idx]['group']
            }
    
    print("\n自定义映射详情:")
    for idx, info in custom_analysis.items():
        print(f"  索引 {idx}: 原始类型={info['original_type']}, 分配方法={info['assigned_method']}, 组={info['group']}")
    
    # 测试映射表的双向性
    print("\n--- 测试3: 映射表双向性测试 ---")
    # 创建一个 scalar 类型方法，但使用 int32 数据
    scalar_methods = {
        "scalar_op": {
            "input_count": 1,
            "input_types": ["scalar"],  # 方法接受 scalar
            "output_count": 1
        }
    }
    
    # 使用 int32 数据测试 scalar 方法
    int_graph = nx.MultiDiGraph()
    for i in range(2):
        int_graph.add_edge(f"node_{i}", "collapse", data_type="int32", value=i)
    
    path_gen_scalar = PathGenerator(int_graph, scalar_methods)
    result_scalar = path_gen_scalar.process_random_method(shuffle_indices=False)
    
    print("scalar 方法匹配 int32 数据:")
    print(f"  - valid_groups: {result_scalar['valid_groups']}")
    print(f"  - unmatched: {result_scalar['unmatched']}")
    
    scalar_collapse_edges = list(int_graph.in_edges("collapse", data=True))
    scalar_analysis = {}
    for idx, (u, v, data) in enumerate(scalar_collapse_edges):
        data_type = data.get("data_type")
        if idx in result_scalar['index_map']:
            assigned_method = result_scalar['index_map'][idx]['method']
            scalar_analysis[idx] = {
                'original_type': data_type,
                'assigned_method': assigned_method,
                'group': result_scalar['index_map'][idx]['group']
            }
    
    print("\n双向映射详情:")
    for idx, info in scalar_analysis.items():
        print(f"  索引 {idx}: 原始类型={info['original_type']}, 分配方法={info['assigned_method']}, 组={info['group']}")
    
    # 测试无映射情况
    print("\n--- 测试4: 无映射关系的类型测试 ---")
    no_map_graph = nx.MultiDiGraph()
    for i in range(2):
        no_map_graph.add_edge(f"node_{i}", "collapse", data_type="unknown_type", value=i)
    
    path_gen_no_map = PathGenerator(no_map_graph, methods)
    result_no_map = path_gen_no_map.process_random_method(shuffle_indices=False)
    
    print("无映射类型的结果:")
    print(f"  - valid_groups: {result_no_map['valid_groups']}")
    print(f"  - unmatched: {result_no_map['unmatched']}")
    
    # 统计映射效果
    print("\n--- 映射效果总结 ---")
    total_data = len(collapse_edges)
    matched_data = len([idx for idx in result['index_map'] if result['index_map'][idx]['method'] != 'unmatched'])
    print(f"总数据量: {total_data}")
    print(f"成功匹配: {matched_data}")
    print(f"匹配率: {matched_data/total_data*100:.1f}%")
    
    # 验证映射表是否正确加载
    print(f"\n当前映射表内容: {path_gen.type_equivalence_map}")
    
    print("\n=== 类型映射系统测试完成 ===")

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

def test_mapping_edge_cases():
    """测试映射系统的边界情况"""
    print("\n=== 测试映射系统边界情况 ===")
    
    # 测试空映射表
    print("\n--- 测试空映射表 ---")
    empty_mapping_graph = nx.MultiDiGraph()
    for i in range(2):
        empty_mapping_graph.add_edge(f"node_{i}", "collapse", data_type="int32", value=i)
    
    methods = {
        "test_method": {
            "input_count": 1,
            "input_types": ["int32"],  # 精确匹配
            "output_count": 1
        }
    }
    
    path_gen_empty = PathGenerator(empty_mapping_graph, methods, 
                                 type_equivalence_map={})
    result_empty = path_gen_empty.process_random_method()
    print("空映射表结果:", result_empty)
    
    # 测试多重映射
    print("\n--- 测试多重映射 ---")
    multi_map_graph = nx.MultiDiGraph()
    for i in range(4):
        if i < 2:
            multi_map_graph.add_edge(f"node_{i}", "collapse", data_type="int32", value=i)
        else:
            multi_map_graph.add_edge(f"node_{i}", "collapse", data_type="float64", value=float(i))
    
    multi_methods = {
        "multi_op": {
            "input_count": 2,
            "input_types": ["scalar", "scalar"],
            "output_count": 1
        }
    }
    
    # 自定义映射：int32 映射到 both，float64 也映射到 both
    complex_mapping = {
        "both": ["int32", "float64"]
    }
    
    path_gen_multi = PathGenerator(multi_map_graph, multi_methods,
                                 type_equivalence_map=complex_mapping)
    result_multi = path_gen_multi.process_random_method()
    print("多重映射结果:", result_multi)
    
    multi_collapse_edges = list(multi_map_graph.in_edges("collapse", data=True))
    multi_analysis = {}
    for idx, (u, v, data) in enumerate(multi_collapse_edges):
        data_type = data.get("data_type")
        if idx in result_multi['index_map']:
            assigned_method = result_multi['index_map'][idx]['method']
            multi_analysis[idx] = {
                'original_type': data_type,
                'assigned_method': assigned_method,
                'group': result_multi['index_map'][idx]['group']
            }
    
    print("\n多重映射详情:")
    for idx, info in multi_analysis.items():
        print(f"  索引 {idx}: {info['original_type']} -> {info['assigned_method']} (组: {info['group']})")

def test_weighted_selection():
    """测试加权随机选择功能"""
    print("\n=== 测试加权选择功能 ===")
    
    # 创建测试图
    graph = nx.MultiDiGraph()
    for i in range(6):
        graph.add_edge(f"node_{i}", "collapse", data_type="int", value=i)
    
    # 定义带权重的方法池
    methods = {
        "high_weight_method": {
            "input_count": 1,
            "input_types": ["int"],
            "output_count": 1,
            "weight": 10.0  # 高权重
        },
        "low_weight_method": {
            "input_count": 1,
            "input_types": ["int"],
            "output_count": 1,
            "weight": 1.0   # 低权重
        },
        "medium_weight_method": {
            "input_count": 1,
            "input_types": ["int"],
            "output_count": 1,
            "weight": 5.0   # 中等权重
        }
    }
    
    print("方法权重分布:")
    for method_name, method_info in methods.items():
        print(f"  {method_name}: weight={method_info.get('weight', 1.0)}")
    
    # 运行多次测试以验证权重效果
    print("\n--- 运行10次测试验证权重分布 ---")
    method_counts = {}
    for i in range(10):
        path_gen = PathGenerator(graph, methods, remove_from_pool=False)
        result = path_gen.process_random_method(shuffle_indices=False)
        
        for idx, info in result['index_map'].items():
            method_name = info['method']
            if method_name != 'unmatched':
                method_counts[method_name] = method_counts.get(method_name, 0) + 1
    
    print("方法选择计数 (10次运行, 6个数据点):")
    for method_name, count in method_counts.items():
        print(f"  {method_name}: {count} 次")
    
    # 验证高权重方法被选择的次数应该最多
    high_weight_count = method_counts.get('high_weight_method', 0)
    low_weight_count = method_counts.get('low_weight_method', 0)
    medium_weight_count = method_counts.get('medium_weight_method', 0)
    
    print(f"\n权重效果验证:")
    print(f"  高权重方法: {high_weight_count} 次")
    print(f"  中等权重方法: {medium_weight_count} 次") 
    print(f"  低权重方法: {low_weight_count} 次")
    
    if high_weight_count >= medium_weight_count >= low_weight_count:
        print("  ✓ 权重选择工作正常")
    else:
        print("  ⚠ 权重选择可能存在问题")

def test_replace_functionality():
    """测试替换功能"""
    print("\n=== 测试替换功能 ===")
    
    # 创建测试图
    graph = nx.MultiDiGraph()
    for i in range(8):
        graph.add_edge(f"node_{i}", "collapse", data_type="int", value=i)
    
    # 定义方法池
    methods = {
        "int_operation": {
            "input_count": 2,
            "input_types": ["int", "int"],
            "output_count": 1
        },
        "single_int": {
            "input_count": 1,
            "input_types": ["int"],
            "output_count": 1
        }
    }
    
    path_gen = PathGenerator(graph, methods, remove_from_pool=False)
    initial_result = path_gen.process_random_method(shuffle_indices=False)
    
    print("初始结果:")
    print(f"  - valid_groups: {initial_result['valid_groups']}")
    print(f"  - unmatched: {initial_result['unmatched']}")
    
    # 测试替换功能 - 替换2个元素
    print("\n--- 替换2个元素 ---")
    try:
        replaced_result = path_gen.replace_random_elements(initial_result, n=2, shuffle_indices=False)
        print("替换后结果:")
        print(f"  - valid_groups: {replaced_result['valid_groups']}")
        print(f"  - unmatched: {replaced_result['unmatched']}")
        
        # 验证替换后组的结构是否合理
        total_indices = sum(len(group) for groups in replaced_result['valid_groups'].values() for group in groups)
        unmatched_count = sum(len(group) for group in replaced_result['unmatched'])
        total_elements = total_indices + unmatched_count
        
        print(f"\n验证: 总元素数 = {total_elements} (应为8)")
        if total_elements == 8:
            print("  ✓ 替换后元素总数保持不变")
        else:
            print("  ⚠ 替换后元素总数异常")
            
    except Exception as e:
        print(f"替换功能异常: {e}")

if __name__ == "__main__":
    test_type_matching_system()
    test_type_mapping_system()
    test_edge_cases()
    test_mapping_edge_cases()
    test_weighted_selection()
    test_replace_functionality()