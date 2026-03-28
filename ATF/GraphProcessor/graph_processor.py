import networkx as nx
import numpy as np
from collections import Counter
import math
from ..CollapseManager.collapse_functions import CollapseFunctionManager, CollapseMethod
import logging
from typing import List, Dict, Set, Optional, Tuple  # add Set here if missing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_ALREADY_WARNED_OLD_MODEL = False

class GraphProcessor:
    def __init__(self, graph: nx.MultiDiGraph, methods: dict, collapse_method=CollapseMethod.SUM):
        """
        初始化图处理器。
        
        参数:
            graph (nx.MultiDiGraph): 初始图结构
            methods (dict): 可用方法字典 {method_name: {"function": ..., "input_count": int, "output_count": int}}
            collapse_method (callable): 聚合函数，默认为 sum
        """
        self.discard_node_method_name = "null"
        self.graph = graph.copy() if graph else nx.MultiDiGraph()
        self.methods = methods
        self.layer = self.get_max_layer_from_graph()  # 记录当前层数（可选）

        self.collapse_manager = CollapseFunctionManager(method=collapse_method)

    def set_graph(self, new_graph):
        if not hasattr(new_graph, 'nodes') or not hasattr(new_graph, 'edges'):
            raise ValueError("new_graph 不是一个有效的图对象")
        self.graph = new_graph
        # 自动同步 layer 状态
        self.layer = self.get_max_layer_from_graph()
        print(f"图结构已更新，当前最大层数：{self.layer}。")

    def set_methods(self, new_methods):
        """更新 methods（函数字典或模块）"""
        if not isinstance(new_methods, dict) and not callable(new_methods):
            raise TypeError("methods 应该是一个函数字典或可调用对象")
        self.methods = new_methods
        print("Methods 已更新。")

    def append_nx_layer(self, result, discard_unmatched='to_discard', discard_node_method_name="null"):
        """
        向图中添加一层新节点。
        """
        import logging

        self.discard_node_method_name = discard_node_method_name
        self.layer += 1
        new_index_edge = 0
        method_counts = {method_name: 0 for method_name in self.methods}
        if discard_node_method_name not in method_counts:
            method_counts[discard_node_method_name] = 0

        v = "collapse"
        collapse_edges = list(self.graph.in_edges(v, data=True))

        # 转换 collapse_edges 为便于查找的映射：data_coord -> (u, data)
        coord_to_edge = {}
        for u, _, data in collapse_edges:
            coord = data.get('data_coord')
            if coord is None:
                raise ValueError(f"Edge to 'collapse' missing 'data_coord': {data}")
            if coord in coord_to_edge:
                raise ValueError(f"Duplicate data_coord {coord} found in collapse edges!")
            coord_to_edge[coord] = (u, data)

        # === 1. 处理 valid_groups ===
        for method_name, groups in result['valid_groups'].items():
            if method_name == discard_node_method_name:
                continue
            for group in groups:
                new_target_node = f"{self.layer}_{method_counts[method_name]}_{method_name}"
                self.graph.add_node(
                    new_target_node,
                    method_name=method_name,
                    layer=self.layer,
                    is_passthrough=False
                )

                # 按 group 的原始顺序分配 input_slot
                for input_slot, coord in enumerate(group):
                    if coord not in coord_to_edge:
                        raise ValueError(f"Coord {coord} in group not found in collapse edges!")
                    u, orig_data = coord_to_edge[coord]
                    self.graph.remove_edge(u, v)
                    # 新增 input_slot 字段，保留其他属性
                    new_edge_data = {**orig_data, 'input_slot': input_slot}
                    self.graph.add_edge(u, new_target_node, **new_edge_data)

                # 添加输出边（到 collapse）
                output_count = self.methods[method_name]["output_count"]
                for local_output_index in range(output_count):
                    self.graph.add_edge(
                        new_target_node, v,
                        output_index=local_output_index,
                        data_coord=new_index_edge,
                        data_type=self.methods[method_name]["output_types"][local_output_index]
                    )
                    new_index_edge += 1

                method_counts[method_name] += 1

        # === 2. 处理 unmatched (discard) ===
        unmatched_groups = result.get('unmatched', [])
        if unmatched_groups and discard_unmatched == 'to_discard':
            for group in unmatched_groups:
                node_name = f"{self.layer}_{method_counts[discard_node_method_name]}_{discard_node_method_name}"
                self.graph.add_node(
                    node_name,
                    method_name=discard_node_method_name,
                    layer=self.layer,
                    is_passthrough=True
                )

                # 按 group 顺序分配 input_slot（透传节点也需要 input_slot！）
                input_data_types = []
                for input_slot, coord in enumerate(group):
                    if coord not in coord_to_edge:
                        raise ValueError(f"Coord {coord} in unmatched group not found in collapse edges!")
                    u, orig_data = coord_to_edge[coord]
                    input_data_types.append(orig_data.get('data_type'))
                    self.graph.remove_edge(u, v)
                    new_edge_data = {**orig_data, 'input_slot': input_slot}
                    self.graph.add_edge(u, node_name, **new_edge_data)

                # 透传节点：输出边数量 = 输入边数量，一一对应
                for local_output_index, used_data_type in enumerate(input_data_types):
                    self.graph.add_edge(
                        node_name, v,
                        output_index=local_output_index,
                        data_coord=new_index_edge,
                        data_type=used_data_type
                    )
                    new_index_edge += 1

                method_counts[discard_node_method_name] += 1

        elif unmatched_groups and discard_unmatched == 'ignore':
            logging.warning(
                "discard_unmatched='ignore' 策略未经充分测试，可能存在未预见行为。"
                "如有问题，请联系开发者。"
            )
            logging.warning(
                "⚠️  UNTESTED PATH: discard_unmatched='ignore' is experimental. "
                "Data will be DROPPED (edges removed from 'collapse'). "
                "Use at your own risk."
            )
            # 扁平化 unmatched_coords
            unmatched_coords = set()
            for group in unmatched_groups:
                unmatched_coords.update(group)
            # 删除对应边（不创建新节点）
            for u, _, data in collapse_edges:
                if data.get('data_coord') in unmatched_coords:
                    if self.graph.has_edge(u, v):
                        self.graph.remove_edge(u, v)

        elif unmatched_groups:
            raise ValueError(f"未知的 discard_unmatched 值：{discard_unmatched}")

        return self.graph

    def remove_last_nx_layer(self):
        """
        删除图中的最后一层节点。
        """
        collapse_node = "collapse"
        incoming_edges = list(self.graph.in_edges(collapse_node, data=True))
        nodes_to_remove = set(u for u, v, d in incoming_edges)

        if not nodes_to_remove:
            print("没有可删除的层。")
            return

        input_edges_map = {
            node: list(self.graph.in_edges(node, data=True)) for node in nodes_to_remove
        }

        for node in nodes_to_remove:
            for prev_node, _, edge_data in input_edges_map[node]:
                self.graph.add_edge(prev_node, collapse_node, **edge_data)

        for node in nodes_to_remove:
            if node != "root":
                self.graph.remove_node(node)
            else:
                print(f"跳过删除节点：{node}（这是保留的根节点）")

        self.layer -= 1
        return self.graph


    # 该部分存在技术残留导致的全量中间缓存，作者有时间可能会改
    def infer_with_graph(self, values):
        """
        使用图结构对输入数据进行推理，支持任意对象（非仅数值）。
        
        参数:
            values: 
                - 若为 numpy array: 必须是 (N, D) shape，dtype 可为 object
                - 若为 list: 必须是 [[feat0, feat1, ...], ...] 的二维结构
        
        返回:
            collapsed_output: 1D numpy array of scalars (shape [N,])
        """
        import numpy as np

        # === 1. 标准化输入为 list of lists（样本 × 输入特征）===
        if isinstance(values, np.ndarray):
            if values.ndim != 2:
                raise ValueError(f"Input values must be 2D, got shape {values.shape}")
            # 转为 list of lists，保留对象引用
            input_samples = [list(row) for row in values]
        elif isinstance(values, list):
            if not all(isinstance(row, (list, tuple)) for row in values):
                raise ValueError("Each sample in values must be a list/tuple of features.")
            input_samples = [list(row) for row in values]
        else:
            raise TypeError("values must be a 2D numpy array or list of lists.")

        num_samples = len(input_samples)
        if num_samples == 0:
            return np.array([])

        # === 2. 初始化节点输出字典 ===
        node_outputs = {}
        node_outputs["root"] = input_samples  # list of lists

        # === 3. 拓扑排序（排除 root 和 collapse）===
        nodes_in_order = list(nx.topological_sort(self.graph))
        nodes_in_order = [n for n in nodes_in_order if n not in {"root", "collapse"}]

        # === 4. 逐节点执行 ===
        for node in nodes_in_order:
            node_data = self.graph.nodes[node]
            method_name = node_data.get("method_name")

            # === 处理 is_passthrough（兼容老模型）===
            if "is_passthrough" not in node_data:
                is_passthrough = (
                    method_name is None or 
                    (isinstance(method_name, str) and method_name.lower() == 'null') or
                    (isinstance(method_name, str) and method_name.lower() == 'unmatched')
                )
                global _ALREADY_WARNED_OLD_MODEL
                if not _ALREADY_WARNED_OLD_MODEL:
                    logger.warning(
                        f'当前使用的为老模型，节点 "{node}" 未显式标记 "is_passthrough"。'
                        f'推断其为 passthrough={is_passthrough}。建议重训练或回退版本。'
                        f'之后的版本可能取消对老版本的兼容'
                    )
                    _ALREADY_WARNED_OLD_MODEL = True
            else:
                is_passthrough = bool(node_data.get("is_passthrough", False))

            # === 收集所有输入特征（按样本对齐）===
            predecessors = list(self.graph.predecessors(node))
            if not predecessors:
                raise ValueError(f"Node '{node}' has no predecessors.")

            # === 收集所有输入特征（按 input_slot 对齐）===
            # === 收集所有输入特征 ===
            input_pairs = []  # 用于新逻辑：(input_slot, extracted)
            input_feature_lists_old = []  # 用于老逻辑：直接 append

            has_input_slot = None  # None=未确定, True=有, False=无

            for src in predecessors:
                edges_from_src = self.graph[src][node]  # MultiEdge dict
                for edge_key, edge_data in edges_from_src.items():
                    output_idx = edge_data.get("output_index")
                    input_slot = edge_data.get("input_slot")  # 可能为 None
                    src_output = node_outputs[src]

                    if output_idx is None:
                        extracted = [sample_output for sample_output in src_output]
                    else:
                        extracted = [sample_output[output_idx] for sample_output in src_output]

                    # 判断是否使用 input_slot
                    current_has_slot = input_slot is not None

                    if has_input_slot is None:
                        has_input_slot = current_has_slot
                    elif has_input_slot != current_has_slot:
                        raise ValueError(
                            f"Node '{node}': mixed use of 'input_slot' in incoming edges. "
                            "All edges must either have 'input_slot' or all must omit it."
                        )

                    if has_input_slot:
                        if not isinstance(input_slot, int):
                            raise ValueError(f"input_slot must be an integer, got {input_slot} (type {type(input_slot)})")
                        input_pairs.append((input_slot, extracted))
                    else:
                        input_feature_lists_old.append(extracted)

            # === 根据是否有 input_slot 选择路径 ===
            if has_input_slot:
                # --- 新逻辑：按 input_slot 排序 ---
                input_pairs.sort(key=lambda x: x[0])
                input_slots = [slot for slot, _ in input_pairs]
                expected_slots = list(range(len(input_pairs)))
                if input_slots != expected_slots:
                    raise ValueError(
                        f"Node '{node}' has inconsistent input_slot indices. "
                        f"Expected {expected_slots}, got {input_slots}."
                    )
                input_feature_lists = [data for _, data in input_pairs]
            else:
                # --- 老逻辑：保持边遍历顺序 ---
                input_feature_lists = input_feature_lists_old

            if is_passthrough:
                if len(input_feature_lists) != 1:
                    raise ValueError(f"Passthrough node '{node}' must have exactly one input, got {len(input_feature_lists)}")
                
                # input_feature_lists[0] is a list of N elements: [feat_sample0, feat_sample1, ..., feat_sampleN-1]
                extracted_per_sample = input_feature_lists[0]  # length = N
                
                # ✅ 正确格式：每个样本一个 list（即使只有一个输出）
                node_outputs[node] = [[feat] for feat in extracted_per_sample]
                continue

            # === 正常方法执行 ===
            if method_name not in self.methods:
                raise ValueError(f"Unknown method: {method_name}")

            method_info = self.methods[method_name]
            func = method_info["function"]
            expected_input_count = method_info["input_count"]
            expected_output_count = method_info["output_count"]
            is_vectorized = method_info.get("vectorized", False)

            if len(input_feature_lists) != expected_input_count:
                raise ValueError(
                    f"Node '{node}' method '{method_name}' expects {expected_input_count} inputs, "
                    f"but got {len(input_feature_lists)} from edges."
                )

            # 尝试向量化执行（仅当方法标记为 vectorized=True）
            node_output_samples = None
            if is_vectorized and num_samples > 1:
                try:
                    # === 尝试构建批量输入 ===
                    batched_inputs = []
                    for input_list in input_feature_lists:
                        # 检查是否所有元素类型一致且可堆叠
                        first = input_list[0]
                        
                        # 情况1: 全是标量（int/float/np.number）
                        if all(isinstance(x, (int, float, np.number)) for x in input_list):
                            batched = np.array(input_list)
                        # 情况2: 全是 numpy 数组且 shape 一致
                        elif all(isinstance(x, np.ndarray) for x in input_list):
                            shapes = [x.shape for x in input_list]
                            if all(s == shapes[0] for s in shapes):
                                batched = np.stack(input_list, axis=0)  # (N, ...)
                            else:
                                raise ValueError("Array shapes mismatch, cannot vectorize")
                        # 情况3: 其他类型（str, dict 等）→ 无法向量化
                        else:
                            raise ValueError("Non-numeric or mixed types, cannot vectorize")
                            
                        batched_inputs.append(batched)
                    
                    batched_outputs = func(*batched_inputs)  # 应返回 (N, output_count) 或 tuple of (N,)
                    
                    # === 标准化输出为 list of lists ===
                    if isinstance(batched_outputs, tuple):
                        # 多输出：每个是 (N,) 或 (N, ...)
                        if len(batched_outputs) != expected_output_count:
                            raise ValueError(f"Expected {expected_output_count} outputs, got {len(batched_outputs)}")
                        # 转置: [(N,), (N,)] → [(out0_s0, out1_s0), ...]
                        node_output_samples = []
                        for i in range(num_samples):
                            sample_outs = [batched_outputs[j][i] for j in range(expected_output_count)]
                            # 如果输出是数组，保留为数组（不强制标量）
                            node_output_samples.append(sample_outs)
                    else:
                        # 单输出或 (N, output_count)
                        if batched_outputs.ndim == 1:
                            # (N,) → 每个样本一个标量
                            if expected_output_count != 1:
                                raise ValueError(f"Expected {expected_output_count} outputs, but got 1D array")
                            node_output_samples = [[x] for x in batched_outputs]
                        elif batched_outputs.ndim == 2:
                            # (N, output_count)
                            if batched_outputs.shape[1] != expected_output_count:
                                raise ValueError(f"Output shape {batched_outputs.shape} mismatches output_count={expected_output_count}")
                            node_output_samples = batched_outputs.tolist()
                        else:
                            raise ValueError(f"Unsupported output ndim: {batched_outputs.ndim}")
                            
                except Exception as e:
                    # 回退到逐样本模式（保持兼容性）
                    logger.warning(
                        f"Vectorized execution failed for method '{method_name}' (inputs: {[type(x[0]) for x in input_feature_lists]}), "
                        f"fallback to sample-by-sample. Error: {e}"
                    )
                    is_vectorized = False  # 触发下方逐样本逻辑

            # === 回退：逐样本执行（原逻辑）===
            else:
                node_output_samples = []
                for sample_idx in range(num_samples):
                    sample_inputs = [input_list[sample_idx] for input_list in input_feature_lists]
                    try:
                        result = func(*sample_inputs)
                    except Exception as e:
                        raise RuntimeError(
                            f"Error in method '{method_name}' at sample {sample_idx}:\n"
                            f"  Inputs: {sample_inputs}\n"
                            f"  Error: {e}"
                        ) from e

                    if not isinstance(result, (list, tuple)):
                        result = [result]
                    result = list(result)

                    if len(result) != expected_output_count:
                        raise ValueError(
                            f"Method '{method_name}' returned {len(result)} outputs, "
                            f"but expected {expected_output_count} (output_count)."
                        )

                    node_output_samples.append(result)

            # 保存结果
            node_outputs[node] = node_output_samples

        # === 5. 聚合到 collapse 节点 ===
        collapse_inputs = []  # List[Tuple[global_coord, List[Any]]]
        for u, v, data in self.graph.in_edges("collapse", data=True):
            local_idx = data.get("output_index")
            global_coord = data.get("data_coord")
            if local_idx is None or global_coord is None:
                raise ValueError(f"Edge from {u} to collapse missing output_index or data_coord")

            src_output = node_outputs[u]  # list of lists
            feature_values = [sample_output[local_idx] for sample_output in src_output]
            collapse_inputs.append((global_coord, feature_values))

        if not collapse_inputs:
            raise ValueError("No inputs connected to 'collapse' node.")

        # 按 global_coord 排序以保证顺序一致
        collapse_inputs.sort(key=lambda x: x[0])
        all_features_per_sample = list(zip(*[feat_list for _, feat_list in collapse_inputs]))  # transpose

        # === 6. 应用 collapse 函数 ===
        collapsed_results = []
        for sample_features in all_features_per_sample:
            try:
                # collapse 接收 list，返回任意对象（标量、向量、字符串等）
                collapsed_val = self.collapse_manager.collapse(list(sample_features))
                collapsed_results.append(collapsed_val)
            except Exception as e:
                raise RuntimeError(
                    f"Error in collapse function with inputs {list(sample_features)}:\n{e}"
                ) from e

        # 返回结果列表（不强制转为 np.array）
        return np.array(collapsed_results)
    
    def infer_with_task_parallel(self, values, num_workers=4):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import queue
        import numpy as np
        import traceback

        # === 1. 标准化输入为 list of lists（与 infer_with_graph 一致）===
        if isinstance(values, np.ndarray):
            if values.ndim != 2:
                raise ValueError(f"Input values must be 2D, got shape {values.shape}")
            input_samples = [list(row) for row in values]
        elif isinstance(values, list):
            if not all(isinstance(row, (list, tuple)) for row in values):
                raise ValueError("Each sample in values must be a list/tuple of features.")
            input_samples = [list(row) for row in values]
        else:
            raise TypeError("values must be a 2D numpy array or list of lists.")

        # 初始化
        node_outputs = {"root": input_samples}
        lock = threading.Lock()
        in_degree_remaining = {}
        ready_queue = queue.Queue()

        # 构建拓扑依赖图，并初始化每个节点的待完成前驱数
        for node in self.graph.nodes:
            if node in ["root", "collapse"]:
                continue
            preds = list(self.graph.predecessors(node))
            in_degree_remaining[node] = len(preds)
            if len(preds) == 0:
                ready_queue.put(node)

        # 处理 root 的直接后继
        for succ in self.graph.successors("root"):
            if succ in in_degree_remaining:
                in_degree_remaining[succ] -= 1
                if in_degree_remaining[succ] == 0:
                    ready_queue.put(succ)

        # 检查 collapse 是否有输入
        collapse_in_edges = list(self.graph.in_edges("collapse"))
        if not collapse_in_edges:
            return []

        # 工作函数
        def process_node(node):
            try:
                with lock:
                    # === 收集所有输入特征（按输入槽组织）===
                    predecessors = list(self.graph.predecessors(node))
                    if not predecessors:
                        raise ValueError(f"Node '{node}' has no predecessors.")

                    # === 收集所有输入特征（按 input_slot 对齐）===
                    # === 收集所有输入特征 ===
                    input_pairs = []  # 用于新逻辑：(input_slot, extracted)
                    input_feature_lists_old = []  # 用于老逻辑：直接 append

                    has_input_slot = None  # None=未确定, True=有, False=无

                    for src in predecessors:
                        edges_from_src = self.graph[src][node]  # MultiEdge dict
                        for edge_key, edge_data in edges_from_src.items():
                            output_idx = edge_data.get("output_index")
                            input_slot = edge_data.get("input_slot")  # 可能为 None
                            src_output = node_outputs[src]

                            if output_idx is None:
                                extracted = [sample_output for sample_output in src_output]
                            else:
                                extracted = [sample_output[output_idx] for sample_output in src_output]

                            # 判断是否使用 input_slot
                            current_has_slot = input_slot is not None

                            if has_input_slot is None:
                                has_input_slot = current_has_slot
                            elif has_input_slot != current_has_slot:
                                raise ValueError(
                                    f"Node '{node}': mixed use of 'input_slot' in incoming edges. "
                                    "All edges must either have 'input_slot' or all must omit it."
                                )

                            if has_input_slot:
                                if not isinstance(input_slot, int):
                                    raise ValueError(f"input_slot must be an integer, got {input_slot} (type {type(input_slot)})")
                                input_pairs.append((input_slot, extracted))
                            else:
                                input_feature_lists_old.append(extracted)

                    # === 根据是否有 input_slot 选择路径 ===
                    if has_input_slot:
                        # --- 新逻辑：按 input_slot 排序 ---
                        input_pairs.sort(key=lambda x: x[0])
                        input_slots = [slot for slot, _ in input_pairs]
                        expected_slots = list(range(len(input_pairs)))
                        if input_slots != expected_slots:
                            raise ValueError(
                                f"Node '{node}' has inconsistent input_slot indices. "
                                f"Expected {expected_slots}, got {input_slots}."
                            )
                        input_feature_lists = [data for _, data in input_pairs]
                    else:
                        # --- 老逻辑：保持边遍历顺序 ---
                        input_feature_lists = input_feature_lists_old

                # === 获取节点元信息 ===
                node_data = self.graph.nodes[node]
                method_name = node_data.get("method_name")

                # 兼容老模型：推断 is_passthrough
                if "is_passthrough" not in node_data:
                    is_passthrough = (
                        method_name is None or 
                        (isinstance(method_name, str) and method_name.lower() == 'null')
                    )
                else:
                    is_passthrough = bool(node_data.get("is_passthrough", False))

                # === 处理 passthrough 节点 ===
                if is_passthrough:
                    if len(input_feature_lists) != 1:
                        raise ValueError(f"Passthrough node '{node}' must have exactly one input, got {len(input_feature_lists)}")
                    # 输入是 [obj, obj, ...]，输出需为 [[obj], [obj], ...]
                    node_output_samples = [[x] for x in input_feature_lists[0]]
                else:
                    # === 正常方法执行 ===
                    if not method_name:
                        raise ValueError(f"Node '{node}'未指定 method_name 且非 passthrough")

                    if method_name not in self.methods:
                        raise KeyError(f"Method '{method_name}' not registered in self.methods")

                    method_info = self.methods[method_name]
                    func = method_info["function"]
                    expected_input_count = method_info["input_count"]
                    expected_output_count = method_info["output_count"]
                    is_vectorized = method_info.get("vectorized", False)

                    if len(input_feature_lists) != expected_input_count:
                        raise ValueError(
                            f"Node '{node}' method '{method_name}' expects {expected_input_count} inputs, "
                            f"but got {len(input_feature_lists)}."
                        )

                    num_samples = len(input_feature_lists[0]) if input_feature_lists else 0
                    node_output_samples = None

                    # --- 尝试向量化执行 ---
                    if is_vectorized and num_samples > 1:
                        try:
                            batched_inputs = []
                            for input_list in input_feature_lists:
                                first = input_list[0]
                                # 情况1: 全是标量（int/float/np.number）
                                if all(isinstance(x, (int, float, np.number)) for x in input_list):
                                    batched = np.array(input_list)
                                # 情况2: 全是 numpy 数组且 shape 一致
                                elif all(isinstance(x, np.ndarray) for x in input_list):
                                    shapes = [x.shape for x in input_list]
                                    if all(s == shapes[0] for s in shapes):
                                        batched = np.stack(input_list, axis=0)
                                    else:
                                        raise ValueError("Array shapes mismatch, cannot vectorize")
                                # 情况3: 其他类型（str, dict, list, object）→ 无法向量化
                                else:
                                    raise ValueError("Non-numeric or mixed types, cannot vectorize")
                                batched_inputs.append(batched)

                            batched_outputs = func(*batched_inputs)

                            # 标准化输出为 list of lists
                            if isinstance(batched_outputs, tuple):
                                if len(batched_outputs) != expected_output_count:
                                    raise ValueError(f"Expected {expected_output_count} outputs, got {len(batched_outputs)}")
                                node_output_samples = [
                                    [batched_outputs[j][i] for j in range(expected_output_count)]
                                    for i in range(num_samples)
                                ]
                            else:
                                if batched_outputs.ndim == 1:
                                    if expected_output_count != 1:
                                        raise ValueError(f"Expected {expected_output_count} outputs, but got 1D array")
                                    node_output_samples = [[x] for x in batched_outputs]
                                elif batched_outputs.ndim == 2:
                                    if batched_outputs.shape[1] != expected_output_count:
                                        raise ValueError(f"Output shape {batched_outputs.shape} mismatches output_count={expected_output_count}")
                                    node_output_samples = batched_outputs.tolist()
                                else:
                                    raise ValueError(f"Unsupported output ndim: {batched_outputs.ndim}")

                        except Exception as e:
                            logger.warning(
                                f"Vectorized execution failed for method '{method_name}' (inputs: {[type(x[0]) for x in input_feature_lists]}), "
                                f"fallback to sample-by-sample. Error: {e}"
                            )
                            is_vectorized = False

                    # --- 回退：逐样本执行 ---
                    else:
                        node_output_samples = []
                        for sample_idx in range(num_samples):
                            sample_inputs = [slot[sample_idx] for slot in input_feature_lists]
                            try:
                                result = func(*sample_inputs)
                            except Exception as e:
                                raise RuntimeError(
                                    f"Error in method '{method_name}' at sample {sample_idx}:\n"
                                    f"  Inputs: {sample_inputs}\n"
                                    f"  Error: {e}"
                                ) from e

                            if not isinstance(result, (list, tuple)):
                                result = [result]
                            result = list(result)

                            if len(result) != expected_output_count:
                                raise ValueError(
                                    f"Method '{method_name}' returned {len(result)} outputs, "
                                    f"but expected {expected_output_count}."
                                )
                            node_output_samples.append(result)

                # === 保存结果并触发后继节点 ===
                with lock:
                    node_outputs[node] = node_output_samples
                    for succ in self.graph.successors(node):
                        if succ == "collapse":
                            continue
                        if succ in in_degree_remaining:
                            in_degree_remaining[succ] -= 1
                            if in_degree_remaining[succ] == 0:
                                ready_queue.put(succ)

            except Exception as e:
                error_msg = f"[🔥 CRITICAL ERROR in process_node] Node '{node}' failed: {e}"
                print(error_msg)
                traceback.print_exc()
                raise

        # 启动线程池
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            total_nodes_to_execute = len([n for n in self.graph.nodes if n not in {"root", "collapse"}])
            submitted_nodes = set()

            while len(submitted_nodes) < total_nodes_to_execute:
                try:
                    node = ready_queue.get(timeout=1)
                    if node in submitted_nodes:
                        continue
                    submitted_nodes.add(node)
                    futures.append(executor.submit(process_node, node))
                except queue.Empty:
                    continue

            for f in futures:
                f.result()

        # === 5. 聚合到 collapse 节点（与 infer_with_graph 一致）===
        collapse_inputs = []  # List[Tuple[global_coord, List[Any]]]
        for u, v, data in self.graph.in_edges("collapse", data=True):
            local_idx = data.get("output_index")
            global_coord = data.get("data_coord")
            if local_idx is None or global_coord is None:
                raise ValueError(f"Edge from {u} to collapse missing output_index or data_coord")

            src_output = node_outputs[u]  # list of lists
            feature_values = [sample[local_idx] for sample in src_output]
            collapse_inputs.append((global_coord, feature_values))

        if not collapse_inputs:
            return []

        # 按 global_coord 排序以保证顺序一致
        collapse_inputs.sort(key=lambda x: x[0])
        all_features_per_sample = list(zip(*[feat_list for _, feat_list in collapse_inputs]))

        # === 6. 应用 collapse 函数 ===
        collapsed_results = []
        for sample_features in all_features_per_sample:
            try:
                collapsed_val = self.collapse_manager.collapse(list(sample_features))
                collapsed_results.append(collapsed_val)
            except Exception as e:
                raise RuntimeError(
                    f"Error in collapse function with inputs {list(sample_features)}:\n{e}"
                ) from e

        return np.array(collapsed_results)  # 返回 list，与 infer_with_graph 一致

    def infer_with_graph_single(self, sample, use_pipeline=False, num_workers=4):
        # Step 1: 确保 sample 是 1D 可迭代（list/tuple/array）
        if isinstance(sample, np.ndarray):
            if sample.ndim == 0:
                sample = [sample.item()]
            elif sample.ndim == 1:
                sample = sample.tolist()  # 转为 Python list
            else:
                raise ValueError("Single sample must be 0D or 1D")
        elif not isinstance(sample, (list, tuple)):
            sample = [sample]

        # Step 2: 包装成 batch: [[x1, x2, ...]]
        batch_input = [sample]  # ✅ 关键：二维结构

        # Step 3: 调用批处理接口
        if use_pipeline:
            result = self.infer_with_task_parallel(batch_input, num_workers=num_workers)
        else:
            result = self.infer_with_graph(batch_input)

        return result[0]
    
    # 在 GraphProcessor 类中
    def replace_node_method(
        self,
        old_node_id: str,
        new_method_name: str
    ) -> str:
        """
        替换图中一个节点的方法，并更新其 ID 和所有相连的边。
        不做图全节点刷新（全节点刷新耗能高并且推理不依赖具体id，可能后续做个单独方法）
        该方法不做类型检测
        
        :param old_node_id: 要替换的旧节点 ID（如 "2_3_return_value"）
        :param new_method_name: 新的方法名（如 "add_values"）
        :return: 新节点的 ID（如 "2_0_add_values"）
        """
        if new_method_name not in self.methods:
            raise ValueError(f"Method '{new_method_name}' not registered in methods.")
        graph = self.graph

        # === 1. 获取旧节点信息 ===
        if old_node_id not in graph:
            raise ValueError(f"Node '{old_node_id}' not found in graph.")
        
        old_data = graph.nodes[old_node_id]
        old_method = old_data.get("method_name")
        if old_method is None:
            raise ValueError(f"Node '{old_node_id}' has no 'method_name' attribute.")

        # === 2. 解析旧 ID 获取 layer 和 index 前缀 ===
        # 旧 ID 格式: {layer}_{index}_{method_name} 或 {layer}_{index}_unmatched
        id_parts = old_node_id.split('_', 2)  # 最多 split 成 3 部分
        if len(id_parts) < 3:
            raise ValueError(f"Invalid node ID format: '{old_node_id}'")
        
        layer_str, index_str, _ = id_parts
        try:
            layer = int(layer_str)
        except ValueError:
            raise ValueError(f"Invalid layer in node ID: '{old_node_id}'")

        # === 3. 生成新 ID ===
        new_base_name = new_method_name  # ✅ 关键：定义 new_base_name
        
        new_node_id = self._generate_unique_node_id(layer, new_method_name)

        if new_node_id in graph:
            raise RuntimeError(f"Node ID collision: {new_node_id} already exists!")

        # === 4. 保存旧节点的入边和出边 ===
        in_edges = list(graph.in_edges(old_node_id, keys=True, data=True))
        out_edges = list(graph.out_edges(old_node_id, keys=True, data=True))

        # === 5. 删除旧节点 ===
        graph.remove_node(old_node_id)

        # === 6. 添加新节点 ===
        is_passthrough = (new_method_name == self.discard_node_method_name)
        graph.add_node(new_node_id, method_name=new_method_name, layer=layer, is_passthrough=is_passthrough)

        # === 7. 重连入边（source -> new_node_id）===
        for src, _, key, data in in_edges:
            graph.add_edge(src, new_node_id, key=key, **data)

        # === 8. 重连出边（new_node_id -> target）===
        for _, dst, key, data in out_edges:
            graph.add_edge(new_node_id, dst, key=key, **data)

        logger.debug(
            "Replaced node '%s' (%s) with '%s' (%s)",
            old_node_id, old_method, new_node_id, new_method_name
        )
        
        return new_node_id

    def _is_processing_node(self, node):
        """
        判断一个节点是否是需要执行函数的“处理节点”。
        排除 root、collapse 和 passthrough（如 discard/unmatched）节点。
        """
        if node in {"root", "collapse"}:
            return False
        
        node_data = self.graph.nodes.get(node, {})

        # 如果是 passthrough 节点（如 discard_node_method_name 对应的节点），不视为处理节点
        if node_data.get("is_passthrough", False):
            return False
        
        return True

    def get_graph_entropy(self):
        """
        计算图结构的熵值，基于节点和方法类型的分布。
        示例计算方法，可根据实际需求替换。
        :return: 计算得到的图结构熵值
        """
        method_counter = Counter()

        # 统计每种方法的出现次数
        for node in self.graph.nodes:
            data = self.graph.nodes[node]
            method_name = data.get("method_name")
            if method_name and method_name != "null":  # 忽略 null 节点
                method_counter[method_name] += 1

        if not method_counter:
            return 0.0

        # 计算概率分布
        total = sum(method_counter.values())
        probabilities = [count / total for count in method_counter.values()]

        # 计算香农熵
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

        return entropy

    def get_method_counter(self):
        """
        统计图中各 method_name 的出现次数
        """
        from collections import Counter
        method_counter = Counter()

        for node in self.graph.nodes:
            data = self.graph.nodes[node]
            method_name = data.get("method_name")
            if method_name and method_name != "null":  # 忽略 null 节点
                method_counter[method_name] += 1

        return method_counter

    def get_max_layer_from_graph(self):
        max_layer = 0
        for node in self.graph.nodes:
            if node == 'root' or node == 'collapse':
                continue
            if isinstance(node, str) and '_' in node:
                layer = int(node.split('_')[0])
                if layer > max_layer:
                    max_layer = layer
        return max_layer

    def replace_subgraph_with_graph(
        self,
        subgraph_nodes: Set[str],
        replacement_graph: nx.DiGraph,
        input_port_bindings: Dict[str, Tuple[str, str, dict]],   # port_name → (src, key, data)
        output_port_bindings: Dict[str, Tuple[str, str, dict]],  # port_name → (dst, key, data)
        root_placeholder: str = "root",
        collapse_placeholder: str = "collapse"
    ) -> List[str]:
        if not subgraph_nodes:
            raise ValueError("subgraph_nodes cannot be empty")
        if root_placeholder not in replacement_graph or collapse_placeholder not in replacement_graph:
            raise ValueError("replacement_graph must contain 'root' and 'collapse' nodes.")

        graph = self.graph

        # 1. 删除子图
        for node in subgraph_nodes:
            if node in graph:
                graph.remove_node(node)

        # 2. 计算新 layer（基于 replacement_graph 中离 root 的距离）
        internal_nodes = [n for n in replacement_graph.nodes() if n not in (root_placeholder, collapse_placeholder)]
        
        try:
            dist_from_root = nx.single_source_shortest_path_length(replacement_graph, root_placeholder)
        except Exception as e:
            raise ValueError(f"Cannot compute layers from root: {e}")

        # 提取所有合法的 layer 编号
        valid_layers = []
        for node in subgraph_nodes:
            if node in ("root", "collapse"):
                continue
            parts = node.split('_', 1)
            if len(parts) >= 1:
                try:
                    layer = int(parts[0])
                    valid_layers.append(layer)
                except ValueError:
                    continue  # 忽略无法解析 layer 的节点

        # 设定全局偏移：如果有合法 layer，取最小值；否则默认为 0
        global_offset = min(valid_layers) if valid_layers else 0

        # 3. 重命名节点
        node_mapping = {}
        for node in internal_nodes:
            method = replacement_graph.nodes[node].get('method_name')
            if not method:
                raise ValueError(f"Node '{node}' missing 'method_name'")
            layer = global_offset + dist_from_root[node] - 1
            new_name = self._generate_unique_node_id(layer, method)
            node_mapping[node] = new_name

        renamed_replacement = nx.relabel_nodes(replacement_graph, node_mapping, copy=True)

        # 4. 添加内部节点和边
        for node in internal_nodes:
            new_node = node_mapping[node]
            graph.add_node(new_node, **renamed_replacement.nodes[new_node])
        
        for u, v, key, data in renamed_replacement.edges(keys=True, data=True):
            if u != root_placeholder and v != collapse_placeholder:
                graph.add_edge(u, v, key=key, **data)

        # 5. 重连输入：直接按 port_name 查找
        for _, internal_node, key, data in renamed_replacement.out_edges(root_placeholder, keys=True, data=True):
            port_name = data.get('port_name')
            if not port_name:
                raise ValueError("Replacement input edge missing 'port_name'")
            if port_name not in input_port_bindings:
                raise ValueError(f"No input binding for port: {port_name}")
            src, key_in, orig_data = input_port_bindings[port_name]
            merged = {
                'data_type': orig_data.get('data_type', data.get('data_type', 'scalar')),
                'output_index': orig_data.get('output_index', 0),
                'input_slot': data.get('input_slot', orig_data.get('input_slot', 0)),
                'data_coord': data.get('data_coord', 0),
                'port_name': port_name
            }
            graph.add_edge(src, internal_node, key=key_in, **merged)

        # 6. 重连输出
        for internal_node, _, key, data in renamed_replacement.in_edges(collapse_placeholder, keys=True, data=True):
            port_name = data.get('port_name')
            if not port_name:
                raise ValueError("Replacement output edge missing 'port_name'")
            if port_name not in output_port_bindings:
                raise ValueError(f"No output binding for port: {port_name}")
            dst, key_out, orig_data = output_port_bindings[port_name]
            merged = {
                'data_type': orig_data.get('data_type', data.get('data_type', 'scalar')),
                'output_index': data.get('output_index', orig_data.get('output_index', 0)),
                'input_slot': orig_data.get('input_slot', data.get('input_slot', 0)),
                'data_coord': data.get('data_coord', 0),
                'port_name': port_name
            }
            graph.add_edge(internal_node, dst, key=key_out, **merged)

        return list(node_mapping.values())

    def _generate_unique_node_id(self, layer: int, method_name: str) -> str:
        """
        生成符合 {layer}_{index}_{method_name} 格式的唯一节点 ID。
        复用 replace_node_method 中的命名逻辑，供内部使用。
        """
        existing_indices = set()
        prefix = f"{layer}_"
        suffix = f"_{method_name}"
        for nid in self.graph.nodes():
            if nid.startswith(prefix) and nid.endswith(suffix):
                idx_part = nid[len(prefix): -len(suffix)]
                if idx_part.isdigit():
                    existing_indices.add(int(idx_part))
        new_index = 0
        while new_index in existing_indices:
            new_index += 1
        return f"{layer}_{new_index}_{method_name}"