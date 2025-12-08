import networkx as nx
import numpy as np
from collections import Counter
import math
from ..CollapseManager.collapse_functions import CollapseFunctionManager, CollapseMethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.discard_node_method_name = discard_node_method_name

        self.layer += 1
        new_index_edge = 0
        method_counts = {method_name: 0 for method_name in self.methods}
        if discard_node_method_name not in method_counts:
            method_counts[discard_node_method_name] = 0
        v = "collapse"
        collapse_edges = list(self.graph.in_edges(v, data=True))

        # 处理有效分组
        for method_name, groups in result['valid_groups'].items():
            if method_name == discard_node_method_name:
                continue
            for group in groups:
                new_target_node = f"{self.layer}_{method_counts[method_name]}_{method_name}"
                self.graph.add_node(
                    new_target_node,
                    method_name=method_name,
                    layer=self.layer,
                    is_passthrough=False   # ← 关键：显式标记
                )

                for u, _, data in collapse_edges:
                    if data.get('data_coord') in group:
                        self.graph.remove_edge(u, v)
                        self.graph.add_edge(u, new_target_node, **data)

                for local_output_index in range(self.methods[method_name]["output_count"]):
                    self.graph.add_edge(new_target_node, v, output_index=local_output_index, data_coord=new_index_edge, data_type=self.methods[method_name]["output_types"][local_output_index])
                    new_index_edge += 1

                method_counts[method_name] += 1
            
            # # === 新增：检测无入边的新节点 ===
            # for method_name, count in method_counts.items():
            #     if method_name == "unmatched":
            #         continue
            #     for i in range(count):
            #         new_node = f"{self.layer}_{i}_{method_name}"
            #         if self.graph.in_degree(new_node) == 0:
            #             print(f"[警告] 节点 '{new_node}' 没有入边！其属性为：")
            #             print(dict(self.graph.nodes[new_node]))
            #             print(result['valid_groups'])

        # 收集所有输入边的 data_type
        input_data_types = []
        
        # 处理 discard_node_method_name
        unmatched_groups = result.get('unmatched', [])
        if unmatched_groups and discard_unmatched == 'to_discard':
            for group in unmatched_groups:
                node_name = f"{self.layer}_{method_counts[discard_node_method_name]}_{discard_node_method_name}"
                self.graph.add_node(
                    node_name,
                    method_name=discard_node_method_name,
                    layer=self.layer,
                    is_passthrough=True   # ← 关键：显式标记
                )

                # 收集所有输入边的 data_type，并重定向边
                input_data_types = []
                for u, _, data in collapse_edges:
                    if data.get('data_coord') in group:
                        input_data_types.append(data.get('data_type'))  # 提取原始边的数据类型
                        self.graph.remove_edge(u, v)
                        self.graph.add_edge(u, node_name, **data)

                # 按顺序为每条输入边创建一条输出边，继承其 data_type
                for local_output_index, used_data_type in enumerate(input_data_types):
                    self.graph.add_edge(
                        node_name, v,
                        output_index=local_output_index,
                        data_coord=new_index_edge,
                        data_type=used_data_type  # ← 使用对应输入边的类型
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
            # TODO: Add unit tests for 'ignore' mode (2025-10-18)
            logging.warning("Experimental 'ignore' mode: unmatched data is dropped.")
            # === 新增逻辑：彻底丢弃 unmatched 数据 ===
            # 将 unmatched_groups 扁平化为一个集合，便于快速查找
            unmatched_coords = set()
            for group in unmatched_groups:
                unmatched_coords.update(group)

            # 遍历原始 collapse 入边（已缓存），删除属于 unmatched 的边
            for u, _, data in collapse_edges:
                if data.get('data_coord') in unmatched_coords:
                    self.graph.remove_edge(u, v)
            # 注意：不创建新节点，也不添加新边 → 数据被丢弃

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
                    (isinstance(method_name, str) and method_name.lower() == 'null')
                )
            else:
                is_passthrough = bool(node_data.get("is_passthrough", False))

            # === 收集所有输入特征（按样本对齐）===
            predecessors = list(self.graph.predecessors(node))
            if not predecessors:
                raise ValueError(f"Node '{node}' has no predecessors.")

            # 按边收集：每个输入边对应一个“输入特征列表”（长度 = num_samples）
            input_feature_lists = []  # List[List[Any]]: [input_slot][sample_idx]

            for src in predecessors:
                edges_from_src = self.graph[src][node]  # Multi-edge dict
                for edge_key in edges_from_src:
                    edge_data = edges_from_src[edge_key]
                    output_idx = edge_data.get("output_index")
                    src_output = node_outputs[src]  # list of lists

                    if output_idx is None:
                        # 透传整个输出（罕见，通常用于 root）
                        extracted = [sample_output for sample_output in src_output]
                    else:
                        # 提取第 output_idx 个输出特征
                        extracted = [sample_output[output_idx] for sample_output in src_output]
                    input_feature_lists.append(extracted)

            # === 执行节点 ===
            if is_passthrough:
                if len(input_feature_lists) != 1:
                    raise ValueError(f"Passthrough node '{node}' must have exactly one input, got {len(input_feature_lists)}")
                # 透传：输出 = 输入
                node_outputs[node] = input_feature_lists[0]  # list of objects (one per sample)
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
            if is_vectorized:
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
            if not is_vectorized:
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
        return collapsed_results
    
    def infer_with_task_parallel(self, values, num_workers=4):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import queue

        # 初始化
        node_outputs = {"root": values}
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

        for succ in self.graph.successors("root"):
            # 对每个从 root 指向的节点，减少一个依赖（因为 root 已完成）
            if succ in in_degree_remaining:
                in_degree_remaining[succ] -= 1
                if in_degree_remaining[succ] == 0:
                    ready_queue.put(succ)

        # collapse 特殊处理：所有指向它的节点完成后才可执行
        collapse_in_edges = list(self.graph.in_edges("collapse"))
        collapse_deps = len(collapse_in_edges)
        if collapse_deps == 0:
            return np.array([])

        # 工作函数
        import traceback  # 👈 确保文件顶部已导入

        def process_node(node):
            try:
                with lock:
                    predecessors = list(self.graph.predecessors(node))
                    inputs = []
                    for src in predecessors:
                        try:
                            # ✅ 修复：遍历所有从 src 到 node 的边
                            edges_from_src = self.graph[src][node]  # {key: edge_data}
                            for edge_key in edges_from_src:
                                edge_data = edges_from_src[edge_key]
                                output_idx = edge_data.get("output_index")
                                if src not in node_outputs:
                                    raise KeyError(f"前置节点 '{src}' 的输出尚未计算")
                                src_output = node_outputs[src]
                                if output_idx is not None:
                                    col = src_output[:, output_idx:output_idx+1]
                                else:
                                    col = src_output
                                inputs.append(col)
                        except Exception as e:
                            raise RuntimeError(f"构建节点 '{node}' 的输入时出错（来自前置节点 '{src}'）: {e}") from e

                    if len(inputs) == 0:
                        raise ValueError(f"节点 '{node}' 没有输入数据")
                    flat_input = np.hstack(inputs) if len(inputs) > 1 else inputs[0]

                # 执行函数
                node_data = self.graph.nodes[node]  # ✅ 确保 node_data 被定义
                method_name = node_data.get("method_name")

                # ✅ 新增：兼容老版本模型，处理缺失 is_passthrough 的情况
                if "is_passthrough" not in node_data:
                    logger.warning(
                        f"⚠️ 节点 '{node}' 缺少 'is_passthrough' 属性，检测到老版本模型。"
                        f"方法名: {method_name}。未来版本将取消对老模型的支持，请尽快升级模型格式。"
                    )
                    # 推断：method_name 为 None 或字符串 'null'（不区分大小写）时视为 passthrough
                    if method_name is None or (isinstance(method_name, str) and method_name.lower() == 'null'):
                        is_passthrough = True
                    else:
                        is_passthrough = False
                else:
                    is_passthrough = bool(node_data.get("is_passthrough", False))

                # ✅ 使用推断/提取出的 is_passthrough 进行判断
                if is_passthrough:
                    with lock:
                        predecessors = list(self.graph.predecessors(node))
                        if len(predecessors) > 1:
                            raise ValueError(f"节点 {node} 使用了 'passthrough' 方法，但有多个前驱节点。这违反设计约束。")

                        node_outputs[node] = flat_input.copy()

                        # 触发后继节点
                        for succ in self.graph.successors(node):
                            if succ == "collapse":
                                continue
                            if succ in in_degree_remaining:
                                in_degree_remaining[succ] -= 1
                                if in_degree_remaining[succ] == 0:
                                    ready_queue.put(succ)
                    
                    output_shape = node_outputs[node].shape
                    # print(f"[✅ SUCCESS] 节点 {node} (method=unmatched) 执行完成，输出形状: {output_shape}")
                    return  # ⚠️ 直接返回，跳过后续函数执行逻辑

                # ========== 原有函数执行逻辑 ==========
                if not method_name:
                    raise ValueError(f"节点 '{node}' 未指定 method_name")

                if method_name not in self.methods:
                    raise KeyError(f"方法 '{method_name}' 未在 self.methods 中注册")

                func = self.methods[method_name]["function"]
                if not callable(func):
                    raise TypeError(f"方法 '{method_name}' 不是可调用对象")

                outputs = []
                for i, row in enumerate(flat_input):
                    try:
                        res = func(*row)
                        if isinstance(res, (int, float)):
                            res = [res]
                        elif isinstance(res, np.ndarray):
                            res = res.tolist()
                        outputs.append(res)
                    except Exception as e:
                        raise RuntimeError(f"在节点 '{node}' 执行第 {i} 行输入时出错: {e} | 输入数据: {row}") from e

                output_array = np.array(outputs)

                # 写回输出（需加锁）
                with lock:
                    node_outputs[node] = output_array
                    # 触发后继节点检查
                    for succ in self.graph.successors(node):
                        if succ == "collapse":
                            continue
                        if succ not in in_degree_remaining:
                            print(f"[WARNING] 后继节点 '{succ}' 不在 in_degree_remaining 中，跳过依赖更新。")
                            continue
                        in_degree_remaining[succ] -= 1
                        if in_degree_remaining[succ] == 0:
                            ready_queue.put(succ)

                # print(f"[✅ SUCCESS] 节点 {node} 执行完成，输出形状: {output_array.shape}")
                # print(f"[🧵 THREAD DONE] 节点 {node} 线程已完全退出")
                return  # 确保显式返回

            except Exception as e:
                error_msg = f"[🔥 CRITICAL ERROR in process_node] 节点 '{node}' 执行失败: {e}"
                print(error_msg)
                traceback.print_exc()
                # 可选：将错误节点放入特殊队列 or 设置全局错误标志
                # 例如：
                # with lock:
                #     global_error_flag.set()
                #     error_queue.put((node, str(e)))
                raise  # 重新抛出，让外层捕获（如 ThreadPoolExecutor 会标记 future 为失败）

        # 启动线程池
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            # 👇 计算总节点数（排除 root 和 collapse）
            total_nodes_to_execute = len([
                n for n in self.graph.nodes 
                if n not in ["root", "collapse"]
            ])
            submitted_nodes = set()  # 用于去重和计数

            # print(f"[🎯 总共需要执行 {total_nodes_to_execute} 个节点]")

            while len(submitted_nodes) < total_nodes_to_execute:
                try:
                    node = ready_queue.get(timeout=1)
                    if node in submitted_nodes:
                        continue  # 防止重复提交（虽然理论上不会，但安全第一）
                    submitted_nodes.add(node)
                    futures.append(executor.submit(process_node, node))
                    # print(f"[📤 已提交节点 {len(submitted_nodes)}/{total_nodes_to_execute}]: {node}")
                except queue.Empty:
                    # 队列暂时空，但还没提交完所有节点 → 等待子线程生成新节点
                    # print(f"[⏳ 队列空，等待中... 已提交 {len(submitted_nodes)}/{total_nodes_to_execute}]")
                    # time.sleep(0.1)  # 避免忙等，节省 CPU 实际工程中可以使用，这里追求实验精度没写，可以取消注释
                    continue

            # print(f"[✅ 所有 {total_nodes_to_execute} 个节点已提交，共 {len(futures)} 个任务，开始等待执行完成...]")

            # 等待全部完成
            for f in futures:
                f.result()

        # 最后处理 collapse 节点
        # print('正在聚合 collapse 节点...')
        collapse_inputs = []
        for u, v, data in self.graph.in_edges("collapse", data=True):
            local_idx = data["output_index"]
            global_coord = data["data_coord"]
            col_data = node_outputs[u][:, local_idx]
            collapse_inputs.append((global_coord, col_data))

        collapse_inputs.sort(key=lambda x: x[0])
        raw_output = np.column_stack([col for _, col in collapse_inputs])
        result = np.apply_along_axis(self.collapse_manager.collapse, axis=1, arr=raw_output)
        # print('完成')
        return result

    def infer_with_graph_single(self, sample, use_pipeline=False, num_workers=4):
        """
        使用图结构对单个样本进行推理计算，可选择是否使用并行流水线。

        参数:
            sample (np.ndarray or list): 单个样本，形状为 [特征维度]
            use_pipeline (bool): 是否使用多线程流水线推理
            num_workers (int): 流水线使用的线程数（仅当 use_pipeline=True 时有效）

        返回:
            float or np.ndarray: 经过图结构处理后的结果（通过 collapse 输出）
        """
        # 选择推理方式
        if use_pipeline:
            result = self.infer_with_pipeline(values, num_workers=num_workers)
        else:
            result = self.infer_with_graph(values)

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
        
        existing_indices = set()
        prefix = f"{layer}_"
        suffix = f"_{new_base_name}"
        for nid in graph.nodes:
            if nid.startswith(prefix) and nid.endswith(suffix):
                idx_part = nid[len(prefix): -len(suffix)]
                if idx_part.isdigit():
                    existing_indices.add(int(idx_part))

        new_index = 0
        while new_index in existing_indices:
            new_index += 1
        new_node_id = f"{layer}_{new_index}_{new_base_name}"

        if new_node_id in graph:
            raise RuntimeError(f"Node ID collision: {new_node_id} already exists!")

        # === 4. 保存旧节点的入边和出边 ===
        in_edges = list(graph.in_edges(old_node_id, keys=True, data=True))
        out_edges = list(graph.out_edges(old_node_id, keys=True, data=True))

        # === 5. 删除旧节点 ===
        graph.remove_node(old_node_id)

        # === 6. 添加新节点 ===
        is_passthrough = (new_method_name == self.discard_node_method_name)
        graph.add_node(new_node_id, method_name=new_method_name, is_passthrough=is_passthrough)

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