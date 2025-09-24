import networkx as nx
import numpy as np
from collections import Counter
import math
from ..CollapseManager.collapse_functions import CollapseFunctionManager, CollapseMethod

class GraphProcessor:
    def __init__(self, graph: nx.MultiDiGraph, methods: dict, collapse_method=CollapseMethod.SUM):
        """
        初始化图处理器。
        
        参数:
            graph (nx.MultiDiGraph): 初始图结构
            methods (dict): 可用方法字典 {method_name: {"function": ..., "input_count": int, "output_count": int}}
            collapse_method (callable): 聚合函数，默认为 sum
        """
        self.graph = graph.copy() if graph else nx.MultiDiGraph()
        self.methods = methods
        self.layer = 0  # 记录当前层数（可选）

        self.collapse_manager = CollapseFunctionManager(method=collapse_method)

    def set_graph(self, new_graph):
        """更新图结构"""
        if not hasattr(new_graph, 'nodes') or not hasattr(new_graph, 'edges'):
            raise ValueError("new_graph 不是一个有效的图对象")
        self.graph = new_graph
        print("图结构已更新。")

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
        self.layer += 1
        new_index_edge = 0
        method_counts = {method_name: 0 for method_name in self.methods}
        method_counts["unmatched"] = 0
        v = "collapse"
        collapse_edges = list(self.graph.in_edges(v, data=True))

        # 处理有效分组
        for method_name, groups in result['valid_groups'].items():
            if method_name == "unmatched":
                continue
            for group in groups:
                new_target_node = f"{self.layer}_{method_counts[method_name]}_{method_name}"
                self.graph.add_node(new_target_node, method_name=method_name)

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
        
        # 处理 unmatched
        unmatched_groups = result.get('unmatched', [])
        if unmatched_groups and discard_unmatched == 'to_discard':
            for group in unmatched_groups:
                node_name = f"{self.layer}_{method_counts['unmatched']}_unmatched"
                self.graph.add_node(node_name, method_name=discard_node_method_name)

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

                method_counts["unmatched"] += 1

        elif unmatched_groups and discard_unmatched != 'ignore':
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
        使用图结构对输入数据进行推理。
        """
        node_outputs = {}
        nodes_in_order = list(nx.topological_sort(self.graph))
        nodes_in_order.remove("collapse")
        if "root" in nodes_in_order:
            nodes_in_order.remove("root")

        node_outputs["root"] = values.copy()

        for node in nodes_in_order:
            node_data = self.graph.nodes[node]
            method_name = node_data.get("method_name", None)

            if method_name == "null":
                predecessors = list(self.graph.predecessors(node))
                if len(predecessors) > 1:
                    raise ValueError(f"节点 {node} 使用了 'null' 方法，但有多个前驱节点。")
                if predecessors:
                    node_outputs[node] = node_outputs.get(predecessors[0], np.array([]))
                continue

            if method_name not in self.methods:
                raise ValueError(f"未知方法名 {method_name}")

            method_info = self.methods[method_name]
            func = method_info["function"]
            input_count = method_info["input_count"]
            output_count = method_info["output_count"]

            in_edges = list(self.graph.in_edges(node, data=True))
            inputs = []

            for src, dst, edge_data in in_edges:
                if src not in node_outputs:
                    raise ValueError(f"前驱节点 {src} 没有输出")
                output_index = edge_data.get("output_index")
                if output_index is not None:
                    try:
                        selected_column = [[sample[output_index]] for sample in node_outputs[src]]
                    except IndexError:
                        raise ValueError(f"前驱节点 {src} 输出长度不足，无法获取 output_index={output_index}")
                    inputs.append(selected_column)
                else:
                    inputs.extend(node_outputs[src])

            if not inputs:
                raise ValueError("没有可合并的输入列")
            num_samples = len(inputs[0])
            for col in inputs:
                if len(col) != num_samples:
                    raise ValueError("所有输入列样本数必须一致")

            final_inputs = []
            for i in range(num_samples):
                merged = []
                for col in inputs:
                    merged.extend(col[i])
                final_inputs.append(merged)

            flat_inputs = np.array(final_inputs)
            results = []

            for row in flat_inputs:
                result = func(*row)
                if isinstance(result, (int, float)):
                    result = [result]
                elif isinstance(result, np.ndarray):
                    result = result.tolist()
                results.append(result)

            new_output = np.zeros((num_samples, output_count))
            for i, res in enumerate(results):
                for j, val in enumerate(res):
                    new_output[i, j] = val

            node_outputs[node] = new_output

        temp_result = []
        for u, v, data in self.graph.in_edges("collapse", data=True):
            if v == "collapse":
                local_index = data.get('output_index')
                global_coord = data.get('data_coord')
                if local_index is None or global_coord is None:
                    raise ValueError("缺少必要属性")
                try:
                    column_data = node_outputs[u][:, local_index]
                    temp_result.append((global_coord, column_data))
                except IndexError:
                    raise ValueError(f"节点 {u} 输出维度不足，无法提取 output_index={local_index}")

        temp_result.sort(key=lambda x: x[0])
        collapse_result = [col for _, col in temp_result]
        raw_output = np.column_stack(collapse_result)
        collapsed_output = np.apply_along_axis(self.collapse_manager.collapse, axis=1, arr=raw_output)
        return collapsed_output
    
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
                method_name = self.graph.nodes[node].get("method_name")

                # ✅ 新增：处理 null 方法节点
                if method_name == "null":
                    with lock:
                        predecessors = list(self.graph.predecessors(node))
                        if len(predecessors) > 1:
                            raise ValueError(f"节点 {node} 使用了 'null' 方法，但有多个前驱节点。")
                        if predecessors:
                            src = predecessors[0]
                            if src not in node_outputs:
                                raise KeyError(f"前置节点 '{src}' 输出未就绪")
                            node_outputs[node] = node_outputs[src].copy()
                        else:
                            # 无前驱，生成默认输出
                            sample_count = flat_input.shape[0] if 'flat_input' in locals() else 100
                            output_count = 1
                            node_outputs[node] = np.zeros((sample_count, output_count))
                        
                        # 触发后继节点
                        for succ in self.graph.successors(node):
                            if succ == "collapse": continue
                            if succ in in_degree_remaining:
                                in_degree_remaining[succ] -= 1
                                if in_degree_remaining[succ] == 0:
                                    ready_queue.put(succ)
                    
                    output_shape = node_outputs[node].shape
                    # print(f"[✅ SUCCESS] 节点 {node} (method=null) 执行完成，输出形状: {output_shape}")
                    return  # ⚠️ 直接返回，跳过函数执行逻辑

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
        # 确保输入是一维数组
        sample = np.array(sample)
        assert len(sample.shape) == 1, "输入必须是一维数组"

        # 扩展为二维 (1, 特征维度)，适配批量接口
        values = sample.reshape(1, -1)

        # 选择推理方式
        if use_pipeline:
            result = self.infer_with_pipeline(values, num_workers=num_workers)
        else:
            result = self.infer_with_graph(values)

        # 返回单个样本的结果（已经是 (1,) 或标量）
        return result[0]
    
    # 在 GraphProcessor 类中
    def replace_node_method(
        self,
        old_node_id: str,
        new_method_name: str
    ) -> str:
        """
        安全地替换图中一个节点的方法，并更新其 ID 和所有相连的边。
        不做图全节点刷新（全节点刷新耗能高并且推理不依赖具体id，可能后续做个单独方法）
        
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
        if new_method_name == "null":
            new_base_name = "unmatched"
        else:
            new_base_name = new_method_name

        # 查找该层中已存在的同方法节点数量，确定新 index
        existing_same_method = [
            nid for nid in graph.nodes
            if nid.startswith(f"{layer}_") and nid.endswith(f"_{new_base_name}")
        ]
        new_index = len(existing_same_method)
        new_node_id = f"{layer}_{new_index}_{new_base_name}"

        # === 4. 保存旧节点的入边和出边 ===
        in_edges = list(graph.in_edges(old_node_id, keys=True, data=True))
        out_edges = list(graph.out_edges(old_node_id, keys=True, data=True))

        # === 5. 删除旧节点 ===
        graph.remove_node(old_node_id)

        # === 6. 添加新节点 ===
        graph.add_node(new_node_id, method_name=new_method_name)

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