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
    
    def infer_with_graph_pipeline(self, values, num_workers=4):
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
        def process_node(node):
            with lock:
                predecessors = list(self.graph.predecessors(node))
                inputs = []
                for src in predecessors:
                    edge_data = self.graph[src][node][0]  # 简化：假设单边
                    output_idx = edge_data.get("output_index")
                    src_output = node_outputs[src]
                    if output_idx is not None:
                        col = src_output[:, output_idx:output_idx+1]
                    else:
                        col = src_output
                    inputs.append(col)
                # 合并输入
                flat_input = np.hstack(inputs) if len(inputs) > 1 else inputs[0]

            # 执行函数
            method_name = self.graph.nodes[node].get("method_name")
            func = self.methods[method_name]["function"]
            outputs = []
            for row in flat_input:
                res = func(*row)
                if isinstance(res, (int, float)): res = [res]
                elif isinstance(res, np.ndarray): res = res.tolist()
                outputs.append(res)
            output_array = np.array(outputs)

            # 写回输出（需加锁）
            with lock:
                node_outputs[node] = output_array
                # 触发后继节点检查
                for succ in self.graph.successors(node):
                    if succ == "collapse":
                        continue
                    in_degree_remaining[succ] -= 1
                    if in_degree_remaining[succ] == 0:
                        ready_queue.put(succ)

        # 启动线程池
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            while True:
                try:
                    node = ready_queue.get(timeout=1)
                    futures.append(executor.submit(process_node, node))
                except queue.Empty:
                    if len(futures) == 0:
                        break  # 所有任务提交完毕
                    else:
                        continue

            # 等待全部完成
            for f in futures:
                f.result()

        # 最后处理 collapse 节点
        collapse_inputs = []
        for u, v, data in self.graph.in_edges("collapse", data=True):
            local_idx = data["output_index"]
            global_coord = data["data_coord"]
            col_data = node_outputs[u][:, local_idx]
            collapse_inputs.append((global_coord, col_data))

        collapse_inputs.sort(key=lambda x: x[0])
        raw_output = np.column_stack([col for _, col in collapse_inputs])
        result = np.apply_along_axis(self.collapse_manager.collapse, axis=1, arr=raw_output)
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