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

from threading import RLock
from .execution_runtime import ExecutionRuntime, prepare_graph, transactional_graph_edit


class GraphProcessor(ExecutionRuntime):
    def __init__(self, graph: nx.MultiDiGraph, methods: dict, collapse_method=CollapseMethod.SUM):
        """
        初始化图处理器。
        
        参数:
            graph (nx.MultiDiGraph): 初始图结构
            methods (dict): 可用方法字典 {method_name: {"function": ..., "input_count": int, "output_count": int}}
            collapse_method (callable): 聚合函数，默认为 sum
        """
        self._execution_lock = RLock()
        self.discard_node_method_name = "null"
        self.graph = graph.copy() if graph else nx.MultiDiGraph()
        self.methods = methods
        self.layer = self.get_max_layer_from_graph()  # 记录当前层数（可选）

        self.collapse_manager = CollapseFunctionManager(method=collapse_method)

    def set_graph(self, new_graph):
        if not hasattr(new_graph, 'nodes') or not hasattr(new_graph, 'edges'):
            raise ValueError("new_graph 不是一个有效的图对象")
        with self._execution_lock:
            prepare_graph(new_graph, self.methods)
            self.graph = new_graph
            self.layer = self.get_max_layer_from_graph()
        print(f"图结构已更新，当前最大层数：{self.layer}。")

    def set_methods(self, new_methods):
        """验证新方法池后原子替换，避免在旧计划执行期间换入新函数。

        直接修改共享字典无法由本接口拦截，调用方不得与推理并发做这种修改。
        注册表的验证使用图副本；失败不会改变已接受模型的端口或执行计划。
        """
        import copy
        if not isinstance(new_methods, dict):
            raise TypeError("methods 必须是方法名到方法元数据的字典")
        with self._execution_lock:
            candidate = copy.deepcopy(self.graph)
            prepare_graph(candidate, new_methods)
            self.methods = new_methods
        print("Methods 已更新。")

    @transactional_graph_edit
    def append_nx_layer(self, result, discard_unmatched='to_discard', discard_node_method_name="null"):
        """
        向图中添加一层新节点。
        """
        import logging

        # 入口一致性守卫：collapse 入边必须满足「列表位置 == data_coord」。
        # 该不变式一旦被打破（典型来源是 remove_last_nx_layer 把旧 data_coord 搬回来），
        # PathGenerator 按位置取列就会取错，且**不会报错**。
        # 这里发现不一致就按 data_coord 重排修回来，并给出警告，避免静默带偏搜索。
        _ce = list(self.graph.in_edges("collapse", keys=True, data=True))
        _bad = [i for i, (_u, _v, _k, _d) in enumerate(_ce)
                if i != _d.get("data_coord")]
        if _bad:
            logging.getLogger(__name__).warning(
                "collapse 入边的 data_coord 与列表位置不一致（错位位置: %s）；"
                "已按 data_coord 重排以恢复不变式「位置 == data_coord」。"
                "这通常意味着此前的 remove_last_nx_layer() 之后没有重整顺序。",
                _bad[:10],
            )
            self._restore_collapse_order()

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

    def _restore_collapse_order(self):
        """把 collapse 入边按 **data_coord 升序** 重排，使「列表位置 == data_coord」恢复成立。

        为什么必须有这一步（原来的隐含假设）：
            PathGenerator 用 **列表位置** 索引 collapse 入边
                path_generator.py:177   _, _, data = collapse_edges[idx]
            而本类用 **data_coord** 查找同一条边
                append_nx_layer():      coord_to_edge[data['data_coord']]
            两者能对上，完全依赖一条**没有断言、没有注释**的隐式不变式：
                「collapse 入边的列表位置 == 它的 data_coord」

        这条不变式在「初始建图」与「纯 append」时天然成立
        （append 给新边从 0 连续编号，恰好等于新位置），
        但 `remove_last_nx_layer()` 会把被删层的输入边**连旧 data_coord 一起**
        放回 collapse（见该方法的 add_edge）。此时列表顺序由插入历史决定，
        与旧 data_coord 不再一致 —— 不变式被打破。

        后果：下一次 append 时，PathGenerator 会按位置取到**错误的列**，
        把方法节点接到不匹配的输入上。它既不抛异常也没有警告
        （因为类型匹配也是按位置做的，两边自洽），
        只会静默地把合成结果与搜索决策带偏。

        **为什么是"排序"而不是"把 data_coord 重编号为位置"**：
        两种做法都能恢复「位置 == data_coord」，但只有排序能保证
        **回退是透明的**（append -> rollback -> append 同一个方案，
        结果与"从未回退过"逐项相同）。重编号会保留被搬回来的边的**位置**
        而改写它们的**编号**，于是位置上的内容与回退前不同：
        实测 'pair_num' 之类的中间结果会出现在本该是基础列的位置上，
        导致后续 'null' 透传节点带的输入与"未回退"时间线不一致。
        排序则把每条边放回它编号对应的位置 —— 语义上等于撤销了这次 append。

        实现说明：networkx 的 `in_edges` 顺序 = 插入顺序，没有原地重排 API，
        因此按目标顺序 **删边再重加**。边的全部属性原样保留（含 data_coord）。
        """
        g = self.graph
        edges = list(g.in_edges("collapse", keys=True, data=True))
        if all(i == d.get("data_coord") for i, (_u, _v, _k, d) in enumerate(edges)):
            return len(edges)                      # 已经有序，什么都不做
        ordered = sorted(edges, key=lambda e: e[3].get("data_coord"))
        for u, v, k, _d in edges:
            g.remove_edge(u, v, key=k)
        for u, v, _k, d in ordered:
            g.add_edge(u, v, **d)
        return len(ordered)

    @transactional_graph_edit
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

        # 被搬回来的边带的是旧 data_coord，与新的列表顺序不再一致；
        # 这里按 data_coord 重排，把不变式「列表位置 == data_coord」修回来，
        # 从而让这次回退在语义上等价于「从未 append 过」。
        self._restore_collapse_order()

        self.layer -= 1
        return self.graph


    # 该部分存在技术残留导致的全量中间缓存，作者有时间可能会改
    def infer_with_graph(self, values):
        """按冻结的语义顺序串行执行；隐式状态读写也属于程序顺序的一部分。"""
        return self._infer_with_execution_plan(values)

    def infer_with_task_parallel(self, values, num_workers=4):
        """仅并行执行数据依赖和副作用依赖都已经满足的节点。"""
        return self._infer_with_execution_plan(values, workers=num_workers)

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
    def replace_node_method(self, old_node_id: str, new_method_name: str) -> str:
        """替换方法，但保留节点的稳定身份和语义位置。

        显示名称可以改变，_atf_uid 不能因删除重建而丢失，否则候选评分和
        接受后的状态访问顺序可能不同。先在副本上完成端口与计划验证，再
        原地更新数据图，既保证失败不污染原模型，也维持 PathGenerator 引用。
        """
        import copy
        with self._execution_lock:
            self.prepare_execution_plan()
            if new_method_name not in self.methods:
                raise ValueError(f"Method {new_method_name!r} is not registered")
            if old_node_id not in self.graph:
                raise ValueError(f"Node {old_node_id!r} was not found")
            old = self.graph.nodes[old_node_id]
            if old.get("method_name") is None:
                raise ValueError(f"Node {old_node_id!r} has no method")
            layer = self.get_node_layer(old_node_id)
            new_id = self._generate_unique_node_id(layer, new_method_name)
            if new_id == old_node_id:
                return old_node_id
            candidate = nx.relabel_nodes(copy.deepcopy(self.graph), {old_node_id: new_id}, copy=True)
            candidate.nodes[new_id].update(method_name=new_method_name, layer=layer,
                                          is_passthrough=new_method_name == self.discard_node_method_name)
            prepare_graph(candidate, self.methods)
            self.graph.clear()
            self.graph.graph.update(candidate.graph)
            self.graph.add_nodes_from(candidate.nodes(data=True))
            self.graph.add_edges_from(candidate.edges(keys=True, data=True))
            return new_id

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

    def get_max_layer_from_graph(self) -> int:
        max_layer = 0
        for node in self.graph.nodes():
            layer = self.get_node_layer(node)
            if layer > max_layer:
                max_layer = layer
        return max_layer

    @transactional_graph_edit
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
            layer = self.get_node_layer(node)
            if layer >= 0:  # 自动排除了 root/collapse 返回的 -1
                valid_layers.append(layer)

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

    def get_node_layer(self, node_id: str, graph: Optional[nx.DiGraph] = None) -> int:
        """
        安全提取节点层号（带回退机制）
        
        优先级:
            1. 特殊节点 ("root", "collapse") 返回 -1
            2. 节点属性 'layer'（最可靠，创建节点时显式赋值）
            3. 从节点ID解析 "{层}_{索引}_{方法}"
            4. 默认层 0（带警告）
        """
        target_graph = graph if graph is not None else self.graph
        
        if node_id in ("root", "collapse"):
            return -1 
        
        # 优先级 1: 节点属性
        node_data = target_graph.nodes.get(node_id, {})
        if 'layer' in node_data:
            try:
                return int(node_data['layer'])
            except (ValueError, TypeError):
                pass # 如果属性存在但无法转为 int，则回退到 ID 解析
        
        # 优先级 2: 从ID解析
        try:
            return int(node_id.split('_')[0])
        except (ValueError, IndexError):
            logger.warning(
                f"Cannot parse layer from node '{node_id}'. Using default layer 0. | "
                f"无法从节点 '{node_id}' 解析层号，使用默认层 0"
            )
            return 0
