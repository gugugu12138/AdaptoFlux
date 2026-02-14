# file: adaptoflux/model_trainer/method_pool_evolver.py
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from collections import defaultdict
import os
import copy
import logging
from datetime import datetime
import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

from ...core.evolved_method import EvolvedMethod

logger = logging.getLogger(__name__)


class MethodPoolEvolver:
    """
    方法池进化器：封装知识沉积机制（Algorithm 2）
    Method Pool Evolver: Encapsulates knowledge sedimentation mechanism (Algorithm 2)
    
    核心设计原则 | Core Design Principles:
        - 拓扑签名 (layer, in_coords, out_coords) 实现跨任务知识对齐，与节点ID/方法名无关
          Topological signatures enable cross-task alignment independent of node IDs/method names
        - 共识阈值过滤噪声模式，确保抽象质量
          Consensus threshold filters noisy patterns for abstraction quality
        - 图同构聚类避免冗余抽象，提升方法池密度
          Graph isomorphism clustering avoids redundant abstractions
    """
    
    def __init__(self, trainer: 'GraphEvoTrainer'):
        """
        初始化：持有 GraphEvoTrainer 反向引用
        Initialization: Hold back-reference to GraphEvoTrainer
        
        参数 | Args:
            trainer: 父级 GraphEvoTrainer 实例，提供 consensus_threshold 等配置
                     Parent GraphEvoTrainer instance providing configurations like consensus_threshold
        """
        self.trainer = trainer
    
    def evolve(
        self,
        adaptoflux_instance,
        snapshots: List[Any],
        max_methods: int = 1,
        enable_graph_isomorphism_clustering: bool = True,
        evolved_methods_save_dir: Optional[str] = None,
        subgraph_selection_policy: str = "largest"
    ) -> Dict[str, Any]:
        """
        进化主入口：执行 Algorithm 2 的 4 阶段流水线
        Main entry point: Executes 4-stage pipeline of Algorithm 2
        
        阶段流程 | Stage Pipeline:
            1. 拓扑签名频次统计 → 2. 共识图构建 → 3. 子图选择 → 4. 方法抽象注册
            1. Topological signature counting → 2. Consensus graph → 3. Subgraph selection → 4. Method registration
        
        参数 | Args:
            adaptoflux_instance: 目标 AdaptoFlux 实例（新方法将注册至此）
                             Target AdaptoFlux instance (new methods register here)
            snapshots: 历史图快照列表（来自训练周期）
                     List of historical graph snapshots (from training cycles)
            max_methods: 每次进化最多抽象的新方法数
                       Max number of new methods to abstract per evolution
            enable_graph_isomorphism_clustering: 是否启用图同构聚类去重
                                               Enable structure-aware deduplication
            evolved_methods_save_dir: 可选，保存进化方法的目录路径
                                    Optional path to save evolved methods
            subgraph_selection_policy: 子图选择策略 ("largest"|"most_frequent"|"smallest"|"balanced")
                                     Subgraph selection policy
        
        返回 | Returns:
            dict 包含:
                - 'methods_added': int (新增方法数)
                - 'new_method_names': List[str] (新方法名列表)
                - 'consensus_graph': nx.MultiDiGraph (用于调试的共识图)
        """
        # 打印警告：该功能仍在实验阶段 | Print warning: experimental feature
        if self.trainer.verbose:
            logger.warning(
                "MethodPoolEvolver is experimental. Report unexpected behavior to developers. | "
                "MethodPoolEvolver 处于实验阶段，遇到异常行为请报告开发者"
            )
            logger.info(
                f"[Phase 3] Method Pool Evolution: Aligning nodes via (layer, in_coords, out_coords) "
                f"across {len(snapshots)} snapshots... | "
                f"[阶段 3] 方法池进化：基于 (层, 输入坐标, 输出坐标) 对齐 {len(snapshots)} 个快照中的节点..."
            )
        
        # 安全检查：空快照直接返回 | Safety check: empty snapshots return early
        if not snapshots:
            return {
                'methods_added': 0, 
                'new_method_names': [], 
                'consensus_graph': None
            }
        
        # === 阶段 1: 构建拓扑签名频次映射 | Stage 1: Build topological signature frequency map ===
        signature_freq = self._build_signature_frequency_map(snapshots)
        
        # === 阶段 2: 构建共识映射（高频方法）| Stage 2: Build consensus map (high-frequency methods) ===
        consensus_map = self._build_consensus_map(signature_freq)
        
        # 无共识签名则跳过进化 | Skip evolution if no consensus signatures
        if not consensus_map:
            if self.trainer.verbose:
                logger.info(
                    "[Phase 3] No consensus signatures found. Skipping evolution. | "
                    "[阶段 3] 未找到共识签名，跳过进化"
                )
            return {
                'methods_added': 0, 
                'new_method_names': [], 
                'consensus_graph': None
            }
        
        # === 阶段 3: 构建共识图 | Stage 3: Construct consensus graph ===
        consensus_graph, node_id_to_sig = self._construct_consensus_graph(
            snapshots[0].graph_processor.graph, consensus_map
        )
        
        # 共识图为空则返回 | Return if consensus graph empty
        if consensus_graph.number_of_nodes() == 0:
            return {
                'methods_added': 0, 
                'new_method_names': [], 
                'consensus_graph': consensus_graph
            }
        
        # === 阶段 4: 选择进化候选子图 | Stage 4: Select evolution candidate subgraphs ===
        selected_graphs, selected_counts = self._select_evolution_candidates(
            consensus_graph,
            node_id_to_sig,
            max_methods,
            enable_graph_isomorphism_clustering,
            subgraph_selection_policy
        )
        
        # 无候选子图则返回 | Return if no candidates
        if not selected_graphs:
            return {
                'methods_added': 0, 
                'new_method_names': [], 
                'consensus_graph': consensus_graph
            }
        
        # === 阶段 5: 重建可执行图（添加 root/collapse + 重命名）| Stage 5: Reconstruct executable graphs ===
        reconstructed_graphs = self._reconstruct_executable_graphs(
            selected_graphs, consensus_graph, node_id_to_sig, snapshots[0]
        )
        
        # === 阶段 6: 注册进化方法 | Stage 6: Register evolved methods ===
        result = self._register_evolved_methods(
            adaptoflux_instance,
            reconstructed_graphs,
            selected_counts,
            evolved_methods_save_dir,
            enable_graph_isomorphism_clustering,
            subgraph_selection_policy
        )
        
        result['consensus_graph'] = consensus_graph
        return result
    
    # -------------------------------------------------------------------------
    # 阶段 1: 拓扑签名频次统计 | Stage 1: Topological Signature Frequency Counting
    # -------------------------------------------------------------------------
    
    def _build_signature_frequency_map(
        self,
        snapshots: List[Any]
    ) -> Dict[Tuple, Dict[str, int]]:
        """
        统计每个拓扑签名上不同方法的出现频次
        Count method frequencies per topological signature across snapshots
        
        签名格式 | Signature format: 
            (layer: int, in_coords: Tuple[int,...], out_coords: Tuple[int,...])
        该签名唯一标识节点在数据流图中的拓扑角色，与节点ID/方法名无关
        This signature uniquely identifies node's topological role, independent of ID/method name
        
        返回 | Returns:
            嵌套字典: signature_freq[signature][method_name] = count
            Nested dict: signature_freq[signature][method_name] = count
        """
        signature_freq = defaultdict(lambda: defaultdict(int))
        
        for snap_idx, snap in enumerate(snapshots):
            graph = snap.graph_processor.graph
            
            for node_id in graph.nodes():
                try:
                    # 提取拓扑签名 | Extract topological signature
                    sig = self._extract_node_signature(graph, node_id)
                    # 获取方法名，缺失则标记为 'unknown' | Get method name, 'unknown' if missing
                    method = graph.nodes[node_id].get('method_name', 'unknown')
                    # 累加频次 | Accumulate frequency
                    signature_freq[sig][method] += 1
                except ValueError:
                    # 跳过特殊节点（root/collapse）或不可解析节点 | Skip special/unparseable nodes
                    continue
        
        # 打印统计日志 | Log statistics
        if self.trainer.verbose:
            total_signatures = len(signature_freq)
            total_occurrences = sum(sum(counts.values()) for counts in signature_freq.values())
            logger.debug(
                f"Built signature frequency map: {total_signatures} unique signatures, "
                f"{total_occurrences} total node occurrences across {len(snapshots)} snapshots. | "
                f"构建签名频次映射：{total_signatures} 个唯一签名，{total_occurrences} 次节点出现（共 {len(snapshots)} 个快照）"
            )
        
        return signature_freq
    
    def _extract_node_signature(
        self,
        graph: nx.DiGraph,
        node_id: str
    ) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
        """
        提取节点的拓扑签名：(层号, 输入坐标元组, 输出坐标元组)
        Extract node's topological signature: (layer, sorted_in_coords, sorted_out_coords)
        
        该签名对节点命名不敏感，实现跨任务拓扑对齐
        Signature is invariant to node naming, enabling cross-task topological alignment
        """
        # 特殊节点无签名 | Special nodes have no signature
        if node_id in ("root", "collapse"):
            raise ValueError("Special nodes (root/collapse) have no topological signature | 特殊节点无拓扑签名")
        
        # 从节点ID解析层号（格式: "{层}_{索引}_{方法}"）| Parse layer from node ID (format: "{layer}_{index}_{method}")
        try:
            layer = self.get_node_layer(node_id, graph)
        except (ValueError, IndexError):
            raise ValueError(f"Cannot parse layer from node ID: {node_id} | 无法从节点ID解析层号: {node_id}")
        
        # 收集输入边坐标（指向该节点的边）| Collect input edge coordinates (edges TO this node)
        in_coords = []
        for _, _, edge_data in graph.in_edges(node_id, data=True):
            coord = edge_data.get('data_coord')
            if coord is not None:
                in_coords.append(coord)
        in_coords = tuple(sorted(in_coords))
        
        # 收集输出边坐标（从该节点出发的边）| Collect output edge coordinates (edges FROM this node)
        out_coords = []
        for _, _, edge_data in graph.out_edges(node_id, data=True):
            coord = edge_data.get('data_coord')
            if coord is not None:
                out_coords.append(coord)
        out_coords = tuple(sorted(out_coords))
        
        return (layer, in_coords, out_coords)
    
    # -------------------------------------------------------------------------
    # 阶段 2: 共识映射构建 | Stage 2: Consensus Map Construction
    # -------------------------------------------------------------------------
    
    def _build_consensus_map(
        self,
        signature_freq: Dict[Tuple, Dict[str, int]]
    ) -> Dict[Tuple, str]:
        """
        为每个拓扑签名选择高频方法作为共识（需满足共识阈值）
        Select dominant method per signature as consensus (must meet threshold)
        
        参数 | Args:
            signature_freq: 来自 _build_signature_frequency_map 的输出
                          Output from _build_signature_frequency_map
        
        返回 | Returns:
            dict: signature -> consensus_method_name
        """
        consensus_map = {}
        threshold = self.trainer.consensus_threshold  # 可为 None（无阈值）| May be None (no threshold)
        
        for sig, method_counts in signature_freq.items():
            total = sum(method_counts.values())
            max_method, max_count = max(method_counts.items(), key=lambda x: x[1])
            
            # 应用共识阈值（若配置）| Apply consensus threshold if configured
            if threshold is None or (max_count / total) >= threshold:
                consensus_map[sig] = max_method
        
        # 打印日志 | Log
        if self.trainer.verbose:
            logger.debug(
                f"Selected consensus methods for {len(consensus_map)} node roles "
                f"(out of {len(signature_freq)} total signatures). | "
                f"为 {len(consensus_map)} 个节点角色选择共识方法（共 {len(signature_freq)} 个签名）"
            )
        
        return consensus_map
    
    # -------------------------------------------------------------------------
    # 阶段 3: 共识图构建 | Stage 3: Consensus Graph Construction
    # -------------------------------------------------------------------------
    
    def _construct_consensus_graph(
        self,
        template_graph: nx.DiGraph,
        consensus_map: Dict[Tuple, str]
    ) -> Tuple[nx.MultiDiGraph, Dict[str, Tuple]]:
        """
        构建共识图：节点为拓扑签名，边保留原始连接关系
        Build consensus graph: nodes are signatures, edges preserve original connections
        
        保留关键节点属性：
            - method_name（来自共识映射）
            - is_passthrough（来自原始节点，影响执行短路优化）
        Preserve critical node attributes:
            - method_name (from consensus map)
            - is_passthrough (from original node, affects execution short-circuiting)
        
        返回 | Returns:
            元组 (共识图, 节点ID到签名的映射)
            Tuple (consensus_graph, node_id_to_signature_mapping)
        """
        consensus_graph = nx.MultiDiGraph()
        node_id_to_sig = {}
        
        # 添加节点 | Add nodes
        for node_id in template_graph.nodes():
            if node_id in ("root", "collapse"):
                continue
            
            try:
                sig = self._extract_node_signature(template_graph, node_id)
                if sig not in consensus_map:
                    continue
                
                # 保留原始节点关键属性 | Preserve critical attributes from original node
                orig_attrs = template_graph.nodes[node_id]
                new_attrs = {
                    'method_name': consensus_map[sig],
                    'is_passthrough': orig_attrs.get('is_passthrough', False),  # ← 关键：保留 passthrough 属性
                }
                
                consensus_graph.add_node(sig, **new_attrs)
                node_id_to_sig[node_id] = sig
            except ValueError:
                continue  # 跳过不可解析节点 | Skip unparseable nodes
        
        # 添加边 | Add edges
        for src_id, dst_id, edge_data in template_graph.edges(data=True):
            if src_id in node_id_to_sig and dst_id in node_id_to_sig:
                src_sig = node_id_to_sig[src_id]
                dst_sig = node_id_to_sig[dst_id]
                consensus_graph.add_edge(src_sig, dst_sig, **edge_data)
        
        # 打印日志 | Log
        if self.trainer.verbose:
            logger.debug(
                f"Constructed consensus graph with {consensus_graph.number_of_nodes()} nodes "
                f"and {consensus_graph.number_of_edges()} edges. | "
                f"构建共识图：{consensus_graph.number_of_nodes()} 个节点，{consensus_graph.number_of_edges()} 条边"
            )
        
        return consensus_graph, node_id_to_sig
    
    # -------------------------------------------------------------------------
    # 阶段 4: 子图选择（含可选同构聚类）| Stage 4: Subgraph Selection (with Optional Isomorphism Clustering)
    # -------------------------------------------------------------------------
    
    def _select_evolution_candidates(
        self,
        consensus_graph: nx.MultiDiGraph,
        node_id_to_sig: Dict[str, Tuple],
        max_methods: int,
        enable_graph_isomorphism_clustering: bool,
        subgraph_selection_policy: str
    ) -> Tuple[List[nx.MultiDiGraph], List[int]]:
        """
        提取并选择候选子图用于方法抽象
        Extract and select candidate subgraphs for method abstraction
        
        步骤 | Steps:
            1. 分割共识图为弱连通分量
               Split consensus graph into weakly connected components
            2. 按 min_subgraph_size_for_evolution 过滤
               Filter by min_subgraph_size_for_evolution
            3. 可选：同构聚类去重
               Optional: cluster isomorphic structures
            4. 按策略排序并选择 top-K
               Sort by policy and select top-K
        
        返回 | Returns:
            元组 (选中子图列表, 出现频次列表)
            Tuple (selected_subgraphs, occurrence_counts)
        """
        # 步骤 1: 提取弱连通分量 | Step 1: Extract weakly connected components
        components = list(nx.weakly_connected_components(consensus_graph))
        comp_sizes = [len(c) for c in components]
        
        if self.trainer.verbose:
            logger.debug(
                f"[Connectivity Analysis] Found {len(components)} weakly connected components. Sizes: {comp_sizes} | "
                f"[连通性分析] 发现 {len(components)} 个弱连通分量，尺寸: {comp_sizes}"
            )
        
        # 步骤 2: 按最小尺寸过滤 | Step 2: Filter by minimum size
        min_size = self.trainer.min_subgraph_size_for_evolution
        filtered_components = [c for c in components if len(c) >= min_size]
        
        if self.trainer.verbose:
            filtered_sizes = [len(c) for c in filtered_components]
            logger.info(
                f"[Component Filtering] Kept {len(filtered_components)} components (min size ≥ {min_size}). Sizes: {filtered_sizes} | "
                f"[分量过滤] 保留 {len(filtered_components)} 个分量（最小尺寸 ≥ {min_size}），尺寸: {filtered_sizes}"
            )
        
        # 无有效分量则返回 | Return if no valid components
        if not filtered_components:
            return [], []
        
        # 步骤 3: 可选同构聚类 | Step 3: Optional isomorphism clustering
        if enable_graph_isomorphism_clustering:
            selected_graphs, counts = self._cluster_isomorphic_subgraphs(
                consensus_graph, filtered_components, subgraph_selection_policy, max_methods
            )
        else:
            # 无聚类：直接取前 K 个分量 | No clustering: take first K components
            selected_graphs = [
                consensus_graph.subgraph(comp).copy() 
                for comp in filtered_components[:max_methods]
            ]
            counts = [1] * len(selected_graphs)
            
            if self.trainer.verbose:
                logger.info(
                    f"[No Clustering] Selected first {len(selected_graphs)} components out of {len(filtered_components)}. | "
                    f"[无聚类] 从 {len(filtered_components)} 个分量中直接选取前 {len(selected_graphs)} 个"
                )
        
        return selected_graphs, counts
    
    def _cluster_isomorphic_subgraphs(
        self,
        consensus_graph: nx.MultiDiGraph,
        components: List[Set],
        policy: str,
        max_k: int
    ) -> Tuple[List[nx.MultiDiGraph], List[int]]:
        """
        使用 DiGraphMatcher 聚类结构等价子图
        Cluster structurally equivalent subgraphs using DiGraphMatcher
        
        节点匹配: method_name 相等
        边匹配: 禁用（聚焦拓扑+方法语义）
        Node matching: method_name equality
        Edge matching: disabled (focus on topology + method semantics)
        
        返回 | Returns:
            元组 (唯一子图列表, 出现频次列表)
            Tuple (unique_subgraphs, occurrence_counts)
        """
        unique_graphs = []
        graph_counts = []
        
        # 构建匹配器 | Build matchers
        node_match: Callable = lambda n1, n2: n1.get('method_name') == n2.get('method_name')
        edge_match = None  # 禁用边属性匹配以增强鲁棒性 | Disable edge attribute matching for robustness
        
        # 聚类分量 | Cluster components
        for comp in components:
            subg = consensus_graph.subgraph(comp).copy()
            matched = False
            
            for i, rep_g in enumerate(unique_graphs):
                if DiGraphMatcher(rep_g, subg, node_match=node_match, edge_match=edge_match).is_isomorphic():
                    graph_counts[i] += 1
                    matched = True
                    break
            
            if not matched:
                unique_graphs.append(subg)
                graph_counts.append(1)
        
        # 按策略排序 | Sort by policy
        if policy == "most_frequent":
            sort_key = lambda x: (x[1],)  # 仅按频次 | Only frequency
        elif policy in ("largest", "balanced"):
            # 频次优先，其次节点数，最后边数 | Frequency first, then node count, then edge count
            sort_key = lambda x: (x[1], x[0].number_of_nodes(), x[0].number_of_edges())
        elif policy == "smallest":
            # 频次优先，但偏好小结构 | Frequency first, but prefer smaller structures
            sort_key = lambda x: (x[1], -x[0].number_of_nodes(), -x[0].number_of_edges())
        else:
            raise ValueError(f"Unknown subgraph_selection_policy: {policy} | 未知子图选择策略: {policy}")
        
        sorted_pairs = sorted(zip(unique_graphs, graph_counts), key=sort_key, reverse=True)
        top_k = sorted_pairs[:max_k]
        
        selected_graphs = [g for g, _ in top_k]
        selected_counts = [c for _, c in top_k]
        
        # 打印聚类日志 | Log clustering results
        if self.trainer.verbose:
            logger.info(
                f"[Graph Clustering] Identified {len(unique_graphs)} unique structures via isomorphism. "
                f"Selected top-{len(selected_graphs)} using policy '{policy}'. | "
                f"[图聚类] 通过同构检测识别 {len(unique_graphs)} 种唯一结构，按策略 '{policy}' 选取前 {len(selected_graphs)} 个"
            )
            for idx, (g, cnt) in enumerate(sorted_pairs[:5]):  # 仅记录前5个 | Log top 5 only
                logger.debug(
                    f"  Structure {idx+1}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, frequency={cnt} | "
                    f"  结构 {idx+1}: {g.number_of_nodes()} 节点, {g.number_of_edges()} 边, 频次={cnt}"
                )
        
        return selected_graphs, selected_counts
    
    # -------------------------------------------------------------------------
    # 阶段 5: 可执行图重建 | Stage 5: Executable Graph Reconstruction
    # -------------------------------------------------------------------------
    
    def _reconstruct_executable_graphs(
        self,
        selected_graphs: List[nx.MultiDiGraph],
        consensus_graph: nx.MultiDiGraph,
        node_id_to_sig: Dict[str, Tuple],
        template_snapshot: Any
    ) -> List[nx.MultiDiGraph]:
        """
        将抽象共识子图转换为可执行 GDFG：
            1. 添加 root/collapse 节点
            2. 重连入口/出口节点并恢复边属性
            3. 重命名为标准格式 "{层}_{索引}_{方法}"
            4. 按层重分配 data_coord 确保执行正确性
        Convert abstract consensus subgraphs to executable GDFGs:
            1. Add root/collapse nodes
            2. Reconnect entry/exit nodes with original edge attributes
            3. Rename to standard format "{layer}_{index}_{method}"
            4. Reassign data_coord per layer for execution correctness
        
        返回 | Returns:
            可执行图列表（可直接用于方法抽象）
            List of fully executable graphs ready for method abstraction
        """
        reconstructed = []
        template_graph = template_snapshot.graph_processor.graph
        
        # 构建反向映射：签名 → 原始节点ID（用于恢复边属性）| Build reverse mapping: signature → original node ID
        sig_to_original_node = {}
        for node_id, sig in node_id_to_sig.items():
            if sig not in sig_to_original_node:
                sig_to_original_node[sig] = node_id
        
        for subg in selected_graphs:
            # 深拷贝避免污染 | Deep copy to avoid mutation
            g_full = subg.copy()
            
            # 添加特殊节点 | Add special nodes
            g_full.add_node("root")
            g_full.add_node("collapse")
            
            # 识别入口/出口节点 | Identify entry/exit nodes
            internal_nodes = [n for n in g_full.nodes() if n not in ("root", "collapse")]
            entry_nodes = [n for n in internal_nodes if g_full.in_degree(n) == 0]
            exit_nodes = [n for n in internal_nodes if g_full.out_degree(n) == 0]
            
            # === 连接 root 到入口节点 | Connect root to entry nodes ===
            edge_counter = 0
            for sig in entry_nodes:
                orig_node = sig_to_original_node.get(sig)
                if orig_node is None:
                    # 回退：创建默认边 | Fallback: create default edge
                    g_full.add_edge(
                        "root", sig,
                        data_type='scalar',
                        output_index=edge_counter,
                        data_coord=edge_counter
                    )
                    edge_counter += 1
                    continue
                
                # 从模板图恢复原始边属性 | Recover original edge attributes from template graph
                in_edges = list(template_graph.in_edges(orig_node, data=True))
                if not in_edges:
                    g_full.add_edge(
                        "root", sig,
                        data_type='scalar',
                        output_index=edge_counter,
                        data_coord=edge_counter
                    )
                    edge_counter += 1
                    continue
                
                # 按层和 data_coord 排序确保确定性 | Sort by layer and data_coord for determinism
                edge_info = []
                for u, v, attrs in in_edges:
                    if not self._validate_edge_attrs(attrs, {'data_coord', 'data_type'}):
                        continue
                    layer_u = self.get_node_layer(u, template_graph)
                    edge_info.append((layer_u, attrs['data_coord'], attrs))
                
                edge_info.sort(key=lambda x: (x[0], x[1]))
                
                for _, _, attrs in edge_info:
                    g_full.add_edge(
                        "root", sig,
                        data_type=attrs['data_type'],
                        output_index=edge_counter,
                        data_coord=edge_counter
                    )
                    edge_counter += 1
            
            # === 连接出口节点到 collapse | Connect exit nodes to collapse ===
            all_exit_edges = []  # (sig, output_index, data_type, layer_u, orig_data_coord)
            
            for sig in exit_nodes:
                orig_node = sig_to_original_node.get(sig)
                if orig_node is None:
                    raise RuntimeError(
                        f"No original node found for exit signature: {sig} | "
                        f"未找到出口签名 {sig} 对应的原始节点"
                    )
                
                out_edges = list(template_graph.out_edges(orig_node, data=True))
                if not out_edges:
                    layer = self.get_node_layer(orig_node, template_graph)
                    all_exit_edges.append((sig, 0, 'scalar', layer, 0))
                    continue
                
                for u, v, attrs in out_edges:
                    if not self._validate_edge_attrs(attrs, {'data_coord', 'data_type', 'output_index'}):
                        continue
                    layer_u = self.get_node_layer(u, template_graph)
                    all_exit_edges.append((
                        sig,
                        attrs['output_index'],
                        attrs['data_type'],
                        layer_u,
                        attrs['data_coord']
                    ))
            
            # 全局排序：高层优先，同层按原始 data_coord 升序 | Global sort: higher layer first, then by original data_coord
            all_exit_edges.sort(key=lambda x: (-x[3], x[4]))
            
            # 全局分配 data_coord (0,1,2,...) | Global assignment of data_coord
            for data_coord, (sig, output_idx, data_type, _, _) in enumerate(all_exit_edges):
                g_full.add_edge(
                    sig, "collapse",
                    output_index=output_idx,
                    data_coord=data_coord,
                    data_type=data_type
                )
            
            # === 重命名节点为标准格式 | Rename nodes to standard format ===
            try:
                # 通过最短路径计算层号（比解析名称更鲁棒）| Compute layer via shortest path (more robust than name parsing)
                lengths = nx.single_source_shortest_path_length(g_full, "root")
            except nx.NetworkXError:
                # 回退（理论上不应发生）| Fallback (should not happen)
                lengths = {n: 0 for n in g_full.nodes()}
                lengths["root"] = 0
            
            # 重命名内部节点: "{层}_{索引}_{方法}" | Rename internal nodes: "{layer}_{index}_{method}"
            mapping = {}
            layer_method_counter = defaultdict(lambda: defaultdict(int))
            
            for node in internal_nodes:
                method = g_full.nodes[node].get('method_name', 'unknown')
                layer = lengths.get(node, 1)
                idx = layer_method_counter[layer][method]
                layer_method_counter[layer][method] += 1
                mapping[node] = f"{layer}_{idx}_{method}"
            
            g_full = nx.relabel_nodes(g_full, mapping)
            
            # === 按层重分配 data_coord（执行关键）| Reassign data_coord per layer (execution critical) ===
            layer_edges = defaultdict(list)  # layer → 边列表 | layer → list of edges
            
            for src, dst, key, attrs in list(g_full.edges(keys=True, data=True)):
                if src == "root" or dst == "collapse":
                    continue  # 跳过特殊边 | Skip special edges
                
                try:
                    layer = int(src.split('_')[0])
                except (ValueError, IndexError):
                    logger.warning(
                        f"Cannot parse layer from node '{src}'. Skipping edge {src}->{dst}. | "
                        f"无法从节点 '{src}' 解析层号，跳过边 {src}->{dst}"
                    )
                    continue
                
                layer_edges[layer].append((src, dst, key, attrs))
            
            # 每层内重分配 data_coord (0,1,2,...) | Reassign data_coord within each layer
            for layer, edges in layer_edges.items():
                edges.sort(key=lambda x: x[1])  # 按目标节点排序确保确定性 | Sort by destination for determinism
                for new_coord, (src, dst, key, attrs) in enumerate(edges):
                    g_full.edges[src, dst, key]['data_coord'] = new_coord
            
            reconstructed.append(g_full)
            
            # 打印重建日志 | Log reconstruction
            if self.trainer.verbose:
                internal = [n for n in g_full.nodes() if n not in ("root", "collapse")]
                logger.debug(
                    f"Reconstructed executable graph: {len(internal)} internal nodes, "
                    f"{g_full.number_of_edges()} edges. | "
                    f"重建可执行图：{len(internal)} 个内部节点，{g_full.number_of_edges()} 条边"
                )
        
        return reconstructed
    
    # -------------------------------------------------------------------------
    # 阶段 6: 方法注册 | Stage 6: Method Registration
    # -------------------------------------------------------------------------
    
    def _register_evolved_methods(
        self,
        adaptoflux_instance,
        reconstructed_graphs: List[nx.MultiDiGraph],
        occurrence_counts: List[int],
        save_dir: Optional[str] = None,
        enable_graph_isomorphism_clustering: bool = True,  # ← 新增参数
        subgraph_selection_policy: str = "largest"          # ← 新增参数
    ) -> Dict[str, Any]:
        """
        将重建图包装为 EvolvedMethod 并注册到方法池
        Wrap reconstructed graphs as EvolvedMethod and register to method pool
        
        关键步骤 | Critical steps:
            1. 生成唯一方法名（含冲突检测）
               Generate unique method name (with conflict detection)
            2. 从 root/collapse 边推断输入/输出类型
               Infer I/O types from root/collapse edges
            3. 创建 EvolvedMethod 包装器
               Create EvolvedMethod wrapper
            4. 注册到 adaptoflux_instance.methods
               Register to adaptoflux_instance.methods
            5. 重建 graph_processor 以识别新方法
               Rebuild graph_processor to recognize new methods
            6. 可选：保存到磁盘
               Optional: save to disk
        
        返回 | Returns:
            dict 包含 'methods_added' 和 'new_method_names'
        """
        new_method_names = []
        
        # 创建保存目录（如需要）| Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for i, (g_full, count) in enumerate(zip(reconstructed_graphs, occurrence_counts)):
            # 生成唯一方法名（含冲突检测）| Generate unique method name with conflict detection
            base_name = f"evolved_method_{len(adaptoflux_instance.methods) + i + 1}"
            name = base_name
            counter = 1
            while name in adaptoflux_instance.methods:
                name = f"{base_name}_{counter}"
                counter += 1
            
            # 从 root/collapse 边推断 I/O 类型 | Infer I/O types from root/collapse edges
            input_types = [
                edge['data_type'] 
                for _, _, edge in g_full.out_edges("root", data=True)
            ]
            output_types = [
                edge['data_type'] 
                for _, _, edge in g_full.in_edges("collapse", data=True)
            ]
            
            # 构建元数据 | Build metadata
            meta = {
                'name': name,
                'occurrence_count': count,
                'subgraph_node_count': g_full.number_of_nodes(),
                'is_clustered': enable_graph_isomorphism_clustering,
                'selection_policy': subgraph_selection_policy,
                'evolved_at': datetime.now().isoformat(),
                'input_count': len(input_types),
                'input_types': input_types,
                'output_types': output_types,
                'output_count': len(output_types),
                'group': 'evolved',
                'weight': 1.0,
                'vectorized': False,
                'is_evolved': True
            }
            
            # 创建进化方法包装器 | Create evolved method wrapper
            evolved_method = EvolvedMethod(
                name=name,
                graph=g_full,
                methods_ref=adaptoflux_instance.methods,
                metadata=meta
            )
            
            # 注册到方法池 | Register to method pool
            adaptoflux_instance.methods[name] = {
                'function': evolved_method,
                'input_count': meta['input_count'],
                'output_count': meta['output_count'],
                'input_types': meta['input_types'],
                'output_types': meta['output_types'],
                'group': meta['group'],
                'weight': meta['weight'],
                'vectorized': meta['vectorized'],
                'is_evolved': True
            }
            
            new_method_names.append(name)
            
            # 【关键】重建 graph_processor 以识别新方法 | 【Critical】Rebuild graph_processor to recognize new methods
            adaptoflux_instance.path_generator.methods = adaptoflux_instance.methods
            current_layer = adaptoflux_instance.graph_processor.layer
            adaptoflux_instance.graph_processor = adaptoflux_instance._create_graph_processor(
                graph=adaptoflux_instance.graph
            )
            adaptoflux_instance.graph_processor.layer = current_layer
            
            # 可选保存 | Optional save
            if save_dir:
                evolved_method.save(save_dir)
                if self.trainer.verbose:
                    logger.info(
                        f"[Saved] Evolved method '{name}' to {save_dir} | "
                        f"[已保存] 进化方法 '{name}' 到 {save_dir}"
                    )
            
            # 打印注册日志 | Log registration
            if self.trainer.verbose:
                logger.info(
                    f"[Evolved Method] Registered '{name}': {meta['input_count']} inputs → "
                    f"{meta['output_count']} outputs, occurrence count={count} | "
                    f"[进化方法] 注册 '{name}': {meta['input_count']} 输入 → {meta['output_count']} 输出, 出现频次={count}"
                )
        
        return {
            'methods_added': len(new_method_names),
            'new_method_names': new_method_names
        }
    
    # -------------------------------------------------------------------------
    # 工具方法（关键修复）| Utility Methods (Critical Fixes)
    # -------------------------------------------------------------------------
    
    def get_node_layer(self, node_id: str, graph: nx.DiGraph) -> int:
        """
        安全提取节点层号（带回退机制）
        Safely extract node layer with fallback mechanisms
        
        优先级 | Priority:
            1. 节点属性 'layer'（最可靠）
               Node attribute 'layer' (most reliable)
            2. 从节点ID解析 "{层}_{索引}_{方法}"
               Parse from node ID format "{layer}_{index}_{method}"
            3. 默认层 0（带警告）
               Default layer 0 (with warning)
        
        修复原实现缺失的 get_node_layer() | Fixes missing get_node_layer() in original implementation
        """
        if node_id in ("root", "collapse"):
            return -1  # 特殊节点设为 -1 | Special nodes set to -1
        
        # 优先级 1: 节点属性 | Priority 1: Node attribute
        if 'layer' in graph.nodes[node_id]:
            return graph.nodes[node_id]['layer']
        
        # 优先级 2: 从ID解析 | Priority 2: Parse from ID
        try:
            return int(node_id.split('_')[0])
        except (ValueError, IndexError):
            logger.warning(
                f"Cannot parse layer from node '{node_id}'. Using default layer 0. | "
                f"无法从节点 '{node_id}' 解析层号，使用默认层 0"
            )
            return 0
    
    def _validate_edge_attrs(self, edge_attrs: Dict[str, Any], required: Set[str]) -> bool:
        """
        验证边是否包含所有必需属性
        Validate edge has all required attributes
        
        替换脆弱的"理论上不该..."注释，提供显式错误处理
        Replaces fragile "theoretically shouldn't happen" comments with explicit error handling
        """
        missing = required - set(edge_attrs.keys())
        if missing:
            logger.error(
                f"Edge missing required attributes {missing}. Full attrs: {edge_attrs} | "
                f"边缺失必需属性 {missing}，完整属性: {edge_attrs}"
            )
            return False
        return True