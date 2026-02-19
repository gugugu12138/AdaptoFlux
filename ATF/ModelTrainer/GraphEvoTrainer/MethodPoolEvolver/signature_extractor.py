# file: adaptoflux/model_trainer/method_pool_evolver/signature_extractor.py
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import logging
import networkx as nx

# 类型别名：拓扑签名 = (层号, 输入坐标元组, 输出坐标元组)
TopologicalSignature = Tuple[int, Tuple[int, ...], Tuple[int, ...]]

logger = logging.getLogger(__name__)


class SignatureExtractor:
    """
    拓扑签名提取器：负责节点签名提取与跨快照频次统计
    
    核心职责 | Core Responsibilities:
        - 从节点提取拓扑签名 (layer, in_coords, out_coords)
        - 签名对节点ID/方法名不敏感，实现跨任务拓扑对齐
        - 统计每个签名上不同方法的出现频次，支撑共识构建
    
    签名格式 | Signature Format:
        (
            layer: int,                    # 节点所在层号
            in_coords: Tuple[int, ...],    # 输入边 data_coord 排序元组
            out_coords: Tuple[int, ...]    # 输出边 data_coord 排序元组
        )
    """
    
    # 特殊节点ID集合，不参与签名提取
    SPECIAL_NODES: Set[str] = {"root", "collapse"}
    
    def __init__(self, trainer: 'GraphEvoTrainer'):
        """
        初始化签名提取器
        
        参数 | Args:
            trainer: GraphEvoTrainer 实例，提供 verbose 等配置
        """
        self.trainer = trainer
    
    def build_frequency_map(
        self, 
        snapshots: List[Any]
    ) -> Dict[TopologicalSignature, Dict[str, int]]:
        """
        统计每个拓扑签名上不同方法的出现频次
        
        遍历所有快照中的所有节点，提取签名并累加方法出现次数。
        跳过特殊节点和无法解析的节点。
        
        参数 | Args:
            snapshots: 历史图快照列表，每个快照需有 graph_processor.graph 属性
        
        返回 | Returns:
            嵌套字典: signature_freq[signature][method_name] = count
            示例:
                {
                    (0, (), (0,)): {'add': 5, 'multiply': 2},
                    (1, (0,), (1,)): {'relu': 10}
                }
        """
        signature_freq: Dict[TopologicalSignature, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        for snap_idx, snap in enumerate(snapshots):
            graph = snap.graph_processor.graph
            
            for node_id in graph.nodes():
                try:
                    # 提取拓扑签名
                    sig = self.extract_signature(graph, node_id)
                    
                    # 获取方法名，缺失则标记为 'unknown'
                    method = graph.nodes[node_id].get('method_name', 'unknown')
                    
                    # 累加频次
                    signature_freq[sig][method] += 1
                    
                except ValueError as e:
                    # 跳过特殊节点或不可解析节点（预期行为，记录 debug 日志）
                    if self.trainer.verbose:
                        logger.debug(
                            f"Skipping node '{node_id}' in snapshot {snap_idx}: {e} | "
                            f"跳过快照 {snap_idx} 中的节点 '{node_id}': {e}"
                        )
                    continue
                except Exception as e:
                    # 非预期错误，记录警告但不中断流程
                    logger.warning(
                        f"Unexpected error processing node '{node_id}': {e} | "
                        f"处理节点 '{node_id}' 时发生未预期错误: {e}"
                    )
                    continue
        
        # 打印统计日志（verbose 模式）
        if self.trainer.verbose:
            total_signatures = len(signature_freq)
            total_occurrences = sum(
                sum(counts.values()) for counts in signature_freq.values()
            )
            logger.info(
                f"[SignatureExtractor] Built frequency map: {total_signatures} unique signatures, "
                f"{total_occurrences} total node occurrences across {len(snapshots)} snapshots. | "
                f"[签名提取器] 构建频次映射：{total_signatures} 个唯一签名，"
                f"{total_occurrences} 次节点出现（共 {len(snapshots)} 个快照）"
            )
        
        return signature_freq
    
    def extract_signature(
        self, 
        graph: nx.DiGraph, 
        node_id: str
    ) -> TopologicalSignature:
        """
        提取单个节点的拓扑签名
        
        签名由三部分组成：
            1. layer: 节点所在层号（从节点属性或ID解析）
            2. in_coords: 输入边 data_coord 的排序元组
            3. out_coords: 输出边 data_coord 的排序元组
        
        该签名对节点命名不敏感，仅依赖拓扑结构和坐标，
        实现跨任务、跨快照的节点角色对齐。
        
        参数 | Args:
            graph: 当前处理的有向图
            node_id: 目标节点ID
        
        返回 | Returns:
            TopologicalSignature: (layer, sorted_in_coords, sorted_out_coords)
        
        异常 | Raises:
            ValueError: 节点为特殊节点或无法解析层号
        """
        # 特殊节点无拓扑签名
        if node_id in self.SPECIAL_NODES:
            raise ValueError(
                f"Special nodes {self.SPECIAL_NODES} have no topological signature | "
                f"特殊节点 {self.SPECIAL_NODES} 无拓扑签名"
            )
        
        # === 步骤1: 提取层号 ===
        layer = self._get_node_layer(node_id, graph)
        
        # === 步骤2: 收集输入边坐标（指向该节点的边）===
        in_coords = []
        for src, dst, edge_data in graph.in_edges(node_id, data=True):
            coord = edge_data.get('data_coord')
            if coord is not None:
                # 确保坐标为整数
                try:
                    in_coords.append(int(coord))
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid data_coord '{coord}' on in-edge to '{node_id}'. Skipping. | "
                        f"输入边 data_coord '{coord}' 无效，跳过节点 '{node_id}'"
                    )
                    continue
        in_coords = tuple(sorted(in_coords))
        
        # === 步骤3: 收集输出边坐标（从该节点出发的边）===
        out_coords = []
        for src, dst, edge_data in graph.out_edges(node_id, data=True):
            coord = edge_data.get('data_coord')
            if coord is not None:
                try:
                    out_coords.append(int(coord))
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid data_coord '{coord}' on out-edge from '{node_id}'. Skipping. | "
                        f"输出边 data_coord '{coord}' 无效，跳过节点 '{node_id}'"
                    )
                    continue
        out_coords = tuple(sorted(out_coords))
        
        return (layer, in_coords, out_coords)
    
    def _get_node_layer(self, node_id: str, graph: nx.DiGraph) -> int:
        """
        安全提取节点层号（带多级回退机制）
        
        优先级 | Priority:
            1. 节点属性 'layer'（最可靠，显式设置）
            2. 从节点ID解析 "{层}_{索引}_{方法}" 格式
            3. 默认返回 0（记录警告日志）
        
        参数 | Args:
            node_id: 节点ID字符串
            graph: 当前图，用于访问节点属性
        
        返回 | Returns:
            int: 节点层号，特殊节点返回 -1
        """
        # 特殊节点层号设为 -1
        if node_id in self.SPECIAL_NODES:
            return -1
        
        # 优先级1: 节点属性 'layer'
        node_attrs = graph.nodes.get(node_id, {})
        if 'layer' in node_attrs:
            layer = node_attrs['layer']
            # 确保返回整数
            try:
                return int(layer)
            except (ValueError, TypeError):
                logger.warning(
                    f"Node '{node_id}' has invalid layer attribute '{layer}'. "
                    f"Falling back to ID parsing. | "
                    f"节点 '{node_id}' 的 layer 属性 '{layer}' 无效，尝试从ID解析"
                )
        
        # 优先级2: 从节点ID解析（格式: "{layer}_{index}_{method}"）
        try:
            parts = node_id.split('_')
            if len(parts) >= 1:
                return int(parts[0])
        except (ValueError, IndexError) as e:
            logger.debug(
                f"Cannot parse layer from node ID '{node_id}': {e} | "
                f"无法从节点ID '{node_id}' 解析层号: {e}"
            )
        
        # 优先级3: 默认层 0（记录警告）
        logger.warning(
            f"Using default layer 0 for node '{node_id}' (parsing failed). | "
            f"节点 '{node_id}' 解析失败，使用默认层 0"
        )
        return 0
    
    def is_valid_signature(self, sig: TopologicalSignature) -> bool:
        """
        验证拓扑签名的基本合法性
        
        检查项:
            - layer >= 0（特殊节点 -1 除外）
            - coords 为整数元组
        
        参数 | Args:
            sig: 待验证的拓扑签名
        
        返回 | Returns:
            bool: 签名是否合法
        """
        layer, in_coords, out_coords = sig
        
        # 允许特殊节点签名（layer=-1）
        if layer < -1:
            return False
        
        # 检查坐标元组元素类型
        for coord in in_coords + out_coords:
            if not isinstance(coord, int):
                return False
        
        return True
    
    def signature_to_str(self, sig: TopologicalSignature) -> str:
        """
        将拓扑签名转换为可读字符串（用于日志/调试）
        
        参数 | Args:
            sig: 拓扑签名元组
        
        返回 | Returns:
            str: 格式化字符串，如 "L0_in[]_out[0,1]"
        """
        layer, in_coords, out_coords = sig
        in_str = ','.join(map(str, in_coords)) if in_coords else '∅'
        out_str = ','.join(map(str, out_coords)) if out_coords else '∅'
        return f"L{layer}_in[{in_str}]_out[{out_str}]"