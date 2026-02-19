# file: adaptoflux/model_trainer/method_pool/signature_analyzer.py
"""
拓扑签名分析器：负责从快照中提取签名、统计频次、构建共识映射
"""
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class SignatureAnalyzer:
    """
    封装拓扑签名相关的分析逻辑
    """
    
    def __init__(self, consensus_threshold: Optional[float] = None):
        self.consensus_threshold = consensus_threshold
    
    def build_signature_frequency_map(
        self, 
        snapshots: List[Any]
    ) -> Dict[Tuple, Dict[str, int]]:
        """阶段 1: 统计每个拓扑签名上不同方法的出现频次"""
        # ... 原有 _build_signature_frequency_map 逻辑 ...
    
    def extract_node_signature(
        self,
        graph: nx.DiGraph,
        node_id: str
    ) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
        """提取节点的拓扑签名 (layer, in_coords, out_coords)"""
        # ... 原有 _extract_node_signature 逻辑 ...
    
    def build_consensus_map(
        self,
        signature_freq: Dict[Tuple, Dict[str, int]]
    ) -> Dict[Tuple, str]:
        """阶段 2: 为每个签名选择高频方法作为共识"""
        # ... 原有 _build_consensus_map 逻辑 ...
    
    @staticmethod
    def get_node_layer(node_id: str, graph: nx.DiGraph) -> int:
        """工具: 安全提取节点层号"""
        # ... 原有 get_node_layer 逻辑 ...