# file: adaptoflux/model_trainer/method_pool/graph_processor.py
"""
图处理器：负责共识图构建、子图选择、去重检测、可执行图重建
"""
from typing import List, Dict, Tuple, Set, Callable, Optional
import hashlib
import json
import logging
import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

logger = logging.getLogger(__name__)


class EvolutionGraphProcessor:
    """
    封装图操作相关的进化逻辑
    """
    
    def __init__(self, 
                 min_subgraph_size: int = 3,
                 verbose: bool = False):
        self.min_subgraph_size = min_subgraph_size
        self.verbose = verbose
        # 去重缓存
        self._fingerprint_cache: Dict[str, str] = {}
        self._fingerprint_to_method: Dict[str, str] = {}
    
    # === 图构建与选择 ===
    
    def construct_consensus_graph(
        self,
        template_graph: nx.DiGraph,
        consensus_map: Dict[Tuple, str]
    ) -> Tuple[nx.MultiDiGraph, Dict[str, Tuple]]:
        """阶段 3: 构建共识图"""
        # ... 原有 _construct_consensus_graph 逻辑 ...
    
    def select_evolution_candidates(
        self,
        consensus_graph: nx.MultiDiGraph,
        node_id_to_sig: Dict[str, Tuple],
        max_methods: int,
        enable_clustering: bool,
        policy: str
    ) -> Tuple[List[nx.MultiDiGraph], List[int]]:
        """阶段 4: 选择候选子图（含同构聚类）"""
        # ... 原有 _select_evolution_candidates + _cluster_isomorphic_subgraphs ...
    
    # === 去重模块（独立子模块）===
    
    def check_duplicate(
        self,
        new_graph: nx.MultiDiGraph,
        existing_methods: Dict[str, Any],
        mode: str = "hybrid",
        granularity: str = "semantic"
    ) -> Tuple[bool, Optional[str], str]:
        """检查是否与现有方法重复"""
        # ... 原有 _check_duplicate 逻辑 ...
    
    def compute_graph_fingerprint(
        self, 
        graph: nx.MultiDiGraph, 
        granularity: str = "semantic"
    ) -> str:
        """计算图的规范化指纹"""
        # ... 原有 _compute_graph_fingerprint 逻辑 ...
    
    def is_duplicate_via_isomorphism(
        self,
        new_graph: nx.MultiDiGraph,
        existing_graph: nx.MultiDiGraph,
        granularity: str
    ) -> bool:
        """通过图同构检测判断重复"""
        # ... 原有同构检测逻辑 ...
    
    def initialize_fingerprint_cache(self, methods: Dict[str, Any], granularity: str):
        """初始化指纹缓存"""
        # ... 缓存初始化逻辑 ...
    
    def clear_cache(self):
        """清除去重缓存"""
        self._fingerprint_cache = {}
        self._fingerprint_to_method = {}
    
    # === 图重建 ===
    
    def reconstruct_executable_graphs(
        self,
        selected_graphs: List[nx.MultiDiGraph],
        consensus_graph: nx.MultiDiGraph,
        node_id_to_sig: Dict[str, Tuple],
        template_snapshot: Any
    ) -> List[nx.MultiDiGraph]:
        """阶段 5: 重建可执行图"""
        # ... 原有 _reconstruct_executable_graphs 逻辑 ...
    
    @staticmethod
    def validate_edge_attrs(edge_attrs: Dict[str, Any], required: Set[str]) -> bool:
        """工具: 验证边属性"""
        # ... 原有 _validate_edge_attrs 逻辑 ...