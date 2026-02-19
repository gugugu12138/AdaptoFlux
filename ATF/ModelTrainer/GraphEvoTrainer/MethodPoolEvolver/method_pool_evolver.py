# file: adaptoflux/model_trainer/method_pool/evolver.py
"""
方法池进化器主入口：编排各组件完成 Algorithm 2 流水线
"""
from typing import List, Dict, Any, Optional
import logging
import os
from datetime import datetime

from ...core.evolved_method import EvolvedMethod
from .signature_analyzer import SignatureAnalyzer
from .graph_processor import EvolutionGraphProcessor

logger = logging.getLogger(__name__)


class MethodPoolEvolver:
    """
    方法池进化器：封装知识沉积机制（Algorithm 2）
    
    通过组合 SignatureAnalyzer 和 EvolutionGraphProcessor 
    实现 6 阶段进化流水线
    """
    
    def __init__(self, trainer: 'GraphEvoTrainer'):
        self.trainer = trainer
        # 组合子模块
        self._analyzer = SignatureAnalyzer(
            consensus_threshold=trainer.consensus_threshold
        )
        self._graph_proc = EvolutionGraphProcessor(
            min_subgraph_size=trainer.min_subgraph_size_for_evolution,
            verbose=trainer.verbose
        )
    
    def evolve(
        self,
        adaptoflux_instance,
        snapshots: List[Any],
        max_methods: int = 1,
        enable_graph_isomorphism_clustering: bool = True,
        evolved_methods_save_dir: Optional[str] = None,
        subgraph_selection_policy: str = "largest",
        enable_deduplication: bool = True,
        deduplication_mode: str = "hybrid",
        deduplication_granularity: str = "semantic"
    ) -> Dict[str, Any]:
        """
        进化主入口：编排 6 阶段流水线
        
        阶段流程:
            1. 签名分析 (SignatureAnalyzer)
            2. 共识构建 (SignatureAnalyzer)  
            3. 共识图构建 (EvolutionGraphProcessor)
            4. 子图选择 (EvolutionGraphProcessor)
            5. 图重建 (EvolutionGraphProcessor)
            6. 方法注册 (本类)
        """
        # 安全检查
        if not snapshots:
            return self._empty_result()
        
        if self.trainer.verbose:
            logger.info(f"[Phase 3] Starting evolution with {len(snapshots)} snapshots...")
        
        # === 阶段 1-2: 签名分析 ===
        signature_freq = self._analyzer.build_signature_frequency_map(snapshots)
        consensus_map = self._analyzer.build_consensus_map(signature_freq)
        
        if not consensus_map:
            return self._empty_result()
        
        # === 阶段 3-4: 图处理 ===
        template_graph = snapshots[0].graph_processor.graph
        consensus_graph, node_id_to_sig = self._graph_proc.construct_consensus_graph(
            template_graph, consensus_map
        )
        
        if consensus_graph.number_of_nodes() == 0:
            return self._empty_result(consensus_graph)
        
        selected_graphs, selected_counts = self._graph_proc.select_evolution_candidates(
            consensus_graph, node_id_to_sig, max_methods,
            enable_graph_isomorphism_clustering, subgraph_selection_policy
        )
        
        if not selected_graphs:
            return self._empty_result(consensus_graph)
        
        # === 阶段 5: 图重建 ===
        reconstructed = self._graph_proc.reconstruct_executable_graphs(
            selected_graphs, consensus_graph, node_id_to_sig, snapshots[0]
        )
        
        # === 阶段 6: 方法注册（本类核心职责）===
        return self._register_methods(
            adaptoflux_instance,
            reconstructed,
            selected_counts,
            evolved_methods_save_dir,
            enable_deduplication,
            deduplication_mode,
            deduplication_granularity,
            consensus_graph
        )
    
    def _register_methods(
        self,
        adaptoflux_instance,
        graphs: List[nx.MultiDiGraph],
        counts: List[int],
        save_dir: Optional[str],
        enable_dedup: bool,
        dedup_mode: str,
        dedup_granularity: str,
        consensus_graph
    ) -> Dict[str, Any]:
        """内部方法：注册进化方法（含去重检查）"""
        # ... 原有 _register_evolved_methods 逻辑，但调用 self._graph_proc.check_duplicate ...
        # 注意：这里需要传入 adaptoflux_instance.methods 给 graph_proc
    
    def _empty_result(self, consensus_graph=None):
        """辅助方法：返回空结果"""
        return {
            'methods_added': 0,
            'new_method_names': [],
            'consensus_graph': consensus_graph,
            'duplicates_skipped': 0,
            'deduplication_mode': 'disabled'
        }
    
    # === 代理方法（方便外部调用）===
    
    def clear_deduplication_cache(self):
        """清除去重缓存"""
        self._graph_proc.clear_cache()