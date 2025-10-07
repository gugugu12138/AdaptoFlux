import networkx as nx
import random
from typing import Optional, Set

class BFSSubgraphSampler:
    """
    使用 BFS 随机采样连通子图。
    """
    def __init__(self, max_nodes: int = 5, exclude_nodes: Set[str] = None):
        self.max_nodes = max_nodes
        self.exclude_nodes = exclude_nodes or {"root", "collapse"}

    def sample(self, graph: nx.DiGraph) -> Optional[nx.DiGraph]:
        """返回一个连通子图，若无法采样则返回 None"""
        candidate_starts = [n for n in graph.nodes() if n not in self.exclude_nodes]
        if not candidate_starts:
            return None

        start = random.choice(candidate_starts)
        visited = {start}
        queue = [start]

        while queue and len(visited) < self.max_nodes:
            node = queue.pop(0)
            for nb in graph.successors(node):
                if nb not in visited and nb not in self.exclude_nodes:
                    visited.add(nb)
                    queue.append(nb)
                    if len(visited) >= self.max_nodes:
                        break

        # 至少需要2个节点才有压缩意义
        if len(visited) < 2:
            return None
        
        return graph.subgraph(visited).copy()