from typing import Any, List, Optional, Dict
import networkx as nx
from networkx.readwrite import json_graph
from ..GraphManager.graph_processor import GraphProcessor

import logging
logger = logging.getLogger(__name__)


class EvolvedMethod:
    """
    表示一个从方法池进化阶段（Method Pool Evolution）中抽象出的可执行子图方法。

    此类封装了一个完整的子图（包含 root 和 collapse 节点），并提供与普通方法相同的调用接口，
    使其可无缝注册到 AdaptoFlux 的全局方法池（methods）中，被图构建器（如 append_nx_layer）直接调用。

    其核心作用是将高频、鲁棒的连通子图结构转化为可复用的“计算原语”，实现知识的自动沉淀与跨任务迁移，
    符合论文第 3.3.4 节“方法池进化”机制的设计目标。
    """

    def __init__(
        self,
        name: str,
        graph: nx.MultiDiGraph,
        methods_ref: Dict[str, Any],          # ← 新增：对全局 methods 的引用
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化一个进化方法实例。

        Parameters
        ----------
        name : str
            方法名称，需全局唯一，通常格式为 'evolved_method_{index}'。
        graph : nx.MultiDiGraph
            完整的子图结构，必须包含 'root' 和 'collapse' 节点，边需携带 data_type、data_coord 等元信息。
        metadata : dict, optional
            元属性字典，建议包含以下字段（与 @method_profile 兼容）：
            - 'input_types': List[str]，输入数据类型列表
            - 'output_types': List[str]，输出数据类型列表
            - 'output_count': int，输出数量
            - 'group': str，方法分组（如 'evolved'）
            - 'weight': float，选择权重
            - 'is_evolved': bool，标记为进化方法
            - 'occurrence_count': int，出现频次
            - 'evolved_at': str，进化时间（ISO 格式）
        """
        self.name = name
        self.graph = graph
        self.methods_ref = methods_ref        # 全局方法池引用
        self.metadata = metadata or {}
        self._processor: Optional[GraphProcessor] = None

    def __call__(self, *inputs) -> List[Any]:
        """
        执行该进化方法的前向推理。

        内部通过 GraphProcessor 对子图进行调度执行，输入数据从 'root' 注入，结果从 'collapse' 收集。

        Parameters
        ----------
        *inputs : Any
            输入数据，数量和类型需与 metadata['input_types'] 一致。

        Returns
        -------
        List[Any]
            输出结果列表，长度等于 metadata['output_count']。
        """
        if self._processor is None:
            # 传入全局 methods_ref
            self._processor = GraphProcessor(
                graph=self.graph,
                methods=self.methods_ref,     # ← 关键：使用主类的 methods
                collapse_method=CollapseMethod.IDENTITY  # 或从 metadata 读取
            )
        return self._processor.run(*inputs)
    @classmethod
    def from_files(cls, base_path: str):
        """
        从磁盘文件加载一个 EvolvedMethod 实例。

        期望存在两个文件：
        - {base_path}.json：图结构（由 json_graph.node_link_data 生成）
        - {base_path}.meta.json：元属性（JSON 格式）

        Parameters
        ----------
        base_path : str
            文件路径前缀（不含扩展名），例如 'evolved_methods/evolved_method_1'

        Returns
        -------
        EvolvedMethod
            重建的进化方法实例。

        Raises
        ------
        FileNotFoundError
            若 .json 或 .meta.json 文件缺失。
        JSONDecodeError
            若文件内容非合法 JSON。
        """
        import json
        # 加载图结构
        with open(base_path + ".json", 'r', encoding='utf-8') as f:
            g_data = json.load(f)
        graph = json_graph.node_link_graph(g_data, edges="edges")
        # 加载元属性
        with open(base_path + ".meta.json", 'r', encoding='utf-8') as f:
            meta = json.load(f)
        return cls(name=meta['name'], graph=graph, metadata=meta)

    def save(self, dir_path: str):
        """
        将当前进化方法持久化到指定目录。

        生成三个文件：
        - {dir_path}/{self.name}.json：图结构（程序可读，用于加载）
        - {dir_path}/{self.name}.gexf：图结构（可视化友好，支持 Gephi 等工具）
        - {dir_path}/{self.name}.meta.json：元属性（结构化存储，含 method_profile 字段）

        此格式便于版本控制、人工审查、可视化调试与主类自动加载。

        Parameters
        ----------
        dir_path : str
            保存目录路径，若不存在将自动创建。
        """
        import os
        import json

        os.makedirs(dir_path, exist_ok=True)
        base = os.path.join(dir_path, self.name)

        try:
            # 1. 保存 GEXF（可视化）
            nx.write_gexf(self.graph, base + ".gexf")

            # 2. 保存 JSON（程序加载）
            data = json_graph.node_link_data(self.graph, edges="edges")
            with open(base + ".json", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # 3. 保存元属性
            with open(base + ".meta.json", 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

            # 4. 打印日志（中英双语，风格统一）
            internal_nodes = [n for n in self.graph.nodes() if n not in ("root", "collapse")]
            logger.info(
                f"[Method Export] Evolved method: name='{self.name}', "
                f"nodes={len(internal_nodes)}, edges={self.graph.number_of_edges()}, "
                f"occurrence={self.metadata.get('occurrence_count', 'N/A')}. "
                f"进化方法：名称='{self.name}'，内部节点数={len(internal_nodes)}，"
                f"边数={self.graph.number_of_edges()}，出现次数={self.metadata.get('occurrence_count', 'N/A')}"
            )
            logger.info(
                f"[File Saved] Successfully saved evolved method to: {base}.{{gexf,json,meta.json}}. "
                f"文件已保存：进化方法已写入 {base}.{{gexf,json,meta.json}}"
            )

        except Exception as e:
            logger.error(
                f"[Save Failed] Failed to save evolved method {self.name}: {e}. "
                f"保存失败：无法保存进化方法 {self.name}：{e}"
            )