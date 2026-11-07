from typing import Any, List, Optional, Dict
import networkx as nx
from networkx.readwrite import json_graph
from ..GraphProcessor.graph_processor import GraphProcessor
from ..CollapseManager.collapse_functions import CollapseMethod
import numpy as np
from ..GraphProcessor.execution_plan import aggregate_effects, normalize_effect_spec, restore_execution_plan

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
        self.methods_ref = methods_ref        # 全局方法池引用
        self.metadata = dict(metadata or {})
        # 复合方法的副作用来自内部方法，不能只信任外部传入的“pure”标签。
        # 构造时核对已保存声明，可发现加载时方法库效果契约发生了变化。
        self.effects = aggregate_effects(graph, methods_ref)
        saved_effects = self.metadata.get("effects")
        if saved_effects is not None and normalize_effect_spec(saved_effects) != self.effects:
            raise ValueError("Composite effect declaration differs from its constituent methods")
        self.metadata["effects"] = self.effects
        self._processor: Optional[GraphProcessor] = GraphProcessor(
            graph=graph, methods=methods_ref, collapse_method=CollapseMethod.IDENTITY)
        # 注册进方法池前就冻结内部计划；不能等首次被并行调用时才确定状态顺序。
        # 运行锁属于 GraphProcessor，其复制/恢复钩子负责重建，不写入模型元数据。
        self._processor.prepare_execution_plan()
        self.graph = self._processor.graph
        
    def __call__(self, *inputs):
        """
        将输入组织为列表，直接调用子图的 infer_with_graph_single，并透传其返回值。
        不做任何标准化、拆分或格式转换。
        """
        # 方法池引用是共享的。若调用方后来修改了某个内部方法的读写声明，
        # 外层调度器掌握的复合效果就可能过期；在任何内部副作用执行前拒绝调用。
        if aggregate_effects(self.graph, self.methods_ref) != self.effects:
            raise ValueError("Constituent effects changed after composite registration")
        # === 懒初始化 processor ===
        if self._processor is None:
            self._processor = GraphProcessor(
                graph=self.graph,
                methods=self.methods_ref,
                collapse_method=CollapseMethod.IDENTITY
            )

        # === 将 *inputs 组织为 sample（list 或 tuple）===
        sample = list(inputs)  # 也可以用 tuple(inputs)，取决于 infer_with_graph_single 的期望

        result = self._processor.infer_with_graph_single(sample)

        # 获取期望的输出数量
        expected_output_count = self.metadata.get('output_count', 1)

        # 情况 1: 期望单输出 → 直接返回（转为标量）
        if expected_output_count == 1:
            if isinstance(result, np.ndarray):
                if result.ndim == 0:
                    return float(result.item())
                elif result.size == 1:
                    return float(result.flat[0])
                else:
                    raise ValueError(f"Expected single output, got array of size {result.size}")
            else:
                val = result

            # 允许 val 为 None
            if val is None:
                return None
            else:
                return float(val)

        # 情况 2: 期望多输出 → 必须拆分为 expected_output_count 个标量
        else:
            # 确保 result 可展平为标量序列
            if isinstance(result, np.ndarray):
                flat = result.flatten()
            elif isinstance(result, (list, tuple)):
                flat = np.array(result).flatten()
            else:
                raise TypeError(f"Unexpected result type: {type(result)}")

            if flat.size != expected_output_count:
                raise ValueError(
                    f"Result has {flat.size} elements, but output_count={expected_output_count}. "
                    f"Result: {result}"
                )

            # 转为 tuple of Python floats（最安全）
            return tuple(float(x) if x is not None else None for x in flat)

    @classmethod
    def from_files(cls, base_path: str, methods_ref: Optional[Dict[str, Any]] = None):
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
        # 图文件只保存结构和声明，不保存任意 Python 函数实现；调用方必须显式
        # 提供原语注册表，随后由构造函数核对声明并恢复内部顺序。
        if methods_ref is None:
            raise ValueError("Loading an evolved method requires its constituent methods_ref registry")
        # 加载图结构
        with open(base_path + ".json", 'r', encoding='utf-8') as f:
            g_data = json.load(f)
        graph = json_graph.node_link_graph(g_data, edges="edges")
        # 总效果并集相同并不表示每个节点的契约相同。例如将一次写声明移到
        # 另一个节点，总 writes 仍相同，但控制依赖可能改变。因此新格式同时
        # 校验逐节点签名/效果、数据端口和冻结顺序，不能只比较 metadata.effects。
        saved_plan = g_data.get("execution_plan")
        if "execution_plan" in g_data and not isinstance(saved_plan, dict):
            raise ValueError("Invalid embedded composite execution plan")
        if saved_plan is not None:
            restore_execution_plan(graph, methods_ref, saved_plan)
        elif any("_atf_uid" in attrs for _, attrs in graph.nodes(data=True)):
            # 新格式的 UID 已表明此程序曾冻结过顺序。计划丢失属于损坏，
            # 不能冒充旧模型按文件插入顺序重新生成一个可能不同的程序。
            raise ValueError("A planned composite model is missing its execution_plan")
        # 没有 UID/计划的历史文件仍可迁移：构造函数会在此次载入边界选择
        # 一次保守顺序并冻结，之后的运行和保存都沿用该顺序。
        # 加载元属性
        with open(base_path + ".meta.json", 'r', encoding='utf-8') as f:
            meta = json.load(f)
        return cls(name=meta['name'], graph=graph, metadata=meta, methods_ref=methods_ref)

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

        # 保存已评价的程序顺序。JSON 节点或边即使被重排，载入后也按保存的
        # 稳定身份恢复语义顺序，而不是根据文件中的先后位置重新选一个拓扑序。
        plan = self._processor.prepare_execution_plan()
        self.graph = self._processor.graph
        effects = aggregate_effects(self.graph, self.methods_ref)
        if effects != self.effects:
            raise ValueError("Constituent effects changed after composite registration")

        try:
            # 1. 保存 GEXF（可视化）
            nx.write_gexf(self.graph, base + ".gexf")

            # 2. 保存 JSON（程序加载）
            data = json_graph.node_link_data(self.graph, edges="edges")
            # 完整计划内嵌在同一个 JSON 中，复合方法不依赖外层模型的侧车文件。
            # 图元数据保存语义顺序；这里的契约快照额外用于拒绝不一致的重载。
            data["execution_plan"] = plan.to_dict()
            with open(base + ".json", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # 3. 保存元属性
            with open(base + ".meta.json", 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

            # 4. 打印日志（中英双语，风格统一）
            internal_nodes = [n for n in self.graph.nodes() if n not in ("root", "collapse")]
            logger.debug(
                f"[Method Export] Evolved method: name='{self.name}', "
                f"nodes={len(internal_nodes)}, edges={self.graph.number_of_edges()}, "
                f"occurrence={self.metadata.get('occurrence_count', 'N/A')}. "
                f"进化方法：名称='{self.name}'，内部节点数={len(internal_nodes)}，"
                f"边数={self.graph.number_of_edges()}，出现次数={self.metadata.get('occurrence_count', 'N/A')}"
            )
            logger.debug(
                f"[File Saved] Successfully saved evolved method to: {base}.{{gexf,json,meta.json}}. "
                f"文件已保存：进化方法已写入 {base}.{{gexf,json,meta.json}}"
            )

        except Exception as e:
            logger.error(
                f"[Save Failed] Failed to save evolved method {self.name}: {e}. "
                f"保存失败：无法保存进化方法 {self.name}：{e}"
            )
            raise
