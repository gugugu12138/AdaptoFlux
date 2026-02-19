@dataclass
class EvolutionConfig:
    """
    方法池进化配置中心 | Method Pool Evolution Configuration
    
    用途 | Purpose:
        集中管理 Algorithm 2 的所有可配置参数，支持:
        - 向后兼容的传统参数传递
        - 模块化嵌套配置（signature/consensus/subgraph 等）
        - 字典/JSON/YAML 外部加载
        - IDE 自动补全和类型检查
    
    示例 | Example:
        >>> config = EvolutionConfig(
        ...     consensus_threshold=0.7,
        ...     subgraph={"max_size": 20, "complexity_penalty": 0.1}
        ... )
        >>> evolver.evolve(..., config=config)
    """
    
    # === 核心参数（保持向后兼容）===
    consensus_threshold: Optional[float] = None
    """
    共识阈值：方法在签名中出现的最小比例
    
    取值范围 | Range: [0.0, 1.0] 或 None（禁用阈值）
    默认行为 | Default: None（仅按 max_vote 选择，不过滤）
    
    示例 | Examples:
        - 0.6: 方法需在该签名出现 ≥60% 才被选为共识
        - None: 总是选择出现次数最多的方法（可能引入噪声）
    
    相关配置 | Related:
        consensus.min_absolute_count: 绝对次数阈值（与比例阈值互补）
    """
    
    min_subgraph_size: int = 2
    """
    进化子图的最小节点数
    
    用途 | Purpose: 过滤过小的子图（单节点抽象价值低）
    建议值 | Recommendation: 2-5（根据任务复杂度调整）
    
    注意 | Note:
        过大会减少候选方法数量，过小可能产生冗余抽象
    """
    
    max_methods: int = 1
    """
    单次进化最多抽象的新方法数量
    
    用途 | Purpose: 控制方法池增长速度和内存占用
    建议 | Tip: 训练初期可设大（3-5），后期设小（1-2）精细优化
    """
    
    subgraph_selection_policy: str = "largest"
    """
    子图选择策略
    
    可选值 | Options:
        - "largest": 优先选择节点数最多的子图（默认）
        - "most_frequent": 优先选择出现频次最高的子图
        - "smallest": 优先选择最小结构（适合轻量抽象）
        - "balanced": 频次×大小的加权平衡策略
    
    影响 | Affects: _cluster_isomorphic_subgraphs() 的排序逻辑
    """
    
    # === 模块化配置（嵌套字典）===
    
    signature: Dict[str, Any] = field(default_factory=dict)
    """
    拓扑签名提取配置
    
    可用键 | Available Keys:
        coord_tolerance: float (默认: 1e-9)
            坐标匹配容差，处理浮点计算误差
        include_edge_types: bool (默认: True)
            签名是否包含边类型信息（增强区分度）
        normalize_coords: bool (默认: False)
            是否归一化坐标（跨任务对齐时启用）
        skip_methods: Set[str] (默认: empty)
            跳过的方法名列表（如 "identity", "pass"）
    
    示例 | Example:
        signature={
            "coord_tolerance": 1e-6,
            "skip_methods": {"identity", "reshape"}
        }
    """
    
    consensus: Dict[str, Any] = field(default_factory=dict)
    """
    共识构建策略配置
    
    可用键 | Available Keys:
        min_absolute_count: Optional[int]
            最小绝对出现次数（与 consensus_threshold 互补）
        aggregation_strategy: str (默认: "max_vote")
            "max_vote" | "weighted_avg" | "first_seen"
        min_snapshots_required: int (默认: 2)
            计算共识所需的最小快照数量
        tie_breaker: str (默认: "alphabetical")
            票数相同时的决胜策略
    
    示例 | Example:
        consensus={
            "min_absolute_count": 3,  # 至少出现3次
            "tie_breaker": "most_recent"  # 选最近出现的方法
        }
    """
    
    subgraph: Dict[str, Any] = field(default_factory=dict)
    """
    子图选择与过滤配置
    
    可用键 | Available Keys:
        max_subgraph_size: Optional[int]
            最大子图节点数限制（防止方法过大）
        complexity_penalty: float (默认: 0.0)
            节点数惩罚系数：score = freq × exp(-penalty × nodes)
        prefer_dag_only: bool (默认: True)
            仅选择 DAG 结构子图（避免循环依赖）
        min_entry_exit_ratio: Tuple[float, float] (默认: (0.1, 10.0))
            入口/出口节点比例约束
        exclude_passthrough_chains: bool (默认: False)
            排除纯 passthrough 链（抽象价值低）
    
    示例 | Example:
        subgraph={
            "max_subgraph_size": 15,
            "complexity_penalty": 0.05,
            "prefer_dag_only": True
        }
    """
    
    isomorphism: Dict[str, Any] = field(default_factory=dict)
    """
    图同构聚类配置
    
    可用键 | Available Keys:
        node_match_attrs: List[str] (默认: ["method_name"])
            节点匹配时比较的属性列表
        edge_match_attrs: Optional[List[str]] (默认: None)
            边匹配属性（None=禁用，聚焦拓扑结构）
        timeout_seconds: Optional[float] (默认: 30.0)
            单次同构检测超时（防止 NP-hard 问题卡死）
        approximate_similarity: Optional[float] (默认: None)
            启用近似同构的阈值 [0,1]（None=精确匹配）
        cluster_by_signature_prefix: bool (默认: False)
            先按签名前缀预分组加速聚类
    
    警告 | Warning:
        启用 approximate_similarity 可能降低抽象质量，
        建议仅在大规模图且性能瓶颈时启用
    
    示例 | Example:
        isomorphism={
            "node_match_attrs": ["method_name", "is_passthrough"],
            "timeout_seconds": 60.0
        }
    """
    
    registration: Dict[str, Any] = field(default_factory=dict)
    """
    方法注册与命名配置
    
    可用键 | Available Keys:
        naming_strategy: str (默认: "sequential")
            "sequential" | "semantic" | "hash_based"
        name_prefix: str (默认: "evolved_method")
            生成方法名的前缀
        include_signature_in_name: bool (默认: False)
            名称中是否包含拓扑签名哈希（增强唯一性）
        conflict_resolution: str (默认: "suffix")
            "suffix" (添加_1) | "overwrite" (覆盖) | "skip" (跳过)
        auto_rebuild_processor: bool (默认: True)
            注册后是否自动重建 graph_processor
        metadata_fields: Optional[List[str]]
            仅保存指定元数据字段（减少存储）
    
    示例 | Example:
        registration={
            "naming_strategy": "semantic",
            "name_prefix": "pattern",
            "conflict_resolution": "skip"
        }
    """
    
    performance: Dict[str, Any] = field(default_factory=dict)
    """
    性能与资源控制配置
    
    可用键 | Available Keys:
        max_signature_entries: Optional[int]
            签名映射最大条目数（LRU 淘汰，防止内存溢出）
        parallel_extraction: bool (默认: False)
            并行提取签名（需确保 thread-safe）
        cache_consensus_graph: bool (默认: True)
            缓存共识图避免重复构建
        batch_process_snapshots: int (默认: 1)
            批量处理快照数（内存/速度权衡）
        early_stop_if_no_progress: bool (默认: True)
            连续 N 轮无新方法时提前终止
    
    示例 | Example:
        performance={
            "max_signature_entries": 10000,
            "parallel_extraction": True,
            "early_stop_if_no_progress": True
        }
    """
    
    debug: Dict[str, Any] = field(default_factory=dict)
    """
    调试与可观测性配置
    
    可用键 | Available Keys:
        export_signature_freq_csv: Optional[str]
            导出频次统计到 CSV 的文件路径
        export_consensus_graph_dot: Optional[str]
            导出共识图为 DOT 格式（可用 Graphviz 可视化）
        log_selected_subgraphs_detail: bool (默认: False)
            详细记录选中子图的结构信息
        track_evolution_history: bool (默认: True)
            追踪历史进化记录（用于分析/回滚）
        validation_hooks: List[Callable] (默认: [])
            自定义验证回调函数列表
    
    示例 | Example:
        debug={
            "export_signature_freq_csv": "./logs/signatures.csv",
            "export_consensus_graph_dot": "./logs/consensus.dot",
            "log_selected_subgraphs_detail": True
        }
    """
    
    strategy: Dict[str, Any] = field(default_factory=dict)
    """
    高级策略配置
    
    可用键 | Available Keys:
        adaptive_threshold: Dict (默认: disabled)
            {
                "enabled": bool,
                "min_threshold": float (0.3),
                "max_threshold": float (0.9),
                "decay_factor": float (0.95)
            }
        cross_layer_alignment: bool (默认: False)
            允许跨层签名对齐（增强泛化但可能降低精度）
        preserve_method_semantics: bool (默认: True)
            抽象时保留方法语义约束（防止无效组合）
        abstraction_depth_limit: Optional[int]
            方法嵌套最大深度（防止递归爆炸）
        feedback_weighting: Dict[str, float]
            基于执行反馈的方法权重调整
    
    示例 | Example:
        strategy={
            "adaptive_threshold": {
                "enabled": True,
                "decay_factor": 0.98  # 阈值随轮次缓慢下降
            },
            "abstraction_depth_limit": 3
        }
    """
    
    # === 类方法 ===
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'EvolutionConfig':
        """
        从字典创建配置实例，支持嵌套更新
        
        参数 | Args:
            config: 配置字典，支持两种格式:
                - 扁平格式: {"consensus_threshold": 0.7, "max_methods": 3}
                - 嵌套格式: {"consensus": {"min_absolute_count": 5}}
        
        返回 | Returns:
            EvolutionConfig 实例，未指定的字段使用默认值
        
        示例 | Examples:
            >>> # 扁平格式（兼容传统参数）
            >>> cfg = EvolutionConfig.from_dict({
            ...     "consensus_threshold": 0.7,
            ...     "max_methods": 3
            ... })
            
            >>> # 嵌套格式（模块化配置）
            >>> cfg = EvolutionConfig.from_dict({
            ...     "consensus": {"min_absolute_count": 5},
            ...     "subgraph": {"max_size": 20}
            ... })
            
            >>> # 混合格式
            >>> cfg = EvolutionConfig.from_dict({
            ...     "max_methods": 2,  # 核心参数
            ...     "debug": {"export_csv": "./log.csv"}  # 嵌套参数
            ... })
        
        注意 | Note:
            - 嵌套字典会 deep update，不会覆盖整个 section
            - 未知键会被忽略（避免配置错误导致崩溃）
        """
        # 分离核心参数和嵌套配置
        core_fields = {
            k: v for k, v in config.items() 
            if k in cls.__dataclass_fields__ and k not in 
            {"signature", "consensus", "subgraph", "isomorphism", 
             "registration", "performance", "debug", "strategy"}
        }
        
        # 处理嵌套配置（deep merge）
        nested_sections = {
            "signature", "consensus", "subgraph", "isomorphism",
            "registration", "performance", "debug", "strategy"
        }
        
        result = {**core_fields}
        for section in nested_sections:
            if section in config:
                # 合并用户配置到默认空字典
                result[section] = {
                    **getattr(cls(section), section, {}),
                    **config[section]
                }
        
        return cls(**result)
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        安全获取嵌套配置值
        
        参数 | Args:
            section: 配置节名称（如 "consensus", "subgraph"）
            key: 配置项键名
            default: 键不存在时的默认返回值
        
        返回 | Returns:
            配置值，或 default（如果 section/key 不存在）
        
        示例 | Examples:
            >>> config = EvolutionConfig(consensus={"min_count": 3})
            >>> config.get("consensus", "min_count")  # 返回 3
            >>> config.get("consensus", "unknown", default=10)  # 返回 10
            >>> config.get("unknown_section", "key", default="N/A")  # 返回 "N/A"
        
        用途 | Use Cases:
            - 在进化逻辑中安全读取可选配置
            - 避免 KeyError，提供降级默认值
        """
        section_data = getattr(self, section, None)
        if isinstance(section_data, dict):
            return section_data.get(key, default)
        return default
    
    def to_dict(self, include_defaults: bool = False) -> Dict[str, Any]:
        """
        将配置序列化为字典（用于保存/传输）
        
        参数 | Args:
            include_defaults: 是否包含默认值字段（默认 False，仅输出修改项）
        
        返回 | Returns:
            字典表示，可直接 JSON 序列化
        
        示例 | Example:
            >>> config = EvolutionConfig(consensus_threshold=0.8)
            >>> config.to_dict()
            {'consensus_threshold': 0.8}  # 仅输出非默认值
            
            >>> config.to_dict(include_defaults=True)
            {'consensus_threshold': 0.8, 'min_subgraph_size': 2, ...}  # 全部字段
        """
        result = {}
        
        # 核心参数
        for field_name in ["consensus_threshold", "min_subgraph_size", 
                          "max_methods", "subgraph_selection_policy"]:
            value = getattr(self, field_name)
            default = self.__dataclass_fields__[field_name].default
            if include_defaults or value != default:
                result[field_name] = value
        
        # 嵌套配置（仅输出非空）
        for section in ["signature", "consensus", "subgraph", "isomorphism",
                       "registration", "performance", "debug", "strategy"]:
            section_data = getattr(self, section)
            if section_data or include_defaults:
                result[section] = dict(section_data)
        
        return result
    
    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """
        生成 JSON Schema，用于 IDE 提示/配置校验/文档生成
        
        返回 | Returns:
            JSON Schema 字典，符合 https://json-schema.org/ 规范
        
        用途 | Use Cases:
            - VSCode/PyCharm 配置自动补全
            - 配置文件校验（如使用 jsonschema 库）
            - 自动生成 API 文档
        
        示例 | Example:
            >>> schema = EvolutionConfig.get_schema()
            >>> with open("config.schema.json", "w") as f:
            ...     json.dump(schema, f, indent=2)
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "EvolutionConfig",
            "description": "方法池进化配置 | Method Pool Evolution Configuration",
            "type": "object",
            "properties": {
                # 核心参数
                "consensus_threshold": {
                    "type": ["number", "null"],
                    "minimum": 0, "maximum": 1,
                    "description": "共识阈值：方法在签名中出现的最小比例"
                },
                "min_subgraph_size": {
                    "type": "integer", "minimum": 1, "default": 2,
                    "description": "进化子图的最小节点数"
                },
                "max_methods": {
                    "type": "integer", "minimum": 1, "default": 1,
                    "description": "单次进化最多抽象的新方法数量"
                },
                "subgraph_selection_policy": {
                    "type": "string",
                    "enum": ["largest", "most_frequent", "smallest", "balanced"],
                    "default": "largest",
                    "description": "子图选择策略"
                },
                # 嵌套配置节（简化示例）
                "consensus": {
                    "type": "object",
                    "properties": {
                        "min_absolute_count": {"type": ["integer", "null"]},
                        "aggregation_strategy": {
                            "type": "string",
                            "enum": ["max_vote", "weighted_avg", "first_seen"]
                        }
                    },
                    "additionalProperties": True
                },
                # ... 其他 section 类似
            },
            "additionalProperties": False
        }