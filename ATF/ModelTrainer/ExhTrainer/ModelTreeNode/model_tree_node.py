class ModelTreeNode:
    def __init__(self, layer_idx, structure, function_combo=None, parent=None):
        self.layer_idx = layer_idx         # 层级索引
        self.structure = structure         # 当前输入结构（如 ['numerical', 'categorical']）
        self.function_combo = function_combo  # 函数组合 [(func_name, info), ...]
        self.parent = parent               # 父节点
        self.children = []                 # 子节点列表
        self.node_info = {}                # 结构化信息 {index_map, valid_groups, unmatched}