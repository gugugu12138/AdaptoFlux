"""为 ATF 数据图生成可保存、可检查的副作用执行计划。

本模块区分两种关系：数据边决定函数实参，控制依赖决定节点何时允许执行。
控制依赖不会写入数据图，因此不改变原算子的输入数量，也不要求把外部状态
改为普通数据端口。运行器只需在数据前驱和控制前驱均完成后放行一个节点。

节点第一次参与编译时获得稳定 UID，并把选定的语义顺序保存到图元数据。
之后节点显示名称、文件中的节点排列或网络库的遍历顺序改变，都不应重新
决定状态读写的先后。训练若主动改变该顺序，必须显式重设并重新评价候选。

这里处理的是“已声明的效果”。它不分析任意 Python 函数体，也不能发现虚假
的 pure 声明或遗漏的资源别名；未声明效果的方法按 unknown 保守排序。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import copy
import json
from uuid import uuid4

import networkx as nx


PLAN_VERSION = 1
UID_KEY = "_atf_uid"
ORDER_KEY = "_atf_execution_order"
VERSION_KEY = "_atf_execution_plan_version"
EFFECT_ORDERING_KEY = "_atf_enable_effect_ordering"
BOUNDARIES = frozenset(("root", "collapse"))


def validate_effect_ordering(value):
    """开关只接受真正的布尔值，避免字符串 'False' 被当作真值误开启。

    构造、切换和文件加载共用此校验；拒绝输入发生在修改图或调用方法之前。
    不使用 bool(value) 强制转换，也不接受 0/1 代替显式的 True/False。
    """
    if type(value) is not bool:
        raise ValueError("enable_effect_ordering must be True or False")
    return value


def _resources(value, label):
    """检查资源名集合；拒绝把一个字符串误当成逐字符的资源列表。"""
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{label} must be an iterable of resource names, not a string or mapping")
    try:
        result = list(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable of resource names") from exc
    if any(not isinstance(name, str) or not name.strip() for name in result):
        raise ValueError(f"{label} resource names must be nonempty strings")
    return sorted(set(result))


def normalize_effects(pure=None, reads=None, writes=None):
    """把注册声明转换为统一的、可 JSON 保存的效果描述。

    完全省略声明意味着 unknown，不能因为没有写出资源就默认纯函数。
    ``pure=False`` 单独使用同样无法说明访问哪些资源，因此仍为 unknown。
    显式给出空的读/写集合则表示没有资源访问，与 pure=True 等价。

    资源 ``*`` 表示全部资源。随机数流、文件、环境状态等都应当被视为资源，
    不能只登记模型内部变量。pure=True 与非空读写集合矛盾时立即报错。
    """
    if pure is not None and not isinstance(pure, bool):
        raise ValueError("pure must be True, False, or None")
    read_set, write_set = _resources(reads, "reads"), _resources(writes, "writes")
    if pure is True:
        if read_set or write_set:
            raise ValueError("A pure method cannot declare reads or writes")
        return {"kind": "pure", "reads": [], "writes": []}
    if reads is None and writes is None:
        return {"kind": "unknown", "reads": [], "writes": []}
    if not read_set and not write_set:
        return {"kind": "pure", "reads": [], "writes": []}
    return {"kind": "declared", "reads": read_set, "writes": write_set}


def normalize_effect_spec(effects):
    """校验保存过的效果描述，也接受 pure/reads/writes 形式的注册声明。

    拒绝拼错的字段与未知 kind，避免“本想声明写操作却被当作纯计算”的情况。
    返回新字典，调用方修改结果不会修改原声明对象。
    """
    if effects is None:
        return normalize_effects()
    if not isinstance(effects, Mapping):
        raise ValueError("effects must be a mapping or None")
    if "kind" not in effects:
        if set(effects) - {"pure", "reads", "writes"}:
            raise ValueError("Unknown effect declaration fields")
        return normalize_effects(**effects)
    if set(effects) - {"kind", "reads", "writes"}:
        raise ValueError("Unknown serialized effect fields")
    kind = effects["kind"]
    reads = _resources(effects.get("reads"), "reads")
    writes = _resources(effects.get("writes"), "writes")
    if kind in ("pure", "unknown"):
        if reads or writes:
            raise ValueError(f"{kind} effects must not contain resource declarations")
        return {"kind": kind, "reads": [], "writes": []}
    if kind != "declared":
        raise ValueError(f"Unknown effect kind: {kind!r}")
    if "reads" not in effects and "writes" not in effects:
        raise ValueError("Declared effects must explicitly provide reads or writes")
    return normalize_effects(reads=reads, writes=writes)


def effects_for_method(method_info):
    """读取一个方法注册项，兼容效果保存在字典或函数属性上的形式。

    注册项中明确给出的 effects 优先级最高；例如显式 effects=None 会退回
    unknown，而不会偷偷采用函数上较宽松的旧声明。旧方法完全没有相关字段
    时按 unknown 处理，保证迁移时不会默认开放未经验证的并行访问。
    """
    if not isinstance(method_info, Mapping):
        return normalize_effects()
    if "effects" in method_info:
        return normalize_effect_spec(method_info["effects"])
    if any(key in method_info for key in ("pure", "reads", "writes")):
        return normalize_effects(**{key: method_info.get(key) for key in ("pure", "reads", "writes")})
    function = method_info.get("function")
    if hasattr(function, "effects"):
        return normalize_effect_spec(function.effects)
    if any(hasattr(function, key) for key in ("pure", "reads", "writes")):
        return normalize_effects(**{key: getattr(function, key, None) for key in ("pure", "reads", "writes")})
    return normalize_effects()


def _passthrough(attributes):
    """与原执行器保持相同的透传判断：透传节点本身不会调用用户函数。"""
    if "is_passthrough" in attributes:
        return bool(attributes["is_passthrough"])
    name = attributes.get("method_name")
    return name is None or (isinstance(name, str) and name.lower() in ("null", "unmatched"))


def _node_effects(attributes, methods):
    if _passthrough(attributes):
        return normalize_effects(pure=True)
    return effects_for_method(methods.get(attributes.get("method_name"), {}))


def aggregate_effects(graph, methods):
    """为提取出的复合方法合并副作用声明。

    任一内部方法未知时，整个复合方法也必须未知；否则取读资源、写资源的
    并集。不能因为复合方法只暴露数值输入输出，就把内部状态写入丢掉。
    """
    reads, writes = set(), set()
    for node, attributes in graph.nodes(data=True):
        if node in BOUNDARIES:
            continue
        effects = _node_effects(attributes, methods)
        if effects["kind"] == "unknown":
            return normalize_effects()
        reads.update(effects["reads"])
        writes.update(effects["writes"])
    return normalize_effects(reads=reads, writes=writes)


def _validate_data_graph(graph):
    """先检查边界与无环性，避免到执行副作用之后才发现图无法调度。"""
    if not graph.is_directed():
        raise ValueError("Execution requires a directed data graph")
    if not BOUNDARIES.issubset(graph.nodes):
        raise ValueError("Execution graph must contain root and collapse")
    if graph.in_degree("root"):
        raise ValueError("root cannot have incoming dependencies")
    if graph.out_degree("collapse"):
        raise ValueError("collapse cannot be a predecessor of an operation")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The data graph contains a cycle")


def _node_uids(graph):
    """获取已有稳定身份，并暂时为新节点分配身份，尚不修改原图。

    显示节点 ID 可以随方法替换而变化，UID 负责把旧语义位置对应回来。
    复制完整模型可以保留 UID；在同一张图里重复插入子图则必须清除新副本
    的 UID，重复身份会被拒绝，而不是把两个节点误认为同一个旧节点。
    """
    uids, seen = {}, set()
    for node, attributes in graph.nodes(data=True):
        if node in BOUNDARIES:
            continue
        uid = attributes.get(UID_KEY)
        if uid is None:
            uid = uuid4().hex
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"Invalid persistent UID for node {node!r}")
        if uid in seen:
            raise ValueError(f"Duplicate persistent node UID {uid!r}; cloned nodes need fresh UIDs")
        uids[node] = uid
        seen.add(uid)
    return uids


def _semantic_order(graph, uids):
    """保留旧节点相对顺序，并把新增节点插入数据依赖允许的位置。

    没有历史计划的新图，首次选择原数据图的一个拓扑序并冻结。已有计划时，
    给幸存旧节点加一条临时排序链来约束本次编译；该链只用于选择语义顺序，
    不会全部成为运行时控制边，因此纯计算或独立资源仍可并行。
    """
    version = graph.graph.get(VERSION_KEY)
    if version is not None and version != PLAN_VERSION:
        raise ValueError(f"Unsupported execution plan version {version!r}")
    old = graph.graph.get(ORDER_KEY)
    if old is None:
        return tuple(node for node in nx.topological_sort(graph) if node not in BOUNDARIES)
    if not isinstance(old, (list, tuple)) or any(not isinstance(uid, str) for uid in old):
        raise ValueError("Stored execution order must be a list of node UIDs")
    if len(old) != len(set(old)):
        raise ValueError("Stored execution order contains duplicate UIDs")
    by_uid = {uid: node for node, uid in uids.items()}
    retained = [by_uid[uid] for uid in old if uid in by_uid]
    # 临时链固定旧节点的相对语义位置。若新的数据边要求相反顺序，就不是
    # 单纯的存储重排，必须作为显式顺序变异重新评价，不能静默“修复”。
    constrained = nx.DiGraph(graph)
    constrained.add_edges_from(zip(retained, retained[1:]))
    if not nx.is_directed_acyclic_graph(constrained):
        raise ValueError("Data dependencies conflict with the frozen execution order; explicitly reorder and reevaluate the candidate")
    old_rank = {node: index for index, node in enumerate(retained)}
    insertion_rank = {node: index for index, node in enumerate(graph.nodes)}
    # 同时就绪时优先保留旧节点；新节点在数据依赖要求的位置插入，否则排在
    # 旧节点之后。新增节点首次顺序由当前插入次序打破平局，随后也会被冻结。
    def rank(node):
        if node == "root":
            return (-1, 0)
        if node in old_rank:
            return (0, old_rank[node])
        return (1, insertion_rank[node])
    return tuple(node for node in nx.lexicographical_topological_sort(constrained, key=rank)
                 if node not in BOUNDARIES)


def _effect_dependencies(order, effects):
    """按资源的读写前沿推导必要控制依赖。

    writers 保存每个资源最近一次写入，readers 保存该次写入后尚未被后续
    写操作覆盖的读取。读取等待前次写入；写入同时等待前次写入及这批读取。
    两次只读或两个独立资源无需人为串联。

    unknown 与写入 * 是全局屏障：它等待所有当前状态访问前沿，后续访问
    再等待这个屏障。这样连续 N 个未知方法只需要 N-1 条控制边，而不是
    把所有节点对都连起来。只读 * 不会阻塞其他读取，但必须等待已有写入，
    并被之后任意资源的写入等待。
    """
    dependencies = set()
    # frontier 是尚未被另一个状态访问显式依赖的末端访问集合；全局屏障
    # 等待这些末端即可间接等待它们之前的全部状态访问。
    writers, readers = {}, {}
    wildcard_readers, frontier = set(), set()
    barrier = None
    for node in order:
        spec = effects[node]
        if spec["kind"] == "pure":
            continue
        read_set, write_set = set(spec["reads"]), set(spec["writes"])
        if spec["kind"] == "unknown" or "*" in write_set:
            previous = set(frontier)
            dependencies.update((source, node) for source in previous)
            # 屏障完成后旧读写均已完成，之后只需依赖屏障，无需保留旧前沿。
            writers.clear()
            readers.clear()
            wildcard_readers.clear()
            frontier = {node}
            barrier = node
            continue
        previous = {barrier} if barrier is not None else set()
        if "*" in read_set:
            previous.update(writers.values())
        for resource in read_set | write_set:
            if resource in writers:
                previous.add(writers[resource])
        for resource in write_set:
            previous.update(readers.get(resource, ()))
            previous.update(wildcard_readers)
        dependencies.update((source, node) for source in previous)
        # 某资源被当前节点写入后，未来访问等待当前节点即可；该资源的旧
        # 读取集合已被本次写入等待，可以清除。其他资源仍保留各自的前沿。
        for resource in write_set:
            writers[resource] = node
            readers.pop(resource, None)
        for resource in read_set - write_set - {"*"}:
            readers.setdefault(resource, set()).add(node)
        if "*" in read_set:
            wildcard_readers.add(node)
        frontier.difference_update(previous)
        frontier.add(node)
    return dependencies


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _export_payload(graph, methods, order, uids, effects, control_edges, enable_effect_ordering):
    """生成可核验的计划快照，记录实际影响执行的端口与方法声明。

    节点身份按 UID 保存；节点显示名称仅供人阅读。数据边以多重集保存，
    不依赖 NetworkX/GEXF 重新分配的多边 key 或文件行序，但重复边不会丢失。
    函数实现本身仍须由模型的代码/依赖提供；此快照不试图序列化 Python 闭包。
    """
    def endpoint(node):
        return ["boundary", node] if node in BOUNDARIES else ["node", uids[node]]
    nodes = []
    for node in order:
        attributes = graph.nodes[node]
        info = methods.get(attributes.get("method_name"), {})
        nodes.append({"uid": uids[node], "node": node,
                      "method_name": attributes.get("method_name"),
                      "passthrough": _passthrough(attributes), "effects": effects[node],
                      "signature": {key: info.get(key) for key in
                                    ("input_count", "output_count", "input_types", "output_types", "vectorized")}})
    data_edges = [{"source": endpoint(source), "target": endpoint(target),
                   "ports": {key: attributes.get(key) for key in
                             ("output_index", "input_slot", "data_coord", "data_type")}}
                  for source, target, attributes in graph.edges(data=True)]
    data_edges.sort(key=_canonical)
    return {"version": PLAN_VERSION, "enable_effect_ordering": enable_effect_ordering,
            "order": [uids[node] for node in order],
            "nodes": nodes, "data_edges": data_edges,
            "control_edges": [[uids[source], uids[target]] for source, target in control_edges]}


@dataclass(frozen=True)
class ExecutionPlan:
    """运行器使用的计划；order 和前驱/后继都只包含内部运算/透传节点。

    root 的输入在开始时已经就绪，collapse 由运行器在全部内部节点结束后
    统一聚合。control_edges 是额外依赖，predecessors/successors 则是其与
    数据依赖的并集。effects 按当前节点 ID 索引，uids 负责持久身份映射。
    """
    order: tuple
    predecessors: dict
    successors: dict
    control_edges: tuple
    effects: dict
    has_effects: bool
    uids: dict
    enable_effect_ordering: bool
    _payload: dict = field(repr=False)

    def to_dict(self):
        return copy.deepcopy(self._payload)


def compile_execution_plan(graph, methods, *, enable_effect_ordering=True):
    """按指定模式生成计划；开启时冻结顺序并合并副作用依赖。

    编译应发生在候选首次评价之前；运行器也会检查计划，避免直接修改图后
    沿用失效计划。旧节点顺序跨存储重排、UID 保留的重命名和正常生长保持。
    要逆转旧顺序，必须调用 set_execution_order 并把它作为新程序重新评价。

    成功前不提交 UID 或顺序元数据；失败不会把半成品计划留在原图。数据边
    始终不变，参数绑定仍由原数据图负责。关闭时不生成副作用依赖，order
    只是本次数据拓扑序快照，不约束下次编译；has_effects 仍反映真实声明，
    以免压缩器把关闭排序错误理解成所有函数都纯净。
    """
    enable_effect_ordering = validate_effect_ordering(enable_effect_ordering)
    previous_mode = validate_effect_ordering(graph.graph.get(EFFECT_ORDERING_KEY, True))
    _validate_data_graph(graph)
    uids = _node_uids(graph)
    # 关闭时只看当前数据图，不让之前冻结的副作用顺序限制结构搜索。
    # 从关闭重新开启也以当前图为新的语义起点，不能复活已经过期的旧顺序。
    # 开启期间的普通重编译仍严格保持原有顺序，与此前版本行为一致。
    if enable_effect_ordering and previous_mode:
        order = _semantic_order(graph, uids)
    else:
        order = tuple(node for node in nx.topological_sort(graph) if node not in BOUNDARIES)
    effects = {node: _node_effects(graph.nodes[node], methods) for node in order}
    order_index = {node: index for index, node in enumerate(order)}
    data_dependencies = {(source, target) for source, target in graph.edges()
                         if source not in BOUNDARIES and target not in BOUNDARIES}
    control = (_effect_dependencies(order, effects) - data_dependencies
               if enable_effect_ordering else set())
    control_edges = tuple(sorted(control, key=lambda pair: (order_index[pair[0]], order_index[pair[1]])))
    dependencies = data_dependencies | control
    predecessors = {node: [] for node in order}
    successors = {node: [] for node in order}
    for source, target in dependencies:
        if order_index[source] >= order_index[target]:
            raise ValueError("Execution dependencies violate the selected semantic order")
        predecessors[target].append(source)
        successors[source].append(target)
    predecessors = {node: tuple(sorted(items, key=order_index.__getitem__)) for node, items in predecessors.items()}
    successors = {node: tuple(sorted(items, key=order_index.__getitem__)) for node, items in successors.items()}
    payload = _export_payload(graph, methods, order, uids, effects, control_edges,
                              enable_effect_ordering)
    # 所有检查和导出均成功后，才提交身份与顺序，避免拒绝候选留下半成品。
    for node, uid in uids.items():
        graph.nodes[node][UID_KEY] = uid
    graph.graph[ORDER_KEY] = [uids[node] for node in order]
    graph.graph[VERSION_KEY] = PLAN_VERSION
    graph.graph[EFFECT_ORDERING_KEY] = enable_effect_ordering
    return ExecutionPlan(order, predecessors, successors, control_edges, effects,
                         any(spec["kind"] != "pure" for spec in effects.values()), uids,
                         enable_effect_ordering, payload)


def set_execution_order(graph, order):
    """显式指定全部内部节点的另一个合法顺序。

    这不是运行时的优化开关：对于冲突副作用，它改变程序语义。训练器必须
    在接受这样的变异之前重新评价候选。遗漏、重复节点或违反数据依赖的
    顺序会被拒绝，root/collapse 不应出现在传入序列中。
    """
    _validate_data_graph(graph)
    selected = tuple(order)
    internal = set(graph.nodes) - BOUNDARIES
    if len(selected) != len(internal) or len(set(selected)) != len(selected) or set(selected) != internal:
        raise ValueError("Explicit order must contain each internal node exactly once")
    rank = {node: index for index, node in enumerate(selected)}
    for source, target in graph.edges():
        if source in internal and target in internal and rank[source] >= rank[target]:
            raise ValueError("Explicit execution order violates data dependencies")
    uids = _node_uids(graph)
    for node, uid in uids.items():
        graph.nodes[node][UID_KEY] = uid
    graph.graph[ORDER_KEY] = [uids[node] for node in selected]
    graph.graph[VERSION_KEY] = PLAN_VERSION
    # 显式指定顺序属于开启模式；运行器关闭时会在调用本函数前拒绝该请求。
    graph.graph[EFFECT_ORDERING_KEY] = True


def export_execution_plan(graph, methods, *, enable_effect_ordering=True):
    """重新校验当前图并导出计划，避免保存失效的旧控制依赖。"""
    return compile_execution_plan(graph, methods, enable_effect_ordering=enable_effect_ordering).to_dict()


def restore_execution_plan(graph, methods, payload):
    """恢复保存的计划，并拒绝任何不一致的图端口或方法/效果声明。

    JSON 图自身已有语义顺序时必须与计划一致；GEXF 未保留图元数据时可以
    从计划补回。节点 UID 必须齐全，不能凭当前遍历顺序猜测旧节点身份。
    先在图副本上检查，完全一致后再提交元数据。该校验不证明函数源码没有
    改变，代码来源仍由外层模型保存/载入及依赖管理负责。
    """
    if not isinstance(payload, Mapping) or payload.get("version") != PLAN_VERSION:
        raise ValueError("Invalid or unsupported saved execution plan")
    required = {"version", "order", "nodes", "data_edges", "control_edges"}
    if set(payload) not in (required, required | {"enable_effect_ordering"}):
        raise ValueError("Saved execution plan has missing or unknown fields")
    # 开关加入前的计划没有此字段，其语义一直是开启。兼容加载必须沿用开启，
    # 不能根据当前实例配置猜测，否则同一个模型会因加载入口不同而改变行为。
    enabled = validate_effect_ordering(payload.get("enable_effect_ordering", True))
    if EFFECT_ORDERING_KEY in graph.graph:
        if validate_effect_ordering(graph.graph[EFFECT_ORDERING_KEY]) != enabled:
            raise ValueError("Graph metadata and saved plan disagree on enable_effect_ordering")
    stored_order = payload["order"]
    if not isinstance(stored_order, list) or any(not isinstance(uid, str) for uid in stored_order):
        raise ValueError("Saved order must be a list of node UIDs")
    if len(stored_order) != len(set(stored_order)):
        raise ValueError("Saved order contains duplicate UIDs")
    if ORDER_KEY in graph.graph and list(graph.graph[ORDER_KEY]) != stored_order:
        raise ValueError("Graph metadata and saved plan disagree on execution order")
    if VERSION_KEY in graph.graph and graph.graph[VERSION_KEY] != PLAN_VERSION:
        raise ValueError("Graph metadata and saved plan disagree on version")
    internal = [node for node in graph if node not in BOUNDARIES]
    actual_uids = [graph.nodes[node].get(UID_KEY) for node in internal]
    if any(not isinstance(uid, str) or not uid for uid in actual_uids) or set(actual_uids) != set(stored_order):
        raise ValueError("Saved plan does not match the graph's persistent node UIDs")
    candidate = graph.copy()
    candidate.graph[ORDER_KEY] = list(stored_order)
    candidate.graph[VERSION_KEY] = PLAN_VERSION
    candidate.graph[EFFECT_ORDERING_KEY] = enabled
    plan = compile_execution_plan(candidate, methods, enable_effect_ordering=enabled)
    actual, expected = plan.to_dict(), copy.deepcopy(dict(payload))
    expected.setdefault("enable_effect_ordering", True)
    # 节点显示名称允许变化；稳定 UID、方法声明、端口和依赖仍必须严格一致。
    try:
        for value in (actual, expected):
            for node in value["nodes"]:
                node.pop("node", None)
            if not enabled:
                # 关闭模式没有冻结顺序保证；文件重排可选择另一合法拓扑序。
                # 仍逐节点核验声明/端口，且必须没有控制边，不能放宽其他契约。
                value["nodes"].sort(key=lambda node: node["uid"])
    except (TypeError, AttributeError, KeyError) as exc:
        raise ValueError("Malformed saved node declarations") from exc
    if not enabled:
        by_uid = {uid: index for index, uid in enumerate(stored_order)}
        for source, target in graph.edges():
            if source not in BOUNDARIES and target not in BOUNDARIES:
                if by_uid[graph.nodes[source][UID_KEY]] >= by_uid[graph.nodes[target][UID_KEY]]:
                    raise ValueError("Saved data-only order violates data dependencies")
        actual["order"] = list(stored_order)
    if _canonical(actual) != _canonical(expected):
        raise ValueError("Saved execution plan differs from graph dependencies or method/effect declarations")
    graph.graph[ORDER_KEY] = list(stored_order)
    graph.graph[VERSION_KEY] = PLAN_VERSION
    graph.graph[EFFECT_ORDERING_KEY] = enabled
    return plan
