import copy
import random
from typing import List, Tuple, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

def refine_random_single_step(
    train,
    adaptoflux_instance,
    input_data: np.ndarray,
    target: np.ndarray,
    processing_nodes: List[str],
    current_loss: float,
    gp: Any  # GraphProcessor 实例
) -> Tuple[bool, float, float, int, List[str]]:
    """
    【策略：随机单点优化】
    在当前图中随机选择一个可优化的处理节点，尝试将其方法替换为所有兼容方法中的最优者（基于损失下降）。
    仅执行一次替换尝试（即一个节点的一次优化），适用于轻量级、低开销的局部搜索。
    注意：本函数会 in-place 修改 adaptoflux_instance 的图结构！

    返回值说明：
    - bool: 本轮是否成功改进（即找到更优方法）
    - float: 改进后的损失值（若未改进则返回原损失）
    - float: 改进后的准确率（若未改进则返回 0.0，调用方应忽略）
    - int: 本次实际执行的优化步数（0 或 1）
    - List[str]: 更新后的可处理节点列表（因图结构可能变化，需重新获取）

    :param adaptoflux_instance: 当前待优化的 AdaptoFlux 实例
    :param input_data: 用于评估的小批量输入数据
    :param target: 对应的真实标签
    :param processing_nodes: 当前图中所有可被优化的处理节点名称列表（已排除冻结节点）
    :param current_loss: 当前模型的损失值（用于比较）
    :param gp: 图处理器实例，用于访问和修改图结构
    :return: (是否改进, 新损失, 新准确率, 步数增量, 更新后的节点列表)
    """
    # 若无可优化节点，直接返回无改进
    if not processing_nodes:
        return False, current_loss, 0.0, 0, processing_nodes

    # 随机选择一个待优化节点
    target_node = random.choice(processing_nodes)
    original_method_name = gp.graph.nodes[target_node]['method_name']
    
    # 获取与该节点兼容的候选方法列表（基于组别或类型匹配）
    candidate_methods = train._get_compatible_methods_for_node(
        adaptoflux_instance, 
        target_node
    )

    best_candidate = None
    best_loss = current_loss  # 初始化为当前损失，用于比较

    attempts_in_step = 0
    
    # 遍历所有候选方法（跳过当前方法）
    for candidate_method_name in candidate_methods:
        if candidate_method_name == original_method_name:
            continue

        # --- 检查是否已达总尝试上限 ---
        if (train.max_total_refinement_attempts is not None and
            train._total_refinement_attempts >= train.max_total_refinement_attempts):
            break

        # --- 计数 +1 ---
        train._total_refinement_attempts += 1
        attempts_in_step += 1

        # 创建临时副本，避免污染原模型
        temp_af = copy.deepcopy(adaptoflux_instance)
        temp_gp = temp_af.graph_processor
        # 尝试替换节点方法
        temp_gp.graph.nodes[target_node]['method_name'] = candidate_method_name

        
        # 评估替换后的损失
        new_loss = train._evaluate_loss(input_data, target, adaptoflux_instance=temp_af)

        # 记录损失更低的最优候选
        if new_loss < best_loss:
            best_loss = new_loss
            best_candidate = candidate_method_name

    # 如果找到更优方法，则应用到原图
    if best_candidate and train._should_accept(current_loss, best_loss):
        # === 替换节点（自动更新 ID 和边） ===
        new_node_id = gp.replace_node_method(target_node, best_candidate)
        
        # 注意：target_node 已被删除，后续操作应使用 new_node_id（但本策略不需要）
        # 如果作者有空而且没忘记可能会在gp里面加一个刷新图节点id的方法提升可读性

        # 重新评估准确率
        new_acc = train._evaluate_accuracy(input_data, target, adaptoflux_instance=adaptoflux_instance)

        # 重新获取处理节点列表（因为节点 ID 已变）
        updated_nodes = [node for node in gp.graph.nodes() if gp._is_processing_node(node)]
        
        return True, best_loss, new_acc, 1, updated_nodes
    
    else:
        # 无改进，返回原始状态
        return False, current_loss, 0.0, 0, processing_nodes
    
def refine_full_sweep_step(
    train,
    adaptoflux_instance,
    input_data: np.ndarray,
    target: np.ndarray,
    processing_nodes: List[str],
    current_loss: float,
    gp: Any
) -> Tuple[bool, float, float, int, List[str]]:
    """
    【策略：完整遍历优化】
    对当前图中所有可优化的处理节点进行一轮完整遍历。
    对每个节点，尝试所有兼容方法，若发现能降低损失的替换，则立即应用（贪心策略）。
    一轮中可能多次修改图结构，适用于更彻底的局部优化，但计算开销较大。
    注意：本函数会 in-place 修改 adaptoflux_instance 的图结构！

    返回值说明：
    - bool: 本轮是否至少有一次成功改进
    - float: 本轮结束后的最终损失值
    - float: 本轮结束后的最终准确率
    - int: 本轮总共执行的方法替换次数
    - List[str]: 最终的可处理节点列表（可能因方法变更而动态变化）

    :param adaptoflux_instance: 当前待优化的 AdaptoFlux 实例
    :param input_data: 用于评估的小批量输入数据
    :param target: 对应的真实标签
    :param processing_nodes: 初始的可优化节点列表
    :param current_loss: 当前损失（作为起点）
    :param gp: 图处理器实例
    :return: (是否改进, 最终损失, 最终准确率, 替换次数, 最终节点列表)
    """
    if not processing_nodes:
        return False, current_loss, 0.0, 0, processing_nodes

    # 随机打乱节点顺序，避免顺序偏差（例如总是先优化靠前的节点）
    nodes_to_try = random.sample(processing_nodes, len(processing_nodes))
    improvement_made = False
    total_replacements = 0
    current_acc = train._evaluate_accuracy(input_data, target, adaptoflux_instance=adaptoflux_instance)

    # 遍历每一个待优化节点
    for target_node in nodes_to_try:
        # 检查节点是否仍然存在于图中（可能被之前的替换操作删除或重命名）
        if target_node not in gp.graph.nodes:
            if train.verbose:
                logger.debug(f"Node '{target_node}' no longer exists in graph; skipping.")
            continue

        original_method_name = gp.graph.nodes[target_node].get('method_name')
        if original_method_name is None:
            if train.verbose:
                logger.warning(f"Node '{target_node}' has no 'method_name'; skipping.")
            continue

        # 获取与该节点兼容的候选方法列表（考虑组别、类型兼容性及冻结规则）
        candidate_methods = train._get_compatible_methods_for_node(
            adaptoflux_instance,
            target_node
        )

        best_candidate = None
        best_loss = current_loss  # 以当前全局损失为基准进行比较

        # 尝试所有兼容方法（跳过当前方法）
        for candidate_method_name in candidate_methods:
            if candidate_method_name == original_method_name:
                continue

            # --- 检查并计数 ---
            if (train.max_total_refinement_attempts is not None and
                train._total_refinement_attempts >= train.max_total_refinement_attempts):
                # 提前退出双层循环
                return improvement_made, current_loss, current_acc, total_replacements, processing_nodes

            train._total_refinement_attempts += 1

            # 创建临时副本进行安全评估，避免污染原模型
            try:
                temp_af = copy.deepcopy(adaptoflux_instance)
                temp_gp = temp_af.graph_processor
                # 安全替换：使用图处理器的标准方法（会处理输入/输出类型变化）
                # 注意：这里我们模拟替换，但不实际调用 replace_node_method（因为只是评估）
                # 所以直接修改 method_name 是安全的，前提是不依赖边结构变化
                temp_gp.graph.nodes[target_node]['method_name'] = candidate_method_name

                new_loss = train._evaluate_loss(input_data, target, adaptoflux_instance=temp_af)

                if new_loss < best_loss:
                    best_loss = new_loss
                    best_candidate = candidate_method_name
            except Exception as e:
                logger.warning(f"Failed to evaluate candidate method '{candidate_method_name}' "
                                f"for node '{target_node}': {e}")
                continue

        # 如果找到更优方法，立即应用到原图（贪心策略）
        if best_candidate and train._should_accept(current_loss, best_loss):
            try:
                # 使用图处理器的标准替换方法，确保图结构一致性（如边更新、ID刷新等）
                new_node_id = gp.replace_node_method(target_node, best_candidate)
                # 更新当前损失和准确率
                current_loss = best_loss
                current_acc = train._evaluate_accuracy(input_data, target, adaptoflux_instance=adaptoflux_instance)
                improvement_made = True
                total_replacements += 1

                if train.verbose:
                    logger.info(f"  Replacement {total_replacements}: Node '{target_node}' "
                                f"{original_method_name} → {best_candidate}, Loss: {current_loss:.6f}")

                # 重要：节点替换后，原 target_node 可能已被删除或重命名（new_node_id）
                # 因此需要重新获取当前所有处理节点，确保后续操作基于最新图状态
                processing_nodes = [
                    node for node in gp.graph.nodes()
                    if gp._is_processing_node(node)
                ]
                # 注意：nodes_to_try 是旧列表，但仅包含节点名，后续节点若仍存在仍可处理；
                # 若需更严格的一致性，可考虑 break 并重启本轮，但会增加开销，此处暂不处理。

            except Exception as e:
                logger.error(f"Failed to apply method replacement for node '{target_node}': {e}")
                # 替换失败，跳过该节点，继续优化其他节点
                continue

    return improvement_made, current_loss, current_acc, total_replacements, processing_nodes

def refine_multi_node_joint_step(
    train,
    adaptoflux_instance,
    input_data: np.ndarray,
    target: np.ndarray,
    processing_nodes: List[str],
    current_loss: float,
    gp: Any
) -> Tuple[bool, float, float, int, List[str]]:
    """
    【TODO】多节点联合优化策略（Multi-Node Joint Refinement）
    该策略将同时选择多个节点（如相邻节点、同一层节点等），
    联合搜索其方法组合，以寻找全局更优的局部结构。
    当前仅为占位实现，抛出 NotImplementedError。
    """
    raise NotImplementedError(
        "Multi-node joint refinement is not yet implemented. "
        "This is a placeholder for future development."
    )