import numpy as np

class DynamicWeightController:
    """动态权重控制器，用于调整AdaptoFlux算法中的指导值权重（α, β, γ）
    
    功能：
    - 分阶段调整权重：探索期 → 过渡期 → 收敛期
    - 根据路径熵反馈动态修正权重
    - 保证权重归一化（α + β + γ = 1）
    """
    
    def __init__(self, total_steps):
        """初始化控制器
        Args:
            total_steps (int): 总训练步数，用于进度计算
        """
        self.total_steps = total_steps  # 总训练步数
        self.phase = "exploration"      # 初始阶段标记（exploration/transition/convergence）
    
    def get_weights(self, current_step, path_entropy):
        """获取当前步的权重组合
        Args:
            current_step (int): 当前训练步数
            path_entropy (int): 路径熵值
        
        Returns:
            tuple: 归一化后的权重 (α, β, γ)
        """
        t = current_step / self.total_steps  # 计算归一化进度（范围0~1）

        # ==================== 阶段划分 ====================
        # 阶段1：探索期（前20%训练步）
        # - 目标：最大化路径多样性（β主导）
        # - α缓慢指数增长：0.3*(1 - e^{-5t})，初期接近0，逐步上升
        if t < 0.2:
            alpha = 0.3 * (1 - np.exp(-5 * t))  # 指数增长公式
            beta = 0.6                          # 固定高β值鼓励探索
            gamma = 0.1                         # 低冗余惩罚

        # 阶段2：过渡期（20%~70%训练步）
        # - 目标：线性转向任务性能优化（α↑, β↓, γ微调）
        elif t < 0.7:
            phase_progress = (t - 0.2) / 0.5    # 过渡期内部进度（0~1）
            alpha = 0.3 + 0.5 * phase_progress  # 线性增长：0.3 → 0.8
            beta = 0.6 - 0.4 * phase_progress   # 线性衰减：0.6 → 0.2
            gamma = 0.1 + 0.1 * phase_progress  # 缓慢增加：0.1 → 0.2

        # 阶段3：收敛期（后30%训练步）
        # - 目标：聚焦任务性能（α主导，维持基础β/γ防僵化）
        else:
            alpha = 0.8   # 高任务权重
            beta = 0.1     # 最低探索权重
            gamma = 0.1    # 基础冗余抑制
            self.phase = "convergence"  # 更新阶段标记

        # ==================== 反馈调整 ====================
        # 当路径熵低于阈值且未到收敛期时，强制增加探索性
        # 目的：防止在过渡期过早失去多样性
        if path_entropy < 0.2 and self.phase != "convergence":
            beta += 0.1               # 提升β鼓励探索
            alpha = max(alpha - 0.05, 0.2)  # 降低α但保持最低值0.2

        # ==================== 权重归一化 ====================
        # 确保权重总和为1，避免数值尺度问题
        total = alpha + beta + gamma
        return alpha/total, beta/total, gamma/total