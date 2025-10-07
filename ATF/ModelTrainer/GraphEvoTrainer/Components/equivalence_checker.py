import numpy as np

class MSEEquivalenceChecker:
    """
    基于 MSE 的功能等效性验证器。
    """
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold  # 相似度阈值

    def is_equivalent(self, original_output: np.ndarray, replacement_output: np.ndarray) -> bool:
        """
        判断两个输出是否功能等效。
        使用归一化 MSE 计算相似度。
        """
        mse = np.mean((original_output - replacement_output) ** 2)
        # 使用原始输出的方差作为归一化因子
        var_original = np.var(original_output) + 1e-8
        similarity = 1.0 - (mse / var_original)
        return similarity >= self.threshold