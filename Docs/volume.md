
# 波形面积分割概率方法

## 1. 方法概述
**波形面积分割概率**（Waveform Area-Split Probability）是一种将时序数据（波形）转化为概率分布的技术。其核心思想是将波形按时间/索引轴均匀分段，计算每一段的绝对面积占总面积的比例，作为该段的概率值。

## 2. 数学原理
给定长度为 `n` 的波形数据 `[y₀, y₁, ..., yₙ₋₁]`：

### 2.1 总面积计算
采用梯形法则计算波形的绝对面积（避免正负抵消）：
```math
A_{\text{total}} = \sum_{i=0}^{n-2} \frac{|y_i + y_{i+1}|}{2}
```

### 2.2 分段面积计算
将波形分为 `k` 个等长段（默认 `k=10`），第 `j` 段的面积为：
```math
A_j = \sum_{i=\text{start}_j}^{\text{end}_j-1} \frac{|y_i + y_{i+1}|}{2}
```

### 2.3 概率分配
第 `j` 段的概率为：
```math
P_j = \frac{A_j}{A_{\text{total}}}
```

## 3. 算法实现
```python
def _volume(self, values):
    """
    计算任意维度数据的体积/面积/长度分割概率（返回概率列表）
    
    参数:
        values (array-like): 任意维度的数组，shape = (N, ...)，其中 N >= 1
    
    返回:
        probabilities (np.ndarray): 每个分段的体积占比，形状为 (self.num_bins,)
    """
    values = np.asarray(values)
    n = values.shape[0]  # 第一维长度

    if n < 2 or self.num_bins <= 0:
        print('使用_volume方法至少需要两个数据点，且分段数大于0')
        return np.zeros(self.num_bins) if self.num_bins > 1 else 0.0

    def compute_volume(arr):
        result = arr.copy()
        while result.ndim > 1:
            result = trapz(result, axis=-1)
        return np.abs(result).sum()

    # 计算总体积
    total_volume = compute_volume(values)

    # 分段计算体积
    segment_length = n / self.num_bins
    probabilities = []

    for k in range(self.num_bins):
        start = int(k * segment_length)
        end = min(int((k + 1) * segment_length), n)
        if start >= end:
            probabilities.append(0.0)
            continue

        segment = values[start:end]
        segment_vol = compute_volume(segment)
        probabilities.append(segment_vol / total_volume if total_volume > 0 else 0.0)

    probabilities = np.array(probabilities)
    total_prob = probabilities.sum()
    if total_prob > 0:
        probabilities /= total_prob
    return probabilities
```

## 4. 扩展方向
1. **加权面积计算**：支持对特定时间段赋予更高权重（可训练的参数）
2. **自适应分段**：根据波形特征动态调整分段边界（可训练的参数）

