
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
def _area(self, values):
    n = len(values)
    if n < 2 or self.num_bins == 0:
        return np.zeros(self.num_bins) if self.num_bins > 1 else 0.0

    # 计算绝对面积（梯形法）
    y = np.array(values)
    total_area = np.sum(np.abs(y[:-1] + y[1:]) * 0.5)

    # 分段计算面积
    segment_length = n / self.num_bins
    probabilities = []
    for k in range(self.num_bins):
        start = int(k * segment_length)
        end = min(int((k + 1) * segment_length), n - 1)
        if start >= end:
            probabilities.append(0.0)
            continue
        segment_y = y[start:end+1]
        segment_area = np.sum(np.abs(segment_y[:-1] + segment_y[1:]) * 0.5)
        probabilities.append(segment_area / total_area if total_area > 0 else 0.0)

    # 归一化处理
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum() if probabilities.sum() > 0 else 1.0
    return probabilities
```

## 4. 扩展方向
1. **加权面积计算**：支持对特定时间段赋予更高权重（可训练的参数）
2. **自适应分段**：根据波形特征动态调整分段边界（可训练的参数）
3. **多维度扩展**：处理多通道波形数据（如3轴加速度计）

