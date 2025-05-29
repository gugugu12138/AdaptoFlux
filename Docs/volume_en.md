# Waveform Area-Split Probability Method

## 1. Method Overview
**Waveform Area-Split Probability** is a technique that transforms time-series data (waveforms) into probability distributions. The core idea is to divide the waveform into equal segments along the time/index axis, compute the absolute area of each segment, and then assign probabilities based on the proportion of each segment's area relative to the total area.

## 2. Mathematical Principles
Given a waveform of length `n`: `[y₀, y₁, ..., yₙ₋₁]`

### 2.1 Total Area Calculation
The total absolute area of the waveform is calculated using the trapezoidal rule to avoid cancellation of positive and negative values:
```math
A_{\text{total}} = \sum_{i=0}^{n-2} \frac{|y_i + y_{i+1}|}{2}
```

### 2.2 Segment Area Calculation
Divide the waveform into `k` equal-length segments (default `k=10`). The area of the j-th segment is:
```math
A_j = \sum_{i=\text{start}_j}^{\text{end}_j-1} \frac{|y_i + y_{i+1}|}{2}
```

### 2.3 Probability Assignment
The probability for the j-th segment is:
```math
P_j = \frac{A_j}{A_{\text{total}}}
```

## 3. Algorithm Implementation
```python
def _area(self, values):
    n = len(values)
    if n < 2 or self.num_bins == 0:
        return np.zeros(self.num_bins) if self.num_bins > 1 else 0.0

    # Compute total absolute area (trapezoidal method)
    y = np.array(values)
    total_area = np.sum(np.abs(y[:-1] + y[1:]) * 0.5)

    # Compute area for each segment
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

    # Normalization
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum() if probabilities.sum() > 0 else 1.0
    return probabilities
```

## 4. Extension Directions
1. **Weighted Area Calculation**: Support assigning higher weights to specific time intervals (trainable parameters)
2. **Adaptive Segmentation**: Dynamically adjust segment boundaries based on waveform characteristics (trainable parameters)
3. **Multidimensional Extension**: Process multi-channel waveform data (e.g., 3-axis accelerometer)