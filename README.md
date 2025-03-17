# AdaptoFlux

**一种基于功能集（包含多种类型的函数的集合）实现智能的算法**

## 项目概述
AdaptoFlux是一种基于功能集的智能算法。不同于传统的深度学习，该算法通过生成一个基于路径的操作流程，实现智能计算和优化。通过对功能集和坍缩函数的操作，该算法拥有极强的兼容性。

## 进展情况
- `main.py` 中的**神经退回部分**仍有部分问题待解决。
- **模型化简**和**新函数生成**部分仍在开发中。
- 正在编写**基于该算法在不修改MLP模型情况优化 MLP**的示例代码，并持续优化。
- 编写**指导值**计算逻辑。

## 未来工作
- 继续完善神经退回部分，修复相关问题。
- 进一步优化模型化简过程，提高计算效率。
- 完善新函数生成机制，以增强算法适用性。
- 完成并优化 MLP 优化示例代码，使其更具参考价值。

# 数据处理模型结构说明
## 数据流
从输入数据到输出结果的处理过程。数据沿着动态生成的路径逐层进行特征提取和转换，每一层级执行特定的任务，并最终通过 **坍缩函数** 将中间表示转化为目标形式的过程。

## 坍缩函数
可选的转换操作，用于在基于动态生成路径的数据流处理过程中，从路径末端提取经过逐层操作后的数据，并将其转化为具有特定目标格式的输出。  
具体而言，当网络中的某些值与 **指导值** 存在直接关系时，坍缩函数通过聚合或总结操作，将复杂的中间表示简化为更简洁且有用的目标形式。  
其输入可以是单个节点的特征向量，也可以是整个路径末端的综合数据，输出则根据任务需求定制，例如概率分布、类别标签或其他所需格式。

## 指导值
用于指导神经的生长或退化。

- 指标分类与层级划分：

  | 类别   | 示例指标       | 调整目标      | 影响权重 |
  | ---- | ---------- | --------- | ---- |
  | 核心任务 | 准确率、F1分数   | 直接优化任务性能  | 高（α） |
  | 路径质量 | 路径熵、路径深度   | 保障探索与架构健康 | 中（β） |
  | 计算效率 | 内存占用、FLOPs | 抑制资源浪费    | 低（γ） |

- **多指标融合公式**

  $$
  指导值 = \sum \omega_i \cdot 核心指标_i + \sum \phi_j \cdot 路径指标_j - \sum \psi_k \cdot 效率指标_k
  $$

- **示例计算公式**：

$$
指导值 = \alpha \cdot 准确率 + \beta \cdot 路径熵 - \gamma \cdot 冗余操作惩罚
$$

#### 路径熵计算

$$
路径熵 = -\sum P(路径_i) \cdot \log P(路径_i)
$$

其中，$P(\text{Path}_i)$ 表示第 $i$ 类路径的出现频率（统计窗口内的占比）。

#### 冗余操作惩罚计算

$$
冗余操作惩罚 = \sum (无效计算次数)
$$

## 功能集（Q）
包含多种类型的函数的集合。

## 函数集（F）
只包含映射函数的功能集。

## 操作集（O）
只包含动作函数的功能集。


$$
G = \{ g_1, g_2, g_3, \dots, g_n \}
$$

$$
F = \{ f_1, f_2, f_3, \dots, f_m \}
$$

$$
O = \{ o_1, o_2, o_3, \dots, o_k \}
$$

![基础结构](./assets/images/基础结构图2.0.png)

## 模型处理流程
1. **输入层处理**  
   - 初始数据点数量：n
   - 按照功能集规则随机分组
   - 对每个分组执行对应函数

2. **迭代处理**  
   - 处理后将数据还原并重新分组
   - 重复流程直至到达路径末端

3. **输出生成**  
   - 对尾部数据应用坍缩函数
   - 生成最终网络输出

---

## 数据量变化公式
### 关键参数定义
- `Iₐ`：函数a的输入/输出数据量比
- `H`：每层数据期望减少比例
- `k`：功能集函数总数
- `Wₐ`：函数a的被选概率

**核心公式**  
```math
H = \sum_{i=1}^{k} W_i I_i
```

### 层间数据量关系
- `n₀`：初始数据量
- `L`：模型层数
- `nₗ`：第L层数据量  
```math
n₀ \cdot H^L = n_L
```


---

## 函数集特性分析
### 分类定义
| 类型       | 特性                          | 反向推导能力        |
|------------|-------------------------------|---------------------|
| 双射函数集 | 所有函数为双射                | 完全可逆            |
| 单射函数集 | 所有函数为单射                | 可逆（需额外信息）  |
| 满射函数集 | 所有函数为满射                | 多输入对应单输出    |

### 特殊函数集示例
```math
F = \begin{cases}
f_1(a,b) = a \cdot c_1 + b \cdot d_1 \\
f_2(a,b) = a \cdot c_2 + b \cdot d_2 \\
\vdots \\
f_n(a,b) = a \cdot c_n + b \cdot d_n
\end{cases}
```
*条件：任意一组(c,d)互质*

---

## 应用特性
### 满射函数集特性
- **输入空间增长公式**  
  ```math
  T = R^C
  ```
  - `T`：输入空间大小
  - `R`：函数输入数量均值
  - `C`：函数调用总次数

### 加密与压缩应用
- 通过添加随机变量实现单射转换
- 支持输出到唯一输入的映射

### 对无直接输入数据时
当没有直接输入时，可以通过使用一个操作集作为功能集，我们可以使用一个周期信号作为输入，通过操作集中的动作函数组合获取数据（或者使得指导函数趋向目标），以此来实现模型的构建。  

**（这部分理论可行，之后会把图和完整概念放上来）**

---

## 模型训练示例（CIFAR-10）
**训练配置**  
- **数据集**：CIFAR-10
- **坍缩算法**：节点值累加求和
- **训练流程**：
  1. 像素值作为初始节点
  2. 随机选取函数进行操作（应该有一种更优的选择方法可以实现）
  3. 坍缩计算结果
  4. 计算指导变量（方差/准确率）
  5. 符合条件则保留路径，否则神经退化重试

**指导值**：
- 坍缩结果与标签方差
- 分类准确率


### 已知的问题
当H小于1时，易在一个空间陷入无限循环无法跳出。
