# FDIA

**一种基于功能集（包含多种类型的函数的集合）实现智能的算法**

## 项目概述
FDIA（Function Set-based Intelligence Algorithm）是一种基于功能集的智能算法。该算法通过不同类型的函数组合，实现智能计算和优化。

## 进展情况
- `main.py` 中的**神经退回部分**仍有部分问题待解决。
- **模型化简**和**新函数生成**部分仍在开发中。
- 正在编写**基于该算法优化 MLP**的示例代码，并持续优化。

## 未来工作
- 继续完善神经退回部分，修复相关问题。
- 进一步优化模型化简过程，提高计算效率。
- 完善新函数生成机制，以增强算法适用性。
- 完成并优化 MLP 优化示例代码，使其更具参考价值。

# 数据处理模型结构说明

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

---

## 模型训练示例（MNIST）
**训练配置**  
- **数据集**：MNIST手写数字
- **坍缩算法**：节点值累加求和
- **训练流程**：
  1. 像素值作为初始节点
  2. 随机选取函数进行操作
  3. 坍缩计算结果
  4. 计算指导变量（方差/准确率）
  5. 符合条件则保留路径，否则神经退化重试

**指导变量**：
- 坍缩结果与标签方差
- 分类准确率

---

> 注：公式中符号使用Unicode数学符号表示，实际应用时建议采用LaTeX数学环境进行渲染。
```
