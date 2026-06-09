# GraphProcessor 图结构与数据流规范文档

## 1. 概述
`GraphProcessor` 是一个用于管理、构建和推理有向多图（`nx.MultiDiGraph`）的核心类。该图结构被设计用于表示**计算图（Computational Graph）**或**神经网络架构**，支持层级化构建、多输入多输出方法映射、向量化推理以及子图替换。

图中的数据流从虚拟的 `root` 节点出发，经过一系列处理节点（执行具体方法），最终汇聚到虚拟的 `collapse` 节点进行聚合输出。

---

## 2. 节点规范 (Node Attributes)

图中的节点分为三种类型：**特殊控制节点**（`root`, `collapse`）、**常规处理节点**和**透传/丢弃节点**。

### 2.1 特殊控制节点
*   **`"root"`**: 虚拟输入节点。不存储业务属性，仅作为所有输入特征（`input_samples`）的起点。
*   **`"collapse"`**: 虚拟聚合节点。不存储业务属性，作为所有最终特征的汇聚点，由 `CollapseFunctionManager` 对其接收到的数据进行最终聚合（如求和、拼接等）。

### 2.2 常规与透传节点属性
在创建常规处理节点和透传节点时，包含以下属性：

| 属性名 | 类型 | 必填 | 描述与意义 |
| :--- | :--- | :---: | :--- |
| **`method_name`** | `str` | 是 | 节点所绑定的计算方法名称。该名称必须存在于 `GraphProcessor.methods` 字典中。对于透传/丢弃节点，通常为 `"null"` 或特定的 `discard_node_method_name`。 |
| **`layer`** | `int` | 是 | 节点所在的拓扑层数（深度）。用于图的层级管理、回退删除（`remove_last_nx_layer`）以及生成唯一节点 ID。 |
| **`is_passthrough`** | `bool` | 否* | 标识该节点是否为**透传节点**。如果为 `True`，推理引擎将跳过 `method_name` 对应的函数执行，直接将输入数据按原样传递到输出（通常用于处理未匹配/被丢弃的数据流）。<br>*注：老版本模型可能缺失此属性，代码会自动推断。 |

### 2.3 节点 ID 命名规范
常规节点的 ID 严格遵循 `{layer}_{index}_{method_name}` 的格式（例如：`2_0_add_values`）。
*   `layer`: 节点所在的层数。
*   `index`: 该层中同种 `method_name` 的局部索引（从 0 开始递增），由 `_generate_unique_node_id` 保证唯一性。
*   `method_name`: 绑定的方法名。

---

## 3. 边规范 (Edge Attributes)

边代表了数据（特征）在节点之间的流动。由于使用的是 `MultiDiGraph`，两个节点之间可以有多条边。边的属性对于**数据对齐**和**排序**至关重要。

| 属性名 | 类型 | 必填 | 描述与意义 |
| :--- | :--- | :---: | :--- |
| **`input_slot`** | `int` | 是 | **目标节点的输入槽位索引**。指示当前边传递的数据应该作为目标节点方法的第几个参数（从 0 开始）。推理时会按此值排序以确保参数顺序正确。 |
| **`output_index`** | `int` | 是 | **源节点的局部输出索引**。如果一个方法返回多个值（如 `output_count > 1`），此属性指示当前边提取的是源节点返回结果中的第几个值（从 0 开始）。 |
| **`data_coord`** | `int` | 是 | **当前层输出特征的局部坐标/索引**。<br>在 `append_nx_layer` 中从 `0` 开始递增。**它不是全局唯一的**，而是每次添加新层时重新计算的局部坐标。由于指向 `collapse` 的边总是代表“当前最新层”的输出，该属性仅用于在最终聚合时，**保证当前层输出的特征送入 `collapse` 函数时的相对顺序正确**。*(注：推理代码中将其命名为 `global_coord` 属于命名误导，本质为局部排序键)*。 |
| **`data_type`** | `str` | 否 | 边上流动的数据类型标识（如 `'scalar'`, `'vector'`, `'object'` 等）。主要用于元数据记录、类型检查或子图替换时的接口匹配。 |
| **`port_name`** | `str` | 否 | **逻辑端口名称**。主要在 `replace_subgraph_with_graph`（子图替换）操作中使用，用于将外部图的边与子图内部的输入/输出边界进行语义绑定。 |

---

## 4. 数据流与推理机制 (Inference Mechanism)

理解边和节点的属性后，以下是数据在 `infer_with_graph` 或 `infer_with_task_parallel` 中的流转逻辑：

### 4.1 数据提取与路由 (Input/Output Routing)
1.  **从源节点提取**：当数据从节点 A 流向节点 B 时，引擎读取边的 `output_index`。如果节点 A 的输出是 `[val0, val1]`，且 `output_index=1`，则提取 `val1`。
2.  **向目标节点注入**：引擎读取边的 `input_slot`。节点 B 会收集所有入边，并按 `input_slot` 从小到大排序，确保传入 `method_name` 对应函数的参数顺序绝对正确。

### 4.2 节点执行逻辑 (Node Execution)
*   **常规节点**：调用 `methods[method_name]["function"]`。支持**向量化执行**（如果 `vectorized=True` 且数据类型允许）或**逐样本循环执行**（作为 fallback）。
*   **透传节点** (`is_passthrough=True`)：不执行函数，直接将输入包装为 `[[feat] for feat in input]` 格式输出。

### 4.3 最终聚合逻辑 (Collapse Aggregation)
这是理解 `data_coord` 局部性的核心环节：
1.  **收集当前层输出**：引擎遍历所有指向 `"collapse"` 节点的入边。由于图的演化机制（`append_nx_layer` 会断开旧边），这些边**必然全部来自当前图的最外层节点**。
2.  **提取局部特征**：通过边的 `output_index` 从源节点提取对应的特征值，并读取该边的 `data_coord`（即当前层的局部索引）。
3.  **局部排序**：引擎根据 `data_coord` 对收集到的特征进行**升序排序**。因为 `data_coord` 是在添加当前层时从 0 连续分配的，这种局部排序足以保证特征的正确顺序。
4.  **执行聚合**：排序后的特征列表被送入 `self.collapse_manager.collapse()`。

---

## 5. 核心图操作说明

### 5.1 添加新层 (`append_nx_layer`)
*   **局部坐标重置**：每次调用时，内部计数器 `new_index_edge` 从 0 开始，为新产生的指向 `collapse` 的边分配局部的 `data_coord`。
*   **边的替换**：首先删除原本直接连向 `collapse` 的旧边（`self.graph.remove_edge(u, v)`），将旧边重定向到新节点并赋予 `input_slot`。
*   **连接新层**：从新节点引出新的边指向 `collapse`，分配 `data_coord`、`output_index` 和 `data_type`。

### 5.2 删除末层 (`remove_last_nx_layer`)
*   收集所有指向 `collapse` 的边（即当前最外层节点的出边）。
*   将这些边重新连接回它们原本的前驱节点与 `collapse` 之间，恢复上一层的图结构。
*   删除当前最外层的所有节点，并将 `self.layer` 减 1。

### 5.3 节点算子替换 (`replace_node_method`)
*   在**不改变图拓扑结构**（入边和出边保持不变）的前提下，仅替换节点的 `method_name`。
*   更新节点的 `is_passthrough` 状态，并生成符合 `{layer}_{new_index}_{new_method_name}` 格式的新节点 ID。常用于神经架构搜索（NAS）或模型微调中的算子替换。

### 5.4 子图替换 (`replace_subgraph_with_graph`)
*   利用 `port_name` 将外部图的边无缝接入被替换的子图（通过 `input_port_bindings` 和 `output_port_bindings`）。
*   自动计算子图内部节点的 `layer`：基于子图内节点距离 `root` 的拓扑最短路径长度，加上全局偏移量（被替换子图的最小 layer）。
*   保留原有的 `data_coord` 和 `input_slot` 逻辑，确保替换后数据流不中断。