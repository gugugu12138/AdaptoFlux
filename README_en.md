English | [中文](README.md)
# AdaptoFlux Project Overview (Simplified Version)

AdaptoFlux is an intelligent algorithm framework based on the concept of "function set + path search". Its core idea is to construct dynamic computing paths by combining basic functions, organize them in a graph structure (DFG), and support path interpretability and structural optimization.

Main features:
- Uses a function set (including MLP, random forest, etc.) as reasoning components;
- Builds dynamic data flow graphs to achieve modular computing and path combinations;
- Supports customizable "collapse functions" as the final output structure;
- Combines "path entropy + redundancy penalty + guidance value" for structural adjustment during training;
- Can be used for symbolic regression, small-sample modeling, structure search, logical combination tasks, and more.

---

# AdaptoFlux

**An intelligence algorithm based on a function set (a collection containing various types of functions)**

## Project Overview
AdaptoFlux is a function-set-based intelligent algorithm. Unlike traditional deep learning, this algorithm achieves intelligent computation and optimization by generating an operation process based on paths. By manipulating the function set and collapse function, this algorithm has extremely strong compatibility and relatively good interpretability.

## Current Status
- The **model simplification** and **new function generation** modules are still under development.
- Writing **example code for optimizing MLP without modifying the MLP model based on this algorithm**, with continuous improvements.
- Refactoring ATF part of the code using DFG structure

## Future Work
- Further optimize the model simplification process to improve computational efficiency.
- Improve the new function generation mechanism to enhance algorithm applicability.
- Complete and optimize MLP optimization example code to make it more valuable as a reference.
- Train on the original model after loading it.
- Extract a portion of the completed path from the model for retraining.
- Select different function sets for path selection based on different input data, grouping data to choose different function sets.
- Cut out a path from the current model, record the number of input and output data points, then train a new path that matches the original input and output to optimize the network.
- Add a function set decorator to directly define positions.
- Use multiple function sets to handle different data separately.

# How to Use
1. Create a new conda environment:

```bash
conda create -n AdaptoFlux python=3.12
conda activate AdaptoFlux
```

2. Clone the repository:

```bash
git clone https://github.com/gugugu12138/AdaptoFlux.git  
cd AdaptoFlux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```
Note: Additional configuration of the environment is required for the function set being used.

# Modifying Function Sets
The training and inference of AdaptoFlux are based on the execution of function sets. By modifying the functions in methods.py, you can provide AdaptoFlux with different options to achieve better results. (You can even include random forests and MLP models into the function set.)

# Data Processing Model Structure Description
## Data Flow
This describes the process from input data to output result. Data is feature-extracted and transformed layer by layer along dynamically generated paths, with each level performing specific tasks, and finally converting the intermediate representation into the target form through a **collapse function**.

## Collapse Function
Optional conversion operations used in the data flow processing based on dynamically generated paths, extracting data processed through layers at the end of the path and converting it into a specific target format. Specifically, when certain values in the network have a direct relationship with the **guidance value**, the collapse function simplifies complex intermediate representations into more concise target forms through aggregation or summarization operations. Its input can be either a feature vector of a single node or comprehensive data at the end of the entire path, and the output is customized according to task requirements, such as probability distributions, class labels, or other desired formats. (Similarly to the function set, the collapse function can flexibly select algorithms, such as using MLP as the collapse function.)

## Guidance Value
Used to guide the growth or degeneration of neurons.

- Metric classification and hierarchy:

  | Category   | Example Metrics       | Adjustment Goal      | Impact Weight |
  | ---- | ---------- | --------- | ---- |
  | Core Tasks | Accuracy, F1 Score   | Directly optimize task performance  | High (α) |
  | Path Quality | Path Entropy, Path Depth   | Ensure exploration and architecture health | Medium (β) |
  | Computational Efficiency | Memory Usage, FLOPs | Suppress resource waste    | Low (γ) |
  | Loss Control | MSE, RMSE, Cross-Entropy | Adjust optimization direction in early training stages, reduce impact later | Variable (δ) |

- **Multi-Metric Fusion Formula**

  $$
  \text{Guidance Value} = \sum \omega_i \cdot \text{Core Metric}_i + \sum \phi_j \cdot \text{Path Metric}_j - \sum \psi_k \cdot \text{Efficiency Metric}_k - \delta \cdot \text{Loss Value}
  $$

- **Example Calculation Formula**:

$$
\text{Guidance Value} = \alpha \cdot \text{Accuracy} + \beta \cdot \text{Path Entropy} - \gamma \cdot \text{Redundant Operation Penalty} - \delta \cdot \text{Loss Value}
$$

### Path Entropy Calculation

$$
\text{Path Entropy} = -\sum P(\text{Path}_i) \cdot \log P(\text{Path}_i)
$$

where $P(\text{Path}_{\text{i}})$ denotes the frequency of occurrence of the i-th type of path (proportion within the statistical window).

### Redundancy Operation Penalty Calculation

$$
\text{Redundancy Operation Penalty} = \sum (\text{Number of Invalid Calculations})
$$

## Function Set (Q)
A collection containing various types of functions.

## Function Set (F)
A function set containing only mapping functions.

## Operation Set (O)
A function set containing only action functions.

```math
G = \left\{ g_1, g_2, g_3, \dots, g_n \right\}
```

```math
F = \left\{ f_1, f_2, f_3, \dots, f_m \right\}
```

```math
O = \left\{ o_1, o_2, o_3, \dots, o_k \right\}
```

![Basic Structure](./assets/images/基础结构图2.0.png)

## Optimization Logic

1. **Generate Initial Model**
Use the process_random_method function to generate multiple models with specified layers (this part does not calculate loss) (or combine replace_random_elements to some extent for optimization). Compare the models and select the best-performing initial model.

2. **Modify Initial Model Graph Nodes**
For each functional node in the initial model, starting from a certain layer in a certain direction, compare each replaceable node (nodes with the same input and output dimensions), compare the loss after changing the node, and select the best-performing node. Repeat the scanning multiple times.

3. **Generate Simplified Model**
Randomly generate a model with the same input and output dimensions as a certain part of the graph, compare their inputs and outputs within a certain range, use the better-performing model to replace all identical parts in the graph, and repeat this process.

4. **Generate New Function Set**
During training, take the well-performing segments, iterate them into a new function set, and use the new function set after a certain number of iterations.

## Model Processing Workflow
1. **Input Layer Processing**  
   - Initial number of data points: n
   - Randomly grouped according to function set rules
   - Apply corresponding functions to each group

2. **Iterative Processing**  
   - Restore and regroup the data after processing
   - Repeat the process until reaching the end of the path

3. **Output Generation**  
   - Apply the collapse function to the tail data
   - Generate the final network output

---

## Data Volume Change Formula
### Key Parameter Definitions
- `Iₐ`: Input/output data volume ratio of function a
- `H`: Expected reduction ratio of data per layer
- `k`: Total number of functions in the function set
- `Wₐ`: Probability of selecting function a

**Core formula**  
```math
H = \sum_{i=1}^{k} W_i I_i
```

By processing with different function sets at different training stages, expansion, modification, and dimensionality reduction of data can be achieved. It can be seen from this formula that by appropriately modifying the random selection method, controlling the dimensions before input and collapse layer is very simple, showing strong compatibility with most activation functions.

### Inter-layer Data Volume Relationship
- `n₀`: Initial data volume
- `L`: Number of model layers
- `nₗ`: Data volume of the L-th layer  
```math
n₀ \cdot H^L = n_L
```
---

## Path Simplification
For already trained paths, we can extract the used function set, generate a list of random data applicable to this function set,
cut this list into multiple two-dimensional lists of different lengths, and use these two-dimensional lists as data for limited (e.g., layer-limited) unsupervised training.
After training multiple paths, compare them; if the input and output are completely the same (or mostly the same), consider those two paths equivalent, evaluate their metrics (such as path depth, running speed, etc.),
and replace the original part in the path with the better one.

## Function Set Characteristic Analysis
### Classification Definition
| Type       | Characteristics                          | Backward Deduction Capability        |
|------------|-------------------------------|---------------------|
| Bijective Function Set | All functions are bijective                | Fully reversible            |
| Injective Function Set | All functions are injective                | Reversible (with additional information)  |
| Surjective Function Set | All functions are surjective                | Multiple inputs correspond to a single output    |

### Special Function Set Examples
```math
F = \begin{cases}
f_1(a,b) = a \cdot c_1 + b \cdot d_1 \\
f_2(a,b) = a \cdot c_2 + b \cdot d_2 \\
\vdots \\
f_n(a,b) = a \cdot c_n + b \cdot d_n
\end{cases}
```
*Condition: Any pair (c,d) is coprime*

---
## Functional Set Characteristics

### Number of Combined Paths Formula

#### **Formula Definition**
$$
N_{\text{paths}} = \prod_{l=1}^{L} \left( |F|^{n_{l-1}} \right)
$$

Where:
- $N_{\text{paths}}$ : Total number of paths.
- $L$ : The depth (number of layers) of the path.
- $n_{l-1}$ : The amount of data at layer $l-1$ (i.e., the input data amount at layer $l$).
- $|F|$ : The size of the function set $F$ (the number of selectable functions).

#### **Key Explanations**
1. **Recursiveness**:
   - Each layer's path selection depends on the output data amount $n_{l-1}$ of the previous layer.
   - Each data point independently selects a function, so the number of branches per layer is $|F|^{n_{l-1}}$.

2. **Example Verification**:
   - **Case 1**: 1 layer, 1 data point, 2 functions  
     $N_{\text{paths}} = 2^1 = 2$ (conforms: paths are $f_1$ or $f_2$).
   - **Case 2**: 2 layers, 1 data point, 2 functions  
     $N_{\text{paths}} = 2^1 \times 2^1 = 4$ (each layer has 2 choices, combined into 4).
   - **Case 3**: 2 layers, 2 data points, 2 functions  
     $N_{\text{paths}} = 2^2 \times 2^2 = 16$ (4 choices in the first layer, 4 in the second, combined into 16).

3. **Dynamic Data Expansion**:
   - If the function may change the data volume (e.g., $n_l 
eq n_{l-1}$ ), additional definition of $n_l$ update rules is needed (e.g., $n_l = \sum_{i=1}^{n_{l-1}} \dim_\text{out}(f_i)$ ).

#### **Full Definition**
1. **Input**:
   - Initial data volume $n_0$ .
   - Function set $F = \{f_1, f_2, \dots, f_m\}$ .
   - Path depth $L$ .

2. **Output**:
   - Total number of paths $N_{\text{paths}}$ .

3. **Constraints**:
   - All functions have input/output data volumes of 1 (by default). If the functions support multiple inputs/outputs, adjust the formula to:

```math
N_{\text{paths}} = \prod_{l=1}^{L} \left( |F|^{n_{l-1}} \times \prod_{i=1}^{n_{l-1}} \dim_\text{out}(f_i) \right)
```

#### **Example Calculation**

##### **Problem Setup**

* **Initial Data Volume**:
  $n_0 = 2$ (two independent data points)

* **Function Set**:
  $F = \{f_1, f_2\}$ , each function:

  * Input: 1 data point
  * Output: 2 data points

* **Layers**: 2 layers

---

##### **Step-by-step Calculation**

###### **Layer 1 ($l=1$)**

* **Input Data Volume**:
  $n_0 = 2$

* **Each Data Point Chooses a Function**:

  * Data point 1 can choose $f_1$ or $f_2$ (2 options)
  * Data point 2 can choose $f_1$ or $f_2$ (2 options)

* **Branch Count**:
  $2 \times 2 = 4$

* **Output Data Volume**:
  Each function outputs 2 data points → For every 1 input processed, 2 outputs are generated.
  $n_1 = 2 \times 2 = 4$

##### **Layer 2 ($l=2$)**

* **Input Data Volume**:
  $n_1 = 4$

* **Each Data Point Chooses a Function**:
  Each data point has 2 options ($f_1$ or $f_2$)

* **Branch Count**:
  $2^4 = 16$

* **Output Data Volume (optional)**:
  $n_2 = 4 \times 2 = 8$ (can continue but doesn't affect the number of paths)

---

##### **Total Path Count**

* Branch count for Layer 1: 4
* Branch count for Layer 2: 16

$$
\text{Total Path Count} = 4 \times 16 = 64
$$

---

##### **Verification Enumeration**

###### **4 Choices for Layer 1**

1. $(f_1, f_1)$
2. $(f_1, f_2)$
3. $(f_2, f_1)$
4. $(f_2, f_2)$

###### **16 Combinations for Each Layer 1 Choice in Layer 2**

Taking the Layer 1 choice $(f_1, f_1)$ as an example:

* **Output Data Volume**:
  $2 \times 2 = 4$

* **Combination Count for Layer 2 Selection**:
  $2^4 = 16$

For example:
$(f_1, f_1, f_1, f_1)$,
$(f_1, f_1, f_1, f_2)$,
...
$(f_2, f_2, f_2, f_2)$

Total path count remains:

$$
4 \times 16 = 64
$$

---

##### **Generalized Formula Match**

According to the general formula:

$$
N_{\text{paths}} = \prod_{l=1}^{L} |F|^{n_{l-1}}
$$

* Layer 1:
  $|F|^{n_0} = 2^2 = 4$

* Layer 2:
  $|F|^{n_1} = 2^4 = 16$

* **Total Path Count**:
  $4 \times 16 = 64$

Consistent with enumeration ✅

---

### Feature Dimension Evolution of Function Combinations

#### Definition

**Background**
Assume we have a very simplified model composed of two layers ($L=2$), where each layer contains two different function combinations. Each function combination accepts certain inputs and produces outputs. Our goal is to calculate the solution space dimension and total solution space dimension of the entire model.

##### Layer 1

- **Combination A**: Input dimension is 2, output dimension is 3.
- **Combination B**: Input dimension is 2, output dimension is 4.

##### Layer 2

- **Combination C**: Input dimension is 3, output dimension is 5.
- **Combination D**: Input dimension is 4, output dimension is 6.

Initial input dimension is 2.

##### Solution Space

**Solution Space** is the set of all output results generated by non-equivalent DFG graphs that the algorithm can explore.

Each DFG graph represents a function combination path, but only when its output differs from others is it considered a new "solution."

**Example**

In the given example, there are four possible path combinations:

1. A → C  
2. A → D  
3. B → C  
4. B → D  

We run these paths and obtain their output results, denoted as:

- Output 1: $o_1$
- Output 2: $o_2$
- Output 3: $o_1$ (equivalent to the first path)
- Output 4: $o_3$

Then the actual **non-equivalent output set** is:

```math
\mathcal{O} = \left\{o_1, o_2, o_3 \right\}
```

Therefore, the size of the solution space is:

```math
|\mathcal{S}| = |\left\{o_1, o_2, o_3 \right\}| = 3
```

###### Formula Description

**Solution Space Definition Formula**  

```math
\mathcal{S} = \left\{ f(p) \mid p \in \mathcal{P} \right\}
```

**Solution Space Size Formula**  

$$
|\mathcal{S}| = \left| \bigcup_{p \in \mathcal{P}} \{f(p)\} \right|
$$

It can also be written as:

$$
|\mathcal{S}| = |\mathcal{P}| - \sum_{i=1}^{k}(n_i - 1)
$$

Where:
- $\mathcal{P}$: The set consisting of all path combinations, $\mathcal{P} = \{p_1, p_2, ..., p_N\}$
- $f(p_i)$: The output result corresponding to path $p_i$ (could be vectors, hash values, or some feature representation)
- $|\mathcal{P}|$: Total number of paths;
- $k$: The number of equivalence classes (i.e., different outputs);
- $n_i$: The number of paths contained in the i-th equivalence class (satisfying $\sum n_i = |\mathcal{P}|$ )

## Application Features
### Surjective Function Set Properties
- **Input Space Growth Formula**  
  ```math
  T = R^C
  ```
  - `T`: Input space size
  - `R`: Average number of function inputs
  - `C`: Total number of function calls

### Encryption and Compression Applications
- Achieve injective transformation by adding random variables
- Support mapping from output to unique input


### When There Is No Direct Input Data
When there is no direct input, by using an operation set as a function set, we can use a periodic signal as input and combine action functions in the operation set to acquire data (or make the guidance function approach the target), thereby building the model.

**(This part is theoretically feasible; diagrams and complete concepts will be added later)**

---


### Application Examples
In integrated operations, taking the outputs of various base classifiers as the model's inputs and using simple bit operations and weighted operations as the function set can combine multiple classifiers.