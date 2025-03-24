English | [中文](README_zh.md)
# AdaptoFlux

**An Intelligent Algorithm Based on Functional Sets (A Collection of Various Types of Functions)**

## Project Overview
AdaptoFlux is an intelligent algorithm based on functional sets. Unlike traditional deep learning, this algorithm achieves intelligent computation and optimization by generating a path-based operational process. Through operations on functional sets and collapse functions, this algorithm offers high compatibility.

## Progress Status
- The **model simplification** and **new function generation** parts are still under development.
- Currently writing example code to **optimize an MLP model without modifying its structure** using this algorithm and continuously improving it.

## Future Work
- Further optimize the model simplification process to enhance computational efficiency.
- Improve the new function generation mechanism to enhance algorithm applicability.
- Complete and refine the MLP optimization example code to make it more valuable as a reference.

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
Tips: The required feature set needs additional environment configuration.

# Modifying Functional Sets
AdaptoFlux's training and inference operate based on functional sets. By modifying the functions in `methods.py`, different selections can be provided for AdaptoFlux to achieve better results.

# Data Processing Model Structure Description
## Data Flow
The process from input data to output results. Data undergoes feature extraction and transformation layer by layer along a dynamically generated path. Each level performs specific tasks, and finally, the **collapse function** converts the intermediate representation into the target format.

## Collapse Function
An optional transformation operation used in dynamically generated path-based data flow processing. It extracts data at the end of the path after undergoing layer-by-layer operations and converts it into a specific target format.

Specifically, when certain values in the network have a direct relationship with **guidance values**, the collapse function simplifies complex intermediate representations into more concise target formats through aggregation or summarization operations. The input can be a feature vector of a single node or comprehensive data from the end of the path, while the output is customized according to task requirements, such as probability distributions, category labels, or other required formats.

## Guidance Values
Used to guide the growth or degradation of neurons.

- **Metric Classification and Hierarchical Division:**

  | Category     | Example Metrics       | Adjustment Target | Impact Weight |
  |-------------|----------------------|------------------|--------------|
  | Core Task   | Accuracy, F1 Score   | Directly optimize task performance | High (\(\alpha\)) |
  | Path Quality | Path entropy, Path depth | Ensure exploration and architecture health | Medium (\(\beta\)) |
  | Computational Efficiency | Memory usage, FLOPs | Suppress resource wastage | Low (\(\gamma\)) |

- **Multi-Indicator Fusion Formula:**

  $$
  Guidance Value = \sum \omega_i \cdot Core Metric_i + \sum \phi_j \cdot Path Metric_j - \sum \psi_k \cdot Efficiency Metric_k
  $$

- **Example Calculation Formula:**

  $$
  Guidance Value = \alpha \cdot Accuracy + \beta \cdot Path Entropy - \gamma \cdot Redundant Operations Penalty
  $$

#### Path Entropy Calculation

  $$
  Path Entropy = -\sum P(Path_i) \cdot \log P(Path_i)
  $$

where \( P(Path_i) \) represents the occurrence frequency of the \( i \)-th category path (proportion within the statistical window).

#### Redundant Operations Penalty Calculation

  $$
  Redundant Operations Penalty = \sum (Invalid Computation Count)
  $$

## Functional Set (Q)
A collection containing various types of functions.

## Function Set (F)
A functional set that only contains mapping functions.

## Operation Set (O)
A functional set that only contains action functions.

$$
G = \{ g_1, g_2, g_3, \dots, g_n \}
$$

$$
F = \{ f_1, f_2, f_3, \dots, f_m \}
$$

$$
O = \{ o_1, o_2, o_3, \dots, o_k \}
$$

![Basic Structure](./assets/images/基础结构图2.0.png)

## Model Processing Flow
1. **Input Layer Processing**  
   - Initial data point count: \( n \)  
   - Random grouping according to functional set rules  
   - Execute corresponding functions for each group  

2. **Iterative Processing**  
   - Restore and regroup data after processing  
   - Repeat the process until reaching the path endpoint  

3. **Output Generation**  
   - Apply collapse function to terminal data  
   - Generate final network output  

---

## Data Volume Change Formula
### Key Parameter Definitions
- \( I_a \): Input/output data volume ratio for function \( a \)
- \( H \): Expected data reduction ratio per layer
- \( k \): Total number of functions in the functional set
- \( W_a \): Selection probability of function \( a \)

**Core Formula**  
```math
H = \sum_{i=1}^{k} W_i I_i
```

### Layer-to-Layer Data Volume Relationship
- \( n_0 \): Initial data volume
- \( L \): Model layers
- \( n_L \): Data volume at layer \( L \)
```math
n_0 \cdot H^L = n_L
```

---

## Function Set Characteristics Analysis
### Classification Definition
| Type       | Characteristics                   | Reverse Deduction Ability |
|------------|----------------------------------|--------------------------|
| Bijection Set | All functions are bijections  | Fully reversible         |
| Injection Set | All functions are injections  | Reversible (requires extra info) |
| Surjection Set | All functions are surjections | Multiple inputs correspond to a single output |

### Special Function Set Example
```math
F = \begin{cases}
f_1(a,b) = a \cdot c_1 + b \cdot d_1 \\
f_2(a,b) = a \cdot c_2 + b \cdot d_2 \\
\vdots \\
f_n(a,b) = a \cdot c_n + b \cdot d_n
\end{cases}
```
*Condition: Any set of \( (c,d) \) must be coprime*

---

## Application Characteristics
### Surjection Function Set Properties
- **Input Space Growth Formula**
```math
T = R^C
```
- \( T \): Input space size
- \( R \): Average function input count
- \( C \): Total function call count

### Encryption and Compression Applications
- Convert to injections by adding random variables
- Support mapping unique outputs to inputs

### When No Direct Input Data Exists
If no direct input exists, a periodic signal can be used as input while action functions in the operation set extract data (or drive guidance function towards targets), enabling model construction.

**(This theoretical approach is feasible, and diagrams and a full concept explanation will be added later.)**

---

### Known Issues

