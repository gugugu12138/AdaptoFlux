# experiments/ExternalBaselines/baselines/pysr_runner.py
import numpy as np
from pysr import pysr, sympy2torch
from ..utils import is_numerical_exact_match, train_test_split_feynman

def run_pysr(X, y, test_size=0.5, random_state=42):
    """
    Run PySR with minimal operator set for fair comparison.
    """
    X_train, X_test, y_train, y_test = train_test_split_feynman(X, y, test_size=test_size, random_state=random_state)
    
    # ⚠️ 关键：仅使用基础操作符，禁用高阶函数和自动常数
    equations = pysr(
        X_train,
        y_train,
        niterations=30,              # ← 从 100 → 30
        populations=10,              # ← 从 20 → 10
        ncyclesperiteration=30,      # ← 从 100 → 30（总表达式 ≈ 900）
        timeout_in_seconds=20,       # ← 单次最多 20 秒（防卡死）
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        maxsize=15,                  # Feynman 公式都很短
        procs=1,
        parallelism="serial",        # 必须！
        deterministic=True,          # 配合 random_state
        random_state=random_state,
        progress=False,
        batching=False,
        warmup_maxsize_by=0.0,
    )
    
    if len(equations) == 0:
        return {
            "exact_match": False,
            "mse": float("inf"),
            "expression": "NO_EQUATION_FOUND"
        }
    
    # 取 Pareto 前沿中 MSE 最小的方程
    best_eq = equations.loc[equations["loss"].idxmin()]
    sympy_expr = best_eq["sympy_format"]
    
    # 编译为可调用函数
    try:
        torch_module = sympy2torch(sympy_expr, variables=[f"x{i}" for i in range(X_test.shape[1])])
        y_pred = torch_module(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
    except Exception as e:
        # 回退：用字符串 eval（不推荐，但保底）
        from sympy import lambdify
        f = lambdify([f"x{i}" for i in range(X_test.shape[1])], sympy_expr, "numpy")
        try:
            y_pred = f(*X_test.T)
        except Exception:
            y_pred = np.full_like(y_test, np.nan)
    
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        mse = float("inf")
        exact = False
    else:
        mse = np.mean((y_pred - y_test) ** 2)
        exact = is_numerical_exact_match(y_pred, y_test)
    
    return {
        "exact_match": bool(exact),
        "mse": float(mse),
        "expression": str(sympy_expr)
    }