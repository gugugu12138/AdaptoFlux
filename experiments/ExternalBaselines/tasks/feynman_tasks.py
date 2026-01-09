import numpy as np

def generate_feynman_E_mc2(n_samples=100, seed=42):
    np.random.seed(seed)
    m = np.random.uniform(1, 10, n_samples)
    c = np.random.uniform(1, 5, n_samples)  # avoid huge numbers
    E = m * (c ** 2)
    X = np.stack([m, c], axis=1).astype(np.float32)
    y = E.astype(np.float32)
    return X, y, "m * c**2", ["m", "c"]

def generate_feynman_F_ma(n_samples=100, seed=42):
    np.random.seed(seed)
    m = np.random.uniform(1, 10, n_samples)
    a = np.random.uniform(0, 10, n_samples)
    F = m * a
    X = np.stack([m, a], axis=1).astype(np.float32)
    y = F.astype(np.float32)
    return X, y, "m * a", ["m", "a"]

def generate_feynman_KE(n_samples=100, seed=42):
    np.random.seed(seed)
    m = np.random.uniform(1, 10, n_samples)
    v = np.random.uniform(0, 10, n_samples)
    const_05_arr = np.full(n_samples, 0.5, dtype=np.float32)
    KE = 0.5 * m * (v ** 2)
    X = np.stack([m, v, const_05_arr], axis=1).astype(np.float32)
    y = KE.astype(np.float32)
    return X, y, "0.5 * m * v**2", ["m", "v", "const_05"]

# ===== 新增任务（仅使用 add, sub, mul, div, square）=====

def generate_feynman_Coulomb(n_samples=100, seed=42):
    # F = q1 * q2 / r^2  (k=1)
    np.random.seed(seed)
    q1 = np.random.uniform(1, 5, n_samples)
    q2 = np.random.uniform(1, 5, n_samples)
    r = np.random.uniform(0.5, 3, n_samples)
    F = q1 * q2 / (r ** 2)
    X = np.stack([q1, q2, r], axis=1).astype(np.float32)
    y = F.astype(np.float32)
    return X, y, "q1 * q2 / r**2", ["q1", "q2", "r"]

def generate_feynman_IdealGas(n_samples=100, seed=42):
    # P = n * T / V  (R=1)
    np.random.seed(seed)
    n = np.random.uniform(1, 5, n_samples)
    T = np.random.uniform(200, 400, n_samples)
    V = np.random.uniform(1, 10, n_samples)
    P = n * T / V
    X = np.stack([n, T, V], axis=1).astype(np.float32)
    y = P.astype(np.float32)
    return X, y, "n * T / V", ["n", "T", "V"]

def generate_feynman_OhmsLaw(n_samples=100, seed=42):
    # V = I * R
    np.random.seed(seed)
    I = np.random.uniform(0.1, 5, n_samples)
    R = np.random.uniform(1, 10, n_samples)
    V = I * R
    X = np.stack([I, R], axis=1).astype(np.float32)
    y = V.astype(np.float32)
    return X, y, "I * R", ["I", "R"]

def generate_feynman_Acceleration(n_samples=100, seed=42):
    # a = (v2 - v1) / t
    np.random.seed(seed)
    v1 = np.random.uniform(0, 10, n_samples)
    v2 = np.random.uniform(v1 + 0.1, v1 + 15, n_samples)  # ensure v2 > v1
    t = np.random.uniform(0.5, 5, n_samples)
    a = (v2 - v1) / t
    X = np.stack([v1, v2, t], axis=1).astype(np.float32)
    y = a.astype(np.float32)
    return X, y, "(v2 - v1) / t", ["v1", "v2", "t"]

def generate_feynman_Power(n_samples=100, seed=42):
    # P = V * I （电功率）
    np.random.seed(seed)
    V = np.random.uniform(1, 20, n_samples)
    I = np.random.uniform(0.1, 5, n_samples)
    P = V * I
    X = np.stack([V, I], axis=1).astype(np.float32)
    y = P.astype(np.float32)
    return X, y, "V * I", ["V", "I"]

# ===== 任务注册表 =====
TASK_REGISTRY = {
    "E_mc2": generate_feynman_E_mc2,
    "F_ma": generate_feynman_F_ma,
    "KE": generate_feynman_KE,
    "Coulomb": generate_feynman_Coulomb,
    "IdealGas": generate_feynman_IdealGas,
    "OhmsLaw": generate_feynman_OhmsLaw,
    "Acceleration": generate_feynman_Acceleration,
    "Power": generate_feynman_Power,
}

def get_task(task_name, n_samples=100, seed=42):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_REGISTRY[task_name](n_samples=n_samples, seed=seed)