# experiments/ExternalBaselines/utils.py
import numpy as np
from sklearn.model_selection import train_test_split

def is_numerical_exact_match(y_pred, y_true, tol=1e-5):
    """判断是否数值上完全匹配（用于 zero-shot 符号回归）"""
    return np.all(np.abs(y_pred - y_true) < tol)

def train_test_split_feynman(X, y, test_size=0.5, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)