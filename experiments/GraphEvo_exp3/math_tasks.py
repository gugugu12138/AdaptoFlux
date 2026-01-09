# math_tasks.py
import numpy as np

SCALAR_TYPE = 'scalar'

# 三阶任务函数
f1 = lambda x: (x + 1) * 2
f2 = lambda x: f1(x) + 3
f3 = lambda x: f2(x) * 0.5

def generate_task_data(func: callable, n: int = 500, low: float = -10.0, high: float = 10.0):
    x = np.random.uniform(low, high, (n, 1)).astype(np.float32)
    y = np.array([func(xi[0]) for xi in x], dtype=np.float32)
    return x, y