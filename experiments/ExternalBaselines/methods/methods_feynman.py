# experiments/ExternalBaselines/methods/methods_feynman.py
from ATF.methods.decorators import method_profile
import numpy as np

@method_profile(output_count=1, input_types=['scalar','scalar'], output_types=['scalar'])
def add(a, b):
    return [a + b]

@method_profile(output_count=1, input_types=['scalar','scalar'], output_types=['scalar'])
def sub(a, b):
    return [a - b]

@method_profile(output_count=1, input_types=['scalar','scalar'], output_types=['scalar'])
def mul(a, b):
    return [a * b]

@method_profile(output_count=1, input_types=['scalar','scalar'], output_types=['scalar'])
def div(a, b):
    return [a / b] if b != 0 else [0.0]

# 由于gp内部可自行复制参数，这里额外引入一个返回两个相同值的方法
@method_profile(
    output_count=2,
    input_types=['scalar'],
    output_types=['scalar', 'scalar'],
)
# Fan-out operator: duplicate input for multi-branch consumption in graph.
def fanout(x):
    return [x, x]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar']
)
def identity(x):
    """Pass-through / identity operation. Essential for skip connections and variable-depth programs."""
    return [x]
