import numpy as np
import math
from ATF.methods.decorators import method_profile
from experiments.minimal_stateful_comparison.env import STATE

# ==========================================
# 1. 核心算子 (专属类型)
# ==========================================
@method_profile(input_types=['scalar'], output_types=['pos_type'], group="state_read")
def read_current_pos(dummy): return float(STATE.pos)

@method_profile(input_types=['scalar'], output_types=['target_type'], group="state_read")
def read_target(dummy): return float(STATE.target)

@method_profile(input_types=['pos_type', 'target_type'], output_types=['scalar'], group="pure_computation")
def calc_direction(pos, target): return float(pos - target)

@method_profile(input_types=['scalar'], output_types=['scalar'], group="stateful_action")
def move_towards_target(diff):
    if diff > 0.5: STATE.pos = max(-2, STATE.pos - 1)
    elif diff < -0.5: STATE.pos = min(2, STATE.pos + 1)
    return float(STATE.pos)

# ==========================================
# 2. 干扰项 (Distractors) 🔥
# ==========================================
# 注意：这些干扰项只接受/输出通用 'scalar'。
# 因为 read_current_pos 输出的是 'pos_type'，类型路由器会直接拒绝非法连接！
@method_profile(input_types=['scalar', 'scalar'], output_types=['scalar'], group="pure_computation")
def add(x, y): return float(x + y)

@method_profile(input_types=['scalar', 'scalar'], output_types=['scalar'], group="pure_computation")
def multiply(x, y): return float(x * y)

@method_profile(input_types=['scalar'], output_types=['scalar'], group="pure_computation")
def sin_op(x): return float(math.sin(x))

@method_profile(input_types=['scalar'], output_types=['scalar'], group="pure_computation")
def cos_op(x): return float(math.cos(x))

@method_profile(input_types=['scalar'], output_types=['scalar'], group="pure_computation")
def identity(x): return float(x)