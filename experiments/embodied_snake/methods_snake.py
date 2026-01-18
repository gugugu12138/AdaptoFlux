# experiments/embodied_snake/methods_snake.py
from ATF.methods.decorators import method_profile
from experiments.embodied_snake.snake_state import SNAKE_STATE
import numpy as np

# Constants
GRID_SIZE = 10
# 注意：不再需要 STEP_SIZE，因为我们用整数坐标


# === Input Methods (保持不变，因为 obs 是 float32) ===
@method_profile(input_types=['raw_signal'], output_types=['head_x'], group="input", weight=1.0)
def get_head_x(x):
    obs = SNAKE_STATE.current_observation
    return [float(obs[0])] if obs is not None else [None]

@method_profile(input_types=['raw_signal'], output_types=['head_y'], group="input", weight=1.0)
def get_head_y(x):
    obs = SNAKE_STATE.current_observation
    return [float(obs[1])] if obs is not None else [None]

@method_profile(input_types=['raw_signal'], output_types=['food_x'], group="input", weight=1.0)
def get_food_x(x):
    obs = SNAKE_STATE.current_observation
    return [float(obs[2])] if obs is not None else [None]

@method_profile(input_types=['raw_signal'], output_types=['food_y'], group="input", weight=1.0)
def get_food_y(x):
    obs = SNAKE_STATE.current_observation
    return [float(obs[3])] if obs is not None else [None]

@method_profile(input_types=['raw_signal'], output_types=['dir_x'], group="input", weight=1.0)
def get_dir_x(x):
    obs = SNAKE_STATE.current_observation
    return [float(obs[4])] if obs is not None else [None]

@method_profile(input_types=['raw_signal'], output_types=['dir_y'], group="input", weight=1.0)
def get_dir_y(x):
    obs = SNAKE_STATE.current_observation
    return [float(obs[5])] if obs is not None else [None]


# === Helper: Convert normalized float to integer grid coord ===
def _to_grid_coord(norm_val):
    """Convert normalized coordinate (e.g., 0.9) to integer grid index (e.g., 9)"""
    # Add small epsilon to handle float32(0.9) = 0.899999976...
    return int(round(norm_val * GRID_SIZE))

def _to_direction_component(val):
    """Convert direction component (should be -1, 0, or 1) to int"""
    return int(round(val))

def _is_trap_position(x, y, snake_body_set, grid_size):
    """
    判断 (x, y) 是否是一个陷阱位置（逃生出口 < 2）
    注意：这里 x=列, y=行，与 env 一致
    """
    if not (0 <= x < grid_size and 0 <= y < grid_size):
        return True  # 墙外当然是陷阱
    
    if (x, y) in snake_body_set:
        return True  # 自身身体，直接危险

    neighbors = [
        (x - 1, y),  # up
        (x + 1, y),  # down
        (x, y - 1),  # left
        (x, y + 1)   # right
    ]
    
    free_neighbors = 0
    for nx, ny in neighbors:
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            if (nx, ny) not in snake_body_set:
                free_neighbors += 1

    # 如果自由邻居 < 2，说明很可能是死胡同
    return free_neighbors < 1

_is_trap_position.is_internal_decorator = True
_to_grid_coord.is_internal_decorator = True
_to_direction_component.is_internal_decorator = True


# === Perception: Danger Detection (INTEGER-BASED, NO FLOAT!) ===
@method_profile(input_types=['head_x', 'head_y', 'dir_x', 'dir_y'],
                output_types=['danger_front'], group="perception", weight=3.0)
def is_danger_ahead(head_x, head_y, dir_x, dir_y):
    if None in (head_x, head_y, dir_x, dir_y):
        return [None]
    
    # Convert to integer coordinates
    x = _to_grid_coord(head_x)
    y = _to_grid_coord(head_y)
    dx = _to_direction_component(dir_x)
    dy = _to_direction_component(dir_y)
    
    # Compute next position (1 step)
    new_x = x + dx
    new_y = y + dy
    
    snake_body_set = SNAKE_STATE.get_snake_body()

    # 1. 立即碰撞
    wall_collision = not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE)
    self_collision = (new_x, new_y) in snake_body_set
    immediate_danger = wall_collision or self_collision

    # 2. 陷阱检测（即使现在安全，但进去就出不来）
    trap_ahead = False
    if not immediate_danger:  # 只有安全时才检查是否是陷阱
        trap_ahead = _is_trap_position(new_x, new_y, snake_body_set, GRID_SIZE)

    danger = immediate_danger or trap_ahead


    print(f"Front danger check (int): from ({x},{y}) + ({dx},{dy}) → ({new_x},{new_y}), danger={danger}")
    return [1.0 if danger else 0.0]


@method_profile(input_types=['head_x', 'head_y', 'dir_x', 'dir_y'],
                output_types=['danger_left'], group="perception", weight=3.0)
def is_danger_left(head_x, head_y, dir_x, dir_y):
    if None in (head_x, head_y, dir_x, dir_y):
        return [None]
    
    x = _to_grid_coord(head_x)
    y = _to_grid_coord(head_y)
    dx = _to_direction_component(dir_x)
    dy = _to_direction_component(dir_y)
    
    # Left turn: (dx, dy) → (-dy, dx)
    new_dx = -dy
    new_dy = dx
    new_x = x + new_dx
    new_y = y + new_dy

    snake_body_set = SNAKE_STATE.get_snake_body()

    # 1. 立即碰撞
    wall_collision = not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE)
    self_collision = (new_x, new_y) in snake_body_set
    immediate_danger = wall_collision or self_collision

    # 2. 陷阱检测（即使现在安全，但进去就出不来）
    trap_ahead = False
    if not immediate_danger:  # 只有安全时才检查是否是陷阱
        trap_ahead = _is_trap_position(new_x, new_y, snake_body_set, GRID_SIZE)

    danger = immediate_danger or trap_ahead


    print(f"Left danger check (int): from ({x},{y}) turn left → ({new_x},{new_y}), danger={danger}")
    return [1.0 if danger else 0.0]


@method_profile(input_types=['head_x', 'head_y', 'dir_x', 'dir_y'],
                output_types=['danger_right'], group="perception", weight=3.0)
def is_danger_right(head_x, head_y, dir_x, dir_y):
    if None in (head_x, head_y, dir_x, dir_y):
        return [None]
    
    x = _to_grid_coord(head_x)
    y = _to_grid_coord(head_y)
    dx = _to_direction_component(dir_x)
    dy = _to_direction_component(dir_y)
    
    # Right turn: (dx, dy) → (dy, -dx)
    new_dx = dy
    new_dy = -dx
    new_x = x + new_dx
    new_y = y + new_dy

    snake_body_set = SNAKE_STATE.get_snake_body()
    
    # 1. 立即碰撞
    wall_collision = not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE)
    self_collision = (new_x, new_y) in snake_body_set
    immediate_danger = wall_collision or self_collision

    # 2. 陷阱检测（即使现在安全，但进去就出不来）
    trap_ahead = False
    if not immediate_danger:  # 只有安全时才检查是否是陷阱
        trap_ahead = _is_trap_position(new_x, new_y, snake_body_set, GRID_SIZE)

    danger = immediate_danger or trap_ahead


    print(f"Right danger check (int): from ({x},{y}) turn right → ({new_x},{new_y}), danger={danger}")
    return [1.0 if danger else 0.0]


# === Compute: Relative Food Vector (keep float for angle calc) ===
@method_profile(input_types=['head_x', 'food_x'], output_types=['rel_food_x'], group="compute", weight=2.0)
def rel_food_x(head_x, food_x):
    return [food_x - head_x] if None not in (head_x, food_x) else [None]

@method_profile(input_types=['head_y', 'food_y'], output_types=['rel_food_y'], group="compute", weight=2.0)
def rel_food_y(head_y, food_y):
    return [food_y - head_y] if None not in (head_y, food_y) else [None]


# === Decision Logic: Keep as-is (uses danger signals, which are now reliable) ===
@method_profile(
    input_types=['danger_front', 'danger_left', 'danger_right', 'rel_food_x', 'rel_food_y', 'dir_x', 'dir_y'],
    output_types=['turn_left_signal', 'turn_right_signal', 'go_straight_signal'],
    group="logic",
    weight=5.0
)
def survival_decision(danger_front, danger_left, danger_right, fx, fy, dx, dy):
    if None in (danger_front, danger_left, danger_right, fx, fy, dx, dy):
        return [None, None, None]

    # Helper: compute angle difference
    food_angle = np.arctan2(fx, fy)
    current_angle = np.arctan2(dx, dy)
    diff = (food_angle - current_angle + np.pi) % (2 * np.pi) - np.pi  # [-π, π]

    # Case 1: Front is safe → try to move toward food
    if danger_front < 0.5:
        # If already roughly aligned with food (within ~45 degrees), go straight
        if abs(diff) < np.pi / 4:
            return [None, None, 1.0]
        # Otherwise, turn toward food if safe
        elif diff > 0 and danger_right < 0.5:  # food is to the right
            return [None, 1.0, None]
        elif diff < 0 and danger_left < 0.5:   # food is to the left
            return [1.0, None, None]
        else:
            # Can't turn toward food safely? Just go straight (fallback)
            return [None, None, 1.0]

    # Case 2: Front is dangerous → must turn
    if danger_left < 0.5 and danger_right < 0.5:
        # Both sides safe: choose direction toward food
        if diff > 0:
            return [None, 1.0, None]  # turn right
        else:
            return [1.0, None, None]  # turn left
    elif danger_left < 0.5:
        return [1.0, None, None]
    elif danger_right < 0.5:
        return [None, 1.0, None]
    else:
        # All directions blocked → random fallback
        import random
        r = random.random()
        if r < 0.33:
            return [1.0, None, None]
        elif r < 0.66:
            return [None, 1.0, None]
        else:
            return [None, None, 1.0]


# === Action Pool: Unchanged ===
@method_profile(input_types=['turn_left_signal'], output_types=['action_signal'], group="action", weight=10.0)
def turn_left(signal):
    if signal is not None and signal > 0:
        SNAKE_STATE.set_action(1)
    return [1.0 if signal is not None else 0.0]

@method_profile(input_types=['turn_right_signal'], output_types=['action_signal'], group="action", weight=10.0)
def turn_right(signal):
    if signal is not None and signal > 0:
        SNAKE_STATE.set_action(2)
    return [1.0 if signal is not None else 0.0]

@method_profile(input_types=['go_straight_signal'], output_types=['action_signal'], group="action", weight=10.0)
def go_straight(signal):
    if signal is not None and signal > 0:
        SNAKE_STATE.set_action(0)
    return [1.0 if signal is not None else 0.0]


# === Internal Utility Methods ===
@method_profile(input_types=['scalar'], output_types=['scalar'], group="util", weight=0.1)
def identity(x):
    return [x] if x is not None else [None]

identity.is_internal_decorator = True

@method_profile(input_types=['scalar'], output_types=['scalar', 'scalar'], group="util", weight=0.1)
def fanout(x):
    return [x, x] if x is not None else [None, None]

fanout.is_internal_decorator = True