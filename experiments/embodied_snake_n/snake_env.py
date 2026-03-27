# experiments/embodied_snake_n/snake_env.py
import numpy as np
from .snake_state import SNAKE_STATE

class SimpleSnakeEnv:
    def __init__(self, grid_size=10, max_steps=10000):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]
        self.direction = (0, 1)  # (dx, dy): (1,0)=Right, (0,1)=Down, (-1,0)=Left, (0,-1)=Up
        self.food = self._place_food()
        self.steps = 0
        self.score = 0
        self.done = False
        # 记录上一帧距离，用于计算进步奖励
        self.prev_dist = self._get_manhattan_dist()
        return self._get_observation()

    def _place_food(self):
        while True:
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if pos not in self.snake:
                return pos

    def _get_manhattan_dist(self):
        head = self.snake[0]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

    def _is_collision(self, point):
        x, y = point
        # 撞墙
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return True
        # 撞自己
        if point in self.snake:
            return True
        return False

    def _get_observation(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dx, dy = self.direction

        # 1. 相对食物位置 (归一化)
        rel_fx = (food_x - head_x) / self.grid_size
        rel_fy = (food_y - head_y) / self.grid_size

        # 2. 当前方向 (One-hot 或 分量)
        dir_x = float(dx)
        dir_y = float(dy)

        # 3. 危险传感器 (左，前，右)
        # 定义方向向量
        # 前
        next_f = (head_x + dx, head_y + dy)
        # 右 (dx, dy) -> (dy, -dx)
        next_r = (head_x + dy, head_y - dx)
        # 左 (dx, dy) -> (-dy, dx)
        next_l = (head_x - dy, head_y + dx)

        danger_f = 1.0 if self._is_collision(next_f) else 0.0
        danger_r = 1.0 if self._is_collision(next_r) else 0.0
        danger_l = 1.0 if self._is_collision(next_l) else 0.0

        # 8 维输入：[RelFX, RelFY, DirX, DirY, DangerL, DangerF, DangerR, Bias]
        return np.array([
            rel_fx, rel_fy, 
            dir_x, dir_y, 
            danger_l, danger_f, danger_r,
            1.0 
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True

        # 修正转向逻辑
        # 0: Straight, 1: Left, 2: Right
        dx, dy = self.direction
        if action == 1:  # turn left: (dy, dx) → (-dx, dy)
            self.direction = (dx, -dy)
        elif action == 2:  # turn right: (dy, dx) → (dx, -dy)
            self.direction = (-dx, dy)
            
        dx, dy = self.direction
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        reward = 0.0
        done = False

        if self._is_collision(new_head):
            done = True
            reward = -1.0  # 死亡惩罚
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.score += 1
                reward = 2.0  # 吃食物奖励
                self.food = self._place_food()
                self.prev_dist = self._get_manhattan_dist()
            else:
                self.snake.pop()
                # 鼓励靠近食物
                curr_dist = self._get_manhattan_dist()
                if curr_dist < self.prev_dist:
                    reward = 0.1
                else:
                    reward = -0.05 # 轻微惩罚远离
                self.prev_dist = curr_dist

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
            
        return self._get_observation(), reward, done

def run_snake_episode(model, action_interval=1, max_steps=10000):
    env = SimpleSnakeEnv(grid_size=10, max_steps=max_steps)
    obs = env.reset()
    survival_time = 0
    total_reward = 0.0
    decision_steps = 0
    
    # 用于计算平均适应度
    foods_eaten = 0

    try:
        for step in range(max_steps):
            if step % action_interval == 0:
                SNAKE_STATE.reset()
                SNAKE_STATE.set_snake_body(env.snake)
                SNAKE_STATE.set_observation(obs)
                # 确保输入形状匹配 (1, 8)
                _ = model.infer_with_graph(obs.reshape(1, -1))
                action = SNAKE_STATE.decided_action
                decision_steps += 1
            else:
                action = 0

            obs, reward, done = env.step(action)
            survival_time += 1
            total_reward += reward
            if reward > 1.0: # 简单判断是否吃到食物
                foods_eaten += 1

            if done:
                break
    finally:
        pass

    # 返回更丰富的指标
    return survival_time, env.score, total_reward