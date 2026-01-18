# experiments/embodied_snake/snake_main.py
# Purpose: Visual demonstration of a handcrafted AdaptoFlux graph in Snake environment
# No training, no evolution — pure zero-shot execution.

import numpy as np
import logging
import sys
import os

# Optional: Try to import pygame for visualization
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("pygame not found. Running headless (no visualization).")

from ATF.core.adaptoflux import AdaptoFlux
from ATF.CollapseManager.collapse_functions import CollapseMethod
from .snake_env import SimpleSnakeEnv  # We'll use the env directly for control

logging.basicConfig(level=logging.INFO)

# === 可视化常量 ===
GRID_SIZE = 10
CELL_SIZE = 40
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 5  # 控制速度

def visualize_episode(model, max_steps=1000):
    """Run one episode with real-time PyGame rendering."""
    if not HAS_PYGAME:
        print("Visualization disabled (pygame not installed).")
        return run_headless(model, max_steps)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("AdaptoFlux Snake Demo")
    clock = pygame.time.Clock()

    env = SimpleSnakeEnv(grid_size=GRID_SIZE, max_steps=max_steps)
    obs = env.reset()
    done = False
    step = 0

    from .snake_state import SNAKE_STATE

    while not done and step < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Get action from AdaptoFlux model
        SNAKE_STATE.reset()
        SNAKE_STATE.set_snake_body(env.snake)
        print("env.snake:", env.snake)
        SNAKE_STATE.set_observation(obs)
        _ = model.infer_with_graph(obs.reshape(1, -1))  # Trigger execution
        action = SNAKE_STATE.decided_action  # Should be 0, 1, or 2

        # Step environment
        obs, reward, done = env.step(action)
        step += 1

        # Render
        screen.fill((0, 0, 0))  # Black background

        # Draw grid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (40, 40, 40), rect, 1)

        # Draw snake
        for i, (y, x) in enumerate(env.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)  # Head brighter
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 100, 0), rect, 1)

        # Draw food
        fy, fx = env.food
        food_rect = pygame.Rect(fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (255, 0, 0), food_rect)

        pygame.display.flip()
        clock.tick(FPS)
        # 在 while 循环内添加
        print(f"Step {step} | Obs: {[round(x, 2) for x in obs]} | Action: {action}")

    print(f"\nDemo finished: survived {step} steps, score = {env.score}")
    pygame.quit()


def run_headless(model, max_steps=1000):
    """Fallback: run without visualization."""
    from .snake_env import run_snake_episode
    survival, score, _ = run_snake_episode(model, action_interval=1, max_steps=max_steps)
    print(f"\nHeadless demo finished: survived {survival} steps, score = {score}")
    return survival, score


def main():
    print("=== AdaptoFlux Snake: Handcrafted Policy Demonstration ===")

    # Initialize AdaptoFlux with dummy input
    dummy_input = np.zeros((1, 6), dtype=np.float32)
    af = AdaptoFlux(
        values=dummy_input,
        labels=None,
        methods_path="experiments/embodied_snake/methods_snake.py",
        input_types_list=['raw_signal'] * 6,
        collapse_method=CollapseMethod.IDENTITY
    )

    # Load pre-constructed graph (should contain Action Pool nodes)
    model_path = "experiments/embodied_snake"

    af.load_model(model_path)
    print("✅ Loaded handcrafted AdaptoFlux graph.")

    # Run visual demo
    visualize_episode(af, max_steps=1000)


if __name__ == "__main__":
    main()