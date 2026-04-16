# experiments/embodied_snake/snake_main.py
# Purpose: Visual demonstration of a handcrafted AdaptoFlux graph in Snake environment
# No training, no evolution — pure zero-shot execution.
# 需要作为模块运行，请使用类似命令行:
# d:/ATF/.conda/python.exe -m experiments.embodied_snake.snake_main --record --output snake_demo.mp4

import numpy as np
import logging
import sys
import os
import argparse

# Optional: Try to import pygame for visualization
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("pygame not found. Running headless (no visualization).")

# Optional: Try to import cv2 for video recording
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("cv2 not found. Video recording disabled. Install with: pip install opencv-python")

from ATF.core.adaptoflux import AdaptoFlux
from ATF.CollapseManager.collapse_functions import CollapseMethod
from .snake_env import SimpleSnakeEnv  # We'll use the env directly for control

logging.basicConfig(level=logging.INFO)

# === 可视化常量 ===
GRID_SIZE = 10
CELL_SIZE = 40
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 5  # 控制速度


def visualize_episode(model, max_steps=1000, record=False, output_path="snake_demo.mp4", record_fps=None):
    """Run one episode with real-time PyGame rendering and optional video recording."""
    if not HAS_PYGAME:
        print("Visualization disabled (pygame not installed).")
        return run_headless(model, max_steps)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("AdaptoFlux Snake Demo")
    clock = pygame.time.Clock()

    # === 视频录制初始化 ===
    video_writer = None
    if record:
        if not HAS_CV2:
            print("⚠️  Warning: cv2 not available, recording disabled.")
            record = False
        else:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            # 使用与显示相同的帧率，或自定义
            record_fps = record_fps if record_fps else FPS
            # OpenCV 默认使用 BGR 格式
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'avc1' for H.264
            video_writer = cv2.VideoWriter(output_path, fourcc, record_fps, (WINDOW_SIZE, WINDOW_SIZE))
            print(f"🎬 Recording started: {output_path} @ {record_fps}fps")

    env = SimpleSnakeEnv(grid_size=GRID_SIZE, max_steps=max_steps)
    obs = env.reset()
    done = False
    step = 0

    from .snake_state import SNAKE_STATE

    while not done and step < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if video_writer:
                    video_writer.release()
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

        # Draw step counter overlay (optional, helpful for video)
        font = pygame.font.Font(None, 24)
        step_text = font.render(f"Step: {step} | Score: {env.score}", True, (255, 255, 255))
        screen.blit(step_text, (5, 5))

        pygame.display.flip()
        
        # === 录制当前帧 ===
        if record and video_writer:
            # 将 pygame surface 转换为 numpy array
            frame = pygame.surfarray.array3d(screen)
            # surfarray 返回的是 (W, H, 3) RGB，需要转置为 (H, W, 3) 并转换为 BGR
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        
        clock.tick(FPS)
        # 在 while 循环内添加
        print(f"Step {step} | Obs: {[round(x, 2) for x in obs]} | Action: {action}")

    print(f"\nDemo finished: survived {step} steps, score = {env.score}")
    
    # === 释放视频资源 ===
    if video_writer:
        video_writer.release()
        print(f"✅ Video saved to: {os.path.abspath(output_path)}")
    
    pygame.quit()


def run_headless(model, max_steps=1000):
    """Fallback: run without visualization."""
    from .snake_env import run_snake_episode
    survival, score, _ = run_snake_episode(model, action_interval=1, max_steps=max_steps)
    print(f"\nHeadless demo finished: survived {survival} steps, score = {score}")
    return survival, score


def main():
    # === 命令行参数解析 ===
    parser = argparse.ArgumentParser(description="AdaptoFlux Snake Demo with Recording")
    parser.add_argument("--record", action="store_true", help="Enable video recording")
    parser.add_argument("--output", type=str, default="snake_demo.mp4", 
                        help="Output video file path (default: snake_demo.mp4)")
    parser.add_argument("--record-fps", type=int, default=None,
                        help="Recording frame rate (default: same as display FPS)")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum steps per episode (default: 1000)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization (faster, no recording)")
    args = parser.parse_args()

    print("=== AdaptoFlux Snake: Handcrafted Policy Demonstration ===")
    if args.record:
        print(f"🎬 Recording enabled → {args.output}")
    if args.headless:
        print("🔇 Running in headless mode (no visualization)")

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
    model_path = "experiments/embodied_snake/snake.json"
    af.load_model(model_path)
    print("✅ Loaded handcrafted AdaptoFlux graph.")

    # Run demo
    if args.headless:
        run_headless(af, max_steps=args.max_steps)
    else:
        visualize_episode(
            af, 
            max_steps=args.max_steps,
            record=args.record,
            output_path=args.output,
            record_fps=args.record_fps
        )


if __name__ == "__main__":
    main()