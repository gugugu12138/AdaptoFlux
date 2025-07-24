import AdaptoFlux
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import os
import multiprocessing
import time # 用于添加一点随机延迟，减少同时创建大量进程时的资源竞争

# --- 定义单个完整训练实验的函数 ---
def run_single_training_experiment(train_data, train_labels, methods_path, epochs_per_run, run_id, process_name):
    """
    运行单次完整的模型训练实验。
    这个函数将在一个单独的进程中运行。
    """
    try:
        # 可选：添加一点随机延迟，避免所有进程同时启动时资源竞争过于激烈
        # import time
        # import random
        # time.sleep(random.uniform(0, 2))

        print(f"[{process_name} - 实验 {run_id}] 开始初始化模型...")
        # 1. 创建模型实例 (根据之前的修正)
        model = AdaptoFlux.AdaptoFlux(
            values=train_data,
            labels=train_labels,
            methods_path=methods_path
        )

        # 2. 导入方法池
        print(f"[{process_name} - 实验 {run_id}] 正在导入方法池: {methods_path}")
        model.import_methods_from_file()
        print(f"[{process_name} - 实验 {run_id}] 成功导入方法池。")

        # 3. 开始训练 (每个实验训练 epochs_per_run 轮)
        print(f"[{process_name} - 实验 {run_id}] 开始训练 {epochs_per_run} 轮...")
        model.training(epochs=epochs_per_run, target_accuracy=None) # 训练指定轮次
        # 注意：根据之前的修改，training 方法结束时会调用 save_model_and_log，
        # 它会将模型和日志保存到一个自动命名的文件夹中（如 models-1, models-2...）
        print(f"[{process_name} - 实验 {run_id}] 训练完成。最终准确率: {model.metrics.get('accuracy', 'N/A')}, 路径长度: {len(model.paths)}")

    except Exception as e:
        print(f"[{process_name} - 实验 {run_id}] 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
# ----------------------------------

# --- 辅助函数：管理并行进程批次 ---
def run_experiments_in_batches(target_func, args_list, max_processes=4):
    """
    分批运行进程，以避免同时启动过多进程。
    :param target_func: 要运行的目标函数
    :param args_list: 包含传递给 target_func 的参数元组的列表
    :param max_processes: 同时运行的最大进程数
    """
    processes = []
    completed = 0
    total = len(args_list)

    # 创建所有进程
    for args in args_list:
        p = multiprocessing.Process(target=target_func, args=args)
        processes.append(p)

    # 分批启动和管理进程
    for i, p in enumerate(processes):
        p.start()
        print(f"已启动进程 {i+1}/{total}")

        # 如果达到最大并行数，或者这是最后一个进程，则等待一批完成
        if (len([proc for proc in processes[:i+1] if proc.is_alive()]) >= max_processes) or (i == len(processes) - 1):
            # 等待当前所有已启动的进程完成
            for j in range(i+1):
                if processes[j].is_alive():
                    processes[j].join()
                    completed += 1
                    print(f"进程 {j+1} 已完成 ({completed}/{total})")

    # 确保所有进程都已完成（作为后备）
    for p in processes:
        if p.is_alive():
            p.join()

# ----------------------------------

# 示例使用
if __name__ == "__main__":
    print("主进程 PID:", os.getpid())

    # 1. 加载 MNIST 数据集
    print("正在加载 MNIST 数据集...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. 数据预处理
    print("正在进行数据预处理...")
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 3. 切割 10% 的数据作为验证集 (保留但 training 方法未使用)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=0.1,
        random_state=42
    )
    print("训练数据展平后的形状:", x_train.reshape(x_train.shape[0], -1).shape)

    # --- 确保展平后的数据 ---
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)
    # -----------------------

    # --- 定义两个不同的方法池文件路径 ---
    base_dir = "AdaptoFlux" # 假设脚本在项目根目录运行
    file_path_1 = os.path.join(base_dir, "models", "methods-1.py")
    file_path_2 = os.path.join(base_dir, "models", "methods.py")
    # ----------------------------------

    # --- 检查文件是否存在 ---
    if not os.path.exists(file_path_1):
        print(f"错误: 文件 {file_path_1} 不存在!")
        exit(1)
    if not os.path.exists(file_path_2):
        print(f"错误: 文件 {file_path_2} 不存在!")
        exit(1)
    # -----------------------

    # --- 配置 ---
    NUM_EXPERIMENTS = 50  # 每个方法池运行 50 次独立实验
    EPOCHS_PER_RUN = 5   # 每次实验训练 20 轮 (根据你的 training 方法默认值)
    MAX_CONCURRENT_PROCESSES = 2 # 同时运行的最大进程数，根据你的 CPU 核心数调整
    # ------------

    print(f"\n--- 开始为每个方法池运行 {NUM_EXPERIMENTS} 次独立训练实验 ---")
    print(f"每次实验训练 {EPOCHS_PER_RUN} 轮。")
    print(f"最大并发进程数: {MAX_CONCURRENT_PROCESSES}")

    # --- 准备所有实验的参数 ---
    all_experiment_args = []
    for run_id in range(1, NUM_EXPERIMENTS + 1):
        # 为 methods-1.py 准备参数
        all_experiment_args.append(
            (x_train_flat, y_train, file_path_1, EPOCHS_PER_RUN, run_id, "方法池1")
        )
        # 为 methods.py 准备参数
        all_experiment_args.append(
            (x_train_flat, y_train, file_path_2, EPOCHS_PER_RUN, run_id, "方法池2")
        )
    # --------------------------

    # --- 分批运行所有实验 ---
    print(f"总共将运行 {len(all_experiment_args)} 个实验进程...")
    run_experiments_in_batches(
        target_func=run_single_training_experiment,
        args_list=all_experiment_args,
        max_processes=MAX_CONCURRENT_PROCESSES
    )
    # --------------------------

    print("\n--- 所有训练实验全部完成 ---")
    print("请检查 models 目录及其自动生成的带序号的子目录，")
    print("里面包含了每次实验的 output.txt, methods.py 和 training_log.csv。")
