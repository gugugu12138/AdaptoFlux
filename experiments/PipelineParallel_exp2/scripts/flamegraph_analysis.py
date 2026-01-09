import time
import statistics
import os
import psutil
import pandas as pd
import numpy as np
import threading
import platform
import subprocess
import logging
from tqdm import tqdm
from ATF.core.adaptoflux import AdaptoFlux
import csv
import functools

# ========================
# 配置区（修改这里即可）
# ========================

# 🔍 选择你要分析的3个模型（根据你的编号）
TARGET_MODELS = [
    "model_8",
    "model_14",
    "model_21"
]

MODEL_BASE_DIR = 'experiments/PipelineParallel_exp2/models'
OUTPUT_FLAMEGRAPH_DIR = 'experiments/PipelineParallel_exp2/results/flamegraph'
OUTPUT_CSV = 'experiments/PipelineParallel_exp2/results/flamegraph_results.csv'

# 推理参数
INFERENCE_PER_RUN = 10      # 每次推理次数（必须足够长让 py-spy 采样）
WARMUP_ITERATIONS = 5        # 每次运行前预热5次（忽略）
REPEAT_CONFIGS = 1           # 每个配置只跑一次（为了节省时间，火焰图只需一次高质量采样）

# 是否启用 sleep 模拟高延迟？
USE_SLEEP = [False, True]    # False: 无sleep (真实负载) | True: 有sleep (模拟延迟)

# py-spy 参数
PYSPY_RATE = 1000            # 采样频率（Hz）
PYSPY_DURATION = 30          # 采集持续时间（秒）
PYSPY_CMD = "py-spy"         # 确保已安装：pip install py-spy

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ========================
# 工具函数
# ========================

def get_model_path(model_name):
    return os.path.join(MODEL_BASE_DIR, model_name)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class FlameGraphExecutor:
    def __init__(self, model_path: str, num_cores: int = 8):
        self.model_path = model_path
        self.num_cores = num_cores
        self.adaptoflux = None
        self._load_model()

    def _load_model(self):
        try:
            self.adaptoflux = AdaptoFlux(
                values=np.zeros((1, 1)),
                labels=np.zeros(1),
                methods_path='experiments/PipelineParallel_exp2/scripts/methods_GIL.py'
            )
            self.adaptoflux.load_model(folder=self.model_path)
            logger.info(f"✅ 模型加载成功: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ 模型加载失败 {self.model_path}: {e}")
            raise e

    def forward(self, values: np.ndarray, use_sleep: bool = False):
        """执行一次推理，可选在方法中插入 sleep"""
        try:
            original_functions = {}  # 保存原始的 function 对象

            if use_sleep:
                logger.info("⚡ 正在为所有方法注入 time.sleep(0.001) 模拟高延迟...")

                for method_name, method_info in list(self.adaptoflux.methods.items()):
                    if not isinstance(method_info, dict) or "function" not in method_info:
                        continue
                    original_func = method_info["function"]
                    if not callable(original_func):
                        continue

                    original_functions[method_name] = original_func

                    def make_wrapped_func(orig_func):
                        def wrapped(*args, **kwargs):
                            result = orig_func(*args, **kwargs)
                            time.sleep(0.001)
                            return result
                        return functools.wraps(orig_func)(wrapped)

                    self.adaptoflux.methods[method_name]["function"] = make_wrapped_func(original_func)

                logger.info("✅ 所有方法已成功注入 sleep 延迟")
            
            else:
                logger.info("⚡ 使用原始方法，无 sleep 延迟")

                for method_name, method_info in list(self.adaptoflux.methods.items()):
                    if not isinstance(method_info, dict) or "function" not in method_info:
                        continue
                    original_func = method_info["function"]
                    if not callable(original_func):
                        continue

                    original_functions[method_name] = original_func

                    def make_wrapped_func(orig_func):
                        def wrapped(*args, **kwargs):
                            result = orig_func(*args, **kwargs)
                            return result
                        return functools.wraps(orig_func)(wrapped)

                    self.adaptoflux.methods[method_name]["function"] = make_wrapped_func(original_func)

            _ = self.adaptoflux.infer_with_task_parallel(values, num_workers=self.num_cores)

            logger.info("🔄 正在恢复原始方法...")
            for method_name, orig_func in original_functions.items():
                if method_name in self.adaptoflux.methods:
                    self.adaptoflux.methods[method_name]["function"] = orig_func
            logger.info("✅ 原始方法已恢复")

        except Exception as e:
            logger.error(f"❌ 推理过程中发生异常: {e}", exc_info=True)
            raise

    def get_graph_node_count(self):
        if not self.adaptoflux or not hasattr(self.adaptoflux, 'graph'):
            return 0
        return len([
            n for n in self.adaptoflux.graph.nodes 
            if n not in ["root", "collapse"]
        ])

def collect_performance_data(executor: FlameGraphExecutor, values: np.ndarray, use_sleep: bool, run_id: str):
    """收集性能数据，不包含火焰图采集"""
    thread_count_before = threading.active_count()

    start_time = time.perf_counter()
    success_count = 0
    for i in range(INFERENCE_PER_RUN):
        if executor.forward(values, use_sleep=use_sleep):
            success_count += 1
    end_time = time.perf_counter()

    thread_count_after = threading.active_count()

    total_time_sec = end_time - start_time
    avg_latency_ms = (total_time_sec * 1000) / INFERENCE_PER_RUN
    throughput = INFERENCE_PER_RUN / total_time_sec

    return {
        'avg_latency_ms': avg_latency_ms,
        'throughput_samples_per_sec': throughput,
        'success_count': success_count,
        'thread_count_before': thread_count_before,
        'thread_count_after': thread_count_after,
        'total_time_sec': total_time_sec
    }

def record_flamegraph(model_name: str, use_sleep: bool, executor: FlameGraphExecutor, values: np.ndarray):
    """启动 py-spy 采集火焰图，并返回输出路径、平均CPU、平均内存"""
    pid = os.getpid()
    output_file = os.path.join(OUTPUT_FLAMEGRAPH_DIR, f"{model_name}_{'sleep' if use_sleep else 'nosleep'}.svg")

    logger.info(f"🔥 正在为 {model_name} {'有sleep' if use_sleep else '无sleep'} 采集火焰图... PID={pid}")

    cmd = [
        PYSPY_CMD,
        "record",
        "--output", output_file,
        "--pid", str(pid),
        "--rate", str(PYSPY_RATE),
        "--duration", str(PYSPY_DURATION)
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 异步监控线程：采集推理期间的 CPU 和内存
    cpu_samples = []
    memory_samples = []
    stop_monitoring = threading.Event()

    def monitor():
        proc_obj = psutil.Process(pid)
        try:
            proc_obj.cpu_percent()  # 初始化基准（重要！）
        except psutil.NoSuchProcess:
            return
        while not stop_monitoring.is_set():
            try:
                cpu_pct = proc_obj.cpu_percent()  # 非阻塞，返回自上次采样以来的值
                mem_mb = proc_obj.memory_info().rss / (1024 * 1024)
                cpu_samples.append(cpu_pct)
                memory_samples.append(mem_mb)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    # 开始推理（同时监控）
    logger.info("▶️  开始执行推理以触发采样...")
    start_time = time.time()

    for _ in range(WARMUP_ITERATIONS):
        executor.forward(values, use_sleep=use_sleep)

    for i in range(INFERENCE_PER_RUN):
        executor.forward(values, use_sleep=use_sleep)
        if i % 20 == 0:
            elapsed = time.time() - start_time
            if elapsed > PYSPY_DURATION * 0.8:
                logger.info(f"⏳ 已运行 {elapsed:.1f}s，接近采样结束时间...")

    # 等待 py-spy 完成
    try:
        stdout, stderr = proc.communicate(timeout=60)
        if proc.returncode != 0:
            logger.error(f"❌ py-spy 失败: {stderr.decode()}")
            return None, None, None
        else:
            logger.info(f"✅ 火焰图已保存至: {output_file}")
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.warning("⚠️ py-spy 超时，可能未采集完整数据，请手动重试。")
        return None, None, None

    # 停止监控
    stop_monitoring.set()
    monitor_thread.join(timeout=2)

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    avg_mem = sum(memory_samples) / len(memory_samples) if memory_samples else 0.0

    return output_file, avg_cpu, avg_mem

def main():
    # 准备输入数据
    try:
        df = pd.read_csv('experiments/PipelineParallel_exp2/data/test_processed.csv')
        if 'Survived' in df.columns:
            values = df.drop(columns=['Survived']).values
        else:
            values = df.values
        values = values[:100].astype(np.float64)
        logger.info(f"✅ 输入数据加载成功: {values.shape}")
    except Exception as e:
        logger.error(f"❌ 输入数据加载失败: {e}")
        return

    # 创建输出目录
    ensure_dir(OUTPUT_FLAMEGRAPH_DIR)

    # 创建结果CSV文件（带表头）
    csv_header = [
        'model_path', 'num_cores', 'avg_latency_ms', 'throughput_samples_per_sec',
        'std_latency_ms', 'timestamp', 'cpu_util_percent', 'memory_mb', 'python_version',
        'graph_node_count', 'thread_count_before', 'thread_count_after',
        'use_sleep', 'flamegraph_file'
    ]
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(csv_header)

    # 遍历每个模型和配置
    for model_name in TARGET_MODELS:
        model_path = get_model_path(model_name)
        logger.info(f"\n{'='*60}\n🎯 正在处理模型: {model_name}\n{'='*60}")

        for use_sleep in USE_SLEEP:
            config_str = "sleep" if use_sleep else "nosleep"
            logger.info(f"⚙️  配置: {config_str}")

            try:
                # 初始化执行器
                executor = FlameGraphExecutor(model_path, num_cores=8)
                graph_node_count = executor.get_graph_node_count()

                # 先收集性能数据（快速）
                perf_data = collect_performance_data(executor, values, use_sleep, run_id=f"{model_name}_{config_str}")

                # 采集火焰图并获取期间的平均 CPU 和内存
                flamegraph_path, avg_cpu, avg_mem = record_flamegraph(model_name, use_sleep, executor, values)

                if not flamegraph_path:
                    logger.warning(f"⚠️ 火焰图采集失败，跳过记录")
                    continue

                # 记录结果到CSV —— 安全处理 None/空值
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                py_version = platform.python_version()

                # 安全格式化：若为 None 或无效，写空字符串
                def safe_float(x, default=""):
                    return f"{x:.1f}" if x is not None and isinstance(x, (int, float)) else default

                row = [
                    model_path, 8,
                    f"{perf_data['avg_latency_ms']:.6f}",
                    f"{perf_data['throughput_samples_per_sec']:.6f}",
                    "0.0",  # std_latency_ms: 只跑一次，设为0
                    timestamp,
                    safe_float(avg_cpu),      # ✅ 安全处理
                    safe_float(avg_mem),      # ✅ 安全处理
                    py_version,
                    graph_node_count,
                    perf_data['thread_count_before'],
                    perf_data['thread_count_after'],
                    use_sleep,
                    flamegraph_path
                ]

                with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                logger.info(f"📊 结果已记录: {model_name} {config_str} | CPU: {avg_cpu:.1f}% | MEM: {avg_mem:.1f}MB")

            except Exception as e:
                logger.error(f"❌ 处理 {model_name} {use_sleep} 时出错: {e}")

    logger.info(f"\n🎉 所有目标模型分析完成！")
    logger.info(f"📁 火焰图保存于: {OUTPUT_FLAMEGRAPH_DIR}")
    logger.info(f"📋 数据保存于: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()