import time
import statistics
import os
import psutil
import pandas as pd
import numpy as np
from ATF.core.adaptoflux import AdaptoFlux
import logging
from tqdm import tqdm
import threading  # ✅ 用于记录活跃线程数
import platform   # ✅ 用于记录 Python 版本

# ========================
# 配置区（请根据你的路径修改）
# ========================
REPEAT_ROUNDS = 3           # 总共跑3轮，丢弃第1轮（预热）
INFERENCE_PER_ROUND = 100 # 每轮推理次数（注意：你循环中是100次，这里应保持一致或修正）
WARMUP_ROUNDS = 1           # 前1轮为预热，不计入统计
TEST_DATA_PATH = 'experiments/PipelineParallel_exp2/data/test_processed.csv'  # 测试数据路径
MODEL_BASE_DIR = 'experiments/PipelineParallel_exp2/models'

# 自动生成 model_1 到 model_30（共30个）
selected_models = [
    os.path.join(MODEL_BASE_DIR, f"model_{i}") 
    for i in range(1, 31)  # 1 to 30
]

# 输出 CSV 路径
OUTPUT_CSV = 'experiments/PipelineParallel_exp2/results_GIL/benchmark_results.csv'

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ========================
# 核心类与函数
# ========================

class PipelineExecutor:
    """封装 AdaptoFlux 模型的流水线推理执行器"""
    def __init__(self, model_path: str, num_cores: int):
        self.model_path = model_path
        self.num_cores = num_cores
        self.adaptoflux = None
        self._load_model()

    def _load_model(self):
        """加载 AdaptoFlux 模型"""
        try:
            # 创建空 AdaptoFlux 实例（values/labels 会在推理时传入）
            self.adaptoflux = AdaptoFlux(
                values=np.zeros((1, 1)),  # 占位，实际推理时传入
                labels=np.zeros(1),       # 占位
                methods_path='experiments/PipelineParallel_exp2/scripts/methods_GIL.py'
            )
            # 加载保存的图结构
            self.adaptoflux.load_model(folder=self.model_path)
            logger.info(f"✅ 模型加载成功: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ 模型加载失败 {self.model_path}: {e}")
            raise e

    def forward(self, values: np.ndarray):
        """执行一次流水线推理（不关心输出，只测性能）"""
        try:
            _ = self.adaptoflux.infer_with_task_parallel(values, num_workers=self.num_cores)
            return True  # 成功
        except Exception as e:
            logger.error(f"❌ 推理失败: {e}")
            return False  # 失败

    def get_graph_node_count(self):
        """获取计算图中实际参与执行的节点数量（排除 root 和 collapse）"""
        if not self.adaptoflux or not hasattr(self.adaptoflux, 'graph'):
            return 0
        return len([
            n for n in self.adaptoflux.graph.nodes 
            if n not in ["root", "collapse"]
        ])

def load_input_data():
    """加载测试数据，取前100行用于推理"""
    try:
        df = pd.read_csv(TEST_DATA_PATH)
        if 'Survived' in df.columns:
            values = df.drop(columns=['Survived']).values
        else:
            values = df.values
        # 取前100行
        values = values[:100].astype(np.float64)
        logger.info(f"✅ 输入数据加载成功: {values.shape}")
        return values
    except Exception as e:
        logger.error(f"❌ 输入数据加载失败: {e}")
        raise e

def log_result(model_path: str, num_cores: int, avg_latency_ms: float,
               throughput: float, std_latency_ms: float,
               graph_node_count: int, thread_count_before: int, thread_count_after: int):
    """将结果追加写入 CSV 文件（若文件不存在则自动创建并写入表头）"""
    import csv
    import os

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    file_exists = os.path.isfile(OUTPUT_CSV)

    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # 写入表头（仅当文件不存在时）
            writer.writerow([
                'model_path', 'num_cores', 'avg_latency_ms',
                'throughput_samples_per_sec', 'std_latency_ms',
                'timestamp', 'cpu_util_percent', 'memory_mb', 'python_version',
                'graph_node_count', 'thread_count_before', 'thread_count_after'
            ])
            logger.info("🆕 创建新结果文件并写入表头")

        # 获取额外信息
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cpu_util = psutil.cpu_percent(interval=None)  # 短暂间隔获取更准确的 CPU 利用率
        memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        py_version = platform.python_version()

        # 写入数据行
        writer.writerow([
            model_path, num_cores, f"{avg_latency_ms:.6f}",
            f"{throughput:.6f}", f"{std_latency_ms:.6f}",
            timestamp, f"{cpu_util:.1f}", f"{memory_mb:.1f}", py_version,
            graph_node_count, thread_count_before, thread_count_after
        ])
    logger.info(f"📊 结果已追加: {model_path} @ {num_cores}核")

# ========================
# 主执行逻辑（带进度条 + ETA + 跳过已测试配置）
# ========================

def main():
    """主函数：遍历所有模型和核心配置，执行性能测试"""
    # 加载输入数据（前100行）
    fixed_input = load_input_data()

    # 构建所有配置组合
    all_configs = [(model_path, num_cores) for model_path in selected_models for num_cores in [1, 2, 4, 8, 16]]

    # ✅ 跳过已测试的配置
    existing_configs = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            for _, row in df_existing.iterrows():
                # 确保字段存在
                if 'model_path' in row and 'num_cores' in row:
                    existing_configs.add((row['model_path'], row['num_cores']))
            logger.info(f"📋 已检测到 {len(existing_configs)} 个已完成的配置，将跳过")
        except Exception as e:
            logger.warning(f"⚠️ 读取已有结果失败，不跳过任何配置: {e}")

    # 过滤掉已测试的配置
    filtered_configs = [
        (mp, nc) for mp, nc in all_configs 
        if (mp, nc) not in existing_configs
    ]
    total_configs = len(filtered_configs)

    if total_configs == 0:
        logger.info("🎉 所有配置均已测试完成，无需重复运行。")
        return

    logger.info(f"🎯 总共要测试 {total_configs} 个配置（{len(selected_models)} 个模型 × 5 核心），跳过 {len(all_configs) - total_configs} 个")

    # 主进度条
    start_time_total = time.time()
    with tqdm(total=total_configs, desc="Overall Progress", unit="config",
              ncols=120, colour="green", dynamic_ncols=True) as pbar:

        for idx, (model_path, num_cores) in enumerate(filtered_configs):
            logger.info(f"\n🚀 开始测试 [{idx+1}/{total_configs}]: {model_path} @ {num_cores} cores")

            try:
                # 初始化执行器
                executor = PipelineExecutor(model_path, num_cores)
                graph_node_count = executor.get_graph_node_count()

                latencies = []
                throughputs = []

                for round_idx in range(REPEAT_ROUNDS):
                    logger.info(f"⏱️  第 {round_idx + 1} 轮推理开始...")

                    start_time_round = time.perf_counter()

                    success_count = 0
                    thread_count_before = threading.active_count()  # ✅ 记录推理前活跃线程数

                    # ✅ 子进度条：每轮 INFERENCE_PER_ROUND 次推理
                    with tqdm(total=INFERENCE_PER_ROUND, desc=f"Round {round_idx+1} Inference",
                            unit="iter", ncols=80, leave=False, colour="blue") as sub_pbar:
                        for i in range(INFERENCE_PER_ROUND):
                            if executor.forward(fixed_input):
                                success_count += 1
                            sub_pbar.update(1)

                    thread_count_after = threading.active_count()
                    end_time_round = time.perf_counter()
                    total_time_sec = end_time_round - start_time_round

                    # ✅ 计算本轮指标（基于 INFERENCE_PER_ROUND 次推理）
                    avg_latency_ms = (total_time_sec * 1000) / INFERENCE_PER_ROUND
                    throughput = INFERENCE_PER_ROUND / total_time_sec

                    if round_idx >= WARMUP_ROUNDS:
                        latencies.append(avg_latency_ms)
                        throughputs.append(throughput)

                    logger.info(
                        f"✅ 第 {round_idx + 1} 轮完成: "
                        f"Latency={avg_latency_ms:.2f}ms, "
                        f"Throughput={throughput:.1f} samples/sec, "
                        f"Success={success_count}/{INFERENCE_PER_ROUND}, "
                        f"Threads: {thread_count_before} → {thread_count_after}"
                    )

                # 计算最终结果（仅使用后两轮）
                if len(latencies) > 0:
                    final_avg_latency = statistics.mean(latencies)
                    final_avg_throughput = statistics.mean(throughputs)
                    final_std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

                    log_result(
                        model_path,
                        num_cores,
                        final_avg_latency,
                        final_avg_throughput,
                        final_std_latency,
                        graph_node_count,
                        thread_count_before,
                        thread_count_after
                    )

                    logger.info(
                        f"📈 最终结果: "
                        f"Avg Latency={final_avg_latency:.2f}ms ± {final_std_latency:.2f}, "
                        f"Throughput={final_avg_throughput:.1f} samples/sec, "
                        f"Graph Nodes={graph_node_count}, "
                        f"Threads {thread_count_before} → {thread_count_after}"
                    )
                else:
                    logger.error("❌ 无有效轮次数据")

            except Exception as e:
                logger.error(f"❌ 模型测试失败 {model_path} @ {num_cores} cores: {e}")

            finally:
                # 更新主进度条 + ETA
                pbar.update(1)
                elapsed_total = time.time() - start_time_total
                completed = pbar.n
                total = pbar.total
                if completed > 0 and total > 0:
                    avg_time_per_config = elapsed_total / completed
                    remaining_configs = total - completed
                    eta_seconds = int(avg_time_per_config * remaining_configs)
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    pbar.set_postfix({"ETA": eta_str}, refresh=True)

    logger.info(f"\n🎉 所有测试完成！结果已追加至: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()