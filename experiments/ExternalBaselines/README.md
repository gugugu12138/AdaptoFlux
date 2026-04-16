ExternalBaselines
=================

说明
----
该目录包含用于生成并比较若干外部基线（symbolic regression / ML baselines）与 ATF 方法实验结果的脚本、模型与处理后数据。文档覆盖目录结构、常用脚本和快速使用示例，方便复现与后续分析。

目录概览
--------
- `adaptoflux_runner.py`：将 ATF 方法纳入统一跑实验流程的运行器（与外部基线对比）。
- `run_baselines_parallel.py` / `run_baselines_with_time.py`：以并行或带时间记录的方式运行外部基线实验（调度多个任务）。
- `baselines/`：放置各个基线实现的子模块（如 `gplearn_runner.py`、`pysr_runner.py`、`xgboost_runner.py`）。
- `best_models/`、`saved_models/`：训练后模型的保存目录（按任务/物理公式分类）。
- `results/`：原始或聚合实验结果 JSON / LaTeX 表格。
- `processed_results/`：进一步处理、用于论文或报告的 LaTeX 表格（例如 `table_atf_stats_*.tex`）。
- `methods/`：基线方法的辅助函数或封装（例如 `methods_feynman.py`）。
- `utils.py`、`utils_viz.py`、`viz2.0.py`：可视化与结果处理工具。
- `tasks/`：定义的任务集（如 `feynman_tasks.py`）。
- 其它脚本：`copy_best_model.py`、`extract_exp5_results.py`、`generate_baseline_tables.py`、`process_full_statistics.py` 等，用于结果整理与表格生成。

快速使用说明
-------------
前提：使用与项目根目录一致的 Python 环境，依赖请参考根目录的 `requirements.txt`。

1. 运行基线实验（示例）

   - 并行运行（会根据脚本内部并行/队列逻辑调度多个任务）：

     python experiments/ExternalBaselines/run_baselines_parallel.py

   - 运行并记录每个任务时间：

     python experiments/ExternalBaselines/run_baselines_with_time.py

   说明：脚本通常会读取任务定义（`tasks/`）并将训练结果保存到 `saved_models/` 与 `results/`。

2. 生成对比表格（LaTeX）

   - 生成用于论文/报告的比较表：

     python experiments/ExternalBaselines/generate_baseline_tables.py

   - 处理并汇总统计信息：

     python experiments/ExternalBaselines/process_full_statistics.py

   运行后，输出会放在 `results/` 与 `processed_results/`，包含 `*.tex` 表格文件与 JSON 汇总。

3. 提取/整理特定实验结果

   - 从特定实验结构中提取结果：

     python experiments/ExternalBaselines/extract_exp5_results.py

4. 可视化与检查

   - 快速绘图/可视化：

     python experiments/ExternalBaselines/viz2.0.py

自定义与开发提示
----------------
- 如果要复现某个 task，请先确认对应的 `tasks/` 中有该任务定义，或在脚本中指定任务名称。
- 模型路径：`saved_models/` 与 `best_models/` 下按任务分类保存多次运行结果。若要加载指定模型进行分析，请根据文件夹名定位。
- 若新增基线方法，可参照 `baselines/` 中现有 runner 模板添加新的 runner，并更新 `generate_baseline_tables.py` 的收集逻辑（如需要）。

常见问题与注意事项
-----------------
- 依赖版本：建议使用与项目根目录 `requirements.txt` 对应的 Python 环境。
- 大规模并行运行：请检查机器并行能力与脚本中的并行设置（避免超载）。
- 数据一致性：运行多个重复试验时，注意将结果输出到不同子目录以免覆盖。

联系方式
-------
如需我帮助补充更详细的用法示例、参数说明或把 README 翻译为英文，请告知。

已知重要文件参考
----------------
- `results/`、`processed_results/`：查看已有的 LaTeX 表格与 JSON 汇总以了解输出格式。
- `baselines/`：查看每个 runner 的实现细节以确定依赖库与参数。

脚本详解
--------
- `run_baselines_parallel.py`：并行控制器。会为每个任务和每种 collapse 模式（`first`、`sum`、`prod`）运行 ATF（AdaptoFlux）若干次（默认 10 次），同时并行提交基线方法（gplearn、XGBoost、PySR）的 best-of-10。输入：`tasks/feynman_tasks` 中的任务；输出：`results/all_results_by_collapse_with_baselines.json`、`results/summaries_by_collapse.json` 和若干 `results/comparison_table_*.tex`、可视化图片（`best_{Task}_collapse_{mode}.png`）。
- `run_baselines_with_time.py`：以“记录运行时间”为目标的基线运行脚本。并行运行 gplearn 与 XGBoost，串行运行 PySR（避免 PySR/Julia 多进程崩溃），最后将基线（best-of-10）结果保存为 `results/baselines_best_of_10_with_time.json`。
- `generate_baseline_tables.py`：从 `results/baselines_best_of_10_with_time.json` 生成 LaTeX 表格：`results/baseline_performance.tex`（基线性能表）和 `results/symbolic_expressions.tex`（基线回归出的符号表达式）。
- `process_full_statistics.py`：处理 `results/all_results_by_collapse_with_baselines.json`（ATF 多次运行 + 基线），计算均值/标准差/中位数/成功率，并生成 `processed_results/table_atf_stats_{mode}.tex`（mode ∈ {first, prod, sum}）以及两张图 `processed_results/atf_mse_distribution_boxplot.png` 与 `processed_results/atf_success_rate_bars.png`。
- `extract_exp5_results.py`：从 `all_results_by_collapse_with_baselines.json` 提取 best-of-10（ATF）与基线的对比表格，并打印与生成 LaTeX（示例脚本主要对 `prod` 模式输出表格）。
- `adaptoflux_runner.py`：负责在单次实验中配置并运行 `AdaptoFlux`（包含不同 collapse 函数），返回评估指标（`mse`、`exact_match`、`method_pool_size`、`save_path` 等）。被 `run_baselines*` 脚本调用。
- `copy_best_model.py`：从 `results/summaries_by_collapse.json` 中读取 prod 模式下的最佳模型 `save_path`，将对应文件夹复制到 `best_models/`（用于后续可视化或分发）。
- `viz2.0.py`、`utils_viz.py`：用于将 AdaptoFlux 保存的图结构（`combined_trainer_temp/final/graph.json`）可视化并导出图片；`viz2.0.py` 提供批处理入口 `batch_visualize_graphs()`。
- `utils.py`：常用小工具，例如数值精确匹配判断 `is_numerical_exact_match()`、任务专用的 `train_test_split_feynman()`。
- `baselines/gplearn_runner.py`：使用 `gplearn` 进行符号回归，返回 `mse`、`exact_match`、表达式字符串。
- `baselines/pysr_runner.py`：使用 `PySR`（Julia 后端）运行符号回归，返回最优方程的 `sympy` 表达式、`mse` 与 `exact_match`。脚本中有对 PySR 的参数压缩与串行化以提高稳定性。
- `baselines/xgboost_runner.py`：XGBoost 回归基线，返回 `mse` 与 `exact_match`（通常为 False）。

`results/` 文件说明
-----------------
- `all_results_by_collapse_with_baselines.json`：主聚合文件。结构为 `{mode: {task: [run1, run2, ...]}}`，每个 run 包含 `adaptoflux` 字段（该次 ATF 运行的字典结果）、以及 `gplearn`、`xgboost`（来自 baseline best-of-10 的记录，通常相同于 runs[0]）与 `ground_truth`。该文件用于统计与绘图。
- `summaries_by_collapse.json`：对 `all_results_by_collapse_with_baselines.json` 取 best-of-10（按 ATF 的 MSE/Exact/Runtime 排序）后生成的简要结果字典，按 collapse mode 和 task 划分，每项指向选定的 `adaptoflux` 运行信息（含 `save_path`，便于复制或可视化）。
- `baselines_best_of_10_with_time.json`：`run_baselines_with_time.py` 的输出，保存 gplearn、pysr、xgboost 的 best-of-10 结果并包含每个方法的运行时间（`runtime_sec`）。
- `baseline_performance.tex`：由 `generate_baseline_tables.py` 生成的 LaTeX 表格，列出基线方法的 Exact、MSE 与运行时间。
- `symbolic_expressions.tex`：由 `generate_baseline_tables.py` 生成，列出 gplearn 与 PySR 回归出的表达式（用于论文展示）。
- `comparison_table_first.tex` / `comparison_table_prod.tex` / `comparison_table_sum.tex`：`run_baselines_parallel.py` 生成的 LaTeX 表（按 collapse 模式），列出 ATF 在各任务上的 best-of-10 结果（Exact、MSE、Runtime、方法池大小等）。
- `baseline_performance.tex`：基线性能表（同上，来自 `generate_baseline_tables.py`）。
- `best_{Task}_collapse_{mode}.png`：`process_full_statistics.py` 或 `run_baselines_parallel.py` 可视化脚本导出的示意图（来自 `combined_trainer_temp/final/graph.json`），每个任务与 mode 一张图，用于展示最终图结构。

`processed_results/` 文件说明
-----------------------
- `table_atf_stats_first.tex`, `table_atf_stats_prod.tex`, `table_atf_stats_sum.tex`：`process_full_statistics.py` 生成的 LaTeX 表格，包含每个任务在对应 collapse 模式下的 MSE（均值±标准差）、中位数、最小值、成功率（Exact%）和运行时间均值±标准差，适合直接插入论文或补充材料。
- `atf_mse_distribution_boxplot.png`：ATF 各任务 MSE 的箱线图（log-scale），用于展示算法在不同任务/模式下的稳定性与分布。
- `atf_success_rate_bars.png`：ATF 在不同任务与 collapse 模式下的 Exact 匹配成功率柱状图。


—— End
