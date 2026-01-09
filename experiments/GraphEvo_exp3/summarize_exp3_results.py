# summarize_exp3_results.py
import os
import json
import pandas as pd

def main():
    base_dir = "experiments/GraphEvo_exp3/exp3_results"
    output_csv = os.path.join(base_dir, "summary_table.csv")
    output_json = os.path.join(base_dir, "summary_table.json")

    summary_data = []

    # 遍历每个实验组目录
    for group_name in os.listdir(base_dir):
        group_path = os.path.join(base_dir, group_name)
        if not os.path.isdir(group_path):
            continue

        agg_file = os.path.join(group_path, "aggregated.json")
        if not os.path.exists(agg_file):
            print(f"⚠️  Missing aggregated.json in {group_name}")
            continue

        with open(agg_file, 'r', encoding='utf-8') as f:
            agg = json.load(f)

        # 提取配置信息（从 group_name 推断）
        # 格式如: Exp_L2_Minimal, Ctrl_L3_Extended, Oracle_L2
        parts = group_name.split('_')
        exp_type = parts[0]  # "Exp", "Ctrl", or "Oracle"
        layer_info = parts[1]  # e.g., "L2", "L3"
        pool_variant = '_'.join(parts[2:]) if len(parts) > 2 else "unknown"

        summary_data.append({
            "group_name": group_name,
            "exp_type": exp_type,
            "max_layers": int(layer_info[1:]) if layer_info.startswith('L') else None,
            "method_pool": pool_variant,
            "avg_task3_accuracy": agg["avg_task3_accuracy"],
            "std_task3_accuracy": agg["std_task3_accuracy"],
            "success_rate": agg["success_rate"],
            "n_repeats": agg["n_repeats"]
        })

    # 转为 DataFrame 并排序
    df = pd.DataFrame(summary_data)
    df = df.sort_values(by=["exp_type", "method_pool", "max_layers"]).reset_index(drop=True)

    # 保存为 CSV 和 JSON
    df.to_csv(output_csv, index=False, encoding='utf-8')
    df.to_json(output_json, orient='records', indent=4, force_ascii=False)

    print(f"✅ Summary saved to:\n  - {output_csv}\n  - {output_json}")
    print("\nSummary Table:")
    print(df.to_string(index=False, float_format="{:.4f}".format))

if __name__ == "__main__":
    main()