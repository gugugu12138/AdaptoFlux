import json
import os
import shutil

# ========== 配置区 ==========
json_file_path = "experiments\\ExternalBaselines\\results\\summaries_by_collapse.json"          # 你的 JSON 标注文件路径
output_dir = "experiments\\ExternalBaselines\\best_models"         # 目标文件夹（存放复制的模型）
# ==========================

def main():
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取 prod 部分
    prod_data = data.get("prod", {})

    copied_count = 0
    for task_name, methods in prod_data.items():
        adaptoflux_info = methods.get("adaptoflux", {})
        save_path = adaptoflux_info.get("save_path")

        if not save_path:
            print(f"⚠️ 任务 {task_name} 中未找到 save_path，跳过。")
            continue

        if not os.path.exists(save_path):
            print(f"⚠️ 路径不存在: {save_path}（任务: {task_name}），跳过。")
            continue

        # 构造目标路径：output_dir/任务名
        dest_path = os.path.join(output_dir, task_name)

        try:
            if os.path.isdir(save_path):
                # 复制整个目录（允许目标已存在）
                shutil.copytree(save_path, dest_path, dirs_exist_ok=True)
            else:
                # 如果是单个文件（虽然不太可能），也支持
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(save_path, dest_path)
            print(f"✅ 已复制: {save_path} → {dest_path}")
            copied_count += 1
        except Exception as e:
            print(f"❌ 复制失败: {save_path} → {dest_path}, 错误: {e}")

    print(f"\n🎉 完成！共成功复制 {copied_count} 个 prod 最佳模型到 '{output_dir}' 文件夹。")

if __name__ == "__main__":
    main()