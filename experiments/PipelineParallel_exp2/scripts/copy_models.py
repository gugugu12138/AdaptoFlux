import csv
import os
import shutil
import json
import re

# 文件路径
csv_file_path = r"experiments/PipelineParallel_exp2/scripts/selected_models_exp2_with_type_corrected.csv"
base_folder_path = r"experiments/LayerGrowTrainer_exp1"
destination_folder = r"experiments/PipelineParallel_exp2/models"  # 目标文件夹

def extract_max_layer_from_graph_json(graph_json_path):
    """从 graph.json 中提取节点 ID 的最大前缀数字（如 '3_xxx' -> 3）"""
    try:
        with open(graph_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {graph_json_path}: {e}")
        return None

    max_prefix = -1
    for node in data.get("nodes", []):
        node_id = node.get("id", "")
        if not isinstance(node_id, str):
            continue
        # 匹配以数字开头，后跟下划线的模式，如 "5_abc"
        match = re.match(r'^(\d+)_', node_id)
        if match:
            prefix = int(match.group(1))
            if prefix > max_prefix:
                max_prefix = prefix
    return max_prefix if max_prefix != -1 else None

def get_model_folders(csv_path, base_path):
    model_folders = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            max_layers = row['max_layers']
            run = row['run']
            model_type = row['model_type']
            expected_layers = int(row['layers'])  # 从 CSV 读取期望的层数

            folder_path = os.path.join(base_path, f"exp_max_layers_{max_layers}", f"run_{run}", model_type)
            graph_json_path = os.path.join(folder_path, "graph.json")

            # 检查文件夹和 graph.json 是否存在
            if not os.path.exists(folder_path):
                print(f"Warning: Folder not found - {folder_path}")
                continue
            if not os.path.exists(graph_json_path):
                print(f"Warning: graph.json not found in - {folder_path}")
                continue

            # 提取实际最大层数
            actual_max_layer = extract_max_layer_from_graph_json(graph_json_path)
            if actual_max_layer is None:
                print(f"Warning: Could not extract layer info from graph.json in {folder_path}")
                continue

            # 验证：实际最大前缀数字应等于 expected_layers
            if actual_max_layer == expected_layers:
                model_folders.append(folder_path)
                print(f"✓ Validated: {folder_path} (layers={expected_layers})")
            else:
                print(f"✗ Layer mismatch: {folder_path} | Expected: {expected_layers}, Got: {actual_max_layer}")
    return model_folders

def copy_and_rename_folders(folders, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for idx, folder in enumerate(folders, start=1):
        new_folder_name = f"model_{idx}"
        new_folder_path = os.path.join(dest_folder, new_folder_name)
        shutil.copytree(folder, new_folder_path)
        print(f"Copied {folder} to {new_folder_path}")

# 执行流程
model_folders = get_model_folders(csv_file_path, base_folder_path)
copy_and_rename_folders(model_folders, destination_folder)