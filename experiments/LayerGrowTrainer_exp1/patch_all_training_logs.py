#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立脚本：用于修补 LayerGrowTrainer 生成的所有 training_log.json 文件。
为每个日志文件添加 'final_model_accuracy' 和 'final_model_layers' 字段。
"""

import os
import json
import argparse
from typing import List, Optional, Tuple

def patch_single_log_file(log_file_path: str, backup: bool = True) -> Tuple[bool, bool]:
    """
    修补单个训练日志文件。

    Args:
        log_file_path (str): 训练日志文件的路径。
        backup (bool): 是否在修改前创建备份文件。

    Returns:
        Tuple[bool, bool]: (是否成功修补, 是否 final_model_accuracy 为空)
    """
    try:
        # 读取日志文件
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        print(f"  [读取] {log_file_path}")
    except FileNotFoundError:
        print(f"  [错误] 文件未找到: {log_file_path}")
        return False, False
    except json.JSONDecodeError as e:
        print(f"  [错误] JSON 格式错误 '{log_file_path}': {e}")
        return False, False
    except Exception as e:
        print(f"  [错误] 读取文件时发生未知错误 '{log_file_path}': {e}")
        return False, False

    # 创建备份
    if backup:
        backup_path = log_file_path + ".bak"
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=4, ensure_ascii=False)
            print(f"  [备份] 已创建: {backup_path}")
        except Exception as e:
            print(f"  [警告] 无法创建备份 '{backup_path}': {e}")

    # --- 核心逻辑：添加缺失字段 ---

    # 1. 添加 final_model_layers
    final_layers = log_data.get("layers_added", 0)
    log_data["final_model_layers"] = final_layers
    print(f"  [添加] final_model_layers = {final_layers}")

    # 2. 添加 final_model_accuracy
    final_accuracy = None
    attempt_history = log_data.get("attempt_history", [])

    if final_layers > 0:
        # 找出所有属于最后一层的记录（可能有多次尝试，因 rollback）
        last_layer_records = [
            record for record in attempt_history
            if record.get("layer") == final_layers
        ]

        if last_layer_records:
            # 取最后一次（按时间顺序最后出现的）最后一层记录
            last_layer_record = last_layer_records[-1]
            attempts = last_layer_record.get("attempts", [])

            # 从后往前遍历尝试，找到最后一个被接受的
            for attempt in reversed(attempts):
                if attempt.get("accepted", False) and attempt.get("status") == "accepted":
                    final_accuracy = attempt.get("new_acc")
                    break

    log_data["final_model_accuracy"] = final_accuracy
    print(f"  [添加] final_model_accuracy = {final_accuracy}")

    # --- 写回文件 ---
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
        print(f"  [成功] 已更新: {log_file_path}")
        return True, final_accuracy is None
    except Exception as e:
        print(f"  [错误] 无法写入文件 '{log_file_path}': {e}")
        return False, False


def find_all_log_files(base_dir: str, filename: str = "training_log.json") -> List[str]:
    """
    递归查找指定目录下所有的训练日志文件。

    Args:
        base_dir (str): 要搜索的根目录。
        filename (str): 要查找的日志文件名。

    Returns:
        List[str]: 所有找到的日志文件的完整路径列表。
    """
    log_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == filename:
                full_path = os.path.join(root, file)
                log_files.append(full_path)
    return log_files


def main():
    parser = argparse.ArgumentParser(description="修补 LayerGrowTrainer 生成的训练日志文件。")
    parser.add_argument(
        "base_dir",
        type=str,
        help="实验根目录，脚本会递归查找其下的所有 'training_log.json' 文件。"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不创建备份文件（默认会创建 .bak 备份）。"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="training_log.json",
        help="要修补的日志文件名（默认为 'training_log.json'）。"
    )

    args = parser.parse_args()

    base_directory = args.base_dir
    create_backup = not args.no_backup
    target_filename = args.filename

    if not os.path.exists(base_directory):
        print(f"错误: 指定的目录 '{base_directory}' 不存在。")
        return

    print(f"开始在目录 '{base_directory}' 中查找所有 '{target_filename}' 文件...")
    all_log_files = find_all_log_files(base_directory, target_filename)

    if not all_log_files:
        print(f"未找到任何名为 '{target_filename}' 的文件。")
        return

    print(f"找到 {len(all_log_files)} 个日志文件，开始修补...\n")

    success_count = 0
    failure_count = 0
    null_accuracy_files = []  # 记录 accuracy 为空的文件

    for log_file in all_log_files:
        print(f"--- 处理文件: {log_file} ---")
        success, is_null_accuracy = patch_single_log_file(log_file, backup=create_backup)
        if success:
            success_count += 1
            if is_null_accuracy:
                null_accuracy_files.append(log_file)
        else:
            failure_count += 1
        print()  # 空行分隔

    print("=" * 50)
    print("修补任务完成！")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {failure_count} 个文件")

    if null_accuracy_files:
        print("\n⚠️  以下文件的 'final_model_accuracy' 为空值（null），请检查其训练历史是否完整：")
        for f in null_accuracy_files:
            print(f"  - {f}")
    else:
        print("\n✅ 所有文件的 'final_model_accuracy' 均已成功提取。")

    print("=" * 50)


if __name__ == "__main__":
    main()