import json

def print_errors_labels(json_file, max_samples=20):
    """
    从 JSON 文件中加载预测错误的样本，并以文本形式打印预测值和真实值。

    参数:
        json_file (str): JSON 文件路径。
        max_samples (int): 最多显示多少个错误样本。
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        errors = json.load(f)

    print(f"共找到 {len(errors)} 个预测错误的样本。\n")
    
    for i, error in enumerate(errors[:max_samples]):
        prediction = error['prediction']
        target = error['target']

        print(f"样本 {i+1:2d} | 预测值: {prediction} | 真实值: {target}")

    if len(errors) > max_samples:
        print(f"\n...还有 {len(errors) - max_samples} 个错误样本未显示。")

print_errors_labels("errors.json", max_samples=30)