import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 可能后面做成类
# 1. 读取 JSON 数据
file_path = 'models/training_log.json'  # 请替换为您的实际文件路径

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 2. 提取所有尝试的数据
# 我们需要一个全局的尝试计数器来表示时间/顺序
global_attempt_num = 0
attempt_numbers = []
layers = []
accuracies = []
losses = []
accepted = []  # 用于区分接受和拒绝的点

# 遍历每层的历史
for history in data['attempt_history']:
    layer_num = history['layer']
    # 遍历该层的每一次尝试
    for attempt in history['attempts']:
        global_attempt_num += 1  # 每次尝试都递增
        attempt_numbers.append(global_attempt_num)
        layers.append(layer_num)
        accuracies.append(attempt['new_acc'])
        losses.append(attempt['new_loss'])
        accepted.append(attempt['accepted'])

# 转换为 numpy 数组以便操作
attempt_numbers = np.array(attempt_numbers)
layers = np.array(layers)
accuracies = np.array(accuracies)
losses = np.array(losses)
accepted = np.array(accepted)

# 3. 创建三维图形
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 4. 为被接受和被拒绝的点设置不同的颜色和标记
# 被接受的点 (绿色, 圆圈)
accepted_mask = accepted
rejected_mask = ~accepted

scat1 = ax.scatter(attempt_numbers[accepted_mask], layers[accepted_mask], accuracies[accepted_mask], 
                   c='green', marker='o', s=60, alpha=0.8, label='Accepted')

# 被拒绝的点 (红色, 十字)
scat2 = ax.scatter(attempt_numbers[rejected_mask], layers[rejected_mask], accuracies[rejected_mask], 
                   c='red', marker='x', s=40, alpha=0.6, label='Rejected')

# 5. 标记最佳模型
best_layer = data['best_model_layers']
best_accuracy = data['best_model_accuracy']
# 找到最佳模型被接受时的全局尝试号
# 这需要在数据中找到最后一次在 best_layer 层被接受的尝试
best_attempt_num = 0
for i, (layer, acc, is_accepted) in enumerate(zip(layers, accuracies, accepted)):
    if layer == best_layer and acc == best_accuracy and is_accepted:
        best_attempt_num = attempt_numbers[i]  # 更新为最后一次达到最佳模型的尝试号

if best_attempt_num > 0:
    ax.scatter([best_attempt_num], [best_layer], [best_accuracy], 
               c='gold', s=150, edgecolors='black', linewidth=2, marker='*', zorder=5, label=f'Best Model (Layer {best_layer})')

# 6. 设置坐标轴标签和标题
ax.set_xlabel('Global Attempt Number', fontsize=12)
ax.set_ylabel('Model Layer', fontsize=12)
ax.set_zlabel('Accuracy', fontsize=12)
ax.set_title('3D Visualization of Model Growth: Attempts, Layers, and Accuracy', fontsize=14, pad=20)

# 7. 添加图例
ax.legend(loc='upper left')

# 8. 优化视角和布局
# 调整视角，以便更好地观察数据
ax.view_init(elev=20, azim=45)  # elev 是仰角, azim 是方位角

# 9. 显示图形
plt.tight_layout()
plt.show()

# 另一个 3D 图：Loss 版本
fig2 = plt.figure(figsize=(14, 10))
ax2 = fig2.add_subplot(111, projection='3d')

ax2.scatter(attempt_numbers[accepted_mask], layers[accepted_mask], losses[accepted_mask],
           c='blue', marker='o', s=60, alpha=0.8, label='Accepted')
ax2.scatter(attempt_numbers[rejected_mask], layers[rejected_mask], losses[rejected_mask],
           c='orange', marker='x', s=40, alpha=0.6, label='Rejected')

ax2.set_xlabel('Global Attempt Number')
ax2.set_ylabel('Model Layer')
ax2.set_zlabel('Loss')
ax2.set_title('3D Visualization: Attempts, Layers, and Loss')
ax2.legend()
ax2.view_init(elev=20, azim=45)
plt.tight_layout()
plt.show()

# 可选：打印一些信息
print(f"Total number of attempts visualized: {len(attempt_numbers)}")
print(f"Number of accepted attempts: {np.sum(accepted)}")
print(f"Number of rejected attempts: {np.sum(~accepted)}")