import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar10


# 1. 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 数据预处理
x_train = x_train / 255.0  # 将像素值归一化到 [0, 1]
x_test = x_test / 255.0

# 3. 构建神经网络模型
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),  # 将 28x28 的图像展平为一维向量
    Dense(64, activation='relu'),   # 全连接层，64 个神经元，ReLU 激活函数
    Dense(10, activation='softmax') # 输出层，10 个神经元，对应 10 个类别，使用 softmax 激活函数
])

# 4. 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 训练模型
model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.1)

# 6. 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print(f"测试损失: {loss}")
print(f"测试准确率: {accuracy}")

# 7. 保存模型
model.save('cifar10_model.keras')

# 8. 加载模型并测试（可选）
# loaded_model = tf.keras.models.load_model('mnist_model.h5')
# loaded_model.evaluate(x_test, y_test)
