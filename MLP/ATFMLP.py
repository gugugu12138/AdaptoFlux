import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
import sys
sys.path.append('../ATF/')
from ATF.AdaptoFlux import *
from ATF.DynamicWeightController import *

methods_path = "MLP/MLP_methods.py"
absolute_path = os.path.abspath(methods_path)
print(f"文件的绝对路径是: {absolute_path}")
# 1. 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

loaded_model = tf.keras.models.load_model('MLP/mnist_model52%.keras')
# 2. 数据预处理（仅对验证集合）
x_test = x_test / 255.0

x_copy_test, y_copy_test = x_test, y_test

def mlp_predict(x):
    x = x.reshape(1, 28, 28)
    return loaded_model.predict(x, verbose=0)

MLP_AdaptoFlux = AdaptoFlux(x_copy_test.reshape(x_copy_test.shape[0], -1), y_copy_test, methods_path)
MLP_AdaptoFlux.import_methods_from_file()
MLP_AdaptoFlux.set_custom_collapse(mlp_predict)
MLP_AdaptoFlux.training(target_accuracy=0.6)
MLP_AdaptoFlux.evaluate(x_copy_test,y_copy_test)
loss, accuracy = loaded_model.evaluate(x_test, y_test)
print(f"测试损失: {loss}")
print(f"测试准确率: {accuracy}")

