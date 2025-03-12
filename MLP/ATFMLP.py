import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import cifar10
import sys
sys.path.append('../ATF/')
from ATF.AdaptoFlux import *

# 1. 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

loaded_model = tf.keras.models.load_model('MLP\mnist_model.h5')
# 2. 数据预处理
x_test = x_test / 255.0

x_copy_test, y_copy_test = x_test, y_test

loaded_model.evaluate(x_test, y_test)

