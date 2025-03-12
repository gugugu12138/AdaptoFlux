import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
import ATF

# 1. 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

loaded_model = tf.keras.models.load_model('mnist_model.h5')
x_copy_test, y_copy_test = x_test, y_test

loaded_model.evaluate(x_test, y_test)

