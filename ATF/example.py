import AdaptoFlux
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


# 示例使用
if __name__ == "__main__":
    # 1. 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. 数据预处理
    x_train = x_train / 255.0  # 将像素值归一化到 [0, 1]
    x_test = x_test / 255.0

    # 3. 切割 10% 的数据作为验证集
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, 
        test_size=0.1,      # 10% 的数据作为验证集
        random_state=42     # 保证结果的可重复性
    )

    print(x_train.reshape(x_train.shape[0], -1).shape)
    
    # 3. 载入模型和标签
    # model = Figure_Link_Network(x_train.reshape(x_train.shape[0], -1),
    #                             y_train,
    #                             x_val.reshape(x_val.shape[0], -1),
    #                             y_val)

    file_path = "methods.py"
    model = AdaptoFlux.AdaptoFlux(x_train.reshape(x_train.shape[0], -1),
                            y_train, file_path)

    
    model.import_methods_from_file()
    model.training(target_accuracy=0.7)