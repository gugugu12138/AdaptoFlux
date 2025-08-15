import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """
    泰坦尼克号数据预处理类：
    - 填补缺失值
    - 特征构造
    - 特征离散化
    - 对数变换
    - One-Hot 编码
    - 特征缩放
    - 删除无用列
    """
    def __init__(self):
        # 各类称谓对应的年龄中位数，用于缺失值填补
        self.title_age_medians = {
            'Mr': 32.32,
            'Miss': 21.68,
            'Mrs': 35.86,
            'Master': 4.57
        }
        self.scaler = StandardScaler()  # 标准化工具
        self.embarked_mode = None       # 登船港口众数
        self.fare_median = None         # 票价中位数
        self.numeric_cols = None        # 数值型特征列名

    def _extract_title(self, df):
        """ 从 Name 中提取称谓（Title），并进行统一处理 """
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        # 替换同义称谓
        title_mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
        df['Title'] = df['Title'].replace(title_mapping)
        # 将不在常见称谓中的分类按性别归类为 Mr 或 Mrs
        title_mask = ~df['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master'])
        df.loc[title_mask, 'Title'] = df.loc[title_mask, 'Sex'].map({'male': 'Mr', 'female': 'Mrs'})
        return df

    def _fill_missing_values(self, df, train=True):
        """ 填补缺失值（Age, Embarked, Fare） """
        if train:
            self.embarked_mode = df['Embarked'].mode()[0]  # 训练集记录众数
            self.fare_median = df['Fare'].median()         # 训练集记录票价中位数
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode)
        df['Fare'] = df['Fare'].fillna(self.fare_median)
        # 按称谓填补年龄
        for title, median in self.title_age_medians.items():
            df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = median
        return df

    def _create_features(self, df):
        """ 构造新特征 """
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1      # 家庭人数
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)   # 是否独自一人
        df['Age*Class'] = df['Age'] * df['Pclass']            # 年龄与船舱等级交互项
        df['Age*Fare'] = df['Age'] * df['Fare']               # 年龄与票价交互项
        return df

    def _discretize_features(self, df):
        """ 对 Age 和 Fare 进行分箱（离散化） """
        df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, np.inf], labels=[0, 1, 2, 3, 4]).astype(int)
        df['FareBand'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3]).astype(int)
        return df

    def _log_transform(self, df):
        """ 对 Fare 取对数变换（减少长尾效应） """
        df['Fare_log'] = np.log1p(df['Fare'])
        return df

    def _one_hot_encoding(self, df):
        """ 对类别变量进行 One-Hot 编码 """
        df = pd.get_dummies(df, columns=['Sex', 'Pclass', 'Embarked', 'Title'], drop_first={
            'Sex': True, 
            'Pclass': True, 
            'Embarked': True, 
            'Title': False  # 保留所有 Title 列
        })
        return df

    def _scale_features(self, df, train=True):
        """ 标准化数值型特征 """
        self.numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if train:
            self.scaler.fit(df[self.numeric_cols])
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        return df

    def _drop_features(self, df):
        """ 删除无用特征列 """
        columns_to_drop = ['Name', 'Ticket', 'Cabin', 'Title', 'Fare', 'SibSp', 'Parch']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        return df.drop(existing_columns, axis=1)

    def _ensure_numeric(self, df):
        """
        确保所有特征都是数值型：
        - 将布尔值转成 int
        - 将 object / category 转成数值
        """
        for col in df.columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)
            elif not np.issubdtype(df[col].dtype, np.number):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df


    def fit(self, X, y=None):
        X = X.copy().set_index('PassengerId')
        X = self._extract_title(X)
        X = self._fill_missing_values(X, train=True)
        X = self._create_features(X)
        X = self._discretize_features(X)
        X = self._log_transform(X)
        X = self._one_hot_encoding(X)
        X = self._drop_features(X)
        X = self._scale_features(X, train=True)
        X = self._ensure_numeric(X)  # ✅ 确保纯数值
        return self

    def transform(self, X):
        X = X.copy().set_index('PassengerId')
        X = self._extract_title(X)
        X = self._fill_missing_values(X, train=False)
        X = self._create_features(X)
        X = self._discretize_features(X)
        X = self._log_transform(X)
        X = self._one_hot_encoding(X)
        X = self._drop_features(X)
        X = self._scale_features(X, train=False)
        X = self._ensure_numeric(X)  # ✅ 确保纯数值
        return X



def preprocess_and_save(train_path, test_path):
    """
    数据预处理主函数：
    - 读取原始 CSV
    - 调用 TitanicPreprocessor 进行处理
    - 输出为 train_processed.csv 与 test_processed.csv
    """
    # 读取原始数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 分离特征与标签
    X_train = train_data.drop('Survived', axis=1)
    y_train = train_data['Survived']

    # 初始化并执行预处理
    preprocessor = TitanicPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(test_data)

    # 训练集保存时合并标签
    train_processed = pd.concat([y_train.reset_index(drop=True), X_train_processed.reset_index(drop=True)], axis=1)
    train_processed.to_csv('examples/kaggle/titanic/output/train_processed.csv', index=False)

    # 测试集直接保存
    X_test_processed.to_csv('examples/kaggle/titanic/output/test_processed.csv', index=False)

    print("✅ 预处理完成：train_processed.csv 和 test_processed.csv 已生成！")


if __name__ == "__main__":
    preprocess_and_save(
        'examples/kaggle/titanic/input/train.csv',
        'examples/kaggle/titanic/input/test.csv'
    )