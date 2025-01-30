import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
import joblib


def linear01():
    """
    线性回归预测房子价格 - 通过正规方程优化
    :return: None
    """
    # 1、获取数据
    data = pd.read_csv("./datasets/boston.csv")
    # 准备特征值和目标值
    x = data.iloc[:, 1:14]
    y = data.iloc[:, 14]

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 3、特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、线性回归预估器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5、得出模型
    print("正规方程-权重系数为：\n", estimator.coef_)
    print("正规方程-偏置为：\n", estimator.intercept_)

    # 6、模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程-均方误差为：\n", error)

    return None


def linear02():
    """
    线性回归预测房子价格 - 通过梯度下降优化
    :return: None
    """
    # 1、获取数据
    data = pd.read_csv("./datasets/boston.csv")
    # 准备特征值和目标值
    x = data.iloc[:, 1:14]
    y = data.iloc[:, 14]

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 3、特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、线性回归预估器
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)

    # 5、得出模型
    print("梯度下降-权重系数为：\n", estimator.coef_)
    print("梯度下降-偏置为：\n", estimator.intercept_)

    # 6、模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降-均方误差为：\n", error)

    return None


def linear03():
    """
    岭回归预测房子价格
    :return: None
    """
    # 1、获取数据
    data = pd.read_csv("./datasets/boston.csv")
    # 准备特征值和目标值
    x = data.iloc[:, 1:14]
    y = data.iloc[:, 14]

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 3、特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、预估器
    # estimator = Ridge()
    # estimator.fit(x_train, y_train)

    # 保存模型
    # joblib.dump(estimator, "my_ridge.pkl")

    # 加载模型
    estimator = joblib.load("my_ridge.pkl")

    # 5、得出模型
    print("岭回归-权重系数为：\n", estimator.coef_)
    print("岭回归-偏置为：\n", estimator.intercept_)

    # 6、模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归-均方误差为：\n", error)

    return None


if __name__ == '__main__':
    # linear01()
    # linear02()
    linear03()
