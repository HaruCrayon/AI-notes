import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def random_forest_titanic():
    """
    用随机森林对泰坦尼克号乘客生存进行预测
    :return: None
    """
    # 1、获取数据
    data = pd.read_csv("./datasets/titanic.csv")

    # 筛选特征值和目标值
    x = data[["Pclass", "Age", "Sex"]]
    y = data["Survived"]

    # 2、数据处理
    # 1) 缺失值处理
    x.fillna({"Age": x["Age"].mean()}, inplace=True)
    # 2) 特征值转换成字典
    x = x.to_dict(orient="records")

    # 3、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 4、特征工程：字典特征抽取
    transfer = DictVectorizer(sparse=False)
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 5、随机森林预估器
    estimator = RandomForestClassifier()
    # 添加网格搜索和交叉验证
    # 参数准备
    param_dict = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)

    estimator.fit(x_train, y_train)

    # 6、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)

    # 结果分析：
    # 最佳参数
    print("最佳参数:\n", estimator.best_params_)
    # 在交叉验证中验证的最好结果
    print("最佳结果:\n", estimator.best_score_)
    # 最好的参数模型
    print("最佳估计器:\n", estimator.best_estimator_)
    # 交叉验证结果
    print("交叉验证结果:\n", estimator.cv_results_)
    return None


if __name__ == '__main__':
    random_forest_titanic()
