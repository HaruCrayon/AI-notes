import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def decision_tree_titanic():
    """
    用决策树对泰坦尼克号乘客生存进行预测
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

    # 5、决策树预估器
    estimator = DecisionTreeClassifier(criterion="entropy", max_depth=8)
    estimator.fit(x_train, y_train)

    # 6、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)

    # 可视化决策树
    export_graphviz(estimator, out_file="./titanic_tree.dot", feature_names=transfer.get_feature_names_out())

    return None


if __name__ == '__main__':
    decision_tree_titanic()
