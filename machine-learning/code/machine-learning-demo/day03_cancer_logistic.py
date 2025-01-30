import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def logistic_regression_cancer():
    """
    逻辑回归进行癌症预测
    :return: None
    """
    # 1、获取数据
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv("./datasets/cancer.csv", names=column_name)

    # 2、数据处理
    # 替换缺失值
    data = data.replace(to_replace='?', value=np.nan)
    # 删除缺失值
    data.dropna(inplace=True)

    # 筛选特征值和目标值
    x = data.iloc[:, 1:-1]
    y = data["Class"]

    # 3、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 4、特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 5、逻辑回归预估器
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    # 逻辑回归的模型参数：回归系数和偏置
    print("逻辑回归-权重系数为：\n", estimator.coef_)
    print("逻辑回归-偏置为：\n", estimator.intercept_)

    # 6、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)

    # 7、分类的评估方法
    # 1) 精确率与召回率
    report = classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性'])
    print("分类评估报告:\n", report)

    # 2) ROC曲线与AUC指标
    # y_true:每个样本的真实类别，必须为0(反例),1(正例)标记
    # 将y_test转换成 0 1
    y_true = np.where(y_test > 3, 1, 0)
    auc = roc_auc_score(y_true, y_predict)
    print("AUC指标：\n", auc)

    return None


if __name__ == '__main__':
    logistic_regression_cancer()
