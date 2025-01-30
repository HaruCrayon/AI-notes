from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import jieba
import pandas as pd


def datasets_demo():
    """
    对鸢尾花数据集的演示
    :return: None
    """
    # 1、获取鸢尾花数据集
    iris = load_iris()
    # print("鸢尾花数据集的返回值：\n", iris)
    # 返回值是一个继承自字典的Bunch
    print("鸢尾花的特征值:\n", iris.data, iris.data.shape)
    print("鸢尾花的目标值：\n", iris.target)
    print("鸢尾花特征的名字：\n", iris.feature_names)
    print("鸢尾花目标值的名字：\n", iris.target_names)
    print("鸢尾花的描述：\n", iris.DESCR)

    # 2、对鸢尾花数据集进行分割
    # 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值:\n", x_train, x_train.shape)

    return None


def dict_demo():
    """
    对字典类型的数据进行特征抽取
    :return: None
    """
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    # 1、实例化一个转换器类
    transfer = DictVectorizer(sparse=False)
    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("字典特征抽取的结果:\n", data_new)
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


def text_count_demo():
    """
    对文本进行特征抽取，countvetorizer
    :return: None
    """
    data = ["life is short,life i like like python", "life is too long,i dislike python"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["is", "too"])
    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("文本特征抽取的结果：\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


def cut_word(text):
    """
    对中文进行分词
    "我爱北京天安门" ————> "我 爱 北京 天安门"
    :param text:
    :return: text_new
    """
    # 用jieba对中文字符串进行分词
    text_new = " ".join(list(jieba.cut(text)))

    return text_new


def text_chinese_count_demo():
    """
    对中文进行特征抽取
    :return: None
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将原始数据转换成分好词的形式
    text_list = []
    for sent in data:
        text_list.append(cut_word(sent))
    print(text_list)

    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["一种", "所以"])
    # 2、调用fit_transform
    data_new = transfer.fit_transform(text_list)
    print("文本特征抽取的结果：\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


def text_chinese_tfidf_demo():
    """
    对中文进行特征抽取
    :return: None
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将原始数据转换成分好词的形式
    text_list = []
    for sent in data:
        text_list.append(cut_word(sent))
    print(text_list)

    # 1、实例化一个转换器类
    transfer = TfidfVectorizer(stop_words=['一种', '不会', '不要'])
    # 2、调用fit_transform
    data_new = transfer.fit_transform(text_list)
    print("文本特征抽取的结果：\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


def minmax_demo():
    """
    归一化
    :return: None
    """
    data = pd.read_csv("datasets/dating.txt", sep="\t", usecols=['milage', 'Liters', 'Consumtime'])
    print(data)
    # 1、实例化一个转换器类
    transfer = MinMaxScaler(feature_range=(2, 3))
    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("最小值最大值归一化处理的结果：\n", data_new)

    return None


def stand_demo():
    """
    标准化
    :return: None
    """
    data = pd.read_csv("datasets/dating.txt", sep="\t", usecols=['milage', 'Liters', 'Consumtime'])
    print(data)
    # 1、实例化一个转换器类
    transfer = StandardScaler()
    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("标准化的结果:\n", data_new)
    print("每一列特征的平均值：\n", transfer.mean_)
    print("每一列特征的方差：\n", transfer.var_)

    return None


def variance_demo():
    """
    删除低方差特征——特征选择
    :return: None
    """
    data = pd.read_csv("datasets/factor_returns.csv")
    data = data.iloc[:, 1:10]
    print(data)
    # 1、实例化一个转换器类
    transfer = VarianceThreshold(threshold=1)
    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("删除低方差特征的结果：\n", data_new)
    print("形状：\n", data_new.shape)

    return None


def pearsonr_demo():
    """
    相关系数计算
    :return: None
    """
    data = pd.read_csv("datasets/factor_returns.csv")

    factor = ['pe_ratio', 'pb_ratio', 'market_cap', 'return_on_asset_net_profit', 'du_return_on_equity', 'ev',
              'earnings_per_share', 'revenue', 'total_expense']

    for i in range(len(factor)):
        for j in range(i, len(factor) - 1):
            print(
                "指标%s与指标%s之间的相关性大小为%f" % (
                    factor[i], factor[j + 1], pearsonr(data[factor[i]], data[factor[j + 1]])[0]))

    return None


def pca_demo():
    """
    对数据进行PCA降维
    :return: None
    """
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]

    # 1、实例化PCA, 小数——保留百分之多少的信息
    transfer1 = PCA(n_components=0.9)
    # 2、调用fit_transform
    data1 = transfer1.fit_transform(data)

    print("保留90%的信息，降维结果为：\n", data1)

    # 1、实例化PCA, 整数——指定降维到的维数
    transfer2 = PCA(n_components=3)
    # 2、调用fit_transform
    data2 = transfer2.fit_transform(data)
    print("降维到3维的结果：\n", data2)

    return None


if __name__ == '__main__':
    # datasets_demo()
    # dict_demo()
    # text_count_demo()
    # text_chinese_count_demo()
    # text_chinese_tfidf_demo()
    # minmax_demo()
    # stand_demo()
    # variance_demo()
    # pearsonr_demo()
    pca_demo()
