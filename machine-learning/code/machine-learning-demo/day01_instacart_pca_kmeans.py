import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def pca_kmeans_instacart():
    """
    1、主成分分析PCA 实现特征的降维
    2、K-Means对Instacart Market用户聚类
    :return: None
    """
    # 1、获取数据集
    products = pd.read_csv("./datasets/instacart/products.csv")
    order_products = pd.read_csv("./datasets/instacart/order_products__prior.csv")
    orders = pd.read_csv("./datasets/instacart/orders.csv")
    aisles = pd.read_csv("./datasets/instacart/aisles.csv")

    # 2、合并表，将user_id和aisle放在一张表上
    tab1 = pd.merge(orders, order_products, on=["order_id", "order_id"])
    tab2 = pd.merge(tab1, products, on=["product_id", "product_id"])
    tab3 = pd.merge(tab2, aisles, on=["aisle_id", "aisle_id"])

    # 3、交叉表处理，把user_id和aisle进行分组
    table = pd.crosstab(tab3["user_id"], tab3["aisle"])

    print("降维前: ", table.shape)  # (206209, 134)

    # 4、主成分分析的方法进行降维
    # 1）实例化一个转换器类PCA
    transfer = PCA(n_components=0.95)
    # 2）fit_transform
    data = transfer.fit_transform(table)

    print("降维后: ", data.shape)  # (206209, 44)

    # 5、KMeans聚类预估器
    estimator = KMeans(n_clusters=3)
    estimator.fit(data)

    y_predict = estimator.predict(data)
    print("聚类结果：\n", y_predict)  # [0 0 0 ... 0 0 1 0 2 0]

    # 6、模型评估：轮廓系数
    score = silhouette_score(data, y_predict)
    print("所有样本的平均轮廓系数：\n", score)  # 0.5365951628715876

    return None


if __name__ == '__main__':
    pca_kmeans_instacart()
