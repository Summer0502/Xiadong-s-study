from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Load iris dataset from sklearn
iris = datasets.load_iris()

# Declare an of the KNN classifier class with the value with neighbors.
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model with training data and target values
knn.fit(iris['data'], iris['target'])

# Provide data whose class labels are to be predicted
X = [
    [5.9, 1.0, 5.1, 1.8],
    [3.4, 2.0, 1.1, 4.8],
]

# Prints the data provided
print(X)

# Store predicted class labels of X
prediction = knn.predict(X)

# Prints the predicted class labels of X
print(prediction)

#这里，  0 对应 Versicolor（杂色鸢尾） 1 对应 Virginica（维吉尼亚鸢尾） 2 对应 Setosa（山鸢尾）
# 基于给定输入，使用 KNN 分类器，两张图中的花都被预测为 Virginica。

#原文链接https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650737297&idx=3&sn=cc16437f343cac2d2eabdd8a9205c87a&chksm=871aceefb06d47f90233a703711385db2e1dc807876740bf10d50b785f59ec42a3199c83be86&mpshare=1&scene=23&srcid=07030hPBZntEE0DWIvfu8n9I#rd
