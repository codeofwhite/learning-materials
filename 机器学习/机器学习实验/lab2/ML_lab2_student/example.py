from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 分割数据集为训练集和测试集

# DecisionTreeClassifier model
dt_cart = []


def visualize_tree(model, feature_names, class_names):
    dot_data = export_graphviz(
        model, out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True, rounded=True,
        special_characters=True
    )
    return graphviz.Source(dot_data)

# 可视化每棵树
tree_cart = visualize_tree(dt_cart, iris.feature_names, iris.target_names)

# 显示决策树
tree_cart.view()
