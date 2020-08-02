import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from Q4 import data_freq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.preprocessing import Normalizer
import seaborn as sns
import pprint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import sklearn

data = pd.read_csv("coursework1.csv")


# class Question:
def gini_impurity(data, feature, label):
    all_feature_categories = data[feature].unique().tolist()
    all_label_categories = data[label].unique().tolist()
    ls = []
    for f_cate in all_feature_categories:
        ls_f = []
        for l_cate in all_label_categories:
            num = len(data[(data[feature] == f_cate) & (data[label] == l_cate)])
            ls_f.append(num)
            impurity = 1
            c_total = sum(ls_f)
            for i in ls_f:
                impurity -= (i / c_total) ** 2
                impurity *= c_total
                ls.append(impurity)
    return sum(ls) / len(data[label])


class Decision_tree:

    def init(self):
        self.tree = 0
        self.features_dic = {}
        self.label_ls = []
        self.model = 0

    def build_tree(self, data_freq, features, label):
        pass


def extract_dic(self, data, features, label):
    features_dic = {}
    for feature in features:
        key = f"{feature}"
        items = data[feature].unique().tolist()
        features_dic[key] = items

        self.features_dic = features_dic
        self.label_ls = data[label].unique().tolist()
    return


def arrange(data, features, label, func=gini_impurity):
    X = features
    Y = [func(data, x, label) for x in X]
    return [x for _, x in sorted(zip(Y, X), reverse=False)]


def gini_impurity(data, feature, label):
    all_feature_categories = data[feature].unique().tolist()
    all_label_categories = data[label].unique().tolist()

    ls = []
    for f_cate in all_feature_categories:
        ls_f = []
    for l_cate in all_label_categories:
        num = len(data[(data[feature] == f_cate) & (data[label] == l_cate)])
        ls_f.append(num)
    impurity = 1
    c_total = sum(ls_f)
    for i in ls_f:
        impurity -= (i / c_total) ** 2
    impurity *= c_total
    ls.append(impurity)
    return sum(ls) / len(data[label])


def build_tree(self, data, features, label):
    features = self.arrange(data, features, label)
    self.extract_dic(data, features, label)
    self.model = self.grow(data, features, label)
    return


def grow(self, data, features, label, level=0):
    feature = features[level]
    node = {}
    level += 1
    f_cate_list = data[feature].unique().tolist()

    # f_cate_list = self.features_dic[feature]

    for f_cate in f_cate_list:
        edge = f"{f_cate}"
        data_selected = data[data[feature] == f_cate]
        if feature == features[-1]:
           leafs = []

        for leaf in self.label_ls:
                if len(data_selected) == 0:
                   val = 0
        else:
            val = len(data_selected[data_selected[label] == leaf]) / len(data_selected)
            leafs.append(val)
            node[edge] = leafs
    else:
        node[edge] = self.grow(data_selected, features, label, level)
    return node


# create a tree class
dec_tree = Decision_tree()
# setup features and label name
features = data_freq.columns[:-1].tolist()
label = data_freq.columns[-1]
# Build the tree
dec_tree.build_tree(data_freq, features, label)
# print the tree
printer = pprint.PrettyPrinter()
#printer.pprint(dec_tree.model)
# with label index
#print('\n')
#for j in enumerate(dec_tree.label_ls):
 #   print(j)
