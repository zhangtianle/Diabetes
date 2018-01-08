from util import normal
import pandas as pd
import numpy as np


class Feature:
    def __init__(self, train):
        self.train = train

    def get_train(self):
        return self.train

    def long_tail(self):
        """
        长尾特征
        :return:
        """
        columns = ["*r-谷氨酰基转换酶",
                   "*球蛋白",
                   "低密度脂蛋白胆固醇",
                   "乙肝e抗体",
                   "红细胞计数",
                   "中性粒细胞%",
                   ]
        for column in columns:
            self.train[column] = self.train[column].apply(lambda x: np.log1p(x))

    def normal_value(self):
        self.train['甘油三酯_normal'] = self.train['甘油三酯'].apply(normal, args=(1.69,))
        self.train['尿素_normal'] = self.train['尿素'].apply(normal, args=(7.1,))
        self.train['尿酸_normal'] = self.train.apply(
            lambda x: 0 if (x['性别'] == '女' and x['尿酸'] > 357) or (x['性别'] == '男' and x['尿酸'] > 416) else 1, axis=1)
        self.train['*天门冬氨酸氨基转换酶_normal'] = self.train['*天门冬氨酸氨基转换酶'].apply(normal, args=(40,))

    def one_hot(self, feature_list):
        for features in feature_list:
            one_hot = pd.get_dummies(self.train[features], prefix=features)
            self.train.pop(features)
        self.train = pd.concat([one_hot, self.train], axis=1)

    def fix_missing(self):
        """
            处理缺失值
        :return:
        """
        columns = self.train.columns.tolist()
        for column in columns:
            self.train[column] = self.train[column].fillna(self.train[column].mean())

    def drop_feature(self, feature_list):
        self.train.drop(feature_list, axis=1, inplace=True)

    def statistics(self):
        # 添加统计量
        # columns = total.columns.tolist()
        columns = ["age", "甘油三酯", "红细胞平均体积", "尿素", "尿酸", "*碱性磷酸酶", "红细胞平均血红蛋白浓度", "*天门冬氨酸氨基转换酶",
                   "*r-谷氨酰基转换酶"]
        for column in columns:
            if column != "date" and column != "gender":
                max_value = self.train[column].max()
                min_value = self.train[column].min()
                avg_value = self.train[column].mean()
                std_value = self.train[column].std()

                self.train[column + "max"] = self.train[column].apply(lambda x: x - max_value)
                self.train[column + "min"] = self.train[column].apply(lambda x: x - min_value)
                self.train[column + "avg"] = self.train[column].apply(lambda x: x - avg_value)
                self.train[column + "std"] = self.train[column].apply(lambda x: (x - avg_value) / std_value)


    def age_group(self):
        # 年龄分组
        age_cut = pd.cut(self.train['年龄'],
                         [1, 20, 30, 35, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88])
        age_cut = pd.DataFrame({'age': age_cut})
        # age_cut.rename(columns={'年龄': 'age'}, inplace=True)
        train_m = pd.concat([self.train, age_cut], axis=1)
        # train_m.drop(['年龄'], axis=1, inplace=True)
        self.train = pd.get_dummies(train_m, columns=['性别', 'age'])
