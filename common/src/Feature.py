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
        # gender 1 为 男 ，2 为 女
        # 返回值 1 不正常 ，0 为 正常
        self.train['甘油三酯_normal'] = self.train['甘油三酯'].apply(normal, args=(1.69,))
        self.train['尿素_normal'] = self.train['尿素'].apply(normal, args=(7.1,))
        self.train['尿酸_normal'] = self.train.apply(
            lambda x: 1 if (x['gender'] == 0 and x['尿酸'] > 357) or (x['gender'] == 1 and x['尿酸'] > 416) else 0, axis=1)
        self.train['肌酐_normal'] = self.train.apply(
            lambda x: 1 if (x['gender'] == 0 and x['肌酐'] > 108) or (x['gender'] == 1 and x['肌酐'] > 133) else 0, axis=1)
        self.train['*天门冬氨酸氨基转换酶_normal'] = self.train['*天门冬氨酸氨基转换酶'].apply(normal, args=(40,))
        self.train['总胆固醇_normal'] = self.train['总胆固醇'].apply(normal, args=(5.2,))
        self.train['高密度脂蛋白胆固醇_normal'] = self.train['高密度脂蛋白胆固醇'].apply(normal, args=(1.21,))

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
            # self.train[column] = self.train[column].fillna(self.train[column].mean())
            self.train[column] = self.train[column].fillna(self.train[column].median())

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
        age_cut = pd.cut(self.train['age'],
                         [1, 20, 30, 35, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88])
        age_cut = pd.DataFrame({'age_cut': age_cut})
        # age_cut.rename(columns={'年龄': 'age'}, inplace=True)
        train_m = pd.concat([self.train, age_cut], axis=1)
        # train_m.drop(['年龄'], axis=1, inplace=True)
        self.train = pd.get_dummies(train_m, columns=['age_cut'])

    def combine_feature(self):
        columns = ["甘油三酯", "尿酸","红细胞计数","白细胞计数"]
        source = "age"
        for column in columns:
            self.train[source + "*" + column] = self.train.apply(lambda x: x[source] * x[column], axis=1)

        columns = ["尿酸"]
        source = "尿素"
        for column in columns:
            self.train[source + "*" + column] = self.train.apply(lambda x: x[source] * x[column], axis=1)

        columns = ["红细胞平均血红蛋白浓度"]
        source = "尿酸"
        for column in columns:
            self.train[source + "*" + column] = self.train.apply(lambda x: x[source] * x[column], axis=1)		
        columns = ["红细胞体积分布宽度"]
        source = "红细胞平均体积"
        for column in columns:
            self.train[source + "*" + column] = self.train.apply(lambda x: x[source] * x[column], axis=1)    
			
        columns = ["红细胞体积分布宽度"]
        source = "红细胞平均血红蛋白量"
        for column in columns:
            self.train[source + "*" + column] = self.train.apply(lambda x: x[source] * x[column], axis=1)       	
        self.train["红细胞平均血红蛋白浓度"+'_onedivide']=self.train["红细胞平均血红蛋白浓度"].apply(lambda x: 1.0/x if x!=0 else 1e8)
        columns = ["红细胞平均血红蛋白浓度_onedivide"]
        source = "红细胞平均体积"
        for column in columns:
            self.train[source + "*" + column] = self.train.apply(lambda x: x[source] * x[column], axis=1)         	
			
			
			
			
			
			

        # total['中性粒细胞/淋巴细胞'] = 1.0* total['中性粒细胞%'] / total['淋巴细胞%'] #0.94
        self.train['红细胞计数^2'] = self.train['红细胞计数'] * self.train['红细胞计数']  # 0.93581
        self.train['尿素^2'] = self.train['尿素'] * self.train['尿素']  # 0.93540
        # total['肌酐^2'] = total['肌酐'] * total['肌酐'] #0.93624
        self.train['尿酸^2'] = self.train['尿酸'] * self.train['尿酸']  # 0.93712
        self.train['总胆固醇^2'] = self.train['总胆固醇'] * self.train['总胆固醇']  # 0.93471
        self.train['甘油三酯^2'] = self.train['甘油三酯'] * self.train['甘油三酯']  # 0.93392
        # total['红细胞计数'] = 1.2*total['红细胞计数'] #0.93703

        '''
        columns = ["高密度脂蛋白胆固醇", "低密度脂蛋白胆固醇"]
        source = "总胆固醇"
        self.train["其他胆固醇和"] = self.train.apply(lambda x: x[source] - x[columns[0]] - x[columns[1]], axis=1)

        columns = ["白蛋白", "*球蛋白"]
        source = "*总蛋白"
        self.train["其他蛋白和"] = self.train.apply(lambda x: x[source] - x[columns[0]] - x[columns[1]], axis=1)

        columns = ["*天门冬氨酸氨基转换酶", "*丙氨酸氨基转换酶", "*碱性磷酸酶", "*r-谷氨酰基转换酶"]
        self.train["xx酶总和"] = self.train.apply(lambda x: x[columns[0]] + x[columns[1]] + x[columns[2]] + x[columns[3]], axis=1)
        '''
    def missing_num(self):
        missing_num = pd.DataFrame({'missing_num': len(self.train.columns) - self.train.T.count()})
        self.train = pd.concat([self.train, missing_num], axis=1)

    def week_day(self):
        self.train['date'] = pd.to_datetime(self.train['date'])
        self.train['weekday'] = self.train['date'].dt.weekday
        self.train.drop(['date'], axis=1, inplace=True)
        self.train = pd.get_dummies(self.train, columns=['weekday'])
