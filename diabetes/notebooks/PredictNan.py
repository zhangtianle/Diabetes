import pandas as pd
import numpy as np
import copy as cp
import re
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
class PredictNan:
    def __init__(self, raw_data, raw_ID):
        data = cp.deepcopy(raw_data)

        ID = cp.deepcopy(raw_ID)
        columns = data.columns.tolist()
        boolen_matrix = np.any(data.isnull(), axis=0)

        self.continue_columns = []
        self.discrete_columns = []


        self.targets = cp.deepcopy(ID)
        self.train = cp.deepcopy(data)

        
        for i, res in enumerate(boolen_matrix):
            if res:
                if len(data[columns[i]].unique()) > 10:
                    self.continue_columns.append(columns[i])
                else:
                    self.discrete_columns.append(columns[i])
                self.targets = pd.concat([pd.DataFrame(self.train[columns[i]]), self.targets],axis=1)
                self.train.pop(columns[i])

        columns = set(self.train.columns.tolist())
        for column in columns:
            if self.train[column].dtype != "object":
                self.train[column] = MinMaxScaler().fit_transform(self.train[column].as_matrix().reshape(-1,1))                
                
        self.train = pd.concat([ID, self.train], axis=1)

        print(len(self.continue_columns), len(self.discrete_columns))
        for i, column in enumerate(self.discrete_columns):
            self.predict_discrete(column)
            print(column, i)


        for i, column in enumerate(self.continue_columns):
            self.predict_continue(column)
            print(column, i)


    #
    def predict_continue(self, column):
        test_id = self.targets.loc[self.targets[column].isnull()]["id"]
        train_id = self.targets.loc[~self.targets[column].isnull()]["id"]
        train_x = self.train.loc[self.train["id"].isin(train_id)]
        train_x.pop("id")
        train_x = train_x.as_matrix()
        test_x = self.train.loc[self.train["id"].isin(test_id)]
        test_x.pop("id")
        test_x = test_x.as_matrix()
        train_y = self.targets.loc[~self.targets[column].isnull()][column].as_matrix()
        clf = lgb.LGBMRegressor(objective="regression",
                boosting_type="GBDT",
                num_leaves=31,
                learning_rate=0.01,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                n_estimators=100)
        clf.fit(train_x, train_y)
        predict = clf.predict(test_x)
        for i, index in enumerate(test_id.index.tolist()):
            self.targets[column][index] = predict[i]

    def predict_discrete(self, column):
        test_id = self.targets.loc[self.targets[column].isnull()]["id"]
        train_id = self.targets.loc[~self.targets[column].isnull()]["id"]
        train_x = self.train.loc[self.train["id"].isin(train_id)]
        train_x.pop("id")
        train_x = train_x.as_matrix()
        test_x = self.train.loc[self.train["id"].isin(test_id)]
        test_x.pop("id")
        test_x = test_x.as_matrix()
        train_y = self.targets.loc[~self.targets[column].isnull()][column].as_matrix()
        clf = DecisionTreeRegressor(max_depth=3)
        clf.fit(train_x, train_y)

        predict = clf.predict(test_x)
        for i, index in enumerate(test_id.index.tolist()):
            self.targets[column][index] = predict[i]
def main():
    train = pd.read_csv("../data/raw/train.csv")
    ID = pd.DataFrame({"ID":train.pop("ID")})

    columns = train.columns.tolist()
    for column in columns:
        if train[column].dtype != "object":
            mean = str(train[column].mean())
            if re.match(r"2017\d+", mean) == None and re.match(r"2016\d+", mean) == None:
                continue
            else:
                train.pop(column)
    # 删除空的特征
    columns = train.columns.tolist()
    for column in columns:
        if train[column].dtype != "object" and train[column].count() == 0:
            train.pop(column)


    predict = PredictNan(train, ID)

    train[predict.continue_columns[0]] = predict.targets[predict.continue_columns[0]]
    train[predict.discrete_columns[0]] = predict.targets[predict.discrete_columns[0]]


if __name__ == '__main__':
    main()
