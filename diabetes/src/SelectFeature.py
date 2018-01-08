import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectPercentile, f_regression
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
import copy as cp



def get_score(df_x, df_y):
    X = df_x.as_matrix()
    Y = df_y.as_matrix()
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'mse'},
        'num_leaves': 31,
        'learning_rate': 0.01,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'cat_smooth': 10,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 5,
        'verbose': 0
    }

    N = 5
    kf = KFold(n_splits=N, random_state=42)
    result_mean = 0.0

    for train_index, test_index in kf.split(X):
        training_features, training_target = X[train_index], Y[train_index]
        testing_features, testing_target = X[test_index], Y[test_index]

        lgb_train = lgb.Dataset(training_features, training_target)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=500)
        results = gbm.predict(testing_features)

        result_mean += np.round(mean_squared_error(testing_target, results), 5)
    result_mean /= N
    return "Mean squared error: %.5f" % (result_mean / 2)


def main():
    train = pd.read_csv("../data/processed/train.csv")
    train.pop("id")
    target = train.pop("血糖")


    base_score = get_score(train, target)
    print(base_score)

    columns = ["date","age","*天门冬氨酸氨基转换酶","*丙氨酸氨基转换酶",
               "*碱性磷酸酶","*r-谷氨酰基转换酶","*总蛋白",
               "白蛋白","*球蛋白","白球比例","甘油三酯","总胆固醇","高密度脂蛋白胆固醇","低密度脂蛋白胆固醇",
               "尿素","肌酐","尿酸","乙肝表面抗原","乙肝表面抗体","乙肝e抗原","乙肝e抗体",
               "乙肝核心抗体","白细胞计数","红细胞计数","血红蛋白","红细胞压积",
               "红细胞平均体积","红细胞平均血红蛋白量","红细胞平均血红蛋白浓度","红细胞体积分布宽度",
               "血小板计数","血小板平均体积","血小板体积分布宽度","血小板比积","中性粒细胞%","淋巴细胞%","单核细胞%",
               "嗜酸细胞%","嗜碱细胞%"]

    results = []
    for column in columns:
        tmp = cp.deepcopy(train)
        tmp.pop(column)
        score = get_score(tmp, target)
        print(column, score)
        if score < base_score:
            print("greater")
            results.append((column,score))
    print(results)

if __name__ == '__main__':
    main()