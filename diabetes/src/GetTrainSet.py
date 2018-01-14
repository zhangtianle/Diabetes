import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union, Pipeline
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
from sklearn.preprocessing import MaxAbsScaler, Normalizer
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from imblearn.under_sampling import NearMiss ,RandomUnderSampler
from xgboost import XGBRegressor
from imblearn.ensemble import EasyEnsemble
from sklearn.metrics import f1_score


def score(train_x, train_y):
    N = 5
    kf = KFold(n_splits=N, random_state=42)
    result_mean = 0.0
    for train_index, test_index in kf.split(train_x):
        training_features, training_target = train_x[train_index], train_y[train_index]
        testing_features, testing_target = train_x[test_index], train_y[test_index]
        exported_pipeline = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("SVR", StackingEstimator(
                estimator=LinearSVR(C=0.01, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.001))),
            ("RidgeCV", StackingEstimator(estimator=RidgeCV())),
            ("LGB", lgb.LGBMRegressor(objective='regression',
                                      boosting_type="GBDT",
                                      num_leaves=17,
                                      learning_rate=0.01,
                                      feature_fraction=0.5,
                                      bagging_fraction=0.5,
                                      bagging_freq=5,
                                      reg_alpha=0.1,
                                      reg_lambda=0.5,
                                      n_estimators=400))]
        )
        exported_pipeline.fit(training_features, training_target)
        results = exported_pipeline.predict(testing_features)
        result_mean += np.round(mean_squared_error(testing_target, results), 5)

    result_mean /= (N)
    return result_mean / 2
def main():
    train = pd.read_csv("../data/processed/train.csv")
    train.pop("id")
    target = train.pop("血糖")

    train_x = train.as_matrix()
    train_y = target.as_matrix()

    best_score = 10
    best_left = 0
    for left in np.arange(3.5,5.0,0.02):
        label_Y = np.zeros(train_y.shape[0])
        for i in range(train_y.shape[0]):
            if train_y[i] >= left and train_y[i] <= 6.1:
                label_Y[i] = 0
            elif train_y[i] < left:
                label_Y[i] = 1
            else:
                label_Y[i] = 2
        nm = NearMiss(ratio={0: 3000, 1: len(np.where(label_Y == 1)[0]), 2: len(np.where(label_Y == 2)[0])},
                      random_state=42, return_indices=True, version=2,n_neighbors=10)
        X_res, y_res, index = nm.fit_sample(train_x, label_Y)
        new_x = train_x[index]
        new_y = train_y[index]
        s = score(new_x, new_y)
        if s < best_score:
            best_score = s
            best_left = left
            print("greater")
            print(best_score, best_left)
        print(best_score, best_left)
    print(best_score, best_left)




if __name__ == '__main__':
    main()