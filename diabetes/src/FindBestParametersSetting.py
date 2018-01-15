import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
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
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def MSE(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    return mean_squared_error(y_test, y_pred) / 2.0
def offline_GBDT_gird(x_train,y_train):
    make_scorer(MSE, greater_is_better=False)
    piple = Pipeline([
        ("scaler", MaxAbsScaler()),
        ("SVR", StackingEstimator(estimator=LinearSVR(C=0.01, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.001))),
        ("RidgeCV", StackingEstimator(estimator=RidgeCV())),
        ("LGB", lgb.LGBMRegressor(objective='regression',
                                  boosting_type="GBDT",
                                  bagging_freq=5,))]
    )
    param_grid = [
        {
            'LGB__num_leaves': range(10, 31, 1),
            'LGB__n_estimators': range(100, 600, 100),
            'LGB__learning_rate': [0.01,0.02],
            'LGB__feature_fraction': [0.5,0.6,0.7,0.8,0.9,1.0],
            'LGB__bagging_fraction': [0.5,0.6,0.7,0.8,0.9,1.0],
            'LGB__reg_alpha': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
            'LGB__reg_lambda': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        },
    ]
    girdSearch = GridSearchCV(piple,param_grid=param_grid, scoring=MSE, iid=False, cv=5, verbose=10)
    girdSearch.fit(x_train, y_train)
    print(girdSearch.cv_results_)
    print("Best param with grid search: ", girdSearch.best_params_)
    print("Best score with grid search: ", girdSearch.best_score_)

def main():
    train = pd.read_csv("../data/processed/train.csv")
    train.pop("id")
    target = train.pop("血糖")
    X = train.as_matrix()
    Y = target.as_matrix()
    offline_GBDT_gird(X, Y)

if __name__ == '__main__':
    main()