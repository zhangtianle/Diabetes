import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
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
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
import util

train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

train.pop("id")
test.pop("id")
target = train.pop("血糖")

train_x = train.as_matrix()
train_y = target.as_matrix()
test_x = test.as_matrix()

lgb_train = lgb.Dataset(train_x, train_y)

exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    StackingEstimator(
        estimator=LinearSVR(C=0.01, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.001)),
    StackingEstimator(estimator=RidgeCV()),
    lgb.LGBMRegressor(objective='regression',
                      boosting_type="GBDT",
                      num_leaves=17,
                      learning_rate=0.01,
                      feature_fraction=0.5,
                      bagging_fraction=0.5,
                      bagging_freq=5,
                      reg_alpha=1,
                      reg_lambda=0.5,
                      n_estimators=500)
)
exported_pipeline.fit(train_x, train_y)
predict = exported_pipeline.predict(test_x)

util.save(pd.DataFrame(predict), '1.8-0.95963')
