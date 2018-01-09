import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline, Pipeline
import lightgbm as lgb
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
                                  reg_alpha=0.5,
                                  reg_lambda=0.5,
                                  n_estimators=400))]
    )
exported_pipeline.fit(train_x, train_y)
predict = exported_pipeline.predict(test_x)

util.save(pd.DataFrame(predict), 'tpot-0.93561')
