import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from sklearn.pipeline import Pipeline
from util import save
import math


train = pd.read_csv("../data/processed/train.csv")
test_X = pd.read_csv("../data/processed/test.csv")
test_X.pop("id")
test_X = test_X.as_matrix()

train.pop("id")
target = train.pop("血糖")

X = train.as_matrix()
Y = target.as_matrix()

# 多项式特征
# from sklearn.preprocessing import PolynomialFeatures
# X = PolynomialFeatures().fit_transform(X)
# test_X = PolynomialFeatures().fit_transform(test_X)

N = 5
kf = KFold(n_splits=N, shuffle=True, random_state=201801)
# kf = KFold(n_splits=N, random_state=42)
result_mean = 0.0
i = 0
test_preds = np.zeros((test_X.shape[0], N))
for train_index, test_index in kf.split(X):
    training_features, training_target = X[train_index], Y[train_index]
    testing_features, testing_target = X[test_index], Y[test_index]

    exported_pipeline = Pipeline([
        # ("scaler", MaxAbsScaler()),
        ("SVR", StackingEstimator(
            estimator=LinearSVR(C=0.01, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.001))),
        ("RidgeCV", StackingEstimator(estimator=RidgeCV())),
        ("LGB", lgb.LGBMRegressor(objective='regression',
                                  boosting_type="GBDT",
                                  num_leaves=31,
                                  learning_rate=0.01,
                                  feature_fraction=0.5,
                                  bagging_fraction=0.5,
                                  bagging_freq=5,
                                  reg_alpha=0.5,
                                  reg_lambda=0.5,
                                  n_estimators=400))]
    )

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)

    clf = SVR(C=12, epsilon=0.02, gamma=0.02)
    clf.fit(training_features, training_target)
    results_clf = clf.predict(testing_features)
    results = (0.9 * results + 0.1 * results_clf)

    # 直接加权融合
    test_pred = exported_pipeline.predict(test_X)
    test_pred_clf = clf.predict(test_X)
    test_preds[:, i] = 0.9 * test_pred + 0.1 * test_pred_clf

    i += 1
    result_mean += np.round(mean_squared_error(testing_target, results), 5)
    print(np.round(mean_squared_error(testing_target, results), 5) / 2)
result_mean /= N
print("Mean squared error: %.5f" % (result_mean / 2))

submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
save(submission, 'tpop_kfold_True_201801_mean_0.95579')