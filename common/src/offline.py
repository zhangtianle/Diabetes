import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from sklearn.pipeline import Pipeline
from util import save


train = pd.read_csv("../data/processed/train.csv")
test_X = pd.read_csv("../data/processed/test.csv")
test_X.pop("id")
test_X = test_X.as_matrix()

train.pop("id")
target = train.pop("血糖")

X = train.as_matrix()
Y = target.as_matrix()

N = 5
kf = KFold(n_splits=N, random_state=42)
result_mean = 0.0
i = 0
test_preds = np.zeros((test_X.shape[0], N))
for train_index, test_index in kf.split(X):
    training_features, training_target = X[train_index], Y[train_index]
    testing_features, testing_target = X[test_index], Y[test_index]

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

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)

    test_preds[:, i] = exported_pipeline.predict(test_X)
    i += 1
    result_mean += np.round(mean_squared_error(testing_target, results), 5)
    print(np.round(mean_squared_error(testing_target, results), 5))
result_mean /= N
print("Mean squared error: %.5f" % (result_mean / 2))

submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
save(submission, 'tpop_kfold_0.93561')