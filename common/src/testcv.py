import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb

# 读数据
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")
train.pop("id")
test.pop("id")
target = train.pop("血糖")

train_x = train.as_matrix()
train_y = target.as_matrix()
test_x = test.as_matrix()


N = 5
kf = KFold(n_splits=N, random_state=42, shuffle=True)
i = 0
result_mean = 0.0
for train_index, test_index in kf.split(train_x):
    training_features, training_target = train_x[train_index], train_y[train_index]
    testing_features, testing_target = train_x[test_index], train_y[test_index]

    ############  lgb ##############
    lgb_train = lgb.Dataset(training_features, training_target)
    lgb_eval = lgb.Dataset(testing_features, testing_target)
    params = {
        'boosting': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': 31,
        'learning_rate': 0.02,
        'lambda_l1': 1,
        'lambda_l2': 0.5,
        'cat_smooth': 10,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1
    }
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets=lgb_eval,
                    verbose_eval=False,
                    early_stopping_rounds=50)

    # predict
    testing_results = gbm.predict(testing_features, num_iteration=gbm.best_iteration)
    result_mean += np.round(mean_squared_error(testing_target, testing_results), 5)
    print('CV_ROUND (', i, ') mse -> ', np.round(mean_squared_error(testing_target, testing_results), 5) / 2)
    i += 1

# 线下CV
result_mean /= N
print("offline CV Mean squared error: %.5f" % (result_mean / 2))

