import pandas as pd
from util import read_data, save, plt_encoding_error, error, normal
import matplotlib.pyplot as plt

train = pd.read_csv("../common/data/processed/train.csv")
test_X = pd.read_csv("../common/data/processed/test.csv")
test_X.pop("id")
test_X = test_X.as_matrix()

train.pop("id")
target = train.pop("血糖")
features = train.columns.tolist()

features = train.columns.tolist()
train_X = train.as_matrix()
train_Y = target.as_matrix()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=17)

params = {
    'boosting': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'learning_rate': 0.02,
    'lambda_l1': 1,
    'lambda_l2': 0.2,
    'cat_smooth': 10,
    'feature_fraction': 0.5,
    'bagging_freq': 5,
    'verbosity': -1
}

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train, feature_name=features)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, feature_name=features)

train_all = lgb.Dataset(train_X, train_Y, feature_name=features)

# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
#
# train_all = lgb.Dataset(train_X, train_Y)

# specify your configurations as a dict

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=283,
                valid_sets=lgb_eval,
                early_stopping_rounds=50)

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
error(y_test, y_pred)

# online

# predict = gbm.predict(test_X, num_iteration=gbm.best_iteration)
# data1 = pd.DataFrame(predict)
# save
# save(data1, 'lgb')

# gbm_online = lgb.train(params,
#                 train_all,
#                 num_boost_round=280)
# # predict
# predict = gbm_online.predict(test_X)
# data1 = pd.DataFrame(predict)
# # save
# save(data1, 'lgb')


plt_encoding_error()
lgb.plot_importance(gbm)
plt.show()
