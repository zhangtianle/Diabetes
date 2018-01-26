import numpy as np
from catboost import Pool, CatBoostRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from util import save

train = pd.read_csv("../common/data/processed/train.csv")
test = pd.read_csv("../common/data/processed/test.csv")
train.pop("id")
test.pop("id")
target = train.pop("血糖")

train_x = train.as_matrix()
train_y = target.as_matrix()
test_x = test.as_matrix()

N = 5
kf = KFold(n_splits=N, random_state=42)
i = 0
result_mean = 0.0
test_preds = np.zeros((test_x.shape[0], N))
# 构造一个用于存储异常值的字典,存储方式：id:[].例如{938:[14,14,14],314:[13]}
outlier = {}
for train_index, test_index in kf.split(train_x):
    training_features, training_target = train_x[train_index], train_y[train_index]
    testing_features, testing_target = train_x[test_index], train_y[test_index]


    ########## catboost #############
    # train_pool = Pool(training_features, training_target)
    # eval_pool = Pool(testing_features, testing_target)
    #
    # # specify the training parameters
    # model = CatBoostRegressor(iterations=3000,
    #                           depth=6,
    #                           learning_rate=0.005,
    #                           loss_function='RMSE',
    #                           l2_leaf_reg=1,
    #                           rsm=0.5,
    #                           border_count=128,
    #                           logging_level='Silent'
    #                           )
    # # train the model
    # model.fit(train_pool, eval_set=eval_pool)
    # # make the prediction using the resulting model
    # testing_results = model.predict(testing_features)
    # result_mean += np.round(mean_squared_error(testing_target, testing_results), 5)
    # print('CV_ROUND (', i, ') mse -> ', np.round(mean_squared_error(testing_target, testing_results), 5) / 2)

    ############  lgb ##############
    lgb_train = lgb.Dataset(training_features, training_target)
    lgb_eval = lgb.Dataset(testing_features, testing_target)
    params = {
        'boosting': 'gbdt',
        'application': 'regression',
        'metric': 'mse',
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'learning_rate': 0.02,
        'lambda_l1': 5,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.8,
        'bagging_freq': 8,
        'verbosity': 0
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
    print(gbm.best_iteration, ' CV_ROUND (', i, ') mse -> ', np.round(mean_squared_error(testing_target, testing_results), 5) / 2)

    test_pred = gbm.predict(test_x)
    test_preds[:, i] = test_pred
    i += 1

result_mean /= N
print("offline CV Mean squared error: %.5f" % (result_mean / 2))

results = test_preds.mean(axis=1)
ouput = pd.DataFrame()
ouput[0] = results
# save(ouput, 'lgb')
