import lightgbm as lgb


def set_missing(df, predict_list):
    # df.drop(['性别', '体检日期'], axis=1, inplace=True)
    for predict_feature in predict_list:

        # 原始数据分为已知和未知的
        known = df[df[predict_feature].notnull()]
        unknown = df[df[predict_feature].isnull()]

        # 训练集构造，从已知的部分构造
        y = known[predict_feature].as_matrix()
        X = known.drop(predict_feature, axis=1).as_matrix()

        # 测试集，从未知的部分构造
        test_X = unknown.drop(predict_feature, axis=1).as_matrix()

        # 用lgb模型进行训练
        predicted_feature = _model_predict(X, y, test_X)

        # 用得到的预测结果填补原缺失数据
        df.loc[(df[predict_feature].isnull()), predict_feature] = predicted_feature

    return df


def _model_predict(X, y, test_X):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'boosting': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'learning_rate': 0.015,
        #     'lambda_l1':1,
        #     'lambda_l2':0.2,
        'cat_smooth': 10,
        'feature_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': 0
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=70)

    # 用得到的模型进行未知年龄结果预测
    predicted_feature = gbm.predict(test_X, num_iteration=gbm.best_iteration)
    print("---------best_iteration: ", gbm.best_iteration)
    return predicted_feature
