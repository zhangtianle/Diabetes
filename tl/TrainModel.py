class TrainModel:

    lgb_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'mse'},
        'num_leaves': 31,
        'learning_rate': 0.02,
        'lambda_l1': 1,
        # 'lambda_l2': 1,
        'cat_smooth': 10,
        'feature_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    def __init__(self, train_X, train_Y, test_X):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X

    def get_model_params(self, name):
        return name
