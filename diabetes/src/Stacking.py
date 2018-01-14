import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
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
from sklearn.preprocessing import MaxAbsScaler, Normalizer
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from imblearn.under_sampling import NearMiss
from xgboost import XGBRegressor
from imblearn.ensemble import EasyEnsemble


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit(self, X, y):
        return

    def fit_predict(self, X, y, T):
        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print ("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res




def main():
    train = pd.read_csv("../data/processed/train.csv")
    train.pop("id")
    target = train.pop("血糖")
    train_x = train.as_matrix()
    train_y = target.as_matrix()



    N = 5
    kf = KFold(n_splits=N, random_state=42)
    result_mean = 0.0
    for train_index, test_index in kf.split(train_x):
        training_features, training_target = train_x[train_index], train_y[train_index]
        testing_features, testing_target = train_x[test_index], train_y[test_index]

        scaler = MaxAbsScaler()
        scaler.fit(training_features)

        training_features = scaler.transform(training_features)
        testing_features = scaler.transform(testing_features)

        knn = KNeighborsRegressor(n_neighbors=9, p=1, weights="distance")
        linear_svr = LinearSVR(C=0.01, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.001)
        ridge = RidgeCV()
        gbm = lgb.LGBMRegressor(objective='regression',
                                boosting_type="GBDT",
                                num_leaves=17,
                                learning_rate=0.01,
                                feature_fraction=0.5,
                                bagging_fraction=0.5,
                                bagging_freq=5,
                                reg_alpha=0.1,
                                reg_lambda=0.5,
                                n_estimators=400)
        lr = LinearRegression()
        en = ElasticNetCV(l1_ratio=0.1, tol=0.01)
        xgb = XGBRegressor(learning_rate=0.01, max_depth=8, min_child_weight=8, n_estimators=100, nthread=1,
                      subsample=0.15000000000000002)
        et = ExtraTreesRegressor(bootstrap=True, max_features=0.35000000000000003, min_samples_leaf=3, min_samples_split=12, n_estimators=100)

        rf = RandomForestRegressor(bootstrap=True, max_features=0.9500000000000001, min_samples_leaf=15, min_samples_split=6,
                              n_estimators=100)


        exported_pipeline0 = make_pipeline(
            StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.1, tol=0.01)),
            ExtraTreesRegressor(bootstrap=False, max_features=0.2, min_samples_leaf=3, min_samples_split=16,
                                n_estimators=100)
        )


        exported_pipeline1 = Pipeline([
            ("SVR", StackingEstimator(estimator=LinearSVR(C=0.01, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.001))),
            ("RidgeCV", StackingEstimator(estimator=RidgeCV())),
            ("LGB", lgb.LGBMRegressor(objective='regression',
                                      boosting_type="GBDT",
                                      num_leaves=17,
                                      learning_rate=0.01,
                                      feature_fraction=0.5,
                                      bagging_fraction=0.5,
                                      bagging_freq=5,
                                      reg_alpha=0.1,
                                      reg_lambda=0.5,
                                      n_estimators=400))]
        )

        exported_pipeline2 = make_pipeline(
            StackingEstimator(estimator=RidgeCV()),
            StackingEstimator( estimator=XGBRegressor(learning_rate=0.01, max_depth=8, min_child_weight=8, n_estimators=100, nthread=1,subsample=0.15000000000000002)),
            ExtraTreesRegressor(bootstrap=True, max_features=0.35000000000000003, min_samples_leaf=3, min_samples_split=12, n_estimators=100)
        )


        stack = Ensemble(n_splits=10,
                         stacker=LinearRegression(),
                         base_models=(rf, knn,lr,linear_svr, ridge,en, gbm, xgb, et, exported_pipeline0, exported_pipeline1, exported_pipeline2))

        results = stack.fit_predict(X=training_features, y=training_target, T=testing_features)
        result_mean += np.round(mean_squared_error(testing_target, results), 5)

    result_mean /= (N)
    print("Mean squared error: %.5f" % (result_mean / 2))
if __name__ == '__main__':
    main()