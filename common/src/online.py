import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import  Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifierCV,LogisticRegression
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from sklearn.ensemble import BaggingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor,XGBClassifier
from util import save
from sklearn.ensemble import GradientBoostingRegressor

#自定义回归模型
'''
融合GBDT和LinearRegressor
利用GBDT生成新特征，再用新特征one-hot编码，最后利用LinearRegressor做回归预测。
'''
class myStackingFeaturesRegressor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.estimator = None
        self.lgb = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                    n_estimators=100,
                                    learning_rate=0.01,
                                    max_depth=8,
                                    subsample=0.8,
                                    min_samples_split=9,
                                    random_state=1,
                                    max_leaf_nodes=10)
        self.grd_enc = OneHotEncoder()
        self.lr = RidgeCV()
        self.classes_ = [-1,1]
    def fit(self, X, y=None, **fit_params):
        self.lgb.fit(X, y)
        self.grd_enc.fit(self.lgb.apply(X))
        self.lr.fit(self.grd_enc.transform(self.lgb.apply(X)), y)
    def predict(self, X):
        return self.lr.predict(self.grd_enc.transform(self.lgb.apply(X)))



#自定义分类器
'''
融合lgb和LR
利用lgb生成新特征，再用新特征one-hot编码，最后利用LR做分类预测。
'''
class myStackingFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.estimator = None
        self.lgb = lgb.LGBMClassifier(boosting_type="GBDT",
                                      num_leaves=31,
                                      learning_rate=0.02,
                                      colsample_bytree=0.8,
                                      subsample=0.5,
                                      subsample_freq=5,
                                      n_estimators=200,
                                      random_state=1801,
                                      n_jobs=-1)
        self.grd_enc = OneHotEncoder()
        self.lr = LogisticRegression()
        self.classes_ = [-1,1]
    def fit(self, X, y=None, **fit_params):
        self.lgb.fit(X, y)
        self.grd_enc.fit(self.lgb.apply(X))
        self.lr.fit(self.grd_enc.transform(self.lgb.apply(X)), y)
    def predict_proba(self, X):
        return self.lr.predict_proba(self.grd_enc.transform(self.lgb.apply(X)))
    def predict(self, X):
        return self.lr.predict(self.grd_enc.transform(self.lgb.apply(X)))


def modif_value(training_features, training_labels, testing_features, X, Y):
    #构造分类器
    exported_pipeline = Pipeline([
        ("scaler", MaxAbsScaler()),
        ("SVR", StackingEstimator(estimator=SVC())),
        ("RidgeCV", StackingEstimator(estimator=RidgeClassifierCV())),
        ("BaggingClassifier", BaggingClassifier(base_estimator=myStackingFeatures(), random_state=201801))])
    exported_pipeline.fit(training_features, training_labels)
    prob = exported_pipeline.predict_proba(testing_features)
    predicts = np.zeros((prob.shape[0],))
    for i in range(prob.shape[0]):
        if prob[i, 1] > 0.5:
            predicts[i] = 1
        else:
            predicts[i] = -1
    negative_pred_list = list(np.where(predicts == -1)[0])
    #预测异常点
    negative_results = None
    if len(negative_pred_list) == 0:
        negative_results = []
    else:
        exported_pipeline = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("SVR", StackingEstimator(
                estimator=LinearSVR(C=0.01, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.001))),
            ("RidgeCV", StackingEstimator(estimator=RidgeCV())),
            # ("LGB", lgb.LGBMRegressor(objective='regression',
            #                           boosting_type="GBDT",
            #                           num_leaves=31,
            #                           learning_rate=0.01,
            #                           feature_fraction=0.5,
            #                           bagging_fraction=0.5,
            #                           bagging_freq=5,
            #                           n_estimators=400))
            ("XGB", myStackingFeaturesRegressor())
        ]
        )
        exported_pipeline.fit(X, Y)
        negative_results = exported_pipeline.predict(testing_features[negative_pred_list])


    return negative_results, negative_pred_list
def main():
    #读数据
    train = pd.read_csv("../data/processed/train.csv")
    test = pd.read_csv("../data/processed/test.csv")
    train.pop("id")
    test.pop("id")
    target = train.pop("血糖")

    train_x = train.as_matrix()
    train_y = target.as_matrix()
    test_x = test.as_matrix()

    high_labels = np.zeros((train_y.shape[0],))
    for i in range(train_y.shape[0]):
        if train_y[i] < 11.2:  #训练集的高值判断
            high_labels[i] = 1
        else:
            high_labels[i] = -1
    #预测结果取5次平均
    N = 5
    kf = KFold(n_splits=N, random_state=42)
    i = 0
    result_mean = 0.0
    test_preds = np.zeros((test_x.shape[0], N))
    #构造一个用于存储异常值的字典,存储方式：id:[].例如{938:[14,14,14],314:[13]}
    outlier = {}
    for train_index, test_index in kf.split(train_x):
        training_features, training_target = train_x[train_index], train_y[train_index]
        testing_features, testing_target = train_x[test_index], train_y[test_index]
        #构造模型，预测血糖值
        exported_pipeline = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("SVR", StackingEstimator(estimator=LinearSVR(C=0.01, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.001))),
            ("RidgeCV", StackingEstimator(estimator=RidgeCV())),
            # ("LGB", StackingEstimator(estimator=lgb.LGBMRegressor(objective='regression',
            #                           boosting_type="GBDT",
            #                           num_leaves=31,
            #                           learning_rate=0.01,
            #                           feature_fraction=0.5,
            #                           bagging_fraction=0.5,
            #                           bagging_freq=5,
            #                           n_estimators=400))),
            ("XGB", XGBRegressor(max_depth=8,
                                 n_estimators=100,
                                 colsample_bytree=0.8,
                                 subsample=0.8,
                                 tweedie_variance_power=1.4,
                                 eta=0.01,
                                 booster="gbtree",
                                 random_state=1015,
                                 gamma=1,
                                 silent=1,
                                 min_child_weight=5,
                                 objective="reg:tweedie",
                                 n_jobs=-1))]
        )

        exported_pipeline.fit(training_features, training_target)

        test_pred = exported_pipeline.predict(test_x)

        ############  lgb ##############
        # lgb_train = lgb.Dataset(training_features, training_target)
        # lgb_eval = lgb.Dataset(testing_features, testing_target)
        # params = {
        #     'boosting': 'gbdt',
        #     'objective': 'regression',
        #     'metric': 'mse',
        #     'num_leaves': 31,
        #     # 'min_data_in_leaf': 20,
        #     'learning_rate': 0.02,
        #     'lambda_l1': 1,
        #     'lambda_l2': 0.5,
        #     'cat_smooth': 10,
        #     'feature_fraction': 0.5,
        #     'bagging_fraction': 0.8,
        #     'bagging_freq': 3,
        #     'verbosity': -1
        # }
        # gbm = lgb.train(params,
        #                 lgb_train,
        #                 num_boost_round=300,
        #                 valid_sets=lgb_eval,
        #                 verbose_eval=False,
        #                 early_stopping_rounds=50)
        #
        # # predict
        # lgb_testing_results = gbm.predict(testing_features, num_iteration=gbm.best_iteration)
        # lgb_test_pred = gbm.predict(test_x)
        #
        # lgb_per = 0.4

        ############## end lgb ##########################



        #预测异常值
        high_results, pred_high_list = modif_value(training_features, high_labels[train_index],
                                                         test_x, train_x[np.where(high_labels == -1)[0]],
                                                         train_y[np.where(high_labels == -1)[0]])
        #存储异常值
        if len(high_results) != 0 and len(pred_high_list) != 0:
            for ii, jj in enumerate(pred_high_list):
                if jj not in outlier:
                    outlier[jj] = []
                outlier[jj].append(high_results[ii])
        for index, value in zip(high_results, pred_high_list):
            print(index, value)

        # 线下CV
        testing_results = exported_pipeline.predict(testing_features)

        # # lgb
        # testing_results = (1 - lgb_per) * testing_results + lgb_per * lgb_testing_results

        # 改值
        cv_high_results, cv_pred_high_list = modif_value(training_features, high_labels[train_index],
                                                         testing_features, train_x[np.where(high_labels == -1)[0]],
                                                         train_y[np.where(high_labels == -1)[0]])

        if len(cv_high_results) != 0 and len(cv_pred_high_list) != 0:
            for ii, jj in enumerate(cv_pred_high_list):
                testing_results[jj] = cv_high_results[ii]

        result_mean += np.round(mean_squared_error(testing_target, testing_results), 5)
        print('CV_ROUND (', i, ') mse -> ', np.round(mean_squared_error(testing_target, testing_results), 5) / 2)

        ############ lgb
        # test_preds[:, i] = (1 - lgb_per) * test_pred + lgb_per * lgb_test_pred
        test_preds[:, i] = test_pred
        i += 1
    results = test_preds.mean(axis=1)


    #修改异常值
    for index in outlier:
        print(index, outlier[index])
        results[index] = max(outlier[index])



    # 线下CV
    result_mean /= N
    print("offline CV Mean squared error: %.5f" % (result_mean / 2))

    ouput = pd.DataFrame()
    ouput[0] = results
    #ouput.to_csv("../result/1.25-WQX-PolyFeatures.csv", header=None, index=False, encoding="utf-8")
    # ouput.to_csv(r'../result/test{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
    #            header=None,index=False, float_format='%.4f')
    save(ouput, 'xgb_class_DaPaiCHong')
    print(ouput.describe())
    print(ouput.loc[ouput[0] > 8])

if __name__ == '__main__':
    main()