import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR,SVC
from sklearn.linear_model import LinearRegression, ElasticNet,Ridge,Lasso,RidgeClassifierCV
from sklearn.model_selection import KFold  
from sklearn.feature_selection import SelectPercentile, f_regression
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
import copy as cp
from sklearn.preprocessing import MaxAbsScaler, Normalizer
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from imblearn.under_sampling import NearMiss ,RandomUnderSampler, NeighbourhoodCleaningRule, OneSidedSelection, AllKNN
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor
from imblearn.ensemble import EasyEnsemble 
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def modif_value(training_features, training_target, training_labels, testing_features):
    exported_pipeline = Pipeline([
        ("scaler",MaxAbsScaler()),
        ("SVR",StackingEstimator(estimator=SVC())),
        ("RidgeCV",StackingEstimator(estimator=RidgeClassifierCV())),
        ("BaggingClassifier",BaggingClassifier(base_estimator=lgb.LGBMClassifier(
                          boosting_type="GBDT",
                          num_leaves=17,
                          learning_rate=0.01,
                          feature_fraction=0.5,
                          bagging_fraction=0.5,
                          bagging_freq=5,
                          reg_alpha=0.5,
                          reg_lambda=0.5,
                          n_estimators=400),random_state=0))
    ]
    )
    exported_pipeline.fit(training_features, training_labels)
    prob = exported_pipeline.predict_proba(testing_features)
    predicts = np.zeros((prob.shape[0],))
    for i in range(prob.shape[0]):
        if prob[i,1]>0.45:
            predicts[i] = 1
        else:
            predicts[i] = -1
    pred_high_list = list(np.where(predicts==-1)[0])
    
    if len(pred_high_list)==0:
        return [],[]
    exported_pipeline = Pipeline([
    ("scaler",MaxAbsScaler()),
    ("SVR",StackingEstimator(estimator=LinearSVR(C=0.01, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.001))),
    ("RidgeCV",StackingEstimator(estimator=RidgeCV())),
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
    high_training_labels_list = list(np.where(training_labels==-1)[0])
    
    exported_pipeline.fit(training_features[high_training_labels_list], training_target[high_training_labels_list])
    high_results = exported_pipeline.predict(testing_features[pred_high_list])  
    return high_results,pred_high_list

train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")
train.pop("id")
test.pop("id")
target = train.pop("血糖")

train_x= train.as_matrix()
train_y = target.as_matrix()
test_x = test.as_matrix()

high_labels = np.zeros((train_y.shape[0],))
for i in range(train_y.shape[0]):
    if train_y[i]<6.68:
        high_labels[i] = 1
    else:
        high_labels[i] = -1
low_labels = np.zeros((train_y.shape[0],))
for i in range(train_y.shape[0]):
    if train_y[i]>3.9:
        low_labels[i] = 1
    else:
        low_labels[i] = -1
		
		
N = 5
kf = KFold(n_splits=N, random_state=42)
result_mean = 0.0
i = 0
test_preds = np.zeros((test_x.shape[0], N))
for train_index, test_index in kf.split(train_x):
    training_features, training_target = train_x[train_index], train_y[train_index]
    testing_features, testing_target = train_x[test_index], train_y[test_index]

    exported_pipeline = Pipeline([
        ("scaler", MaxAbsScaler()),
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
                                  n_estimators=400))]
    )

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)

    
    # 直接加权融合
    test_pred = exported_pipeline.predict(test_x)
    #分类，判断是否为高血糖，如果是则重新用回归模型预测
    high_results,pred_high_list = modif_value(training_features, training_target, high_labels[train_index], test_x)
    
    if len(high_results) !=0 and len(pred_high_list)!=0:
        for ii,jj in enumerate(pred_high_list):
            test_pred[jj] = high_results[ii]
    for index, value in zip(high_results, pred_high_list):
        print(index,value)
    
    test_preds[:, i] = test_pred
    i += 1

results = test_preds.mean(axis=1)
#results[313] = 15.4860937360076
#results[938] = 17.5400019823901
ouput = pd.DataFrame()
ouput[0] = results
print(ouput.describe())
print(ouput.loc[ouput[0]>8])
#ouput.to_csv("../result/Test.csv", header=None, index=False,encoding="utf-8")
results = test_preds.mean(axis=1)