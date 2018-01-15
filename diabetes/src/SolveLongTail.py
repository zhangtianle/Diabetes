import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
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
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator



def get_score(df_x, df_y):
    X = df_x.as_matrix()
    Y = df_y.as_matrix()
    N = 5
    kf = KFold(n_splits=N, random_state=42)
    result_mean = 0.0

    for train_index, test_index in kf.split(X):
        training_features, training_target = X[train_index], Y[train_index]
        testing_features, testing_target = X[test_index], Y[test_index]

        exported_pipeline = Pipeline([
            ("scaler", MaxAbsScaler()),
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
        exported_pipeline.fit(training_features, training_target)
        results = exported_pipeline.predict(testing_features)

        exported_pipeline.fit(training_features, training_target)
        results = exported_pipeline.predict(testing_features)

        result_mean += np.round(mean_squared_error(testing_target, results), 5)
    result_mean /= N
    return "Mean squared error: %.5f" % (result_mean / 2)


def main():
    train = pd.read_csv("../data/processed/train.csv")
    train.pop("id")
    target = train.pop("血糖")


    base_score = get_score(train, target)
    print(base_score)

    columns = train.columns.tolist()

    index = []
    scores = []
    for column in columns:
        tmp = cp.deepcopy(train)
        if tmp[column].min() < 0:
            continue
        tmp[column] = tmp[column].apply(lambda x : np.log1p(x))
        score = get_score(tmp, target)
        print(column, score)
        if score < base_score:
            print("greater")
            index.append(column)
            scores.append(score)


    pd.set_option("max_rows", 200)
    features = pd.DataFrame({"index": index, "scores": scores})
    features = features.sort_values(by=['scores'], ascending=False)

    print(features)

if __name__ == '__main__':
    main()