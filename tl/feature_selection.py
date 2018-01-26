import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_selection import SelectFromModel

from util import read_data, save, plt_encoding_error, error, normal
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train = pd.read_csv("../common/data/processed/train.csv")
test_X = pd.read_csv("../common/data/processed/test.csv")
test_X.pop("id")
test_X = test_X.as_matrix()

train.pop("id")
target = train.pop("血糖")

features = train.columns.tolist()

train_X = train.as_matrix()
train_Y = target.as_matrix()

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

train_X = PolynomialFeatures().fit_transform(train_X)

X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

print('Start training...')
# train
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.02,
                        colsample_bytree=0.8,
                        subsample=0.5,
                        subsample_freq=5,
                        n_estimators=200)
clf = Pipeline([
    ("scaler", StandardScaler()),
    ('feature_selection', SelectFromModel(gbm, threshold="10*mean", prefit=False)),
    ('regression', gbm)
])

clf.fit(X_train, y_train)


print('Start predicting...')
# predict
y_pred = clf.predict(X_test)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) * 0.5)

# # feature importances
# print('Feature importances:', list(gbm.feature_importances_))
#
# plt_encoding_error()
# lgb.plot_importance(gbm)
# plt.show()
