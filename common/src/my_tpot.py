from sys import path

path.append('../')

from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# 读取数据

train_x = pd.read_csv("../data/processed/train.csv")

train_x.pop("id")
target = train_x.pop("血糖")

train_X = train_x.as_matrix()
train_Y = target.as_matrix()


X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

tpot = TPOTRegressor(generations=60, population_size=80, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
