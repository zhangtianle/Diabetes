from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
if __name__ == '__main__':
    train = pd.DataFrame(pd.read_csv("../data/processed/train.csv"))
    target = train.pop("血糖")

    X = train.as_matrix()
    Y = target.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=0.75, test_size=0.25)
    tpot = TPOTRegressor(generations=50, population_size=100, cv=5, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export("../TPOT-Mode/tpot_boston_pipeline_{}_{}_{}_{}.py".format(datetime.now().month, datetime.now().day, datetime.now().hour,
                                                  datetime.now().minute))
