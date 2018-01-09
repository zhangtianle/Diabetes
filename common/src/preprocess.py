from Feature import Feature
from util import read_data, save_data
import pandas as pd

total, target, train_id, test_id, total_ID = read_data()

feature = Feature(total)
feature.drop_feature(['date'])
feature.fix_missing()
feature.long_tail()
feature.statistics()
feature.combine_feature()
feature.drop_feature(['尿素std'])
feature.one_hot(['gender'])

total = feature.get_train()


train = total[0:len(train_id)]
train = pd.concat([train_id, train, target], axis=1)

test = total[len(train_id):]
test = pd.concat([test_id, test], axis=1)

save_data([train, test], ['../data/processed/train.csv', '../data/processed/test.csv'])
