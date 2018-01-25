from Feature import Feature
from util import read_data, save_data
import pandas as pd

total, target, train_id, test_id, total_ID = read_data()

feature = Feature(total)
feature.week_day()
# feature.missing_num()
#feature.normal_value()
feature.fix_missing()
feature.long_tail()
feature.statistics()
feature.combine_feature()
#feature.drop_feature(['尿素std', '红细胞平均血红蛋白浓度std', '红细胞平均血红蛋白浓度avg', '甘油三酯avg', '乙肝核心抗体'])
feature.one_hot(['gender'])
total = feature.get_train()

#读取多项式特征
poly = pd.read_csv("../data/poly_feature/poly_feature _90.csv")
poly = poly.iloc[:,0:5]
poly = poly.reset_index(drop=True)
#
total = total.reset_index(drop=True)
total = pd.concat([total,poly],axis=1)

train = total[0:len(train_id)]
train = pd.concat([train_id, train, target], axis=1)

test = total[len(train_id):]
test = pd.concat([test_id, test], axis=1)

save_data([train, test], ['../data/processed/train.csv', '../data/processed/test.csv'])
