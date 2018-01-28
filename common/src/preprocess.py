from Feature import Feature
from util import read_data, save_data
import pandas as pd

# test_tpye = 'A' OR 'B'
total, target, train_id, test_id, total_ID = read_data(test_type='B')

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

######## start 读取多项式特征 ######
# TODO 重新生成Test_B的多项式特征文件
# poly = pd.read_csv("../data/poly_feature/poly_feature _90.csv")
# poly = poly.iloc[:,0:4]
#
# poly = poly.reset_index(drop=True)
# total = total.reset_index(drop=True)
#
# total = pd.concat([total,poly],axis=1)
########  end 多项式特征  ########

train = total[0:len(train_id)]
train = pd.concat([train_id, train, target], axis=1)

test = total[len(train_id):].reset_index(drop=True)
test = pd.concat([test_id, test], axis=1)

save_data([train, test], ['../data/processed/train.csv', '../data/processed/test.csv'])
