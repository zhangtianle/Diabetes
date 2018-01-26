import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
color = sns.color_palette()
import lightgbm as lgb
import xgboost as xgb
import time
import datetime
import numpy as np
import pandas as pd
from dateutil.parser import parse
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
#data
train = pd.read_csv('D:/contest/diabetes/d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('D:/contest/diabetes/d_test_A_20180102.csv', encoding='gb2312')
from util import plt_encoding_error
plt_encoding_error()

def evalerror(pred, dtrain):
    labels = dtrain.get_label()
    score = mean_squared_error(labels, pred) * 0.5
    return ('mse', score)

def is_weekday(x):
    wd=x.weekday()
    return wd

def chang_sex(x):
    if str(x)=='1':
        return int(1)
    elif str(x)=='??':
        return int(2)
    else:
        return int(0)
#数据处理
together=pd.concat([train,test],axis=0)
together['体检日期']=pd.to_datetime(together['体检日期'])
together['month'] = together['体检日期'].apply(lambda x: x.month)
together['day'] = together['体检日期'].apply(lambda x: x.day)
together['num_week'] = together['体检日期'].apply(lambda x: x.isoweekday())
together['weekday'] = together['体检日期'].apply(is_weekday)
together['体检日期'] = (together['体检日期'] - parse('2017-10-09')).dt.days
# together.drop(['tijian_date'],inplace=True,axis=1)
together['性别'] = together['性别'].map({'男': 1, '女': 0,'??':2})

labels=together['血糖'][:5642]
together.drop(['血糖'],inplace=True,axis=1)
#缺失值中位数填充
together_fill=together.fillna(together.median(axis=0))

# feature
together['年龄*红细胞计数'] = together['年龄'] * together['红细胞计数']
together['中性粒细胞/淋巴细胞'] = 1.0* together['中性粒细胞%'] / together['淋巴细胞%']
together['红细胞计数^2'] = together['红细胞计数']*together['红细胞计数']
together['尿素^2'] = together['尿素'] * together['尿素']
together['肌酐^2'] = together['肌酐'] * together['肌酐']
together['尿酸^2'] = together['尿酸'] * together['尿酸']   #0.9603574006958835
together['总胆固醇^2'] = together['总胆固醇'] * together['总胆固醇'] #0.9578951256983005
together['甘油三酯^2'] = together['甘油三酯'] * together['甘油三酯'] #0.9578951256983005
together['红细胞计数'] = 1.2*together['红细胞计数']

#取训练集测试集
train_feat=together_fill[:5642]
test_feat=together_fill[5642:]
train_feat.drop('id',inplace=True,axis=1)
test_id=test_feat['id']
test_feat.drop('id',inplace=True,axis=1)



dtrain=xgb.DMatrix(train_feat,label=labels)
param={
   'colsample_bytree':0.8,
    'subsample':0.8,
    'objective':'reg:tweedie',
    # 'eval_metric':'rmse',
    'tweedie_variance_power':1.4,
    'eta':0.008,
    'max_depth':5,
    'booster':'gbtree',
    'gamma': 1,
    'silent': 1,
    'min child weight':5
}
num_rounds=10000
def tianci_loss(pred,dtrain):
    labels=dtrain.get_label()
    re=sum((pred-labels)*(pred-labels))*1.0/(len(pred)*2)
    return 'ali_s',re
# ss2=xgb.cv(param,dtrain=dtrain,early_stopping_rounds=250,num_boost_round=num_rounds,nfold=10,
#            feval=tianci_loss, maximize=False,verbose_eval=100,seed=0)

model=xgb.train(param,dtrain,num_rounds,feval=evalerror, maximize=False,evals=[(dtrain,'train')])
test=xgb.DMatrix(test_feat[train_feat.columns.values])
res=pd.DataFrame(model.predict(test))
res.to_csv(r'xgb{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),header=None,
                  index=False, float_format='%.4f')

#feature importance

feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importance')
plt.ylabel('Feature Importance Score')
plt.show()
