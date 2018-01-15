from sys import path

path.append('../')
import configparser
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error
import math


def plt_encoding_error():
    # coding:utf-8
    import matplotlib
    # matplotlib.use('qt4agg')
    # 指定默认字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    # 解决负号'-'显示为方块的问题
    matplotlib.rcParams['axes.unicode_minus'] = False


def normal(feature, value, height=1):
    """
    Determine whether the value is normal or missing
    :param feature:
    :param value: feature's value
    :param height: 1 means over the value is abnormal, 0 means under the value is abnormal
    :return: if normal return 0, abnormal return 1, missing return -1
    """
    if math.isnan(feature):
        return -1
    elif (height == 1 and feature > value) or (height == 0 and feature < value):
        return 1
    else:
        return 0


def get_url():
    conf = configparser.ConfigParser()
    conf.read("./dia.conf")

    root_dir = conf.get("local", "data_dir")
    train_url = conf.get("local", "train_url")
    feature_url = conf.get("local", "feature_url")
    return root_dir, train_url, feature_url


def read_data():
    train = pd.read_csv("../data/raw/d_train.csv")
    test = pd.read_csv("../data/raw/d_test_A.csv")

    # important_feature = ['*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯',
    #                      '总胆固醇', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体',
    #                      '乙肝核心抗体']
    # trian_imp = train.loc[:, important_feature]
    # row_index = trian_imp[trian_imp.T.count() == 0].index
    # train.drop(row_index, inplace=True)

    test_id = test.pop("id")
    train_id = train.pop("id")

    total_ID = pd.concat([train_id, test_id])

    target = train.pop("血糖")
    total = pd.concat([train, test])
    return total, target, train_id, test_id, total_ID


def save_data(data, url):
    for d, u in zip(data, url):
        d.to_csv(u, index=False, encoding="utf8")


def error(y_train, predict):
    result = mean_squared_error(y_train, predict) / 2
    print("1/2 Mean squared error: %.6f" % result)
    return result


def save(result, name):
    now = datetime.datetime.now()
    my_time = now.strftime('%m_%d_%H_%M')
    result.to_csv("../result/" + name + '_' + my_time + '.csv', header=None, index=False, encoding="utf-8")


if __name__ == '__main__':
    pass
