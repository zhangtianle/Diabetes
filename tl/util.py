from sys import path

path.append('../')
import configparser
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error


def plt_encoding_error():
    # coding:utf-8
    import matplotlib
    matplotlib.use('qt4agg')
    # 指定默认字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    # 解决负号'-'显示为方块的问题
    matplotlib.rcParams['axes.unicode_minus'] = False


def get_url():
    conf = configparser.ConfigParser()
    conf.read("./dia.conf")

    root_dir = conf.get("local", "data_dir")
    train_url = conf.get("local", "train_url")
    feature_url = conf.get("local", "feature_url")
    return root_dir, train_url, feature_url


def read_data():
    root_dir, train_url, feature_url = get_url()

    train = pd.read_csv(root_dir + 'd_train_20180102.csv')
    test_A = pd.read_csv(root_dir + 'd_test_A_20180102.csv')
    sample = pd.read_csv(root_dir + 'd_sample_20180102.csv')
    return train, test_A, sample


def error(y_train, predict):
    print("1/2 Mean squared error: %.6f" % (mean_squared_error(y_train, predict) / 2))


def save(result, name):
    now = datetime.datetime.now()
    my_time = now.strftime('%m_%d_%H_%M')
    result.to_csv("result/" + name + '_' + my_time + '.csv', header=None, index=False, encoding="utf-8")


if __name__ == '__main__':
    pass