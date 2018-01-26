import pandas as pd

keda = pd.read_csv("xgb20180119_162157_off0.32021.csv", header=None)
xidian = pd.read_csv("1.18-LiuYuJIA.csv", header=None)

new_result = keda.loc[:, 0] * 0.6 + xidian.loc[:, 0] * 0.4
new_result.to_csv("merge_0.81xx_0.8151.csv", header=None, index=False, encoding="utf-8")
print(keda.shape)