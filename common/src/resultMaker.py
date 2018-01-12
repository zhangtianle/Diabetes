import pandas as pd

result = pd.read_csv("../result/tpop_kfold_True_201801_96643_01_11_10_10.csv", header=None)

print(result.head())

result[0] = result[0].apply(lambda x: 1.7 * x if x > 7 else x)

result.to_csv("../result/xx.csv", header=None, index=False, encoding="utf-8")
print(result.head())
