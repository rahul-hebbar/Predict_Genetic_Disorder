import pandas as pd
import numpy as np

df = pd.read_csv("./dataset/train.csv")

df.replace({True:1,False:0},inplace=True)

obj_arr = []
oth_arr = []
for i in df:
	if df[i].dtype == "object":
		obj_arr.append(i)
	else:
		oth_arr.append(i)

col = oth_arr + obj_arr
df = df[col]
print(df.info())