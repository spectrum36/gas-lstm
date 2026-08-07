import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from priceDf import getPriceDf

df = getPriceDf(False)
series = df.filter(like="EMM_EPMR_PTE_YORD_DPG")
series = series.dropna()
series = series.reset_index(drop=True)

result = adfuller(series)

print("raw data")
print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:")
for key, value in result[4].items():
    print(f"   {key}: {value}")

dict = series.to_dict()

list = []
#list2 = []
for key, value in dict.items():
    #for i, p in value.items():
        #list2.append(p)
    list2 = value
    for i in range(len(list2) - 1):
        list.append((list2[i+1] - list2[i]) / list2[i])

series = pd.Series(list)

result = adfuller(series)

print("percent change")
print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:")
for key, value in result[4].items():
    print(f"   {key}: {value}")
